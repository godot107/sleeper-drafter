"""Browser dashboard.

Runs alongside the terminal UI rather than replacing it: one window for the
board and the charts, one for the CLI, one for the Sleeper app itself.

A single background thread owns polling and recomputation and publishes an
immutable snapshot; every callback reads that snapshot. The browser never
triggers a Sleeper request, so opening a second tab costs nothing and a slow
network cannot wedge the UI.

Freshness is treated as first-class. A draft dashboard that silently stops
updating is worse than no dashboard, so the header always states when data last
changed and turns amber, then red, as that ages.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from dash import Dash, Input, Output, dash_table, dcc, html

from .charts import decision_scatter, outlook_table, scarcity_bar, standings_bar, value_cliff
from .config import settings
from .optimizer import rank_with_lookahead
from .rating import rate_teams
from .state import DraftState
from .theme import FONT, palette

logger = logging.getLogger(__name__)

POSITIONS = ("QB", "RB", "WR", "TE", "K", "DEF")
RISKS = ("steady", "neutral", "volatile", "unknown")
ROLES = ("workhorse", "starter", "committee", "backup", "unknown")


@dataclass
class Snapshot:
    """Everything the UI needs, computed once per poll."""

    ranked: pd.DataFrame = field(default_factory=pd.DataFrame)
    board: pd.DataFrame = field(default_factory=pd.DataFrame)
    state: object = None
    rated: pd.DataFrame = field(default_factory=pd.DataFrame)
    window: list[int] = field(default_factory=list)
    expected: dict[str, float] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)
    roster: list[dict] = field(default_factory=list)
    changed_at: float = 0.0
    polled_at: float = 0.0
    age_s: float | None = None
    error: str | None = None


class Publisher:
    """Thread-safe holder for the latest snapshot."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot = Snapshot()

    def get(self) -> Snapshot:
        with self._lock:
            return self._snapshot

    def set(self, snapshot: Snapshot) -> None:
        with self._lock:
            self._snapshot = snapshot


def build_snapshot(
    state: DraftState,
    board: pd.DataFrame,
    my_slot: int,
    *,
    previous: Snapshot | None = None,
    age_s: float | None = None,
    error: str | None = None,
) -> Snapshot:
    from .opponent_model import selection_probs

    ranked, window = rank_with_lookahead(state, board, my_slot)
    rated = rate_teams(state, board)

    expected: dict[str, float] = {}
    if window and not ranked.empty:
        positions = ranked["pos"].to_numpy()
        for pick_no in window:
            probs = selection_probs(ranked, state.team_at(pick_no), state.schema, pick_no)
            for pos in POSITIONS:
                expected[pos] = expected.get(pos, 0.0) + float(probs[positions == pos].sum())

    my_roster_id = state.slot_to_roster.get(my_slot, my_slot)
    team = state.teams_by_roster.get(my_roster_id)
    indexed = board.set_index("player_id")
    roster_rows = []
    for i, pid in enumerate(team.roster if team else [], 1):
        if pid in indexed.index:
            row = indexed.loc[pid]
            roster_rows.append({
                "round": i, "name": row["name"], "pos": row["pos"],
                "proj": round(float(row["proj_pts"])), "risk": row.get("risk", ""),
            })

    pick = state.current_pick_no
    my_row = rated[rated["roster_id"] == my_roster_id] if not rated.empty else pd.DataFrame()
    info = {
        "name": (state.draft.get("metadata") or {}).get("name") or state.draft_id,
        "pick": pick,
        "round": (pick - 1) // state.teams + 1,
        "in_round": (pick - 1) % state.teams + 1,
        "teams": state.teams,
        "rounds": state.rounds,
        "scoring": state.scoring,
        "my_slot": my_slot,
        "my_roster": my_roster_id,
        "is_my_turn": state.is_my_turn(my_slot),
        "picks_until_turn": len(window),
        "grade": (my_row["grade"].iloc[0] if not my_row.empty else "—"),
        "rank": (int(my_row.index[0]) + 1 if not my_row.empty else None),
        "holes": (int(my_row["holes"].iloc[0]) if not my_row.empty else 0),
    }

    now = time.time()
    changed = now
    if previous is not None and previous.info.get("pick") == pick and previous.changed_at:
        changed = previous.changed_at

    return Snapshot(
        ranked=ranked, board=board, state=state, rated=rated, window=window, expected=expected,
        info=info, roster=roster_rows, changed_at=changed, polled_at=now,
        age_s=age_s, error=error,
    )


# ------------------------------------------------------------------- pollers

def start_live_poller(
    publisher: Publisher, client, draft_id: str, state: DraftState,
    board: pd.DataFrame, my_slot: int, stop: threading.Event,
) -> threading.Thread:
    def loop() -> None:
        while not stop.is_set():
            error = None
            age = None
            try:
                state.ingest(client.picks(draft_id))
                age = client.last_age_s
            except Exception as exc:  # never let the dashboard thread die
                logger.warning("poll failed: %s", exc)
                error = str(exc)
            publisher.set(build_snapshot(
                state, board, my_slot,
                previous=publisher.get(), age_s=age, error=error,
            ))
            near = publisher.get().info.get("picks_until_turn", 99)
            stop.wait(settings.poll_interval_near_s if near <= settings.poll_near_threshold
                      else settings.poll_interval_s)

    thread = threading.Thread(target=loop, name="sleeper-poll", daemon=True)
    thread.start()
    return thread


def start_mock_poller(
    publisher: Publisher, state: DraftState, board: pd.DataFrame,
    my_slot: int, stop: threading.Event, *, seconds_per_pick: float = 3.0,
    seed: int = 0,
) -> threading.Thread:
    """Advance a simulated draft on a timer, so the dashboard can be exercised
    end to end without a live draft to join."""
    from .opponent_model import sample_pick

    rng = np.random.default_rng(seed)

    def loop() -> None:
        publisher.set(build_snapshot(state, board, my_slot))
        while not stop.is_set() and state.current_pick_no <= state.total_picks:
            pick_no = state.current_pick_no
            available = board[~board["player_id"].isin(state.drafted)]
            if available.empty:
                break
            if state.slot_at(pick_no) == my_slot:
                ranked, _ = rank_with_lookahead(state, board, my_slot)
                player_id = str(ranked.iloc[0]["player_id"])
            else:
                player_id = sample_pick(
                    available, state.team_at(pick_no), state.schema, pick_no, rng
                )
            state.ingest([{
                "pick_no": pick_no, "round": (pick_no - 1) // state.teams + 1,
                "draft_slot": state.slot_at(pick_no),
                "roster_id": state.roster_at(pick_no), "player_id": player_id,
            }])
            publisher.set(build_snapshot(
                state, board, my_slot, previous=publisher.get()
            ))
            stop.wait(seconds_per_pick)

    thread = threading.Thread(target=loop, name="mock-draft", daemon=True)
    thread.start()
    return thread


# ---------------------------------------------------------------------- app

INDEX = """<!DOCTYPE html><html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
  :root {
    --surface:#fcfcfb; --plane:#f9f9f7; --text:#0b0b0b; --text2:#52514e;
    --muted:#898781; --border:rgba(11,11,11,0.10); --grid:#e1e0d9;
    --accent:#2a78d6; --you:#eb6834;
    --good:#0ca30c; --warning:#fab219; --critical:#d03b3b;
  }
  [data-theme="dark"] {
    --surface:#1a1a19; --plane:#0d0d0d; --text:#ffffff; --text2:#c3c2b7;
    --muted:#898781; --border:rgba(255,255,255,0.10); --grid:#2c2c2a;
    --accent:#3987e5; --you:#d95926;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--plane); color:var(--text);
         font-family:system-ui,-apple-system,"Segoe UI",sans-serif; font-size:14px; }
  .wrap { max-width:1680px; margin:0 auto; padding:16px 20px 48px; }
  .bar { display:flex; align-items:center; gap:18px; flex-wrap:wrap;
         background:var(--surface); border:1px solid var(--border);
         border-radius:10px; padding:12px 16px; margin-bottom:14px; }
  .tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
           gap:12px; margin-bottom:14px; }
  .tile { background:var(--surface); border:1px solid var(--border);
          border-radius:10px; padding:12px 14px; }
  .tile .k { color:var(--muted); font-size:11px; text-transform:uppercase;
             letter-spacing:.05em; margin-bottom:6px; }
  .tile .v { font-size:24px; font-weight:600; color:var(--text); line-height:1.15; }
  .tile .s { color:var(--text2); font-size:12px; margin-top:3px; }
  .card { background:var(--surface); border:1px solid var(--border);
          border-radius:10px; padding:14px 16px; margin-bottom:14px; }
  .card h3 { margin:0 0 2px; font-size:14px; font-weight:600; color:var(--text); }
  .card p.sub { margin:0 0 10px; font-size:12px; color:var(--muted); }
  .grid2 { display:grid; grid-template-columns:1.7fr 1fr; gap:14px; }
  .grid2b { display:grid; grid-template-columns:1.2fr 1fr; gap:14px; }
  @media (max-width:1100px){ .grid2,.grid2b{ grid-template-columns:1fr; } }
  .filters { display:flex; gap:22px; align-items:flex-start; flex-wrap:wrap; }
  .filters label { color:var(--muted); font-size:11px; text-transform:uppercase;
                   letter-spacing:.05em; display:block; margin-bottom:5px; }
  .dot { width:9px; height:9px; border-radius:50%; display:inline-block; margin-right:7px; }
  .badge { padding:4px 11px; border-radius:999px; font-weight:600; font-size:12px; }
  .turn { background:var(--good); color:#fff; }
  .wait { background:var(--accent); color:#fff; }
  input, .Select-control { font-family:inherit !important; }
  #tbl .dash-spreadsheet td, #tbl .dash-spreadsheet th { font-variant-numeric:tabular-nums; }
</style></head>
<body data-theme="light">{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer>
<script>
  window.addEventListener('load', function () {
    var mq = window.matchMedia('(prefers-color-scheme: dark)');
    var set = function () {
      if (!document.body.dataset.userTheme) {
        document.body.setAttribute('data-theme', mq.matches ? 'dark' : 'light');
      }
    };
    set(); mq.addEventListener('change', set);
  });
</script>
</body></html>"""


def _tile(key: str, value: str, sub: str = "") -> html.Div:
    return html.Div(className="tile", children=[
        html.Div(key, className="k"),
        html.Div(value, className="v"),
        html.Div(sub, className="s"),
    ])


def create_app(publisher: Publisher) -> Dash:
    app = Dash(__name__, title="Sleeper Draft Co-Pilot", update_title=None)
    app.index_string = INDEX

    app.layout = html.Div(className="wrap", children=[
        dcc.Interval(id="tick", interval=1500, n_intervals=0),
        dcc.Store(id="theme", data="light"),

        html.Div(id="bar", className="bar"),
        html.Div(id="tiles", className="tiles"),

        html.Div(className="card", children=[
            html.Div(className="filters", children=[
                html.Div(style={"minWidth": "260px"}, children=[
                    html.Label("positions"),
                    dcc.Checklist(
                        id="f-pos",
                        options=[{"label": f" {p} ", "value": p} for p in POSITIONS],
                        value=list(POSITIONS), inline=True,
                    ),
                ]),
                html.Div(style={"minWidth": "260px"}, children=[
                    html.Label("risk profile"),
                    dcc.Checklist(
                        id="f-risk",
                        options=[{"label": f" {r} ", "value": r} for r in RISKS],
                        value=list(RISKS), inline=True,
                    ),
                ]),
                html.Div(style={"minWidth": "250px"}, children=[
                    html.Label("role (depth chart + volume)"),
                    dcc.Checklist(
                        id="f-role",
                        options=[{"label": f" {r} ", "value": r} for r in ROLES],
                        value=list(ROLES), inline=True,
                    ),
                ]),
                html.Div(style={"minWidth": "170px"}, children=[
                    html.Label("max tier"),
                    dcc.Slider(id="f-tier", min=1, max=9, step=1, value=9,
                               marks={i: str(i) for i in range(1, 10)}),
                ]),
                html.Div(style={"minWidth": "190px"}, children=[
                    html.Label("search player"),
                    dcc.Input(id="f-name", type="text", value="", debounce=False,
                              placeholder="name contains…",
                              style={"width": "100%", "padding": "6px 8px",
                                     "borderRadius": "6px", "border": "1px solid var(--border)",
                                     "background": "var(--plane)", "color": "var(--text)"}),
                ]),
                html.Div(style={"minWidth": "210px"}, children=[
                    html.Label("show"),
                    dcc.Checklist(
                        id="f-only-need", inline=True,
                        options=[{"label": " only positions I still need", "value": "need"}],
                        value=[],
                    ),
                ]),
            ]),
        ]),

        html.Div(className="grid2", children=[
            html.Div(className="card", children=[
                html.H3("Take now, or wait?"),
                html.P("Upper-left is urgent: high cost to wait, unlikely to last. "
                       "Lower-right you can safely pass on.", className="sub"),
                dcc.Graph(id="g-decision", config={"displayModeBar": False}),
            ]),
            html.Div(className="card", children=[
                html.H3("Positional outlook"),
                html.P("What each position costs you to wait on, and whether its "
                       "top three will still be there.", className="sub"),
                html.Div(id="outlook"),
            ]),
        ]),

        html.Div(className="grid2", children=[
            html.Div(className="card", children=[
                html.H3("Coming off the board"),
                html.P("Expected picks per position before your next turn.", className="sub"),
                dcc.Graph(id="g-scarcity", config={"displayModeBar": False}),
            ]),
        ]),

        html.Div(className="card", children=[
            html.H3("Value cliffs by position"),
            html.P("Projected points down each position's board. Steep segments are the "
                   "cliffs; dotted hairlines are tier breaks.", className="sub"),
            dcc.Graph(id="g-cliff", config={"displayModeBar": False}),
        ]),

        html.Div(className="grid2b", children=[
            html.Div(className="card", children=[
                html.H3("Recommendations"),
                html.P("Sortable. VONA is the decision number; Score is the ranking.",
                       className="sub"),
                html.Div(id="tbl", children=[
                    dash_table.DataTable(
                        id="table", page_size=14, sort_action="native",
                        style_as_list_view=True,
                        style_table={"overflowX": "auto"},
                        style_cell={"fontFamily": FONT, "fontSize": "13px",
                                    "padding": "7px 10px", "border": "none",
                                    "backgroundColor": "transparent"},
                        style_header={"fontWeight": "600", "border": "none",
                                      "borderBottom": "1px solid var(--border)"},
                    ),
                ]),
            ]),
            html.Div(children=[
                html.Div(className="card", children=[
                    html.H3("League"),
                    html.P("Projected starting-lineup points. Yours is highlighted.",
                           className="sub"),
                    dcc.Graph(id="g-standings", config={"displayModeBar": False}),
                ]),
                html.Div(className="card", children=[
                    html.H3("Your roster"),
                    html.Div(id="roster"),
                ]),
            ]),
        ]),
    ])

    def _filtered(snap: Snapshot, positions, risks, roles, max_tier, name, only_need):
        frame = snap.ranked
        if frame.empty:
            return frame
        mask = frame["pos"].isin(positions or list(POSITIONS))
        if "risk" in frame.columns and risks:
            mask &= frame["risk"].isin(risks)
        if "role" in frame.columns and roles:
            mask &= frame["role"].isin(roles)
        if "tier" in frame.columns:
            mask &= frame["tier"] <= (max_tier or 9)
        if name:
            mask &= frame["name"].str.contains(name, case=False, na=False)
        if only_need:
            need = {p for p in POSITIONS
                    if snap.info.get("unfilled", {}).get(p, 0) > 0} or set(POSITIONS)
            mask &= frame["pos"].isin(need)
        return frame[mask]

    @app.callback(
        Output("bar", "children"), Output("tiles", "children"),
        Output("g-decision", "figure"), Output("g-scarcity", "figure"),
        Output("g-cliff", "figure"), Output("g-standings", "figure"),
        Output("table", "data"), Output("table", "columns"),
        Output("outlook", "children"), Output("roster", "children"),
        Input("tick", "n_intervals"), Input("f-pos", "value"),
        Input("f-risk", "value"), Input("f-role", "value"), Input("f-tier", "value"),
        Input("f-name", "value"), Input("f-only-need", "value"),
        Input("theme", "data"),
    )
    def refresh(_n, positions, risks, roles, max_tier, name, only_need, theme):
        snap = publisher.get()
        info = snap.info
        c = palette(theme or "light")

        if not info:
            empty = value_cliff(pd.DataFrame(columns=["pos"]), theme or "light")
            return ([html.Span("waiting for draft data…")], [], empty, empty, empty,
                    empty, [], [], "", "")

        # --- freshness, stated plainly
        since = time.time() - (snap.changed_at or time.time())
        cdn_age = snap.age_s or 0.0
        if snap.error:
            dot, note = c["critical"], f"poll failed — {snap.error[:60]}"
        elif cdn_age > settings.stale_age_warn_s:
            dot, note = c["critical"], f"cached response {cdn_age:.0f}s old"
        elif since > 120:
            dot, note = c["warning"], f"no new pick for {since / 60:.0f}m"
        else:
            dot, note = c["good"], f"updated {since:.0f}s ago"

        turn = (html.Span("YOUR PICK", className="badge turn") if info["is_my_turn"]
                else html.Span(f"{info['picks_until_turn']} picks away", className="badge wait"))
        bar = [
            html.Strong(str(info["name"])),
            html.Span(f"slot {info['my_slot']}", style={"color": "var(--text2)"}),
            html.Span(f"pick {info['pick']} · round {info['round']}.{info['in_round']:02d}",
                      style={"color": "var(--text2)"}),
            html.Span(str(info["scoring"]).upper(), style={"color": "var(--muted)"}),
            turn,
            html.Span(style={"marginLeft": "auto"}, children=[
                html.Span(className="dot", style={"background": dot}),
                html.Span(note, style={"color": "var(--text2)", "fontSize": "12px"}),
                html.Span(f"  ·  polled {time.strftime('%H:%M:%S', time.localtime(snap.polled_at))}",
                          style={"color": "var(--muted)", "fontSize": "12px"}),
            ]),
        ]

        view = _filtered(snap, positions, risks, roles, max_tier, name, only_need)
        state_for_outlook = snap.state
        best = view.iloc[0] if not view.empty else None
        tiles = [
            _tile("on the clock", f"{info['round']}.{info['in_round']:02d}",
                  f"pick {info['pick']} of {info['teams'] * info['rounds']}"),
            _tile("your next turn",
                  "now" if info["is_my_turn"] else f"{info['picks_until_turn']} picks",
                  "back-to-back" if info["picks_until_turn"] == 0 and not info["is_my_turn"]
                  else ""),
            _tile("top recommendation",
                  f"{best['name']}" if best is not None else "—",
                  (f"{best['pos']} · VONA {best['vona']:+.0f} · "
                   f"{best['survival'] * 100:.0f}% to last") if best is not None else ""),
            _tile("your grade", str(info["grade"]),
                  (f"{info['rank']} of {info['teams']}" if info["rank"] else "")
                  + (f" · {info['holes']} unfilled" if info["holes"] else "")),
            _tile("data", note.split("—")[0].strip(),
                  f"CDN age {cdn_age:.0f}s" if cdn_age else "live"),
        ]

        columns = [
            {"name": "Player", "id": "name"}, {"name": "Pos", "id": "pos"},
            {"name": "Tm", "id": "team"}, {"name": "Proj", "id": "proj_pts"},
            {"name": "ADP", "id": "adp"}, {"name": "Tier", "id": "tier"},
            {"name": "Surv%", "id": "surv"}, {"name": "VORP", "id": "vorp"},
            {"name": "VONA", "id": "vona"}, {"name": "Lineup+", "id": "lineup_value"},
            {"name": "Score", "id": "score"},
            {"name": "Role", "id": "role"}, {"name": "Ceil%", "id": "ceil"},
            {"name": "Risk", "id": "risk"},
        ]
        table = view.head(60).copy()
        if not table.empty:
            table["surv"] = (table["survival"] * 100).round(0)
            for col in ("proj_pts", "vorp", "vona", "score", "lineup_value"):
                table[col] = table[col].round(0)
            table["adp"] = table["adp"].round(1)
            table["role"] = table.get("role", "")
            table["ceil"] = (table["ceiling_pct"] * 100).round(0) if "ceiling_pct" in table \
                else ""
        rows = table[[c["id"] for c in columns]].to_dict("records") if not table.empty else []

        roster = html.Table(
            style={"width": "100%", "borderCollapse": "collapse", "fontSize": "13px"},
            children=[html.Tbody([
                html.Tr(style={"borderBottom": "1px solid var(--border)"}, children=[
                    html.Td(f"R{r['round']}", style={"color": "var(--muted)", "padding": "5px 0",
                                                      "width": "34px"}),
                    html.Td(r["name"], style={"padding": "5px 0"}),
                    html.Td(r["pos"], style={"color": "var(--text2)", "width": "44px"}),
                    html.Td(f"{r['proj']}", style={"textAlign": "right", "width": "50px",
                                                   "fontVariantNumeric": "tabular-nums"}),
                ]) for r in snap.roster
            ])]
        ) if snap.roster else html.Div("no picks yet", style={"color": "var(--muted)"})

        outlook = outlook_table(state_for_outlook, view, snap.window, theme or "light") \
            if state_for_outlook is not None else ""

        return (
            bar, tiles,
            decision_scatter(view, theme or "light"),
            scarcity_bar(snap.expected, theme or "light"),
            value_cliff(view if not view.empty else snap.ranked, theme or "light"),
            standings_bar(snap.rated, info["my_roster"], theme or "light"),
            rows, columns, outlook, roster,
        )

    return app
