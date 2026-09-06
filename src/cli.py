"""Terminal dashboard.

Design constraint: this is read under a pick clock, by someone who is also
looking at the Sleeper app. So it optimises for glanceability -- the same
columns in the same order every refresh, colour carrying meaning rather than
decoration, and no layout that reflows as numbers change.
"""

from __future__ import annotations

import pandas as pd
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .config import settings
from .state import DraftState

RISK_STYLE = {
    "steady": "green",
    "neutral": "white",
    "volatile": "yellow",
    "unknown": "dim",
}

POS_STYLE = {
    "QB": "magenta", "RB": "green", "WR": "cyan",
    "TE": "yellow", "K": "dim", "DEF": "blue",
}


def header(state: DraftState, my_slot: int, window: list[int], stale: bool = False) -> Panel:
    pick = state.current_pick_no
    rnd = (pick - 1) // state.teams + 1
    in_round = (pick - 1) % state.teams + 1
    name = (state.draft.get("metadata") or {}).get("name") or state.draft_id

    my_turn = state.is_my_turn(my_slot)
    turn = Text(" YOUR PICK ", style="bold black on green") if my_turn else \
        Text(f" {len(window)} picks until your turn ", style="bold white on blue")

    line = Text.assemble(
        (f"{name}  ", "bold"),
        (f"slot {my_slot}", "cyan"), ("  |  ", "dim"),
        (f"pick {pick} ({rnd}.{in_round:02d})", "white"), ("  |  ", "dim"),
        (f"{state.scoring.upper()}", "magenta"), ("  ", ""),
    )
    line.append_text(turn)
    if stale:
        line.append_text(Text("  ⚠ STALE — retrying ", style="bold black on red"))
    return Panel(line, border_style="green" if my_turn else "blue")


def recommendations_table(ranked: pd.DataFrame, top_n: int = 10) -> Table:
    table = Table(
        title="Recommendations", title_style="bold",
        header_style="bold", expand=True, padding=(0, 1),
    )
    for col, justify in [
        ("#", "right"), ("Player", "left"), ("Pos", "left"), ("Tm", "left"),
        ("Proj", "right"), ("ADP", "right"), ("Tier", "right"),
        ("Surv%", "right"), ("VORP", "right"), ("VONA", "right"),
        ("Score", "right"), ("Risk", "left"),
    ]:
        table.add_column(col, justify=justify, no_wrap=True)

    for rank, (_, row) in enumerate(ranked.head(top_n).iterrows(), 1):
        surv = row.get("survival")
        vona = row.get("vona")
        table.add_row(
            str(rank),
            Text(str(row["name"]), style="bold" if rank == 1 else ""),
            Text(str(row["pos"]), style=POS_STYLE.get(row["pos"], "")),
            str(row.get("team", "")),
            f"{row['proj_pts']:.0f}",
            "—" if row["adp"] >= settings.undrafted_adp else f"{row['adp']:.1f}",
            str(int(row.get("tier", 0))),
            "—" if surv is None or surv != surv else f"{surv * 100:.0f}",
            f"{row['vorp']:.0f}",
            "—" if vona is None or vona != vona else f"{vona:.0f}",
            Text(f"{row['score']:.0f}", style="bold"),
            Text(str(row.get("risk", "")), style=RISK_STYLE.get(row.get("risk"), "")),
        )
    return table


def run_warning(
    state: DraftState, board: pd.DataFrame, window: list[int],
    available: pd.DataFrame | None = None,
) -> Panel | None:
    """Warn when a position is likely to be picked clean before your next turn.

    Petersen (Ch.7) advises against joining a run mid-stream; the value of this
    panel is seeing a run *coming* while you can still get ahead of it.

    Measured as the expected number of players taken at each position over the
    window -- summing the same normalised selection probabilities the survival
    model uses -- rather than by counting how many upcoming teams "need" the
    position. Counting need alone fires constantly in round 1, when nobody has
    filled anything, and it double-counts teams that pick twice in the window.
    """
    if not window:
        return None

    from .opponent_model import selection_probs

    pool = board if available is None else available
    if pool.empty:
        return None

    positions = pool["pos"].to_numpy()
    expected: dict[str, float] = {}
    for pick_no in window:
        team = state.team_at(pick_no)
        probs = selection_probs(pool, team, state.schema, pick_no)
        for pos in ("QB", "RB", "WR", "TE"):
            expected[pos] = expected.get(pos, 0.0) + float(probs[positions == pos].sum())

    alerts = []
    for pos, n in sorted(expected.items(), key=lambda kv: -kv[1]):
        if n < settings.run_warning_opponents:
            continue
        top = pool[pool["pos"] == pos].nlargest(3, "proj_pts")
        if top.empty:
            continue
        cliff = float(top["dropoff"].max())
        if cliff >= settings.tier_cliff_pts:
            alerts.append(
                f"[bold]{pos}[/]: ~{n:.0f} expected off the board before your turn, "
                f"and a {cliff:.0f}-pt cliff is live"
            )

    if not alerts:
        return None
    return Panel("\n".join(alerts), title="⚠ Position run forming",
                 border_style="yellow", title_align="left")


GRADE_STYLE = {
    "A+": "bold green", "A": "bold green", "A-": "green",
    "B+": "cyan", "B": "cyan", "B-": "cyan",
    "C+": "yellow", "C": "yellow", "C-": "yellow",
    "D+": "red", "D": "red", "F": "bold red",
}


def standings_panel(state: DraftState, board: pd.DataFrame, my_slot: int) -> Panel | None:
    """Live draft grades, so a positional hole is a decision and not a verdict."""
    from .rating import rate_teams, weakest_positions

    rated = rate_teams(state, board)
    if rated.empty or rated["picks"].sum() == 0:
        return None

    my_roster = state.slot_to_roster.get(my_slot, my_slot)
    table = Table.grid(padding=(0, 2))
    for _ in range(5):
        table.add_column()

    for rank, (_, row) in enumerate(rated.iterrows(), 1):
        mine = row["roster_id"] == my_roster
        weak = weakest_positions(rated, int(row["roster_id"]))
        note = ""
        if row["holes"]:
            note = f"[red]{int(row['holes'])} unfilled[/]"
        elif weak:
            note = f"[dim]thin at {', '.join(weak)}[/]"
        table.add_row(
            Text(f"{rank:>2}.", style="bold" if mine else "dim"),
            Text(f"slot {int(row['draft_slot']):>2}" + ("  ← you" if mine else ""),
                 style="bold white" if mine else "dim"),
            Text(f"{row['starters_pts']:>7.0f}", style="bold" if mine else ""),
            Text(f"{row['grade']:<2}", style=GRADE_STYLE.get(row["grade"], "")),
            note,
        )

    my_row = rated[rated["roster_id"] == my_roster]
    title = "League (projected starting lineup)"
    if not my_row.empty:
        rank = int(my_row.index[0]) + 1
        title += f" — you are {rank} of {len(rated)}, grade {my_row['grade'].iloc[0]}"
    return Panel(table, title=title, border_style="dim", title_align="left")


def recent_picks(state: DraftState, board: pd.DataFrame, n: int = 5) -> Panel:
    names = board.set_index("player_id")
    lines = []
    for pick in state.picks[-n:][::-1]:
        pid = str(pick.get("player_id"))
        rnd, slot = pick.get("round"), pick.get("draft_slot")
        if pid in names.index:
            row = names.loc[pid]
            lines.append(f"[dim]{rnd}.{slot:02d}[/] {row['name']} [dim]{row['pos']}[/]")
        else:
            lines.append(f"[dim]{rnd}.{slot:02d}[/] [dim]{pid}[/]")
    return Panel("\n".join(lines) or "[dim]no picks yet[/]",
                 title="Recent picks", border_style="dim", title_align="left")


def dashboard(
    state: DraftState,
    board: pd.DataFrame,
    ranked: pd.DataFrame,
    my_slot: int,
    window: list[int],
    *,
    stale: bool = False,
    top_n: int = 10,
    show_standings: bool = True,
) -> Group:
    parts = [header(state, my_slot, window, stale), recommendations_table(ranked, top_n)]
    warning = run_warning(state, board, window, available=ranked)
    if warning:
        parts.append(warning)
    if show_standings:
        standings = standings_panel(state, board, my_slot)
        if standings:
            parts.append(standings)
    parts.append(recent_picks(state, board))
    return Group(*parts)
