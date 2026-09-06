"""Sleeper draft co-pilot.

Read-only: this observes a draft and ranks your options. It never drafts for
you -- Sleeper's public API has no write endpoints, and that is a feature. You
keep the veto when injury news breaks mid-draft.

    python main.py --find-draft <username>       # discover your draft_id
    python main.py --mock --slot 5               # offline dry run
    python main.py --draft-id <id> --slot 5      # draft night
    python main.py --export-cheatsheet           # the paper fallback
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.live import Live

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.cli import dashboard  # noqa: E402
from src.config import settings  # noqa: E402
from src.mock import mock_draft_object, simulate  # noqa: E402
from src.optimizer import rank_with_lookahead  # noqa: E402
from src.sleeper_client import NotFound, SleeperClient, SleeperError  # noqa: E402
from src.state import DraftState  # noqa: E402
from src.valuation import build_board  # noqa: E402

logger = logging.getLogger("sleeper-drafter")
console = Console()

FANTASY_POSITIONS = ("QB", "RB", "WR", "TE", "K", "DEF")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if not settings.projections_csv.exists():
        console.print(
            "[bold red]No projections found.[/] Run: "
            "[bold]python scripts/fetch_projections.py[/]"
        )
        raise SystemExit(2)
    proj = pd.read_csv(settings.projections_csv, dtype={"player_id": str})

    cons_path = settings.data_dir / "consistency.csv"
    cons = pd.read_csv(cons_path, dtype={"player_id": str}) if cons_path.exists() else None
    if cons is None:
        console.print("[yellow]No consistency data — risk labels unavailable.[/] "
                      "Run: python scripts/fetch_consistency.py")
    return proj, cons


def make_board(state: DraftState, proj: pd.DataFrame, cons: pd.DataFrame | None) -> pd.DataFrame:
    starters = {p: state.schema.total_starters(p) for p in FANTASY_POSITIONS}
    return build_board(proj, starters, state.teams, consistency=cons)


def poll_delay(picks_until_turn: int, my_turn: bool) -> float:
    """Poll faster the closer your pick is.

    Twenty picks away, a couple of seconds of lag costs nothing. On the clock,
    it is the difference between seeing the pick that just went and reaching for
    a player who is already gone.
    """
    if my_turn or picks_until_turn <= settings.poll_near_threshold:
        return settings.poll_interval_near_s
    if picks_until_turn <= settings.poll_near_threshold * 3:
        return settings.poll_interval_s
    return settings.poll_interval_far_s


# ----------------------------------------------------------------- subcommands

def cmd_find_draft(username: str, season: str) -> int:
    client = SleeperClient()
    try:
        user = client.user(username)
    except NotFound:
        # Sleeper answers 200 with a null body here, not 404.
        console.print(f"[bold red]No Sleeper user '{username}'.[/]")
        return 1

    user_id = user.get("user_id")
    console.print(f"user [bold]{user.get('display_name')}[/] ({user_id})")
    try:
        drafts = client.user_drafts(user_id, season)
    except SleeperError as exc:
        console.print(f"[bold red]Could not list drafts:[/] {exc}")
        return 1

    if not drafts:
        console.print(f"[yellow]No {season} drafts found for this user.[/]")
        return 1

    for draft in drafts:
        meta = draft.get("metadata") or {}
        order = draft.get("draft_order") or {}
        slot = order.get(user_id, "?")
        console.print(
            f"  [bold]{draft.get('draft_id')}[/]  "
            f"{meta.get('name') or '(unnamed)'}  "
            f"status=[cyan]{draft.get('status')}[/]  "
            f"type={draft.get('type')}  your slot=[bold green]{slot}[/]"
        )
    return 0


def cmd_export_cheatsheet(out: Path, teams: int, rounds: int) -> int:
    """The fallback that works when nothing else does. Print it before the draft."""
    proj, cons = load_inputs()
    draft, league = mock_draft_object(teams=teams, rounds=rounds)
    state = DraftState(draft, proj, league=league)
    board = make_board(state, proj, cons)

    cols = ["name", "pos", "team", "proj_pts", "adp", "tier", "dropoff",
            "vorp", "wk_cv", "risk"]
    sheet = board[board["adp"] < settings.undrafted_adp][cols].copy()
    sheet = sheet.sort_values("adp").reset_index(drop=True)
    sheet.insert(0, "rank", sheet.index + 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(out, index=False)

    console.print(f"[green]wrote[/] {out}  ({len(sheet)} players)")
    console.print("[dim]Print this. If the API is down on draft night, it is what you have.[/]")
    return 0


def cmd_mock(slot: int, teams: int, rounds: int, seed: int, step: bool) -> int:
    proj, cons = load_inputs()
    draft, league = mock_draft_object(teams=teams, rounds=rounds)
    state = DraftState(draft, proj, league=league)
    board = make_board(state, proj, cons)
    rng = np.random.default_rng(seed)

    def choose(st: DraftState, available: pd.DataFrame, round_no: int) -> str:
        ranked, window = rank_with_lookahead(st, board, slot)
        console.print(dashboard(st, board, ranked, slot, window))
        if step:
            console.input("[dim]enter to take the top recommendation…[/] ")
        return str(ranked.iloc[0]["player_id"])

    simulate(state, board, my_slot=slot, rng=rng, on_my_pick=choose)

    names = board.set_index("player_id")
    console.print(f"\n[bold]Final roster (slot {slot})[/]")
    for i, pid in enumerate(state.teams_by_roster[slot].roster, 1):
        row = names.loc[pid]
        console.print(f"  R{i:<2} {row['name']:<26} {row['pos']:<4} "
                      f"{row['proj_pts']:6.1f}  tier {int(row['tier'])}  {row['risk']}")

    from src.cli import standings_panel

    panel = standings_panel(state, board, slot)
    if panel:
        console.print()
        console.print(panel)
    return 0


def cmd_web(draft_id: str | None, slot: int, port: int, host: str,
            mock: bool, teams: int, rounds: int, seed: int,
            seconds_per_pick: float) -> int:
    """Serve the browser dashboard. One poller thread, many browser tabs."""
    import threading

    from src.web import Publisher, create_app, start_live_poller, start_mock_poller

    proj, cons = load_inputs()
    publisher = Publisher()
    stop = threading.Event()

    if mock:
        draft, league = mock_draft_object(teams=teams, rounds=rounds)
        state = DraftState(draft, proj, league=league)
        board = make_board(state, proj, cons)
        start_mock_poller(publisher, state, board, slot, stop,
                          seconds_per_pick=seconds_per_pick, seed=seed)
        console.print(f"[yellow]mock mode[/] — simulating a pick every {seconds_per_pick}s")
    else:
        client = SleeperClient()
        try:
            draft = client.draft(draft_id)
        except NotFound:
            console.print(f"[bold red]No draft '{draft_id}'.[/]")
            return 1
        league = None
        if draft.get("league_id"):
            try:
                league = client.league(draft["league_id"])
            except SleeperError as exc:
                console.print(f"[yellow]League unavailable ({exc}); "
                              f"falling back to draft slot settings.[/]")
        state = DraftState(draft, proj, league=league,
                           traded_picks=client.traded_picks(draft_id))
        board = make_board(state, proj, cons)
        start_live_poller(publisher, client, draft_id, state, board, slot, stop)
        console.print(f"[green]watching[/] {draft_id}  "
                      f"{state.teams} teams x {state.rounds} rounds, {state.scoring}")

    app = create_app(publisher)
    console.print(f"[bold]dashboard:[/] http://{host if host != '0.0.0.0' else 'localhost'}:{port}")
    console.print("[dim]Ctrl+C to stop. The terminal UI can run alongside this "
                  "in another window.[/]")
    try:
        app.run(host=host, port=port, debug=False)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
    return 0


def cmd_live(draft_id: str, slot: int | None, top_n: int) -> int:
    proj, cons = load_inputs()
    client = SleeperClient()

    try:
        draft = client.draft(draft_id)
    except NotFound:
        console.print(f"[bold red]No draft '{draft_id}'.[/]")
        return 1

    league = None
    if draft.get("league_id"):
        try:
            league = client.league(draft["league_id"])
        except SleeperError as exc:
            # Costs us SUPER_FLEX detection, so say so rather than silently degrade.
            console.print(f"[yellow]League unavailable ({exc}); "
                          f"falling back to draft slot settings.[/]")

    state = DraftState(draft, proj, league=league,
                       traded_picks=client.traded_picks(draft_id))
    board = make_board(state, proj, cons)

    if slot is None:
        console.print("[bold red]--slot is required for a live draft.[/] "
                      "Find it with --find-draft.")
        return 2

    console.print(f"[green]watching[/] {draft_id}  "
                  f"{state.teams} teams x {state.rounds} rounds, {state.scoring}")

    stale = False
    window: list[int] = []
    with Live(console=console, refresh_per_second=4, screen=True) as live:
        while True:
            try:
                state.ingest(client.picks(draft_id))
                # A response the CDN has held longer than a pick clock means we
                # may be looking at a cached board, not a quiet one.
                age = client.last_age_s or 0.0
                stale = age > settings.stale_age_warn_s
                if stale:
                    logger.warning("served a %.0fs-old cached response", age)
            except SleeperError as exc:
                # Never crash out of the loop mid-draft; show the last good board.
                logger.warning("poll failed: %s", exc)
                stale = True

            ranked, window = rank_with_lookahead(state, board, slot)
            live.update(dashboard(state, board, ranked, slot, window,
                                  stale=stale, top_n=top_n))

            if state.current_pick_no > state.total_picks:
                break
            time.sleep(poll_delay(len(window), state.is_my_turn(slot)))

    console.print("[green]draft complete.[/]")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sleeper-drafter", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draft-id", help="Sleeper draft id to watch live")
    ap.add_argument("--slot", type=int, help="your 1-indexed draft slot")
    ap.add_argument("--mock", action="store_true", help="run an offline simulated draft")
    ap.add_argument("--find-draft", metavar="USERNAME", help="look up your draft ids")
    ap.add_argument("--export-cheatsheet", action="store_true",
                    help="write the static fallback board")
    ap.add_argument("--season", default=settings.season)
    ap.add_argument("--teams", type=int, default=12)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--top", type=int, default=10, help="recommendations to show")
    ap.add_argument("--seed", type=int, default=0, help="mock draft rng seed")
    ap.add_argument("--step", action="store_true", help="pause at each of your picks")
    ap.add_argument("--web", action="store_true",
                    help="serve the browser dashboard instead of the terminal UI")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--host", default="127.0.0.1",
                    help="use 0.0.0.0 to reach it from another device")
    ap.add_argument("--seconds-per-pick", type=float, default=3.0,
                    help="mock draft speed for --web --mock")
    ap.add_argument("-o", "--out", type=Path,
                    default=settings.data_dir / "cheatsheet.csv")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.find_draft:
        return cmd_find_draft(args.find_draft, args.season)
    if args.export_cheatsheet:
        return cmd_export_cheatsheet(args.out, args.teams, args.rounds)
    if args.web:
        if not args.mock and not args.draft_id:
            ap.error("--web needs either --draft-id or --mock")
        return cmd_web(args.draft_id, args.slot or 5, args.port, args.host,
                       args.mock, args.teams, args.rounds, args.seed,
                       args.seconds_per_pick)
    if args.mock:
        return cmd_mock(args.slot or 5, args.teams, args.rounds, args.seed, args.step)
    if args.draft_id:
        return cmd_live(args.draft_id, args.slot, args.top)

    ap.error("one of --draft-id, --mock, --find-draft, or --export-cheatsheet is required")


if __name__ == "__main__":
    sys.exit(main())
