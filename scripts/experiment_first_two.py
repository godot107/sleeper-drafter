"""Does 'take running backs early' hold at every draft slot?

Peer wisdom says spend the first two picks on RB/WR and never on a quarterback.
That is a claim about where value sits, and where value sits depends on where
you pick -- so it is testable. This forces the first two picks to a given
position pair, lets the engine draft normally afterwards, and scores the final
starting lineup. Everything else (opponents, seed, board) is held fixed, so the
only difference between arms is the constraint.

    python scripts/experiment_first_two.py --seeds 16 --slots 1,4,8,12

Reads data/projections.csv and data/consistency.csv; makes no network calls.
Writes a tidy CSV to data/experiment_first_two.csv.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import load_inputs, make_board  # noqa: E402
from src.mock import mock_draft_object, simulate  # noqa: E402
from src.optimizer import rank_with_lookahead  # noqa: E402
from src.rating import rate_teams  # noqa: E402
from src.state import DraftState  # noqa: E402

# The arms. `None` is the control: the engine picks freely.
ARMS: dict[str, tuple[str, str] | None] = {
    "engine": None,
    "QB+RB": ("QB", "RB"),
    "RB+RB": ("RB", "RB"),
    "RB+WR": ("RB", "WR"),
    "WR+WR": ("WR", "WR"),
    "QB+WR": ("QB", "WR"),
}


def run_one(proj, cons, slot: int, teams: int, rounds: int, seed: int,
            forced: tuple[str, ...] | None) -> tuple[float, list[str]]:
    draft, league = mock_draft_object(teams=teams, rounds=rounds)
    state = DraftState(draft, proj, league=league)
    board = make_board(state, proj, cons)
    rng = np.random.default_rng(seed)
    taken: list[str] = []

    def choose(st: DraftState, available: pd.DataFrame, round_no: int) -> str:
        ranked, _ = rank_with_lookahead(st, board, slot)
        n = len(taken)
        if forced is not None and n < len(forced):
            want = ranked[ranked["pos"] == forced[n]]
            # Forcing a position that is genuinely gone is not a fair arm; fall
            # back to the free choice rather than drafting a replacement-level
            # body and blaming the strategy for it.
            if not want.empty:
                taken.append(forced[n])
                return str(want.iloc[0]["player_id"])
        row = ranked.iloc[0]
        taken.append(str(row["pos"]))
        return str(row["player_id"])

    simulate(state, board, my_slot=slot, rng=rng, on_my_pick=choose)
    rated = rate_teams(state, board)
    mine = rated[rated["draft_slot"] == slot]
    return float(mine["starters_pts"].iloc[0]), taken[:2]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--slots", default="1,4,8,12")
    ap.add_argument("--teams", type=int, default=12)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent / "data"
                    / "experiment_first_two.csv")
    args = ap.parse_args(argv)

    proj, cons = load_inputs()
    slots = [int(s) for s in args.slots.split(",")]
    rows = []
    free_choices: dict[int, Counter] = {s: Counter() for s in slots}

    for slot in slots:
        for name, forced in ARMS.items():
            for seed in range(args.seeds):
                pts, first_two = run_one(proj, cons, slot, args.teams,
                                         args.rounds, seed, forced)
                rows.append({"slot": slot, "arm": name, "seed": seed, "starters_pts": pts})
                if forced is None:
                    free_choices[slot].update(first_two)
            done = [r for r in rows if r["slot"] == slot and r["arm"] == name]
            print(f"slot {slot:>2}  {name:<7} "
                  f"{np.mean([r['starters_pts'] for r in done]):8.1f}", flush=True)

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    print(f"\nwrote {args.out}")
    pivot = frame.pivot_table(index="slot", columns="arm", values="starters_pts")
    baseline = pivot["engine"]
    print("\npoints vs the engine's own free choice:")
    print((pivot.drop(columns="engine").sub(baseline, axis=0)).round(1).to_string())
    print("\nwhat the engine took with its first two picks, unconstrained:")
    for slot in slots:
        print(f"  slot {slot:>2}  {dict(free_choices[slot].most_common())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
