"""Does week-to-week variance measure draft-relevant risk? (It does not.)

Petersen's uncertainty measure (Ch. 6) is the *spread of a player's projections
across sources*. This project has one source -- Sleeper -- so week-to-week
scoring variance from last season stood in for it. That substitution looked
reasonable and is wrong, and this script is what showed it.

CV = sd/mean has the mean in the denominator, so it flags low-volume players
whose small scores bounce around zero, not the boom-or-bust starters the draft
advice is about. Run it and read the terciles: the "volatile" third has a lower
ceiling than the "steady" third, which is the opposite of what a risk label is
supposed to mean.

    python scripts/experiment_cv.py

Reads data/projections.csv and data/consistency.csv. No network calls.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-games", type=int, default=10,
                    help="a CV from four games is noise measuring noise (default: 10)")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "experiment_cv.csv")
    args = ap.parse_args(argv)

    proj = pd.read_csv(ROOT / "data" / "projections.csv", dtype={"player_id": str})
    cons = pd.read_csv(ROOT / "data" / "consistency.csv", dtype={"player_id": str})

    df = cons.merge(proj[["player_id", "name", "pos", "proj_pts"]], on="player_id")
    df = df[(df["games_played"] >= args.min_games) & df["wk_cv"].notna()].copy()
    df["tercile"] = pd.qcut(df["wk_cv"], 3, labels=["steady", "neutral", "volatile"])

    terciles = df.groupby("tercile", observed=True).agg(
        n=("wk_cv", "size"),
        cv=("wk_cv", "mean"),
        pts_per_week=("wk_mean", "mean"),
        ceiling_p90=("wk_ceiling", "mean"),
    ).round(2)

    print(f"{len(df)} players with >= {args.min_games} games\n")
    print(terciles.to_string())
    print("\ncorrelations")
    print(f"  CV      vs points/week   {df['wk_cv'].corr(df['wk_mean']):+.3f}")
    print(f"  CV      vs ceiling       {df['wk_cv'].corr(df['wk_ceiling']):+.3f}")
    print(f"  ceiling vs points/week   {df['wk_ceiling'].corr(df['wk_mean']):+.3f}")
    print("\nCV is negatively correlated with both quality and upside. It is kept"
          "\nas a label for what it measures -- week-to-week bounce -- and `role`"
          "\nand `ceiling` carry the questions it was standing in for.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
