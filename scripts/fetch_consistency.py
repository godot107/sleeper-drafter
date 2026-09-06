"""Build ``data/consistency.csv`` -- per-player weekly scoring variability.

Grounded in Isaac T. Petersen, *Fantasy Football Analytics*, Ch. 6
(Player Evaluation): uncertainty is operationalised as the standard deviation
and **coefficient of variation** of a player's fantasy points, Eq. 6.1::

    CV = s_x / x̄

Petersen's point is that uncertainty is not inherently bad -- it is a different
risk profile. Two players projected for 150 points with SDs of 5 and 30 are not
interchangeable. His draft guidance follows directly: prefer *low*-uncertainty
players for starting slots and *high*-uncertainty players (potential sleepers,
higher upside and lower downside) for bench slots.

Sleeper publishes no uncertainty estimate alongside its projections, and we have
only one projection source, so cross-source SD is unavailable. We substitute
week-to-week variance from the previous completed season, which is what
Petersen's own consistency R code computes.

Run once before the draft::

    python scripts/fetch_consistency.py --season 2025
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings  # noqa: E402
from src.sleeper_client import FANTASY_POSITIONS, SleeperClient  # noqa: E402

logger = logging.getLogger("fetch_consistency")

REGULAR_SEASON_WEEKS = 18


def build(season: str, *, weeks: int = REGULAR_SEASON_WEEKS) -> pd.DataFrame:
    client = SleeperClient()
    # player_id -> list of weekly point totals for games actually played
    weekly: dict[str, list[float]] = {}

    for pos in FANTASY_POSITIONS:
        got = 0
        for week in range(1, weeks + 1):
            try:
                rows = client.weekly_stats(pos, season, week)
            except Exception as exc:  # a missing week must not sink the build
                logger.warning("%s week %d unavailable: %s", pos, week, exc)
                continue
            for rec in rows:
                stats = rec.get("stats") or {}
                pts = stats.get("pts_ppr")
                # gp==0 means inactive; a real 0.0 in a played game still counts.
                if pts is None or not stats.get("gp"):
                    continue
                weekly.setdefault(str(rec.get("player_id")), []).append(float(pts))
                got += 1
            time.sleep(0.05)  # stay well clear of the 1000 calls/min ceiling
        logger.info("%-3s %d player-weeks", pos, got)

    rows_out = []
    for pid, points in weekly.items():
        arr = np.asarray(points, dtype=float)
        mean = float(arr.mean())
        # Sample SD; a single game gives no variance information.
        sd = float(arr.std(ddof=1)) if arr.size > 1 else np.nan
        rows_out.append(
            {
                "player_id": pid,
                "games_played": int(arr.size),
                "wk_mean": mean,
                "wk_sd": sd,
                # Petersen Eq. 6.1. Undefined at mean<=0, which happens for
                # deep bench players who only ever posted zeros.
                "wk_cv": float(sd / mean) if (sd == sd and mean > 0) else np.nan,
                # 90th-percentile week: what his ceiling actually looks like.
                # CV turned out to be a poor upside signal (see valuation.py),
                # so record the ceiling directly rather than inferring it.
                "wk_ceiling": float(np.percentile(arr, 90)),
                "wk_floor": float(np.percentile(arr, 10)),
            }
        )

    frame = pd.DataFrame(rows_out).sort_values("wk_mean", ascending=False)
    return frame.reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fetch_consistency", description=__doc__)
    ap.add_argument("--season", default=str(int(settings.season) - 1),
                    help="completed season to measure (default: last year)")
    ap.add_argument("--weeks", type=int, default=REGULAR_SEASON_WEEKS)
    ap.add_argument("-o", "--out", type=Path,
                    default=settings.data_dir / "consistency.csv")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    frame = build(args.season, weeks=args.weeks)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    usable = int(frame["wk_cv"].notna().sum())
    print(f"wrote {args.out}  ({len(frame)} players, {usable} with a usable CV, season={args.season})")
    print(frame[frame["games_played"] >= 10].head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
