"""Build ``data/projections.csv`` from Sleeper's own projections endpoint.

Run this the day before the draft (and again on the morning of, to pick up
late injury/depth-chart movement)::

    python scripts/fetch_projections.py --scoring ppr

The original spec called for a hand-mocked projections file. That is fine for
unit tests and useless on draft night, so this pulls real numbers instead:
``api.sleeper.com/projections/nfl/<season>`` returns season-long projected
points *and* ADP for every fantasy position, keyed by the same ``player_id`` as
the player directory. Verified 2026-09-04: 631 players with points, 422 with a
real (<999) ADP.

The output is a frozen snapshot, so ``--mock`` runs and draft-day runs both work
with no network.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings  # noqa: E402
from src.sleeper_client import FANTASY_POSITIONS, SleeperClient  # noqa: E402

logger = logging.getLogger("fetch_projections")

# Sleeper's scoring_type -> the projection/ADP columns that go with it.
SCORING_COLUMNS = {
    "ppr": ("pts_ppr", "adp_ppr"),
    "half_ppr": ("pts_half_ppr", "adp_half_ppr"),
    "std": ("pts_std", "adp_std"),
    "2qb": ("pts_ppr", "adp_2qb"),  # superflex boards price QBs completely differently
}


def build(scoring: str, season: str, *, refresh_players: bool = False) -> pd.DataFrame:
    pts_col, adp_col = SCORING_COLUMNS[scoring]
    client = SleeperClient()

    directory = client.players(refresh=refresh_players)
    logger.info("player directory: %d players", len(directory))

    rows: list[dict] = []
    for pos in FANTASY_POSITIONS:
        payload = client.projections(pos, season=season)
        kept = 0
        for rec in payload:
            stats = rec.get("stats") or {}
            pts = stats.get(pts_col)
            if pts is None:
                continue  # no projection -> not draftable, drop it
            pid = str(rec.get("player_id"))
            meta = directory.get(pid, {})
            nested = rec.get("player") or {}
            name = meta.get("full_name") or " ".join(
                filter(None, [nested.get("first_name"), nested.get("last_name")])
            )
            rows.append(
                {
                    "player_id": pid,
                    "name": name,
                    "pos": meta.get("position") or nested.get("position") or pos,
                    "team": meta.get("team") or nested.get("team") or "FA",
                    "proj_pts": float(pts),
                    # Opportunity. Sleeper projects no target share, so receptions
                    # stand in for receiving volume -- an imperfect proxy, since it
                    # bakes in catch rate and so understates high-target receivers
                    # in poor offences. Carries need no proxy.
                    "proj_rec": float(stats.get("rec") or 0.0),
                    "proj_rush_att": float(stats.get("rush_att") or 0.0),
                    # Where he sits on his own team's depth chart: 1 = the starter.
                    "depth_chart_order": meta.get("depth_chart_order"),
                    # 999 is Sleeper's "undrafted" sentinel; keep it as the sentinel
                    # rather than NaN so the ADP kernel stays defined everywhere.
                    "adp": float(stats.get(adp_col) or settings.undrafted_adp),
                    "search_rank": meta.get("search_rank") or 99999,
                    "years_exp": meta.get("years_exp"),
                    "injury_status": meta.get("injury_status"),
                }
            )
            kept += 1
        logger.info("%-3s %4d rows -> %3d projected", pos, len(payload), kept)

    frame = pd.DataFrame(rows)
    # Undrafted players fall back to search_rank order so they still rank sanely.
    frame = frame.sort_values(["adp", "search_rank"]).reset_index(drop=True)
    return frame


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fetch_projections", description=__doc__)
    ap.add_argument("--scoring", choices=sorted(SCORING_COLUMNS), default="ppr",
                    help="league scoring format (default: ppr)")
    ap.add_argument("--season", default=settings.season)
    ap.add_argument("--refresh-players", action="store_true",
                    help="force a refetch of the ~15 MB player directory")
    ap.add_argument("-o", "--out", type=Path, default=settings.projections_csv)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    frame = build(args.scoring, args.season, refresh_players=args.refresh_players)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    real_adp = int((frame["adp"] < settings.undrafted_adp).sum())
    print(f"wrote {args.out}  ({len(frame)} players, {real_adp} with real ADP, scoring={args.scoring})")
    print(frame.head(12).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
