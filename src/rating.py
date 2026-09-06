"""Team ratings: where you actually stand, while you can still fix it.

A post-draft letter grade tells you nothing you can act on. The point of doing
this live is that a hole at tight end in round 9 is a decision; in round 15 it
is a verdict.

Grading is deliberately simple and legible: fill each roster's best legal
starting lineup, sum the projected points, and rank. That mirrors what
commercial draft graders do, and it is the same quantity Petersen's Ch. 7
lineup optimiser maximises -- solved greedily here rather than with an MILP,
which is exact for the slot structures Sleeper supports because every flex slot
is filled from the leftovers of the dedicated ones.

Two things it deliberately does *not* do:

* **No bench credit.** A great bench is real value, but it is not points scored,
  and counting it is how you talk yourself into a roster that cannot start a
  legal lineup.
* **No schedule or bye-week modelling.** Both matter and neither is knowable
  from projections alone.

A hole -- an unfilled starting slot -- is scored at replacement level, not zero,
because in practice you stream someone. It is reported separately, because it
is the failure that actually loses you weeks.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .state import FLEX_ELIGIBILITY, DraftState, RosterSchema

logger = logging.getLogger(__name__)

# Curve over the league, in standard deviations of starting-lineup points.
GRADE_CUTS: list[tuple[float, str]] = [
    (1.5, "A+"), (1.0, "A"), (0.7, "A-"),
    (0.4, "B+"), (0.15, "B"), (-0.15, "B-"),
    (-0.4, "C+"), (-0.7, "C"), (-1.0, "C-"),
    (-1.4, "D+"), (-1.8, "D"),
]


def grade_for(z: float) -> str:
    for cut, letter in GRADE_CUTS:
        if z >= cut:
            return letter
    return "F"


def best_lineup(
    roster: list[str],
    points: dict[str, float],
    positions: dict[str, str],
    schema: RosterSchema,
    replacement: dict[str, float] | None = None,
) -> tuple[float, int, dict[str, float]]:
    """Return (starting points, unfilled slots, points contributed per position).

    Greedy is exact here: dedicated slots take the best players at their own
    position, and flex slots can only ever take what those leave behind. Flex
    slots are filled most-restrictive-first so a SUPER_FLEX does not swallow a
    receiver that a REC_FLEX then has no way to replace.
    """
    replacement = replacement or {}
    by_pos: dict[str, list[float]] = {}
    for pid in roster:
        pos = positions.get(pid)
        if pos:
            by_pos.setdefault(pos, []).append(points.get(pid, 0.0))
    for pos in by_pos:
        by_pos[pos].sort(reverse=True)

    total = 0.0
    holes = 0
    used: dict[str, int] = {}
    contribution: dict[str, float] = {}

    for pos, need in schema.starters.items():
        if not need:
            continue
        have = by_pos.get(pos, [])
        take = have[:need]
        total += sum(take)
        contribution[pos] = contribution.get(pos, 0.0) + sum(take)
        used[pos] = len(take)
        missing = need - len(take)
        if missing:
            holes += missing
            # Streamed off waivers rather than a literal zero.
            total += missing * replacement.get(pos, 0.0)

    leftovers: list[tuple[float, str]] = []
    for pos, values in by_pos.items():
        leftovers += [(v, pos) for v in values[used.get(pos, 0):]]
    leftovers.sort(reverse=True)

    flex_slots = sorted(
        ((name, n) for name, n in schema.flex.items() if n),
        key=lambda kv: len(FLEX_ELIGIBILITY.get(kv[0], frozenset())),
    )
    for name, count in flex_slots:
        eligible_positions = FLEX_ELIGIBILITY.get(name, frozenset())
        for _ in range(count):
            pick = next(
                (i for i, (_, pos) in enumerate(leftovers) if pos in eligible_positions),
                None,
            )
            if pick is None:
                holes += 1
                continue
            value, pos = leftovers.pop(pick)
            total += value
            contribution[pos] = contribution.get(pos, 0.0) + value

    return total, holes, contribution


def rate_teams(state: DraftState, board: pd.DataFrame) -> pd.DataFrame:
    """Rate every roster in the draft on projected starting-lineup points."""
    indexed = board.set_index("player_id")
    points = indexed["proj_pts"].to_dict()
    positions = indexed["pos"].to_dict()
    replacement = (
        board.groupby("pos")["replacement_pts"].first().to_dict()
        if "replacement_pts" in board.columns else {}
    )

    rows = []
    for roster_id, team in state.teams_by_roster.items():
        total, holes, contribution = best_lineup(
            team.roster, points, positions, state.schema, replacement
        )
        rows.append({
            "roster_id": roster_id,
            "draft_slot": team.draft_slot,
            "starters_pts": total,
            "holes": holes,
            "picks": len(team.roster),
            **{f"pts_{pos}": value for pos, value in contribution.items()},
        })

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    spread = frame["starters_pts"].std()
    if spread and spread > 0:
        frame["z"] = (frame["starters_pts"] - frame["starters_pts"].mean()) / spread
    else:
        frame["z"] = 0.0
    frame["grade"] = [grade_for(v) for v in frame["z"]]
    return frame.sort_values("starters_pts", ascending=False).reset_index(drop=True)


def weakest_positions(
    frame: pd.DataFrame,
    roster_id: int,
    top_n: int = 2,
    *,
    threshold_sd: float = 0.75,
) -> list[str]:
    """Positions where this roster is *materially* behind the league.

    Ranking raw gaps alone is useless: every roster is below average somewhere,
    so the best team in the league still gets flagged "thin at WR, TE". A
    position only counts as thin if it trails by a meaningful fraction of how
    much that position actually varies across teams.
    """
    row = frame[frame["roster_id"] == roster_id]
    if row.empty:
        return []

    gaps = []
    for column in [c for c in frame.columns if c.startswith("pts_")]:
        values = frame[column].fillna(0.0)
        spread = float(values.std())
        if not spread or spread <= 0:
            continue
        mine = float(row[column].iloc[0] or 0.0)
        gap = mine - float(values.mean())
        if gap < -threshold_sd * spread:
            gaps.append((gap, column[len("pts_"):]))

    gaps.sort()
    return [pos for _, pos in gaps[:top_n]]
