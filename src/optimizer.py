"""Ranking: what should I actually take with this pick?

Phase 3 implements the static half -- VORP plus roster fit -- which is already
a usable draft aid. Phase 4 layers VONA and denial on top; the ``Score``
column is assembled here so adding those terms does not reshape the pipeline.

``RosterFitBonus`` was named but never defined in the original spec. Without it
a pure value sort happily recommends a fourth running back in round 6, because
the fourth back really does have more raw VORP than the tight end you still
need to start. It is defined here as:

* a bonus for filling an unfilled **dedicated starting** slot,
* a smaller bonus for filling a **flex** slot,
* an escalating **penalty** for depth past what you can start, scaled by round
  (harmless in round 14, disqualifying in round 2),
* a **hard block** on K/DEF until the final rounds -- justified by Petersen
  (§7.4.1), who measures kickers and defenses as having the lowest dropoff of
  any position, so there is no scarcity cost to waiting,
* a **mandatory-slot constraint** once picks run short (see below),
* a **risk adjustment** from Petersen Ch. 6: steady players are worth slightly
  more in a starting slot, volatile players slightly more once you are drafting
  bench upside.

All magnitudes are in projected-fantasy-point units so they are commensurate
with VORP, and all live in ``config.Settings``.

**Why a constraint and not just a bigger bonus.** The first mock draft finished
with six RBs, five TEs, and no QB, K or DEF at all. That was not mis-tuning: it
is what VORP correctly implies. With a bench-cohort baseline, every QB outside
the top dozen has *negative* VORP in a 1QB league -- streaming QB13-24 off
waivers really is better than spending a pick on QB25 -- so no additive bonus
short of an arbitrary one ever makes the model want one. But a roster that
cannot fill its starting QB slot scores zero there on Sunday. The requirement
to field a legal lineup is a hard constraint, so it is enforced as one: when
the picks remaining are no more than the mandatory starting slots still
unfilled, the candidate pool is restricted to positions that fill them. That is
also what a human does -- "three picks left, no QB or K or DEF, I take them
now".
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import settings
from .state import DraftState, Team, RosterSchema

logger = logging.getLogger(__name__)

# Large enough to sink a player to the bottom without poisoning arithmetic.
HARD_BLOCK = -1e6


def roster_fit(
    board: pd.DataFrame,
    team: Team,
    schema: RosterSchema,
    round_no: int,
    *,
    starter_bonus: float | None = None,
    flex_bonus: float | None = None,
    depth_penalty: float | None = None,
    risk_weight: float | None = None,
) -> pd.Series:
    """Vectorised roster-fit adjustment, in projected-point units."""
    starter_bonus = settings.starter_bonus if starter_bonus is None else starter_bonus
    flex_bonus = settings.flex_bonus if flex_bonus is None else flex_bonus
    depth_penalty = settings.depth_penalty if depth_penalty is None else depth_penalty
    risk_weight = settings.risk_weight if risk_weight is None else risk_weight

    positions = board["pos"].to_numpy()
    owned = np.array([team.slot_counts.get(p, 0) for p in positions], dtype=float)
    dedicated = np.array([schema.starters.get(p, 0) for p in positions], dtype=float)
    total_start = np.array([schema.total_starters(p) for p in positions], dtype=float)

    fit = np.zeros(len(board), dtype=float)

    fills_starter = owned < dedicated
    fills_flex = (~fills_starter) & (owned < total_start)
    is_depth = owned >= total_start

    fit[fills_starter] = starter_bonus
    fit[fills_flex] = flex_bonus

    # Depth hurts most early: in round 2 a redundant RB costs you a starter you
    # will never fill; by round 14 you are supposed to be taking upside. The
    # floor matters -- at zero, the final rounds had no depth penalty at all and
    # the engine stacked a fifth tight end.
    rounds_left_frac = max(
        settings.depth_penalty_floor,
        (schema.rounds - round_no) / max(schema.rounds, 1),
    )
    depth_over = np.maximum(0.0, owned - total_start + 1)
    fit[is_depth] -= (depth_penalty * depth_over * rounds_left_frac)[is_depth]

    # Petersen Ch.6: steady for starters, volatile for bench upside.
    if "risk" in board.columns and risk_weight:
        risk = board["risk"].to_numpy()
        starting = fills_starter | fills_flex
        fit[starting & (risk == "steady")] += risk_weight
        fit[starting & (risk == "volatile")] -= risk_weight
        fit[is_depth & (risk == "volatile")] += risk_weight

    # K/DEF have the lowest measured dropoff, so waiting costs nothing.
    gate_round = schema.rounds - settings.kdef_gate_from_end
    if round_no <= gate_round:
        fit[np.isin(positions, ("K", "DEF"))] = HARD_BLOCK

    # Mandatory-slot constraint. Once you have only as many picks left as you
    # have unfilled starting slots, every remaining pick must fill one.
    picks_left = schema.rounds - round_no + 1
    unfilled = {
        pos: max(0, schema.starters.get(pos, 0) - team.slot_counts.get(pos, 0))
        for pos in schema.starters
    }
    mandatory = sum(unfilled.values())
    if mandatory and picks_left <= mandatory:
        required = {pos for pos, n in unfilled.items() if n > 0}
        logger.debug(
            "round %d: %d picks left, %d slots unfilled -> restricting to %s",
            round_no, picks_left, mandatory, sorted(required),
        )
        forced = np.isin(positions, tuple(required))
        # Overrides the K/DEF gate: fielding a kicker beats waiting for one.
        fit = np.where(forced, np.maximum(fit, starter_bonus), HARD_BLOCK)

    return pd.Series(fit, index=board.index, name="roster_fit")


def rank_static(
    board: pd.DataFrame,
    team: Team,
    schema: RosterSchema,
    round_no: int,
) -> pd.DataFrame:
    """Rank available players on VORP + roster fit (no lookahead yet).

    This is the Phase 3 recommender: enough to draft with, and the baseline
    that VONA has to beat.
    """
    out = board.copy()
    out["roster_fit"] = roster_fit(out, team, schema, round_no)
    out["vona"] = np.nan       # filled in by the Phase 4 engine
    out["denial"] = 0.0
    out["score"] = out["vorp"] + out["roster_fit"]
    return out.sort_values("score", ascending=False).reset_index(drop=True)


def recommend(
    state: DraftState,
    board: pd.DataFrame,
    my_slot: int,
    *,
    top_n: int = 10,
) -> pd.DataFrame:
    """Top ``top_n`` recommendations for the roster at ``my_slot`` right now."""
    available = board[~board["player_id"].isin(state.drafted)]
    team = state.team_at(state.current_pick_no) if state.is_my_turn(my_slot) else \
        state.teams_by_roster[state.slot_to_roster.get(my_slot, my_slot)]
    round_no = (state.current_pick_no - 1) // state.teams + 1
    ranked = rank_static(available, team, state.schema, round_no)
    return ranked.head(top_n)


def rank_with_lookahead(
    state: DraftState,
    board: pd.DataFrame,
    my_slot: int,
) -> tuple[pd.DataFrame, list[int]]:
    """Rank available players, annotated with survival to the user's next turn.

    Returns the ranked frame and the intermediate-pick window it was computed
    over, so the caller can show "N picks until your turn" from the same source
    of truth the maths used.
    """
    from .opponent_model import survival_probabilities

    available = board[~board["player_id"].isin(state.drafted)].copy()
    window = state.picks_until_my_turn(my_slot)

    roster_id = state.slot_to_roster.get(my_slot, my_slot)
    team = state.teams_by_roster.setdefault(roster_id, Team(roster_id, my_slot))
    round_no = (state.current_pick_no - 1) // state.teams + 1

    ranked = rank_static(available, team, state.schema, round_no)
    # survival_probabilities preserves row order, so align on the sorted frame.
    ranked["survival"] = survival_probabilities(state, ranked, window)
    return ranked, window
