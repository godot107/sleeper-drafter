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


FANTASY_POSITIONS = ("QB", "RB", "WR", "TE", "K", "DEF")


def slot_a_body_fills(team: Team, schema: RosterSchema) -> dict[str, str]:
    """Per position: does one more body fill a starting slot, or is it depth?

    Returns ``"starter"`` / ``"flex"`` / ``"depth"`` for each position.

    This must be asked of the *whole* roster rather than per position, because
    flex slots are shared across RB/WR/TE. The earlier implementation asked each
    position independently against ``schema.total_starters``, so a roster whose
    two FLEX slots were already filled -- one by a running back, one by a tight
    end -- still credited a fourth running back and a third tight end with a
    flex bonus for slots that no longer existed. Live, that ranked a pure bench
    body above a player who filled a real hole.

    Answered by asking the lineup solver whether an extra body at that position
    actually converts into points.
    """
    from .rating import best_lineup

    points: dict[str, float] = {}
    positions: dict[str, str] = {}
    roster: list[str] = []
    n = 0
    for pos, count in team.slot_counts.items():
        for _ in range(int(count)):
            pid = f"_{n}"
            roster.append(pid)
            points[pid] = 1.0
            positions[pid] = pos
            n += 1

    _, holes_before, _ = best_lineup(roster, points, positions, schema)

    result: dict[str, str] = {}
    for pos in FANTASY_POSITIONS:
        probe = "_probe"
        points[probe] = 1.0
        positions[probe] = pos
        _, holes_after, _ = best_lineup(roster + [probe], points, positions, schema)
        points.pop(probe)
        positions.pop(probe)

        if holes_after < holes_before:
            dedicated_open = schema.starters.get(pos, 0) - team.slot_counts.get(pos, 0)
            result[pos] = "starter" if dedicated_open > 0 else "flex"
        else:
            result[pos] = "depth"
    return result


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
    total_start = np.array([schema.total_starters(p) for p in positions], dtype=float)

    # Shared flex capacity, resolved against the whole roster (see docstring).
    role = slot_a_body_fills(team, schema)
    roles = np.array([role.get(p, "depth") for p in positions])

    fit = np.zeros(len(board), dtype=float)

    fills_starter = roles == "starter"
    fills_flex = roles == "flex"
    is_depth = roles == "depth"

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

    ranked = add_vona(ranked)
    ranked["denial"] = denial_value(state, ranked, window)

    # Score(X) = VONA + denial + roster fit. VORP stays as a visible sanity
    # column: VONA answers "take him now or later", VORP "is he any good".
    ranked["score"] = ranked["vona"] + ranked["denial"] + ranked["roster_fit"]
    return ranked.sort_values("score", ascending=False).reset_index(drop=True), window


def expected_best_available(
    points: np.ndarray,
    survival: np.ndarray,
    replacement: float,
) -> np.ndarray:
    """E[best player at this position at my next pick], excluding each candidate.

    ``points`` must be sorted descending; ``survival`` aligned to it.

    The spec's formula summed over the whole ranked pool, which has two
    problems. It leaves probability mass unaccounted for when *nobody*
    survives, and it includes the candidate himself -- but if you draft X, X is
    precisely who will not be there next time. Excluding X matters most for the
    best player at a position, where it is the difference between "take him
    now" and "he's the same guy you'd get later".

    Both are handled by a forward/backward pass rather than the O(n^2) rebuild:

        F[i] = s[i]*p[i] + (1-s[i])*F[i+1]      value of the pool from i on,
               F[n] = replacement                falling back to replacement
        G[i] = prod_{l<i} (1-s[l])              all better players gone
        H[i] = sum_{l<i} s[l]*p[l]*G[l]         value if one of them survives

        E_excluding_i = H[i] + G[i] * F[i+1]

    which is exact and linear.
    """
    n = points.size
    if n == 0:
        return np.zeros(0)

    # Backward: expected best from i onward, replacement if none survive.
    forward = np.empty(n + 1, dtype=float)
    forward[n] = replacement
    for i in range(n - 1, -1, -1):
        forward[i] = survival[i] * points[i] + (1.0 - survival[i]) * forward[i + 1]

    # Forward: probability every better player is gone, and the value if not.
    gone = np.empty(n, dtype=float)
    value_if_earlier_survives = np.empty(n, dtype=float)
    gone[0] = 1.0
    value_if_earlier_survives[0] = 0.0
    for i in range(1, n):
        gone[i] = gone[i - 1] * (1.0 - survival[i - 1])
        value_if_earlier_survives[i] = (
            value_if_earlier_survives[i - 1]
            + survival[i - 1] * points[i - 1] * gone[i - 1]
        )

    return value_if_earlier_survives + gone * forward[1:]


def add_vona(board: pd.DataFrame) -> pd.DataFrame:
    """Add ``vona`` = projected points minus expected best available next turn.

    Requires ``survival`` and ``replacement_pts`` columns.
    """
    out = board.copy()
    out["vona"] = np.nan

    for pos, group in out.groupby("pos"):
        ordered = group.sort_values("proj_pts", ascending=False)
        points = ordered["proj_pts"].to_numpy(dtype=float)
        survival = ordered["survival"].to_numpy(dtype=float)
        replacement = float(ordered["replacement_pts"].iloc[0])
        expected = expected_best_available(points, survival, replacement)
        out.loc[ordered.index, "vona"] = points - expected

    return out


def denial_value(
    state: DraftState,
    board: pd.DataFrame,
    window: list[int],
    *,
    lam: float | None = None,
) -> np.ndarray:
    """Expected value taken away from opponents by drafting each candidate.

    For every upcoming opponent pick, the chance they take X times what X would
    be worth *to them* over their own replacement at that position. Summed and
    scaled by lambda.

    Deliberately conservative. Petersen (Ch.7) warns against joining a run
    mid-stream, which is exactly what an aggressive denial term encourages, so
    lambda stays low and denial can nudge between comparable players rather
    than override value.
    """
    lam = settings.denial_lambda if lam is None else lam
    if board.empty or not window:
        return np.zeros(len(board))

    from .opponent_model import selection_probs

    points = board["proj_pts"].to_numpy(dtype=float)
    replacement = board["replacement_pts"].to_numpy(dtype=float)
    gain_to_them = np.clip(points - replacement, 0, None)

    denied = np.zeros(len(board), dtype=float)
    for pick_no in window:
        team = state.team_at(pick_no)
        denied += selection_probs(board, team, state.schema, pick_no) * gain_to_them

    return lam * denied
