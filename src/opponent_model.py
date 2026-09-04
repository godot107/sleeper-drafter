"""How likely is each player to survive until my next pick?

The spec defined the selection probability only up to proportionality::

    P(team i drafts X at pick j)  ∝  Urgency(i, pos(X)) · ADPWeight(X, j)

and never resolved the constant. Left unresolved the values sit on an arbitrary
scale, so the survival product is meaningless and every VONA number downstream
is arbitrary too. Exactly one player is taken at pick *j*, which pins it:

    P(X taken at j) = score(X, j) / Σ_{X' available} score(X', j)

That normalisation is the single most important correctness fix in this
project.

Two consequences worth noting:

* The Gaussian's ``1/√(2πσ²)`` factor is constant across players at fixed σ, so
  it cancels in the ratio. We compute the bare exponential.
* Survival multiplies across intermediate picks as if they were independent.
  Real drafts have **runs** -- three receivers in four picks -- so this is
  mildly optimistic mid-run. That is what the position-run warning is for, and
  it is stated in the README rather than hidden.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import settings
from .state import DraftState, RosterSchema, Team

logger = logging.getLogger(__name__)


def position_urgency(team: Team, schema: RosterSchema, pos: str) -> float:
    """How badly ``team`` still needs ``pos``, in [0, 1].

    A team that has filled every starting slot at a position is not indifferent
    -- it may still take a backup -- but it is far less likely to, hence the
    small floor rather than zero.
    """
    total = schema.total_starters(pos)
    if total <= 0:
        return settings.urgency_filled
    unfilled = max(0, total - team.slot_counts.get(pos, 0))
    if unfilled == 0:
        return settings.urgency_filled
    return max(settings.urgency_min, unfilled / total)


def adp_weight(adp: np.ndarray, pick_no: int, sigma: float | None = None) -> np.ndarray:
    """One-sided Gaussian kernel around consensus ADP.

    The spec used a symmetric Gaussian, which says a player is *less* likely to
    be taken the further he falls past his ADP. That is backwards. ADP is an
    average: a player still on the board well past it is overdue, and the next
    manager to need that position takes him immediately. So the kernel is flat
    at its maximum once ``adp <= pick_no`` and only decays for players being
    reached for ahead of schedule.

    Without this the engine badly overestimated survival for elite fallers --
    the ADP-1 player showed a 67% chance of lasting another 15 picks.

    The ``1/√(2πσ²)`` constant is omitted: it is identical for every player at
    fixed σ and cancels in the normalisation.
    """
    sigma = effective_sigma(pick_no) if sigma is None else sigma
    overdue = adp <= pick_no
    return np.where(overdue, 1.0, np.exp(-((adp - pick_no) ** 2) / (2.0 * sigma**2)))


def effective_sigma(pick_no: int) -> float:
    """ADP uncertainty, which grows as the draft goes on.

    A fixed σ is wrong at both ends. The top of round 1 is highly predictable --
    consensus is tight and the board goes roughly to ADP -- while by round 10
    boards have diverged and ADP means little. Holding σ at 8 made early picks
    far too uncertain: the ADP-5 player showed a 41% chance of surviving 14
    picks, when in practice he is gone almost immediately.

    Modelled as σ growing linearly with pick number, floored so it never
    collapses to certainty.
    """
    return max(settings.adp_sigma_min, settings.adp_sigma_frac * pick_no)


def mandatory_positions(team: Team, schema: RosterSchema, round_no: int) -> set[str] | None:
    """Positions this team is forced to draft, or ``None`` if it is still free.

    Mirrors the constraint the optimizer applies to our own roster. Opponents
    obey it too, because real managers do: nobody finishes a draft without a
    kicker. Modelling that matters for survival estimates -- without it the
    engine cannot foresee the late K/DEF run and will overrate the chance that
    a marginal skill player lasts to the final rounds.
    """
    picks_left = schema.rounds - round_no + 1
    unfilled = {
        pos: max(0, schema.starters.get(pos, 0) - team.slot_counts.get(pos, 0))
        for pos in schema.starters
    }
    mandatory = sum(unfilled.values())
    if mandatory and picks_left <= mandatory:
        return {pos for pos, n in unfilled.items() if n > 0}
    return None


def selection_probs(
    board: pd.DataFrame,
    team: Team,
    schema: RosterSchema,
    pick_no: int,
) -> np.ndarray:
    """Normalised probability that each available player goes at ``pick_no``."""
    if board.empty:
        return np.zeros(0)

    positions = board["pos"].to_numpy()
    urgency = np.array(
        [position_urgency(team, schema, p) for p in positions],
        dtype=float,
    )
    adp = board["adp"].to_numpy(dtype=float)
    score = urgency * adp_weight(adp, pick_no)

    # Restrict to a realistic consideration set. Nobody at pick 6 is weighing
    # the ADP-200 player, but normalising across the whole 600-player pool
    # hands him probability mass anyway -- which drains mass from the players
    # who will actually go, and inflates their survival. Capping consideration
    # to the top of the board by ADP is what makes survival numbers behave:
    # before this, the ADP-1 player showed a 44% chance of lasting 22 picks.
    if len(board) > settings.consideration_set:
        cutoff = np.partition(adp, settings.consideration_set)[settings.consideration_set]
        score = np.where(adp <= cutoff, score, 0.0)

    round_no = (pick_no - 1) // max(schema.teams, 1) + 1
    required = mandatory_positions(team, schema, round_no)
    if required:
        score = np.where(np.isin(positions, tuple(required)), score, 0.0)
        if score.sum() <= 0:
            # Forced position, but every candidate is far from its ADP. Spread
            # mass over the required positions by projected points instead.
            eligible = np.isin(positions, tuple(required)).astype(float)
            pts = np.clip(board["proj_pts"].to_numpy(dtype=float), 0, None) * eligible
            if pts.sum() > 0:
                return pts / pts.sum()

    total = score.sum()
    if total <= 0:
        # Every candidate is far from its ADP (deep in the draft, or a pool of
        # undrafted-flagged players). Fall back to projected points so the
        # distribution stays proper instead of collapsing to zeros.
        pts = board["proj_pts"].to_numpy(dtype=float)
        pts = np.clip(pts, 0, None)
        total = pts.sum()
        return pts / total if total > 0 else np.full(len(board), 1.0 / len(board))

    return score / total


def survival_probabilities(
    state: DraftState,
    board: pd.DataFrame,
    pick_window: list[int],
) -> np.ndarray:
    """P(each player in ``board`` is still available after ``pick_window``).

    ``pick_window`` is the list of intermediate picks from
    :meth:`DraftState.picks_until_my_turn`. An empty window (a turn boundary)
    means certain survival.
    """
    if board.empty:
        return np.zeros(0)
    survive = np.ones(len(board), dtype=float)
    for pick_no in pick_window:
        team = state.team_at(pick_no)
        probs = selection_probs(board, team, state.schema, pick_no)
        survive *= 1.0 - probs
    return survive


def sample_pick(
    board: pd.DataFrame,
    team: Team,
    schema: RosterSchema,
    pick_no: int,
    rng: np.random.Generator,
) -> str:
    """Draw one player id from the selection distribution.

    Used by ``--mock`` to advance opponents. Sampling from the same
    distribution the model assumes is deliberately charitable -- it validates
    the arithmetic, not the realism. Only a live mock draft tests realism.
    """
    probs = selection_probs(board, team, schema, pick_no)
    idx = rng.choice(len(board), p=probs)
    return str(board.iloc[idx]["player_id"])
