"""Offline mock drafts, for validating the engine before it matters.

This exercises the arithmetic end to end -- pick order, roster accounting,
survival, ranking -- with no network and no draft clock. What it deliberately
does *not* validate is realism: opponents are sampled from the very
distribution the opponent model assumes, so a mock draft can never disconfirm
that model. Only a live Sleeper mock draft tests whether real humans behave
like it. Both are on the draft-night runbook for that reason.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .state import DraftState
from .opponent_model import sample_pick

logger = logging.getLogger(__name__)

DEFAULT_ROSTER = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "K", "DEF"]


def mock_draft_object(
    *,
    teams: int = 12,
    rounds: int = 15,
    scoring: str = "ppr",
    roster_positions: list[str] | None = None,
) -> tuple[dict, dict]:
    """Build draft + league dicts shaped exactly like Sleeper's payloads."""
    roster = list(roster_positions or DEFAULT_ROSTER)
    roster += ["BN"] * max(0, rounds - len(roster))

    draft = {
        "draft_id": "mock",
        "type": "snake",
        "status": "drafting",
        "season": "2026",
        "league_id": "mock-league",
        "metadata": {"name": "Mock Draft", "scoring_type": scoring},
        "settings": {"teams": teams, "rounds": rounds},
        "slot_to_roster_id": {str(s): s for s in range(1, teams + 1)},
    }
    league = {"league_id": "mock-league", "roster_positions": roster}
    return draft, league


def simulate(
    state: DraftState,
    board: pd.DataFrame,
    my_slot: int,
    *,
    rng: np.random.Generator | None = None,
    on_my_pick=None,
) -> list[dict]:
    """Run a full mock draft, returning every pick made.

    ``on_my_pick(state, available, round_no) -> player_id`` chooses for the user;
    if omitted the engine's own top recommendation is taken, which is what makes
    this a regression test of the recommender rather than just the plumbing.
    """
    rng = rng or np.random.default_rng(0)
    picks: list[dict] = []

    for pick_no in range(1, state.total_picks + 1):
        available = board[~board["player_id"].isin(state.drafted)]
        if available.empty:
            break

        slot = state.slot_at(pick_no)
        team = state.team_at(pick_no)
        round_no = (pick_no - 1) // state.teams + 1

        if slot == my_slot and on_my_pick is not None:
            player_id = on_my_pick(state, available, round_no)
        elif slot == my_slot:
            from .optimizer import rank_static

            ranked = rank_static(available, team, state.schema, round_no)
            player_id = str(ranked.iloc[0]["player_id"])
        else:
            player_id = sample_pick(available, team, state.schema, pick_no, rng)

        pick = {
            "pick_no": pick_no,
            "round": round_no,
            "draft_slot": slot,
            "roster_id": state.roster_at(pick_no),
            "player_id": player_id,
            "picked_by": "",
            "is_keeper": None,
        }
        state.ingest([pick])
        picks.append(pick)

    return picks
