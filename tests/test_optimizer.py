"""Guards roster construction.

The invariant that matters most: a completed draft must be able to field a
legal starting lineup. The first mock draft could not -- it ended with no QB,
K or DEF -- and nothing in the ranking maths was wrong. These tests exist so
that failure mode cannot come back silently.
"""

import numpy as np
import pandas as pd
import pytest

from src.mock import mock_draft_object, simulate
from src.optimizer import (
    HARD_BLOCK,
    rank_static,
    rank_with_lookahead,
    roster_fit,
)
from src.state import DraftState, RosterSchema, Team
from src.valuation import build_board

STARTERS = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "K": 1, "DEF": 1}


@pytest.fixture
def schema():
    return RosterSchema(
        teams=12, rounds=15,
        starters=dict(STARTERS), flex={"FLEX": 1}, bench=5,
    )


@pytest.fixture
def board():
    rows = []
    for pos, pts in [("QB", 300), ("RB", 280), ("WR", 270), ("TE", 200),
                     ("K", 120), ("DEF", 100)]:
        for i in range(4):
            rows.append({
                "player_id": f"{pos}{i}", "name": f"{pos}{i}", "pos": pos,
                "team": "X", "proj_pts": float(pts - i * 10), "adp": float(i + 1),
                "vorp": float(pts - i * 10 - 50), "risk": "neutral",
            })
    return pd.DataFrame(rows)


class TestRosterFit:
    def test_unfilled_starter_gets_bonus(self, board, schema):
        fit = roster_fit(board, Team(1, 1), schema, round_no=1)
        assert (fit[board.pos == "RB"] > 0).all()

    def test_depth_is_penalised(self, board, schema):
        full = Team(1, 1, slot_counts={"RB": 5})
        fit = roster_fit(board, full, schema, round_no=5)
        assert (fit[board.pos == "RB"] < 0).all()

    def test_depth_penalty_survives_the_final_round(self, board, schema):
        """It used to vanish at round 15, which stacked a fifth tight end."""
        full = Team(1, 1, slot_counts={"RB": 5, "QB": 1, "WR": 3, "TE": 1,
                                       "K": 1, "DEF": 1})
        fit = roster_fit(board, full, schema, round_no=15)
        assert (fit[board.pos == "RB"] < 0).all()

    def test_kdef_blocked_early(self, board, schema):
        fit = roster_fit(board, Team(1, 1), schema, round_no=3)
        assert (fit[board.pos.isin(["K", "DEF"])] == HARD_BLOCK).all()

    def test_kdef_allowed_late(self, board, schema):
        team = Team(1, 1, slot_counts={"QB": 1, "RB": 2, "WR": 3, "TE": 1})
        fit = roster_fit(board, team, schema, round_no=14)
        assert (fit[board.pos == "K"] > HARD_BLOCK).all()

    def test_risk_nudge_follows_petersen(self, schema):
        """Steady preferred for a starting slot, volatile for bench depth."""
        rows = [
            {"player_id": "a", "name": "A", "pos": "RB", "team": "X",
             "proj_pts": 200.0, "adp": 10.0, "vorp": 50.0, "risk": "steady"},
            {"player_id": "b", "name": "B", "pos": "RB", "team": "X",
             "proj_pts": 200.0, "adp": 10.0, "vorp": 50.0, "risk": "volatile"},
        ]
        frame = pd.DataFrame(rows)
        starting = roster_fit(frame, Team(1, 1), schema, round_no=2)
        assert starting.iloc[0] > starting.iloc[1]

        # Starters all filled, so the mandatory constraint is dormant and we
        # are genuinely shopping for bench upside.
        deep = Team(1, 1, slot_counts={"QB": 1, "RB": 5, "WR": 3, "TE": 1,
                                       "K": 1, "DEF": 1})
        bench = roster_fit(frame, deep, schema, round_no=10)
        assert bench.iloc[1] > bench.iloc[0]


class TestMandatorySlotConstraint:
    def test_forces_unfilled_slots_when_picks_run_out(self, board, schema):
        # Round 14 of 15 -> 2 picks left, and 2 starting slots unfilled (K, DEF).
        team = Team(1, 1, slot_counts={"QB": 1, "RB": 2, "WR": 3, "TE": 1})
        fit = roster_fit(board, team, schema, round_no=14)
        assert (fit[board.pos.isin(["K", "DEF"])] > HARD_BLOCK).all()
        assert (fit[board.pos.isin(["RB", "WR", "TE"])] == HARD_BLOCK).all()

    def test_does_not_fire_while_picks_remain(self, board, schema):
        team = Team(1, 1, slot_counts={"QB": 1, "RB": 2, "WR": 3, "TE": 1})
        fit = roster_fit(board, team, schema, round_no=8)
        assert (fit[board.pos == "RB"] > HARD_BLOCK).all()

    def test_overrides_the_kdef_gate(self, board, schema):
        """Fielding a kicker beats waiting for a better one you'll never take."""
        team = Team(1, 1, slot_counts={"QB": 1, "RB": 2, "WR": 3, "TE": 1,
                                       "DEF": 1})
        fit = roster_fit(board, team, schema, round_no=15)
        assert (fit[board.pos == "K"] > HARD_BLOCK).all()


class TestRankStatic:
    def test_sorted_by_score(self, board, schema):
        ranked = rank_static(board, Team(1, 1), schema, round_no=1)
        assert ranked["score"].is_monotonic_decreasing

    def test_blocked_positions_rank_last(self, board, schema):
        ranked = rank_static(board, Team(1, 1), schema, round_no=1)
        assert ranked.tail(8)["pos"].isin(["K", "DEF"]).all()


@pytest.fixture(scope="module")
def finished():
    """A completed 12x15 mock draft, shared across the integration checks."""
    rows = []
    for pos, base, n in [("QB", 320, 40), ("RB", 330, 60), ("WR", 315, 70),
                         ("TE", 255, 40), ("K", 125, 30), ("DEF", 110, 30)]:
        for i in range(n):
            rows.append({
                "player_id": f"{pos}{i}", "name": f"{pos}{i}", "pos": pos,
                "team": "X", "proj_pts": float(base - i * 4),
                "adp": float(i * 3 + 1),
            })
    proj = pd.DataFrame(rows)
    draft, league = mock_draft_object(teams=12, rounds=15)
    state = DraftState(draft, proj, league=league)
    starters = {p: state.schema.total_starters(p) for p in STARTERS}
    board = build_board(proj, starters, state.teams)
    simulate(state, board, my_slot=5, rng=np.random.default_rng(7))
    return state, board


class TestFullMockDraft:
    """The integration guard: a finished draft must field a legal lineup."""

    def test_every_team_can_field_a_legal_lineup(self, finished):
        state, _ = finished
        for roster_id, team in state.teams_by_roster.items():
            for pos, need in STARTERS.items():
                assert team.slot_counts.get(pos, 0) >= need, (
                    f"roster {roster_id} cannot start {need} {pos}"
                )

    def test_no_player_drafted_twice(self, finished):
        state, _ = finished
        picked = [p["player_id"] for p in state.picks]
        assert len(picked) == len(set(picked))

    def test_roster_sizes_respect_the_board(self, finished):
        state, _ = finished
        for team in state.teams_by_roster.values():
            assert len(team.roster) <= state.rounds


class TestSurvivalCalibration:
    """Survival must be monotone in ADP and bounded.

    Three separate bugs made it wrong in ways that still looked plausible: a
    symmetric ADP kernel that made fallen elites *less* likely to be taken,
    normalising across all ~600 players instead of a realistic consideration
    set, and a fixed sigma that treated round 1 as being as unpredictable as
    round 10.
    """

    @pytest.fixture
    def state_and_board(self):
        rows = []
        for i in range(60):
            rows.append({
                "player_id": f"p{i}", "name": f"P{i}", "pos": "RB" if i % 2 else "WR",
                "team": "X", "proj_pts": float(300 - i * 3), "adp": float(i + 1),
            })
        proj = pd.DataFrame(rows)
        draft, league = mock_draft_object(teams=12, rounds=15)
        state = DraftState(draft, proj, league=league)
        starters = {p: state.schema.total_starters(p) for p in STARTERS}
        return state, build_board(proj, starters, state.teams)

    def test_probabilities_are_bounded(self, state_and_board):
        state, board = state_and_board
        ranked, _ = rank_with_lookahead(state, board, my_slot=5)
        assert ((ranked["survival"] >= 0) & (ranked["survival"] <= 1)).all()

    def test_early_adp_survives_less_than_late_adp(self, state_and_board):
        state, board = state_and_board
        ranked, window = rank_with_lookahead(state, board, my_slot=5)
        assert window, "slot 5 should have a non-empty window at pick 1"
        early = ranked[ranked["adp"] <= 10]["survival"].mean()
        late = ranked[ranked["adp"] >= 50]["survival"].mean()
        assert early < late

    def test_turn_boundary_means_certain_survival(self, state_and_board):
        state, board = state_and_board
        # Slot 12 picks 12 and 13 back to back -> nothing intervenes.
        for i in range(1, 12):
            state.ingest([{"pick_no": i, "round": 1, "draft_slot": i,
                           "roster_id": i, "player_id": f"p{i - 1}"}])
        ranked, window = rank_with_lookahead(state, board, my_slot=12)
        assert window == []
        assert (ranked["survival"] == 1.0).all()

    def test_sigma_grows_through_the_draft(self):
        from src.opponent_model import effective_sigma
        assert effective_sigma(5) < effective_sigma(50) < effective_sigma(150)


class TestBenchInference:
    """A standalone mock draft exposes no slots_bn.

    Leaving bench at 0 made every round look like a starting slot, so the
    mandatory-slot constraint fired ~3 rounds early and started forcing K/DEF
    in round 10 of a 15-round draft.
    """

    def test_infers_bench_when_slots_bn_is_absent(self):
        draft = {
            "draft_id": "x", "type": "snake",
            "settings": {"teams": 10, "rounds": 15, "slots_qb": 1, "slots_rb": 2,
                         "slots_wr": 2, "slots_te": 1, "slots_flex": 2,
                         "slots_k": 1, "slots_def": 1},
        }
        schema = RosterSchema.from_league(draft, None)
        assert sum(schema.starters.values()) + sum(schema.flex.values()) == 10
        assert schema.bench == 5  # 15 rounds - 10 starting slots

    def test_declared_bench_still_wins(self):
        draft = {
            "draft_id": "x", "type": "snake",
            "settings": {"teams": 10, "rounds": 15, "slots_qb": 1, "slots_rb": 2,
                         "slots_wr": 2, "slots_te": 1, "slots_flex": 2,
                         "slots_k": 1, "slots_def": 1, "slots_bn": 6},
        }
        assert RosterSchema.from_league(draft, None).bench == 6

    def test_kdef_not_forced_too_early_in_a_standalone_mock(self):
        draft = {
            "draft_id": "x", "type": "snake",
            "settings": {"teams": 10, "rounds": 15, "slots_qb": 1, "slots_rb": 2,
                         "slots_wr": 2, "slots_te": 1, "slots_flex": 2,
                         "slots_k": 1, "slots_def": 1},
        }
        schema = RosterSchema.from_league(draft, None)
        frame = pd.DataFrame([
            {"player_id": "a", "name": "A", "pos": "RB", "team": "X",
             "proj_pts": 200.0, "adp": 40.0, "vorp": 60.0, "risk": "steady"},
        ])
        # Round 10 with a bare roster: still far too early to be forced into K/DEF.
        fit = roster_fit(frame, Team(1, 1), schema, round_no=10)
        assert fit.iloc[0] > HARD_BLOCK
