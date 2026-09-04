"""Guards the forward-looking value maths.

VONA is the number the whole engine exists to produce, and it is easy to get
subtly wrong in ways that still look plausible on a board -- an off-by-one in
the exclusion, or a missing replacement tail, shifts every recommendation
without ever raising.
"""

import numpy as np
import pandas as pd
import pytest

from src.optimizer import add_vona, denial_value, expected_best_available


class TestExpectedBestAvailable:
    """Worked by hand so the implementation cannot drift silently."""

    POINTS = np.array([100.0, 90.0, 80.0])
    SURVIVAL = np.array([0.5, 0.5, 0.5])
    REPLACEMENT = 50.0

    def test_matches_hand_computed_values(self):
        got = expected_best_available(self.POINTS, self.SURVIVAL, self.REPLACEMENT)
        # F = [88.75, 77.5, 65, 50]; gone = [1, .5, .25]; H = [0, 50, 72.5]
        assert got == pytest.approx([77.5, 82.5, 85.0])

    def test_excludes_the_candidate_himself(self):
        """Drafting X is exactly what makes X unavailable next turn."""
        got = expected_best_available(self.POINTS, self.SURVIVAL, self.REPLACEMENT)
        # The best player's own 100 must not appear in his own expectation.
        assert got[0] < self.POINTS[0]
        assert got[0] == pytest.approx(77.5)

    def test_no_survivors_falls_back_to_replacement(self):
        """The tail the spec's formula left unaccounted for."""
        got = expected_best_available(
            self.POINTS, np.zeros(3), self.REPLACEMENT
        )
        assert got == pytest.approx([50.0, 50.0, 50.0])

    def test_certain_survival_means_you_get_the_best_one_left(self):
        got = expected_best_available(self.POINTS, np.ones(3), self.REPLACEMENT)
        # Excluding #1 you'd still get #2 (90); excluding #2 you'd get #1 (100).
        assert got == pytest.approx([90.0, 100.0, 100.0])

    def test_empty_pool(self):
        assert expected_best_available(np.zeros(0), np.zeros(0), 50.0).size == 0

    def test_expectation_never_exceeds_the_best_available(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            pts = np.sort(rng.uniform(50, 300, size=12))[::-1]
            surv = rng.uniform(0, 1, size=12)
            got = expected_best_available(pts, surv, 40.0)
            assert (got <= pts[0] + 1e-9).all()
            assert (got >= 40.0 - 1e-9).all()


class TestAddVona:
    @pytest.fixture
    def frame(self):
        return pd.DataFrame({
            "player_id": ["a", "b", "c"],
            "name": ["A", "B", "C"],
            "pos": ["RB"] * 3,
            "proj_pts": [100.0, 90.0, 80.0],
            "survival": [0.5, 0.5, 0.5],
            "replacement_pts": [50.0] * 3,
        })

    def test_vona_is_points_minus_expected(self, frame):
        out = add_vona(frame).set_index("player_id")
        assert out.loc["a", "vona"] == pytest.approx(22.5)
        assert out.loc["b", "vona"] == pytest.approx(7.5)
        assert out.loc["c", "vona"] == pytest.approx(-5.0)

    def test_certain_survival_gives_non_positive_vona(self, frame):
        """If he is certain to last, there is no urgency to take him now."""
        frame["survival"] = 1.0
        out = add_vona(frame).set_index("player_id")
        assert out.loc["b", "vona"] < 0
        assert out.loc["c", "vona"] < 0

    def test_positions_are_valued_independently(self):
        frame = pd.DataFrame({
            "player_id": ["r1", "r2", "w1", "w2"],
            "name": ["R1", "R2", "W1", "W2"],
            "pos": ["RB", "RB", "WR", "WR"],
            "proj_pts": [200.0, 100.0, 190.0, 185.0],
            "survival": [0.0, 0.0, 0.0, 0.0],
            "replacement_pts": [50.0, 50.0, 120.0, 120.0],
        })
        out = add_vona(frame).set_index("player_id")
        # Nobody survives, so each falls back to its own position's replacement.
        assert out.loc["r1", "vona"] == pytest.approx(150.0)
        assert out.loc["w1", "vona"] == pytest.approx(70.0)


class TestDenial:
    @pytest.fixture
    def state_and_board(self):
        from src.mock import mock_draft_object
        from src.state import DraftState
        from src.valuation import build_board

        rows = [{
            "player_id": f"p{i}", "name": f"P{i}",
            "pos": "RB" if i % 2 else "WR", "team": "X",
            "proj_pts": float(300 - i * 5), "adp": float(i + 1),
        } for i in range(40)]
        proj = pd.DataFrame(rows)
        draft, league = mock_draft_object(teams=12, rounds=15)
        state = DraftState(draft, proj, league=league)
        starters = {p: state.schema.total_starters(p)
                    for p in ("QB", "RB", "WR", "TE", "K", "DEF")}
        return state, build_board(proj, starters, state.teams)

    def test_denial_is_non_negative(self, state_and_board):
        state, board = state_and_board
        window = state.picks_until_my_turn(5)
        assert (denial_value(state, board, window) >= 0).all()

    def test_empty_window_denies_nothing(self, state_and_board):
        """At a turn boundary you pick again immediately -- nothing to deny."""
        state, board = state_and_board
        assert (denial_value(state, board, []) == 0).all()

    def test_scales_with_lambda(self, state_and_board):
        state, board = state_and_board
        window = state.picks_until_my_turn(5)
        low = denial_value(state, board, window, lam=0.1)
        high = denial_value(state, board, window, lam=0.5)
        assert high.sum() > low.sum()

    def test_stays_small_relative_to_value(self, state_and_board):
        """Petersen warns against joining runs; denial must not dominate."""
        state, board = state_and_board
        window = state.picks_until_my_turn(5)
        denial = denial_value(state, board, window)
        assert denial.max() < board["vorp"].max()
