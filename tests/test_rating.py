"""Guards the live draft grades.

A grade people act on has to be right about the lineup it assumes. The greedy
fill is exact for Sleeper's slot structures, but only if flex slots are filled
in the correct order -- a SUPER_FLEX taking a receiver that a REC_FLEX then
cannot replace silently understates the roster.
"""

import pandas as pd
import pytest

from src.rating import best_lineup, grade_for, rate_teams, weakest_positions
from src.state import RosterSchema

POINTS = {
    "qb1": 300.0, "qb2": 280.0,
    "rb1": 250.0, "rb2": 200.0, "rb3": 150.0,
    "wr1": 240.0, "wr2": 220.0, "wr3": 180.0, "wr4": 120.0,
    "te1": 190.0, "te2": 100.0,
    "k1": 120.0, "def1": 110.0,
}
POSITIONS = {
    "qb1": "QB", "qb2": "QB",
    "rb1": "RB", "rb2": "RB", "rb3": "RB",
    "wr1": "WR", "wr2": "WR", "wr3": "WR", "wr4": "WR",
    "te1": "TE", "te2": "TE",
    "k1": "K", "def1": "DEF",
}


@pytest.fixture
def schema():
    return RosterSchema(
        teams=12, rounds=15,
        starters={"QB": 1, "RB": 2, "WR": 3, "TE": 1, "K": 1, "DEF": 1},
        flex={"FLEX": 1}, bench=5,
    )


class TestBestLineup:
    def test_fills_dedicated_slots_with_the_best_players(self, schema):
        roster = ["qb1", "qb2", "rb1", "rb2", "wr1", "wr2", "wr3",
                  "te1", "k1", "def1"]
        total, holes, _ = best_lineup(roster, POINTS, POSITIONS, schema)
        # QB1 + RB1,2 + WR1,2,3 + TE1 + K + DEF, then QB2 is flex-ineligible so
        # the FLEX slot goes empty -> one hole.
        assert total == pytest.approx(300 + 250 + 200 + 240 + 220 + 180 + 190 + 120 + 110)
        assert holes == 1

    def test_flex_takes_the_best_leftover(self, schema):
        roster = ["qb1", "rb1", "rb2", "rb3", "wr1", "wr2", "wr3",
                  "te1", "k1", "def1"]
        total, holes, _ = best_lineup(roster, POINTS, POSITIONS, schema)
        assert holes == 0
        # rb3 (150) is the only leftover eligible for FLEX.
        assert total == pytest.approx(300 + 250 + 200 + 240 + 220 + 180 + 190 + 120 + 110 + 150)

    def test_holes_are_scored_at_replacement_not_zero(self, schema):
        roster = ["rb1", "rb2", "wr1", "wr2", "wr3", "te1", "k1", "def1"]
        replacement = {"QB": 275.0}
        no_rep, holes_a, _ = best_lineup(roster, POINTS, POSITIONS, schema)
        with_rep, holes_b, _ = best_lineup(
            roster, POINTS, POSITIONS, schema, replacement
        )
        assert holes_a == holes_b  # a hole is still reported either way
        assert with_rep == pytest.approx(no_rep + 275.0)

    def test_superflex_can_start_a_second_quarterback(self):
        schema = RosterSchema(
            teams=12, rounds=15,
            starters={"QB": 1, "RB": 2, "WR": 3, "TE": 1, "K": 1, "DEF": 1},
            flex={"SUPER_FLEX": 1}, bench=5,
        )
        roster = ["qb1", "qb2", "rb1", "rb2", "wr1", "wr2", "wr3",
                  "te1", "k1", "def1"]
        total, holes, contribution = best_lineup(roster, POINTS, POSITIONS, schema)
        assert holes == 0
        assert contribution["QB"] == pytest.approx(580.0)  # both QBs start

    def test_restrictive_flex_is_filled_first(self):
        """A SUPER_FLEX must not eat the receiver a REC_FLEX needs."""
        schema = RosterSchema(
            teams=12, rounds=15,
            starters={"QB": 1, "RB": 1, "WR": 1, "TE": 1, "K": 0, "DEF": 0},
            flex={"SUPER_FLEX": 1, "REC_FLEX": 1}, bench=5,
        )
        roster = ["qb1", "qb2", "rb1", "wr1", "wr2", "te1"]
        total, holes, _ = best_lineup(roster, POINTS, POSITIONS, schema)
        assert holes == 0
        # REC_FLEX takes wr2 (220); SUPER_FLEX then takes qb2 (280).
        assert total == pytest.approx(300 + 250 + 240 + 190 + 220 + 280)

    def test_empty_roster_is_all_holes(self, schema):
        total, holes, _ = best_lineup([], POINTS, POSITIONS, schema)
        assert total == 0.0
        assert holes == 10  # 9 dedicated + 1 flex


class TestGradeCurve:
    def test_monotone(self):
        grades = [grade_for(z) for z in (2.0, 1.2, 0.5, 0.0, -0.5, -1.2, -2.0)]
        assert grades[0] == "A+"
        assert grades[-1] == "F"
        assert len(set(grades)) == len(grades)

    def test_average_team_lands_in_the_b_range(self):
        """The curve is deliberately a shade harsh: dead average is a B-.

        Real draft graders skew the same way, and a curve where half the league
        gets a B or better reads as flattery rather than information.
        """
        assert grade_for(0.0) == "B-"
        assert grade_for(0.2) == "B"


class TestRateTeams:
    @pytest.fixture
    def rated(self):
        from src.mock import mock_draft_object, simulate
        from src.state import DraftState
        from src.valuation import build_board
        import numpy as np

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
        starters = {p: state.schema.total_starters(p)
                    for p in ("QB", "RB", "WR", "TE", "K", "DEF")}
        board = build_board(proj, starters, state.teams)
        simulate(state, board, my_slot=5, rng=np.random.default_rng(3))
        return rate_teams(state, board), state, board

    def test_every_team_is_rated_and_sorted(self, rated):
        frame, state, _ = rated
        assert len(frame) == state.teams
        assert frame["starters_pts"].is_monotonic_decreasing

    def test_grades_span_the_league(self, rated):
        frame, _, _ = rated
        assert frame["grade"].nunique() > 1

    def test_the_best_team_is_not_flagged_thin_everywhere(self, rated):
        """Every roster is below average somewhere; that is not a weakness."""
        frame, _, _ = rated
        best = int(frame.iloc[0]["roster_id"])
        worst = int(frame.iloc[-1]["roster_id"])
        assert len(weakest_positions(frame, best)) <= len(weakest_positions(frame, worst))

    def test_empty_draft_rates_without_crashing(self):
        from src.mock import mock_draft_object
        from src.state import DraftState
        from src.valuation import build_board

        rows = [{"player_id": f"p{i}", "name": f"P{i}", "pos": "RB", "team": "X",
                 "proj_pts": 100.0, "adp": float(i + 1)} for i in range(30)]
        proj = pd.DataFrame(rows)
        draft, league = mock_draft_object(teams=12, rounds=15)
        state = DraftState(draft, proj, league=league)
        board = build_board(proj, {"RB": 2}, 12)
        frame = rate_teams(state, board)
        assert len(frame) == 12
        assert (frame["picks"] == 0).all()
