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


class TestSharedFlexCapacity:
    """Flex slots are shared across RB/WR/TE, not per position.

    Found live: with both FLEX slots already filled -- one by a running back,
    one by a tight end -- the engine still credited a fourth RB and a third TE
    with a flex bonus, and ranked a pure bench body above a receiver who filled
    a genuinely empty starting slot.
    """

    @pytest.fixture
    def schema(self):
        return RosterSchema(
            teams=10, rounds=15,
            starters={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DEF": 1},
            flex={"FLEX": 2}, bench=5,
        )

    def test_full_flex_makes_extra_bodies_depth(self, schema):
        from src.optimizer import slot_a_body_fills
        # 3 RB (2 start + 1 flex), 2 TE (1 starts + 1 flex), 1 WR -> both flex used
        team = Team(1, 1, slot_counts={"RB": 3, "TE": 2, "QB": 1, "WR": 1})
        role = slot_a_body_fills(team, schema)
        assert role["RB"] == "depth"
        assert role["TE"] == "depth"
        assert role["WR"] == "starter"   # WR2 is still open
        assert role["K"] == "starter"
        assert role["DEF"] == "starter"

    def test_open_flex_is_still_credited(self, schema):
        from src.optimizer import slot_a_body_fills
        team = Team(1, 1, slot_counts={"RB": 2, "WR": 2, "TE": 1, "QB": 1})
        role = slot_a_body_fills(team, schema)
        assert role["RB"] == "flex"      # dedicated full, flex open
        assert role["WR"] == "flex"

    def test_empty_roster_fills_dedicated_first(self, schema):
        from src.optimizer import slot_a_body_fills
        role = slot_a_body_fills(Team(1, 1), schema)
        assert role["RB"] == "starter"
        assert role["WR"] == "starter"

    def test_bench_body_ranks_below_a_hole_filler(self, schema):
        """The live regression, as a test."""
        team = Team(1, 1, slot_counts={"RB": 3, "TE": 2, "QB": 1, "WR": 1})
        frame = pd.DataFrame([
            {"player_id": "rb", "name": "DepthRB", "pos": "RB", "team": "X",
             "proj_pts": 151.0, "adp": 80.0, "vorp": 60.0, "risk": "neutral"},
            {"player_id": "wr", "name": "StarterWR", "pos": "WR", "team": "X",
             "proj_pts": 140.0, "adp": 74.0, "vorp": 45.0, "risk": "neutral"},
        ])
        fit = roster_fit(frame, team, schema, round_no=8)
        assert fit.iloc[1] > fit.iloc[0], "the WR filling WR2 must outrank bench depth"


class TestPositionalOutlook:
    """The allocation view the per-pick score cannot express.

    A single pick only ever compares players against their own position's
    next-best, so it cannot say "receivers are about to vanish and tight ends
    will wait". That distinction is what a turn-slot draft turns on.
    """

    @pytest.fixture
    def live(self):
        rows = []
        for pos, base, n in [("QB", 360, 30), ("RB", 300, 60), ("WR", 260, 60),
                             ("TE", 200, 30), ("K", 120, 20), ("DEF", 110, 20)]:
            for i in range(n):
                rows.append({"player_id": f"{pos}{i}", "name": f"{pos}{i}", "pos": pos,
                             "team": "X", "proj_pts": float(base - i * 5),
                             "adp": float(i * 4 + 1)})
        proj = pd.DataFrame(rows)
        draft, league = mock_draft_object(teams=12, rounds=15)
        state = DraftState(draft, proj, league=league)
        board = build_board(proj, {p: state.schema.total_starters(p) for p in STARTERS},
                            state.teams)
        return state, board

    def test_one_row_per_available_position(self, live):
        from src.optimizer import positional_outlook
        state, board = live
        ranked, window = rank_with_lookahead(state, board, my_slot=6)
        frame = positional_outlook(state, ranked, window)
        assert set(frame["pos"]) <= set(STARTERS)
        assert len(frame) == ranked["pos"].nunique()

    def test_sorted_by_what_the_wait_costs(self, live):
        from src.optimizer import positional_outlook
        state, board = live
        ranked, window = rank_with_lookahead(state, board, my_slot=6)
        frame = positional_outlook(state, ranked, window)
        assert frame["cost_of_waiting"].is_monotonic_decreasing

    def test_survival_is_a_probability(self, live):
        from src.optimizer import positional_outlook
        state, board = live
        ranked, window = rank_with_lookahead(state, board, my_slot=6)
        frame = positional_outlook(state, ranked, window)
        assert ((frame["top_survival"] >= 0) & (frame["top_survival"] <= 1)).all()

    def test_survival_over_does_not_disturb_the_ranking(self, live):
        """The alternate horizon is a view, never an input to the score."""
        from src.optimizer import survival_over
        state, board = live
        ranked, _ = rank_with_lookahead(state, board, my_slot=6)
        alt = survival_over(state, ranked, [state.current_pick_no + i for i in range(1, 20)])
        assert list(alt["player_id"]) == list(ranked["player_id"])
        assert (alt["score"] == ranked["score"]).all()

    def test_a_longer_window_lowers_survival(self, live):
        from src.optimizer import survival_over
        state, board = live
        ranked, _ = rank_with_lookahead(state, board, my_slot=6)
        short = survival_over(state, ranked, [2, 3])
        long = survival_over(state, ranked, list(range(2, 24)))
        assert long["survival"].mean() < short["survival"].mean()


class TestRosterCapsAndExclusions:
    """Two judgement channels the projections cannot supply.

    Replaying a finished draft, the engine wanted a third tight end at three
    separate picks behind a starter it already had, plus a backup quarterback at
    two more -- five bench picks on positions that stream freely off waivers.
    The depth penalty alone never stopped it, because once every slot is filled
    a bench body adds nothing to the lineup, so the ranking falls back on VONA,
    and VONA is largest exactly where the pool is shallowest.
    """

    @pytest.fixture
    def schema(self):
        return RosterSchema(
            teams=12, rounds=15,
            starters={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DEF": 1},
            flex={"FLEX": 1}, bench=6,
        )

    @pytest.fixture
    def frame(self):
        return pd.DataFrame([
            {"player_id": "te", "name": "BackupTE", "pos": "TE", "team": "X",
             "proj_pts": 150.0, "adp": 90.0, "vorp": 40.0, "risk": "steady"},
            {"player_id": "qb", "name": "BackupQB", "pos": "QB", "team": "X",
             "proj_pts": 300.0, "adp": 95.0, "vorp": 25.0, "risk": "steady"},
            {"player_id": "rb", "name": "DepthRB", "pos": "RB", "team": "X",
             "proj_pts": 140.0, "adp": 92.0, "vorp": 35.0, "risk": "steady"},
        ])

    def test_a_filled_capped_position_is_blocked(self, frame, schema):
        team = Team(1, 1, slot_counts={"QB": 1, "TE": 1, "RB": 2, "WR": 2})
        fit = roster_fit(frame, team, schema, round_no=10)
        assert fit.iloc[0] == HARD_BLOCK      # second TE
        assert fit.iloc[1] == HARD_BLOCK      # second QB
        assert fit.iloc[2] > HARD_BLOCK       # another RB is still fine

    def test_the_first_one_is_not_blocked(self, frame, schema):
        fit = roster_fit(frame, Team(1, 1), schema, round_no=3)
        assert fit.iloc[0] > HARD_BLOCK
        assert fit.iloc[1] > HARD_BLOCK

    def test_excluded_players_are_blocked_but_stay_on_the_board(self, frame, schema):
        """They must still be nameable -- someone else will draft them."""
        marked = frame.assign(excluded=[False, False, True])
        fit = roster_fit(marked, Team(1, 1), schema, round_no=3)
        assert fit.iloc[2] == HARD_BLOCK
        assert len(marked) == len(frame), "exclusion must not drop the row"

    def test_exclusion_file_parsing(self, tmp_path):
        from src.valuation import load_do_not_draft
        path = tmp_path / "dnd.txt"
        path.write_text("# a comment\n\nDeebo Samuel   # injury history\nJohn Doe\n")
        assert load_do_not_draft(path) == {"deebo samuel", "john doe"}

    def test_missing_exclusion_file_is_fine(self, tmp_path):
        from src.valuation import load_do_not_draft
        assert load_do_not_draft(tmp_path / "nope.txt") == set()


class TestMarginalLineupValue:
    """The column that is shown but deliberately not ranked on.

    Ranking by it loses: A/B'd over 16 drafts at slots 1, 6 and 12 across VONA
    weights 0.5 to 3.0, it was worse at every setting (-7.2 best, -11.2 worst).
    Marginal lineup value is greedy -- fill the empty slot now -- when the right
    play is often to take the scarce player and fill that slot later from a
    deeper pool, which is what VONA prices.
    """

    @pytest.fixture
    def schema(self):
        return RosterSchema(
            teams=12, rounds=15,
            starters={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DEF": 1},
            flex={"FLEX": 1}, bench=6,
        )

    @pytest.fixture
    def frame(self):
        return pd.DataFrame([
            {"player_id": "wr", "name": "WR", "pos": "WR", "team": "X",
             "proj_pts": 187.0, "adp": 60.0, "vorp": 40.0, "risk": "steady"},
            {"player_id": "rb", "name": "RB", "pos": "RB", "team": "X",
             "proj_pts": 159.0, "adp": 59.0, "vorp": 20.0, "risk": "steady"},
        ])

    REPLACEMENT = {"QB": 277.0, "RB": 137.0, "WR": 149.0, "TE": 111.0,
                   "K": 100.0, "DEF": 84.0}

    def test_an_empty_dedicated_slot_pays_the_full_surplus(self, frame, schema):
        from src.optimizer import marginal_lineup_value
        team = Team(1, 1, slot_counts={"QB": 1, "RB": 2, "TE": 1})
        value = marginal_lineup_value(frame, team, schema, self.REPLACEMENT)
        # WR slots empty -> 187 - WR replacement 149
        assert value[0] == pytest.approx(38.0)

    def test_flex_baseline_only_uses_this_league_s_flex_types(self, frame, schema):
        """SUPER_FLEX must not drag quarterbacks into a plain FLEX baseline.

        It did, and a ~277-point QB replacement swamped everything, so every
        back and receiver scored a marginal value of zero.
        """
        from src.optimizer import marginal_lineup_value
        team = Team(1, 1, slot_counts={"QB": 1, "RB": 2, "WR": 2, "TE": 1})
        value = marginal_lineup_value(frame, team, schema, self.REPLACEMENT)
        # Only the flex is open; baseline is the best of RB/WR/TE replacement (149).
        assert value[0] == pytest.approx(38.0)
        assert value[1] == pytest.approx(10.0)

    def test_never_negative(self, frame, schema):
        from src.optimizer import marginal_lineup_value
        full = Team(1, 1, slot_counts={"QB": 1, "RB": 3, "WR": 3, "TE": 1,
                                       "K": 1, "DEF": 1})
        assert (marginal_lineup_value(frame, full, schema, self.REPLACEMENT) >= 0).all()

    def test_score_is_still_vona_based(self):
        """Guards the A/B result: the ranking must not silently switch."""
        import inspect
        from src import optimizer
        src = inspect.getsource(optimizer.rank_with_lookahead)
        assert 'ranked["score"] = ranked["vona"] + ranked["denial"] + ranked["roster_fit"]' in src
