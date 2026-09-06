"""Guards the valuation definitions taken from Petersen, *Fantasy Football Analytics*.

These are definitional tests. If VORP silently starts measuring against the
last starter instead of a typical bench player, or dropoff starts comparing
across positions, every recommendation shifts and nothing visibly breaks.
"""

import numpy as np
import pandas as pd
import pytest

from src.valuation import (
    _jenks_breaks,
    add_consistency,
    add_dropoff,
    add_vorp,
    assign_tiers,
    build_board,
    replacement_levels,
)


@pytest.fixture
def board():
    """Two positions with deliberately different shapes."""
    rows = []
    for i, pts in enumerate([300, 290, 250, 249, 248, 200, 150, 100]):
        rows.append({"player_id": f"rb{i}", "name": f"RB{i}", "pos": "RB",
                     "team": "X", "proj_pts": float(pts), "adp": float(i + 1)})
    for i, pts in enumerate([280, 275, 270, 265, 260, 255, 250, 245]):
        rows.append({"player_id": f"wr{i}", "name": f"WR{i}", "pos": "WR",
                     "team": "Y", "proj_pts": float(pts), "adp": float(i + 1)})
    return pd.DataFrame(rows)


class TestDropoff:
    """Petersen: points minus the *next-best player at that position*."""

    def test_dropoff_is_within_position(self, board):
        out = add_dropoff(board).set_index("player_id")
        assert out.loc["rb0", "dropoff"] == pytest.approx(10.0)   # 300 -> 290
        assert out.loc["rb1", "dropoff"] == pytest.approx(40.0)   # 290 -> 250
        assert out.loc["wr0", "dropoff"] == pytest.approx(5.0)    # 280 -> 275

    def test_does_not_compare_across_positions(self, board):
        out = add_dropoff(board).set_index("player_id")
        # rb2 (250) -> rb3 (249), NOT to any WR sitting between them.
        assert out.loc["rb2", "dropoff"] == pytest.approx(1.0)

    def test_last_player_has_zero_not_nan(self, board):
        out = add_dropoff(board).set_index("player_id")
        assert out.loc["rb7", "dropoff"] == 0.0
        assert out["dropoff"].notna().all()


class TestReplacementLevel:
    """Replacement is a typical BENCH player, not the last starter."""

    def test_uses_bench_cohort_median(self, board):
        # 2 teams x 1 RB starter = 2 starters; bench cohort = next 2 (250, 249)
        levels = replacement_levels(board, {"RB": 1, "WR": 1}, teams=2, cohort=2)
        assert levels["RB"] == pytest.approx(249.5)

    def test_vorp_is_points_minus_replacement(self, board):
        out = add_vorp(board, {"RB": 1, "WR": 1}, teams=2).set_index("player_id")
        expected = 300.0 - out.loc["rb0", "replacement_pts"]
        assert out.loc["rb0", "vorp"] == pytest.approx(expected)

    def test_replacement_is_below_starter_cutoff(self, board):
        """The whole point: a bench baseline is lower than a last-starter one."""
        levels = replacement_levels(board, {"RB": 1}, teams=2, cohort=2)
        starters = board[board.pos == "RB"].nlargest(2, "proj_pts")["proj_pts"].min()
        assert levels["RB"] < starters

    def test_thin_position_falls_back_to_tail(self):
        thin = pd.DataFrame([
            {"player_id": "k0", "name": "K0", "pos": "K", "team": "Z",
             "proj_pts": 120.0, "adp": 200.0}
        ])
        levels = replacement_levels(thin, {"K": 1}, teams=12)
        assert levels["K"] == pytest.approx(120.0)  # defined, not a crash


class TestTiers:
    def test_tier_never_improves_as_points_fall(self, board):
        out = assign_tiers(board)
        for _, group in out.groupby("pos"):
            ordered = group.sort_values("proj_pts", ascending=False)
            assert ordered["tier"].is_monotonic_increasing

    def test_best_player_is_tier_one(self, board):
        out = assign_tiers(board)
        for _, group in out.groupby("pos"):
            best = group.nlargest(1, "proj_pts").iloc[0]
            assert best["tier"] == 1

    def test_jenks_breaks_are_unique_and_sorted(self):
        values = np.array([1.0, 2, 3, 100, 101, 102, 200, 201], dtype=float)
        starts = _jenks_breaks(values, 3)
        assert starts == sorted(set(starts))
        assert starts[0] == 0

    def test_jenks_finds_the_obvious_gaps(self):
        values = np.array([1.0, 2, 3, 100, 101, 102, 200, 201], dtype=float)
        assert _jenks_breaks(values, 3) == [0, 3, 6]


class TestConsistency:
    def test_labels_split_within_position(self, board):
        cons = pd.DataFrame({
            "player_id": [f"rb{i}" for i in range(8)],
            "wk_cv": [0.1, 0.15, 0.2, 0.4, 0.5, 0.8, 0.9, 1.0],
            "wk_sd": [1.0] * 8, "wk_mean": [10.0] * 8, "games_played": [17] * 8,
        })
        out = add_consistency(board, cons)
        rb = out[out.pos == "RB"]
        assert set(rb["risk"]) >= {"steady", "volatile"}
        # Lowest CV must not be labelled volatile.
        assert rb.loc[rb["wk_cv"].idxmin(), "risk"] == "steady"

    def test_missing_history_is_unknown_not_steady(self, board):
        """A rookie has no variance estimate; that is not low variance."""
        out = add_consistency(board, pd.DataFrame(columns=[
            "player_id", "wk_cv", "wk_sd", "wk_mean", "games_played"]))
        assert (out["risk"] == "unknown").all()


class TestBuildBoard:
    def test_drops_non_rosterable_positions(self, board):
        polluted = pd.concat([board, pd.DataFrame([{
            "player_id": "fb0", "name": "FB0", "pos": "FB", "team": "X",
            "proj_pts": 20.0, "adp": 999.0}])], ignore_index=True)
        out = build_board(polluted, {"RB": 2, "WR": 2}, teams=2)
        assert "FB" not in set(out["pos"])

    def test_sorted_by_vorp(self, board):
        out = build_board(board, {"RB": 2, "WR": 2}, teams=2)
        assert out["vorp"].is_monotonic_decreasing


class TestSharedFlexCapacity:
    """One FLEX slot is one slot, not one per eligible position.

    Counting it in full for RB, WR and TE alike pushed every VORP baseline too
    deep. It bit hardest at tight end, whose pool is shallow: the baseline came
    from TE25-36 in a league where ~13 tight ends start, so every TE looked
    ~30 points better than he was. Across ten mock drafts the engine rostered a
    mean of 3.5 tight ends for a single starting slot.
    """

    def _schema(self, flex):
        from src.state import RosterSchema
        return RosterSchema(
            teams=12, rounds=15,
            starters={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DEF": 1},
            flex=flex, bench=6,
        )

    def test_one_flex_slot_is_split_across_eligible_positions(self):
        schema = self._schema({"FLEX": 1})
        # FLEX takes RB/WR/TE -> a third each, not a whole slot each.
        assert schema.flex_capacity("RB") == pytest.approx(1 / 3)
        assert schema.flex_capacity("TE") == pytest.approx(1 / 3)
        assert schema.flex_capacity("QB") == 0.0

    def test_capacity_never_exceeds_the_slots_that_exist(self):
        schema = self._schema({"FLEX": 2})
        total = sum(schema.flex_capacity(p) for p in ("RB", "WR", "TE"))
        assert total == pytest.approx(2.0), "must sum to the number of real slots"

    def test_superflex_includes_quarterbacks(self):
        schema = self._schema({"SUPER_FLEX": 1})
        assert schema.flex_capacity("QB") == pytest.approx(1 / 4)
        assert sum(schema.flex_capacity(p) for p in ("QB", "RB", "WR", "TE")) \
            == pytest.approx(1.0)

    def test_tight_end_baseline_is_not_pushed_absurdly_deep(self):
        schema = self._schema({"FLEX": 1})
        # 12 teams x total_starters(TE) should land near the real starter count
        # (~13), not out at TE24.
        assert 12 * schema.total_starters("TE") < 18

    def test_replacement_rises_when_flex_is_shared(self):
        rows = []
        for pos, base, n in [("RB", 300, 60), ("WR", 280, 60), ("TE", 200, 40)]:
            for i in range(n):
                rows.append({"player_id": f"{pos}{i}", "name": f"{pos}{i}", "pos": pos,
                             "team": "X", "proj_pts": float(base - i * 5),
                             "adp": float(i + 1)})
        frame = pd.DataFrame(rows)
        shared = self._schema({"FLEX": 1})
        full = {"RB": 3.0, "WR": 3.0, "TE": 2.0}          # the old, triple-counted view
        lv_shared = replacement_levels(
            frame, {p: shared.total_starters(p) for p in ("RB", "WR", "TE")}, 12)
        lv_full = replacement_levels(frame, full, 12)
        assert lv_shared["TE"] > lv_full["TE"], "sharing must lift the TE baseline"
