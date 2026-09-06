"""Guards the calibration harness.

If replay silently mis-scores, every future decision about correcting the
survival model rests on a bad number -- and unlike a ranking bug, nothing on
screen would look wrong.
"""

import numpy as np
import pandas as pd
import pytest

from src.mock import mock_draft_object
from src.replay import append_log, replay
from src.state import DraftState
from src.valuation import build_board

STARTERS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DEF": 1}


@pytest.fixture
def synthetic_draft():
    rows = []
    for pos, base, n in [("QB", 320, 30), ("RB", 330, 50), ("WR", 315, 60),
                         ("TE", 255, 30), ("K", 125, 20), ("DEF", 110, 20)]:
        for i in range(n):
            rows.append({"player_id": f"{pos}{i}", "name": f"{pos}{i}", "pos": pos,
                         "team": "X", "proj_pts": float(base - i * 4),
                         "adp": float(i * 3 + 1)})
    proj = pd.DataFrame(rows)
    draft, league = mock_draft_object(teams=10, rounds=15)
    state = DraftState(draft, proj, league=league)
    board = build_board(proj, {p: state.schema.total_starters(p) for p in STARTERS},
                        state.teams)
    # Deterministic completed draft: everyone takes the best remaining ADP.
    picks, taken = [], set()
    pool = board.sort_values("adp")["player_id"].tolist()
    for pick_no in range(1, state.teams * state.rounds + 1):
        pid = next(x for x in pool if x not in taken)
        taken.add(pid)
        picks.append({"pick_no": pick_no, "round": (pick_no - 1) // 10 + 1,
                      "draft_slot": state.slot_at(pick_no),
                      "roster_id": state.roster_at(pick_no), "player_id": pid})
    return draft, picks, board


class TestReplay:
    def test_produces_a_report(self, synthetic_draft):
        draft, picks, board = synthetic_draft
        r = replay(draft, picks, board, my_slot=3)
        assert r["n_observations"] > 0
        assert 0.0 <= r["predicted_mean"] <= 1.0
        assert 0.0 <= r["actual_mean"] <= 1.0
        assert 0.0 <= r["brier"] <= 1.0

    def test_calibration_buckets_sum_to_the_observations(self, synthetic_draft):
        draft, picks, board = synthetic_draft
        r = replay(draft, picks, board, my_slot=3)
        assert int(r["calibration"]["n"].sum()) == r["n_observations"]

    def test_scores_a_perfectly_predictable_draft_well(self, synthetic_draft):
        """Everyone drafts strictly by ADP, so survival should be easy to call."""
        draft, picks, board = synthetic_draft
        r = replay(draft, picks, board, my_slot=3)
        assert r["brier"] < 0.25, "should beat a coin flip on a deterministic draft"

    def test_one_row_per_pick_in_the_comparison(self, synthetic_draft):
        draft, picks, board = synthetic_draft
        r = replay(draft, picks, board, my_slot=3)
        my_picks = [p["pick_no"] for p in picks if p["draft_slot"] == 3]
        assert len(r["comparisons"]) == len(my_picks)

    def test_rejects_a_slot_that_never_picked(self, synthetic_draft):
        draft, picks, board = synthetic_draft
        with pytest.raises(ValueError):
            replay(draft, picks, board, my_slot=99)


class TestCalibrationLog:
    def test_appends_and_deduplicates(self, synthetic_draft, tmp_path):
        draft, picks, board = synthetic_draft
        r = replay(draft, picks, board, my_slot=3)
        log = tmp_path / "calibration.csv"
        append_log(r, log)
        append_log(r, log)          # same draft twice must not double-count
        assert len(pd.read_csv(log)) == 1
        r2 = dict(r); r2["draft_id"] = "another"
        append_log(r2, log)
        frame = pd.read_csv(log)
        assert len(frame) == 2
        assert "bias" in frame.columns
