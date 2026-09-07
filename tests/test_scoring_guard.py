"""Guards the check that a board is only used in the league it was built for.

Every other bad input to this app announces itself. This one does not: a
full-PPR board loads cleanly in a half-PPR league, every downstream number
computes, and the only symptom is that receivers are worth about a round more
than they should be for the whole draft. It has happened once for real, which
is why the check exists and why these tests pin it.
"""

import pandas as pd
import pytest

from src.state import board_scoring, scoring_format, scoring_mismatch


def frame(scoring=None, n=3):
    data = {"player_id": [str(i) for i in range(n)], "proj_pts": [100.0] * n}
    if scoring is not None:
        data["scoring"] = [scoring] * n
    return pd.DataFrame(data)


def test_board_scoring_reads_the_stamp():
    assert board_scoring(frame("half_ppr")) == "half_ppr"


@pytest.mark.parametrize("bad", [
    pd.DataFrame(),                                   # nothing at all
    frame(None),                                      # snapshot predates the column
])
def test_board_scoring_is_unknown_not_wrong(bad):
    """Unknown must be distinguishable from mismatched: one warns, one blocks."""
    assert board_scoring(bad) is None


def test_board_scoring_rejects_a_mixed_file():
    mixed = frame("ppr")
    mixed.loc[1, "scoring"] = "std"
    assert board_scoring(mixed) is None


def test_matching_formats_do_not_trip_the_guard():
    assert scoring_mismatch(frame("half_ppr"), "half_ppr") is None


def test_unknown_board_does_not_trip_the_guard():
    """An unstamped board cannot be proven wrong, so it must not block a draft."""
    assert scoring_mismatch(frame(None), "std") is None


@pytest.mark.parametrize("built,league", [
    ("ppr", "half_ppr"),
    ("half_ppr", "std"),
    ("std", "ppr"),
    ("ppr", "2qb"),
])
def test_mismatch_reports_both_formats(built, league):
    assert scoring_mismatch(frame(built), league) == (built, league)


def test_guard_agrees_with_the_format_the_league_reports():
    """The two halves of the check must speak the same vocabulary.

    ``scoring_format`` names the league and ``board_scoring`` names the file;
    if they ever drift apart the guard fires on every draft or on none.
    """
    draft = {"metadata": {"scoring_type": "half_ppr"},
             "settings": {"teams": 12, "rounds": 15,
                          "slots_qb": 1, "slots_rb": 2, "slots_wr": 2,
                          "slots_te": 1, "slots_flex": 1, "slots_k": 1,
                          "slots_def": 1, "slots_bn": 6}}
    league_format = scoring_format(draft, None)
    assert scoring_mismatch(frame(league_format), league_format) is None
