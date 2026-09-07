"""Guards the survival calibration correction.

Raw product-form survival assumes intermediate picks are independent. Four
completed drafts (3,360 predictions) say otherwise, consistently and in one
direction: predicted 94.1% against 87.4% actual, with the error concentrated in
the contested middle where herding actually happens.

These tests pin the properties the correction must keep, not the fitted value
itself -- that number is expected to move as `--replay` evidence accumulates.
"""

import numpy as np
import pytest

from src.config import settings
from src.opponent_model import _calibrate


def test_correction_is_pessimistic_never_optimistic():
    """It may only push survival down. An optimistic correction would be worse
    than none: the failure mode is passing on a player who then vanishes."""
    raw = np.array([0.05, 0.3, 0.5, 0.7, 0.95])
    assert np.all(_calibrate(raw) <= raw + 1e-12)


def test_certainty_is_preserved():
    """A turn boundary has an empty window and must stay exactly 1.0.

    p**gamma fixes 0 and 1, which is why an exponent was chosen over a linear
    shrink -- 'he is definitely there, you pick again next' must survive."""
    assert _calibrate(np.array([1.0]))[0] == pytest.approx(1.0)
    assert _calibrate(np.array([0.0]))[0] == pytest.approx(0.0)


def test_ordering_is_preserved():
    """Calibration must not reshuffle the board -- it rescales confidence, it
    does not change who is more likely to last than whom."""
    raw = np.array([0.1, 0.25, 0.4, 0.6, 0.8, 0.99])
    out = _calibrate(raw)
    assert np.all(np.diff(out) > 0)


def test_bites_hardest_in_the_middle():
    """The measured error is concentrated at 40-80%, not at the extremes.

    The 80-100% bucket was already well calibrated (99% predicted, 96% actual)
    and holds 88% of all observations, so a correction that moved it much would
    be trading a large accurate bucket for a small inaccurate one.
    """
    raw = np.array([0.5, 0.99])
    shift = raw - _calibrate(raw)
    assert shift[0] > shift[1] * 5


def test_gamma_of_one_is_a_true_no_op():
    """The escape hatch has to be exact, so a disabled correction is provably
    identical to the pre-calibration engine."""
    original = settings.survival_gamma
    try:
        settings.survival_gamma = 1.0
        raw = np.array([0.13, 0.47, 0.86])
        assert np.array_equal(_calibrate(raw), raw)
    finally:
        settings.survival_gamma = original


def test_fitted_value_is_pessimistic_and_bounded():
    """Sanity-check the shipped constant. Below 1.0 would make the model *more*
    optimistic than the evidence; an extreme value would collapse every
    mid-range survival to zero and make VONA meaningless."""
    assert 1.0 < settings.survival_gamma < 10.0
