"""Closed forms and known values for the scoring maths."""

import math

import numpy as np
import pytest

from agon.scoring import bootstrap_mean, brier, ece, mcnemar_exact, wilson_interval


def test_wilson_matches_a_hand_computed_value() -> None:
    # 81 of 263 with z = 1.96, worked by hand from the formula in PAPER.md:
    # centre = (p + z^2/2n) / (1 + z^2/n), half = z sqrt(p(1-p)/n + z^2/4n^2) / (1 + z^2/n).
    lo, hi = wilson_interval(81, 263, z=1.96)
    assert lo == pytest.approx(0.2553, abs=5e-4)
    assert hi == pytest.approx(0.3662, abs=5e-4)


def test_wilson_closed_form_at_zero_successes() -> None:
    # With p = 0 the interval is [0, z^2 / (n + z^2)].
    z = 1.959963984540054
    lo, hi = wilson_interval(0, 50)
    assert lo == pytest.approx(0.0, abs=1e-12)
    assert hi == pytest.approx(z * z / (50 + z * z))


def test_wilson_is_symmetric_and_narrows() -> None:
    lo, hi = wilson_interval(30, 100)
    lo2, hi2 = wilson_interval(70, 100)
    assert lo == pytest.approx(1 - hi2)
    assert hi == pytest.approx(1 - lo2)
    wide = wilson_interval(50, 100)
    narrow = wilson_interval(500, 1000)
    assert narrow[1] - narrow[0] < wide[1] - wide[0]


def test_brier_limits() -> None:
    assert brier([1.0, 0.0], [True, False]) == 0.0
    assert brier([0.0, 1.0], [True, False]) == 1.0
    assert brier([0.5, 0.5, 0.5], [True, False, True]) == 0.25
    assert brier([0.8], [True]) == pytest.approx(0.04)


def test_ece_is_zero_for_a_calibrated_forecaster() -> None:
    # Confidence 0.8, right four times in five.
    p = [0.8, 0.8, 0.8, 0.8, 0.8]
    y = [True, True, True, True, False]
    assert ece(p, y) == pytest.approx(0.0)


def test_ece_of_a_confident_coin_is_a_half() -> None:
    p = [1.0, 1.0, 0.0, 0.0]
    y = [True, False, True, False]
    assert ece(p, y) == pytest.approx(0.5)


def test_ece_weights_bins_by_size() -> None:
    # Three answers at 0.95, all right (gap 0.05), and one at 0.55, wrong (gap 0.55).
    p = [0.95, 0.95, 0.05, 0.55]
    y = [True, True, False, False]
    assert ece(p, y) == pytest.approx(0.75 * 0.05 + 0.25 * 0.55)


def test_mcnemar_known_values() -> None:
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(0, 5) == pytest.approx(2 / 32)
    # 2 against 10: two-sided binomial tail, 2 * (1 + 12 + 66) / 4096.
    assert mcnemar_exact(2, 10) == pytest.approx(2 * 79 / 4096)
    assert mcnemar_exact(2, 10) == mcnemar_exact(10, 2)


def test_bootstrap_is_seeded_and_close_to_the_normal_interval() -> None:
    rng = np.random.default_rng(3)
    x = list(rng.normal(0.1, 1.0, size=400))
    first = bootstrap_mean(x, n_boot=2000, seed=5)
    assert first == bootstrap_mean(x, n_boot=2000, seed=5)
    mean, lo, hi = first
    half = 1.96 * float(np.std(x, ddof=1)) / math.sqrt(len(x))
    assert hi - mean == pytest.approx(half, rel=0.15)
    assert mean - lo == pytest.approx(half, rel=0.15)


def test_bootstrap_of_a_constant_is_exact() -> None:
    assert bootstrap_mean([0.25] * 30, n_boot=200, seed=1) == (0.25, 0.25, 0.25)
