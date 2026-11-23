"""Accuracy, calibration and the paired comparisons.

Formulas and their sources are listed in PAPER.md.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

Z95 = 1.959963984540054


def wilson_interval(successes: float, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson (1927) score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def brier(p_true: Sequence[float], truth: Sequence[bool]) -> float:
    """Brier (1950) score: mean squared error of the probability. Lower is better."""
    p = np.asarray(p_true, dtype=float)
    y = np.asarray(truth, dtype=float)
    return float(np.mean((p - y) ** 2))


def ece(p_true: Sequence[float], truth: Sequence[bool], bins: int = 5) -> float:
    """Expected calibration error on the confidence in the chosen side.

    Confidence is max(p, 1 - p), which lies in [0.5, 1]. That range is cut into
    ``bins`` equal bins. The error is the weighted mean gap between each bin's
    mean confidence and its accuracy (Naeini et al. 2015; Guo et al. 2017).
    A probability of exactly 0.5 is scored as a wrong answer at confidence 0.5.
    """
    p = np.asarray(p_true, dtype=float)
    y = np.asarray(truth, dtype=bool)
    confidence = np.maximum(p, 1 - p)
    correct = np.where(p == 0.5, False, (p > 0.5) == y)
    edges = np.linspace(0.5, 1.0, bins + 1)
    index = np.clip(np.digitize(confidence, edges[1:-1], right=False), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            total += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
    return float(total)


def bootstrap_mean(
    values: Sequence[float], n_boot: int = 10_000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean with a percentile bootstrap 95% interval over questions.

    Pass per-question differences to get a paired comparison.
    """
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[draws].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


def bootstrap_ece(
    p_true: Sequence[float], truth: Sequence[bool], n_boot: int = 2_000, seed: int = 0
) -> tuple[float, float, float]:
    """ECE with a percentile bootstrap 95% interval over questions."""
    p = np.asarray(p_true, dtype=float)
    y = np.asarray(truth, dtype=bool)
    rng = np.random.default_rng(seed)
    stats: NDArray[np.float64] = np.empty(n_boot)
    for i in range(n_boot):
        draw = rng.integers(0, len(p), size=len(p))
        stats[i] = ece(list(p[draw]), list(y[draw]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return ece(p_true, truth), float(lo), float(hi)


def mcnemar_exact(only_first: int, only_second: int) -> float:
    """Two-sided exact McNemar test on the discordant pairs.

    ``only_first`` counts questions the first system got right and the second
    got wrong; ``only_second`` the reverse. Under the null each discordant pair
    is a fair coin, so the p-value is a two-sided binomial tail.
    """
    n = only_first + only_second
    if n == 0:
        return 1.0
    k = min(only_first, only_second)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return float(min(1.0, 2.0 * tail))
