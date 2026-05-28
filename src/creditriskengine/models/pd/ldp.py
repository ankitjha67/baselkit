"""
Low-Default Portfolio (LDP) PD estimation — Pluto-Tasche.

Reference:
    - Pluto & Tasche (2005) — "Estimating Probabilities of Default for
      Low Default Portfolios".
    - BCBS WP14 (2005) — Studies on the Validation of Internal Rating
      Systems.
    - EBA/GL/2017/16 — PD estimation for low-default portfolios.

For portfolios with few or zero observed defaults, the maximum-
likelihood PD estimate (defaults/obligors) is unreliable or zero.
The Pluto-Tasche "most prudent estimate" produces an upper confidence
bound on PD that is non-zero even when zero defaults are observed.
"""

from __future__ import annotations

import logging

from scipy.stats import beta as beta_dist

logger = logging.getLogger(__name__)


def pluto_tasche_single(
    n_obligors: int,
    n_defaults: int,
    confidence_level: float = 0.90,
) -> float:
    """Most-prudent PD estimate for a single grade (Pluto-Tasche).

    Uses the confidence-bound approach: the PD upper bound is the value
    at which the probability of observing at most ``n_defaults`` defaults
    in ``n_obligors`` trials equals ``1 - confidence_level``.

    For the zero-default case this reduces to:
        PD_upper = 1 - (1 - confidence_level)^(1 / n_obligors)

    The general case uses the Clopper-Pearson upper bound (Beta
    quantile), which is the standard exact binomial confidence bound.

    Args:
        n_obligors: Number of obligors in the grade.
        n_defaults: Number of observed defaults.
        confidence_level: Confidence level for the upper bound
            (e.g., 0.90 = 90th percentile).

    Returns:
        Most-prudent (upper confidence bound) PD estimate.

    Raises:
        ValueError: If inputs are invalid.

    Reference:
        Pluto & Tasche (2005), BCBS WP14.
    """
    if n_obligors <= 0:
        raise ValueError("n_obligors must be positive")
    if n_defaults < 0 or n_defaults > n_obligors:
        raise ValueError("n_defaults must be in [0, n_obligors]")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")

    if n_defaults == 0:
        return 1.0 - (1.0 - confidence_level) ** (1.0 / n_obligors)

    # Clopper-Pearson upper bound: Beta(k+1, n-k) quantile at confidence_level
    return float(
        beta_dist.ppf(confidence_level, n_defaults + 1, n_obligors - n_defaults)
    )


def pluto_tasche_multi_grade(
    obligors_per_grade: list[int],
    defaults_per_grade: list[int],
    confidence_level: float = 0.90,
) -> list[float]:
    """Most-prudent PD estimates across ordered grades with monotonicity.

    Grades must be ordered from best (lowest PD) to worst (highest PD).
    The most-prudent estimate for grade i pools all obligors in grades
    i, i+1, ..., k (i.e., obligors at least as risky as grade i), which
    enforces a monotonic non-decreasing PD term structure.

    Args:
        obligors_per_grade: Obligor count per grade (best → worst).
        defaults_per_grade: Default count per grade (best → worst).
        confidence_level: Confidence level for the upper bound.

    Returns:
        List of most-prudent PD estimates, one per grade, monotonically
        non-decreasing.

    Raises:
        ValueError: If grade lists have different lengths.

    Reference:
        Pluto & Tasche (2005), Section 2.2 (multi-grade).
    """
    if len(obligors_per_grade) != len(defaults_per_grade):
        raise ValueError("obligors and defaults lists must have equal length")

    n_grades = len(obligors_per_grade)
    pds: list[float] = []
    for i in range(n_grades):
        pooled_obligors = sum(obligors_per_grade[i:])
        pooled_defaults = sum(defaults_per_grade[i:])
        pd = pluto_tasche_single(
            pooled_obligors, pooled_defaults, confidence_level
        )
        pds.append(pd)

    # Enforce monotonicity (best grade <= next <= ...). Pooling makes the
    # better grades have larger pools and thus lower bounds, so the
    # sequence is naturally non-increasing in pool size → ensure order.
    for i in range(n_grades - 1):
        if pds[i] > pds[i + 1]:
            pds[i] = pds[i + 1]
    return pds
