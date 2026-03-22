"""
Stability monitoring metrics for credit risk models.

Regulatory context:
- OCC 2011-12: Model Risk Management
- SR 11-7 (Fed): Ongoing monitoring requirements
- ECB Guide to Internal Models: PSI/CSI monitoring
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def population_stability_index(
    actual: np.ndarray,
    expected: np.ndarray,
    bins: int = 10,
    precomputed: bool = False,
) -> float:
    """Population Stability Index (PSI).

    PSI = Sum [(%actual_i - %expected_i) * ln(%actual_i / %expected_i)]

    Interpretation:
    - PSI < 0.10: no significant change
    - 0.10 <= PSI < 0.25: moderate shift
    - PSI >= 0.25: significant shift

    Args:
        actual: Actual distribution values (raw or pre-binned proportions).
        expected: Expected/reference distribution (raw or pre-binned proportions).
        bins: Number of bins (if not precomputed).
        precomputed: If True, actual/expected are already bin proportions.

    Returns:
        PSI value.
    """
    if precomputed:
        pct_actual = np.asarray(actual, dtype=np.float64)
        pct_expected = np.asarray(expected, dtype=np.float64)
    else:
        actual = np.asarray(actual, dtype=np.float64)
        expected = np.asarray(expected, dtype=np.float64)

        # Create bins from expected distribution
        bin_edges = np.percentile(expected, np.linspace(0, 100, bins + 1))
        bin_edges = np.unique(bin_edges)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        actual_counts = np.histogram(actual, bins=bin_edges)[0]
        expected_counts = np.histogram(expected, bins=bin_edges)[0]

        pct_actual = actual_counts / max(len(actual), 1)
        pct_expected = expected_counts / max(len(expected), 1)

    # Replace zeros to avoid log(0)
    pct_actual = np.maximum(pct_actual, 1e-10)
    pct_expected = np.maximum(pct_expected, 1e-10)

    psi = float(np.sum((pct_actual - pct_expected) * np.log(pct_actual / pct_expected)))
    return psi


def characteristic_stability_index(
    actual: np.ndarray,
    expected: np.ndarray,
    bins: int = 10,
) -> float:
    """Characteristic Stability Index (CSI).

    Same formula as PSI but applied at individual feature level.

    Args:
        actual: Actual feature distribution.
        expected: Expected/reference feature distribution.
        bins: Number of bins.

    Returns:
        CSI value (same interpretation as PSI).
    """
    return population_stability_index(actual, expected, bins)


def herfindahl_index(shares: np.ndarray) -> float:
    """Herfindahl-Hirschman Index for concentration.

    HHI = Sum(share_i^2)

    Args:
        shares: Array of shares/proportions (should sum to 1.0).

    Returns:
        HHI value in [0, 1]. Higher = more concentrated.
    """
    shares = np.asarray(shares, dtype=np.float64)
    return float(np.sum(shares ** 2))


def migration_matrix_stability(
    matrix_1: np.ndarray,
    matrix_2: np.ndarray,
) -> dict[str, float]:
    """Compare two transition/migration matrices for stability.

    Uses L2 (Frobenius) norm and maximum absolute difference.

    Args:
        matrix_1: First transition matrix.
        matrix_2: Second transition matrix.

    Returns:
        Dict with frobenius_norm, max_abs_diff, mean_abs_diff.
    """
    diff = matrix_1 - matrix_2
    return {
        "frobenius_norm": float(np.linalg.norm(diff, "fro")),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
    }
