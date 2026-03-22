"""
PD backtesting framework.

Reference: BCBS WP14 (May 2005), EBA GL/2017/16.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def pd_backtest_summary(
    predicted_pds: np.ndarray,
    observed_defaults: np.ndarray,
    rating_grades: np.ndarray | None = None,
) -> dict[str, float]:
    """Summary statistics for PD backtesting.

    Args:
        predicted_pds: Predicted PDs per exposure.
        observed_defaults: Binary default indicator (0/1).
        rating_grades: Optional rating grade labels for grouping.

    Returns:
        Dict with overall metrics.
    """
    predicted_pds = np.asarray(predicted_pds, dtype=np.float64)
    observed_defaults = np.asarray(observed_defaults, dtype=np.int64)

    n = len(predicted_pds)
    n_defaults = int(np.sum(observed_defaults))
    avg_pd = float(np.mean(predicted_pds))
    observed_dr = n_defaults / n if n > 0 else 0.0

    return {
        "n_observations": n,
        "n_defaults": n_defaults,
        "average_predicted_pd": avg_pd,
        "observed_default_rate": observed_dr,
        "ratio_observed_to_predicted": observed_dr / avg_pd if avg_pd > 0 else 0.0,
    }
