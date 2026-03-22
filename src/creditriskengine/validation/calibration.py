"""
Calibration tests — predicted vs observed default rates.

Tests whether predicted PDs are consistent with observed defaults.

Regulatory context:
- BCBS WP14 (May 2005): Traffic Light approach
- SR 11-7 (Fed): Outcomes analysis requirements
- ECB Guide to Internal Models
- EBA GL/2017/16
"""

import logging

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def binomial_test(
    n_defaults: int,
    n_observations: int,
    predicted_pd: float,
    confidence: float = 0.99,
) -> dict[str, float | bool]:
    """Binomial test for PD calibration.

    Tests if observed defaults are consistent with predicted PD,
    assuming independent Bernoulli trials.

    H0: true PD = predicted_pd
    z = (d - N*PD) / sqrt(N*PD*(1-PD))

    Args:
        n_defaults: Observed number of defaults.
        n_observations: Total number of observations.
        predicted_pd: Predicted (average) PD.
        confidence: Confidence level for the test.

    Returns:
        Dict with z_stat, p_value, critical_value, reject_h0.
    """
    if n_observations == 0:
        return {"z_stat": 0.0, "p_value": 1.0, "critical_value": 0.0, "reject_h0": False}

    expected = n_observations * predicted_pd
    std = (n_observations * predicted_pd * (1.0 - predicted_pd)) ** 0.5

    if std < 1e-15:
        return {"z_stat": 0.0, "p_value": 1.0, "critical_value": 0.0, "reject_h0": False}

    z = (n_defaults - expected) / std
    p_value = 1.0 - stats.norm.cdf(z)  # One-sided (upper tail)
    critical = stats.norm.ppf(confidence)

    return {
        "z_stat": float(z),
        "p_value": float(p_value),
        "critical_value": float(critical),
        "reject_h0": z > critical,
    }


def hosmer_lemeshow_test(
    observed_defaults: np.ndarray,
    predicted_pds: np.ndarray,
    group_counts: np.ndarray,
    n_groups: int = 10,
) -> dict[str, float | bool]:
    """Hosmer-Lemeshow goodness-of-fit test.

    H-L = Sum(i=1..g) [(O_i - E_i)^2 / (N_i * pi_i * (1-pi_i))]

    Args:
        observed_defaults: Observed defaults per group.
        predicted_pds: Average predicted PD per group.
        group_counts: Number of observations per group.
        n_groups: Number of groups (for degrees of freedom).

    Returns:
        Dict with hl_stat, p_value, df, reject_h0 (at 5% level).
    """
    expected = group_counts * predicted_pds
    variance = group_counts * predicted_pds * (1.0 - predicted_pds)

    # Avoid division by zero
    mask = variance > 1e-15
    hl = float(np.sum((observed_defaults[mask] - expected[mask]) ** 2 / variance[mask]))

    df = max(n_groups - 2, 1)
    p_value = 1.0 - stats.chi2.cdf(hl, df)

    return {
        "hl_stat": hl,
        "p_value": float(p_value),
        "df": df,
        "reject_h0": p_value < 0.05,
    }


def spiegelhalter_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float | bool]:
    """Spiegelhalter test for overall calibration.

    Tests whether the sum of (predicted - observed)^2 is
    consistent with expectations under correct calibration.

    Args:
        y_true: Binary outcomes (0/1).
        y_pred: Predicted probabilities.

    Returns:
        Dict with z_stat, p_value, reject_h0 (at 5% level).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    n = len(y_true)
    if n == 0:
        return {"z_stat": 0.0, "p_value": 1.0, "reject_h0": False}

    brier = np.sum((y_pred - y_true) ** 2)
    expected_brier = np.sum(y_pred * (1.0 - y_pred))

    # Variance of Brier score under H0
    var_terms = (1.0 - 2.0 * y_pred) ** 2 * y_pred * (1.0 - y_pred)
    var_brier = np.sum(var_terms)

    if var_brier < 1e-15:
        return {"z_stat": 0.0, "p_value": 1.0, "reject_h0": False}

    z = (brier - expected_brier) / (var_brier ** 0.5)
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z)))

    return {
        "z_stat": float(z),
        "p_value": float(p_value),
        "reject_h0": p_value < 0.05,
    }


def traffic_light_test(
    n_defaults: int,
    n_observations: int,
    predicted_pd: float,
) -> str:
    """Basel Committee Traffic Light approach.

    Reference: BCBS WP14 (May 2005) — "Studies on the Validation
    of Internal Rating Systems".

    Green: observed < 95th percentile of binomial
    Yellow: 95th-99.99th percentile
    Red: > 99.99th percentile

    Args:
        n_defaults: Observed defaults.
        n_observations: Total observations.
        predicted_pd: Predicted PD.

    Returns:
        "green", "yellow", or "red".
    """
    if n_observations == 0:
        return "green"

    # Binomial percentiles
    p_95 = stats.binom.ppf(0.95, n_observations, predicted_pd)
    p_9999 = stats.binom.ppf(0.9999, n_observations, predicted_pd)

    if n_defaults <= p_95:
        return "green"
    elif n_defaults <= p_9999:
        return "yellow"
    else:
        return "red"


def jeffreys_test(
    n_defaults: int,
    n_observations: int,
    predicted_pd: float,
    confidence: float = 0.99,
) -> dict[str, float | bool]:
    """Jeffreys test — Bayesian alternative to binomial.

    Uses Beta posterior: Beta(d + 0.5, n - d + 0.5)
    where d = defaults, n = observations.

    Args:
        n_defaults: Observed defaults.
        n_observations: Total observations.
        predicted_pd: Predicted PD.
        confidence: Confidence level.

    Returns:
        Dict with posterior_mean, lower_bound, upper_bound, pd_within_interval.
    """
    alpha = n_defaults + 0.5
    beta_param = n_observations - n_defaults + 0.5

    posterior_mean = alpha / (alpha + beta_param)
    lower = stats.beta.ppf((1 - confidence) / 2, alpha, beta_param)
    upper = stats.beta.ppf((1 + confidence) / 2, alpha, beta_param)

    return {
        "posterior_mean": float(posterior_mean),
        "lower_bound": float(lower),
        "upper_bound": float(upper),
        "pd_within_interval": lower <= predicted_pd <= upper,
    }


def brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Brier Score for probability calibration.

    BS = (1/N) * Sum(PD_i - D_i)^2

    Lower is better. Perfect calibration: BS approaches PD*(1-PD).

    Args:
        y_true: Binary outcomes (0/1).
        y_pred: Predicted probabilities.

    Returns:
        Brier score.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean((y_pred - y_true) ** 2))
