"""
Discriminatory power tests for credit risk models.

Statistical tests measuring a model's ability to separate
defaulters from non-defaulters.

Regulatory context:
- US SR 11-7 (Fed, April 2011)
- ECB Guide to Internal Models
- PRA SS1/23
- EBA GL/2017/16
"""

import logging

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area Under the Receiver Operating Characteristic curve.

    Uses the Mann-Whitney U-statistic formulation for efficiency.

    Args:
        y_true: Binary labels (1=default, 0=non-default).
        y_score: Predicted scores/PDs (higher = riskier).

    Returns:
        AUROC value in [0, 1].
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Sort by score ascending
    order = np.argsort(y_score)
    y_sorted = y_true[order]

    # Mann-Whitney U: for each positive, count negatives ranked below it
    cum_neg = np.cumsum(1 - y_sorted)
    auc = float(np.sum(cum_neg[y_sorted == 1])) / float(n_pos * n_neg)
    return auc


def gini_coefficient(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Gini coefficient (Accuracy Ratio).

    Formula: Gini = 2 * AUC - 1

    Standard metric per SR 11-7, ECB Guide.

    Args:
        y_true: Binary labels.
        y_score: Predicted scores/PDs.

    Returns:
        Gini coefficient in [-1, 1].
    """
    return 2.0 * auroc(y_true, y_score) - 1.0


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic.

    Maximum separation between cumulative distributions
    of defaulters and non-defaulters.

    Args:
        y_true: Binary labels.
        y_score: Predicted scores/PDs.

    Returns:
        KS statistic in [0, 1].
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)

    defaults = y_score[y_true == 1]
    non_defaults = y_score[y_true == 0]

    if len(defaults) == 0 or len(non_defaults) == 0:
        return 0.0

    ks_stat, _ = stats.ks_2samp(defaults, non_defaults)
    return float(ks_stat)


def cap_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative Accuracy Profile (CAP) curve.

    Args:
        y_true: Binary labels.
        y_score: Predicted scores/PDs.

    Returns:
        Tuple of (fraction_of_population, fraction_of_defaults).
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)

    n = len(y_true)
    n_defaults = np.sum(y_true)

    # Sort by score descending (riskiest first)
    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    cum_defaults = np.cumsum(y_sorted)
    frac_pop = np.arange(1, n + 1) / n
    frac_defaults = cum_defaults / n_defaults if n_defaults > 0 else cum_defaults

    return frac_pop, frac_defaults


def accuracy_ratio(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Accuracy Ratio from CAP curve.

    AR = area_under_model_CAP / area_under_perfect_CAP

    Equivalent to Gini coefficient when computed correctly.

    Args:
        y_true: Binary labels.
        y_score: Predicted scores/PDs.

    Returns:
        Accuracy ratio in [-1, 1].
    """
    return gini_coefficient(y_true, y_score)


def information_value(
    feature: np.ndarray,
    target: np.ndarray,
    bins: int = 10,
) -> float:
    """Information Value (IV) for a single feature.

    IV = Sum [(%good_i - %bad_i) * WoE_i]

    Interpretation:
    - IV < 0.02: useless
    - 0.02-0.1: weak
    - 0.1-0.3: medium
    - 0.3-0.5: strong
    - > 0.5: suspicious (possible overfitting)

    Args:
        feature: Feature values.
        target: Binary target (1=default, 0=non-default).
        bins: Number of bins for continuous features.

    Returns:
        Information Value.
    """
    feature = np.asarray(feature, dtype=np.float64)
    target = np.asarray(target, dtype=np.int64)

    total_good = np.sum(target == 0)
    total_bad = np.sum(target == 1)

    if total_good == 0 or total_bad == 0:
        return 0.0

    # Bin the feature
    try:
        bin_edges = np.percentile(feature, np.linspace(0, 100, bins + 1))
        bin_edges = np.unique(bin_edges)
        bin_indices = np.digitize(feature, bin_edges[1:-1])
    except (ValueError, IndexError):
        return 0.0

    iv = 0.0
    for b in range(len(bin_edges) - 1):
        mask = bin_indices == b
        n_good = np.sum((target == 0) & mask)
        n_bad = np.sum((target == 1) & mask)

        pct_good = max(n_good / total_good, 1e-10)
        pct_bad = max(n_bad / total_bad, 1e-10)

        woe = np.log(pct_good / pct_bad)
        iv += (pct_good - pct_bad) * woe

    return float(iv)


def somers_d(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Somers' D statistic.

    Equals Gini coefficient when there are no ties.

    Args:
        y_true: Binary labels.
        y_score: Predicted scores.

    Returns:
        Somers' D value.
    """
    return gini_coefficient(y_true, y_score)


def divergence(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float:
    """Divergence statistic.

    D = (mean_default - mean_non_default)^2 / (0.5 * (var_default + var_non_default))

    Args:
        y_true: Binary labels.
        y_score: Predicted scores.

    Returns:
        Divergence statistic (higher = better separation).
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)

    defaults = y_score[y_true == 1]
    non_defaults = y_score[y_true == 0]

    if len(defaults) < 2 or len(non_defaults) < 2:
        return 0.0

    mean_diff = np.mean(defaults) - np.mean(non_defaults)
    avg_var = 0.5 * (np.var(defaults, ddof=1) + np.var(non_defaults, ddof=1))

    if avg_var < 1e-15:
        return 0.0

    return float(mean_diff ** 2 / avg_var)
