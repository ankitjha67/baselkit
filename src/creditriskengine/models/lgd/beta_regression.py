"""
Beta-distribution LGD modeling.

Reference:
    - Gupton & Stein (2002) — LossCalc LGD modeling.
    - EBA/GL/2017/16 — LGD estimation.
    - Huang & Oosterlee (2011) — Generalised beta regression for LGD.

LGD is bounded in [0, 1] and frequently bimodal (mass near 0 and 1),
making the beta distribution a natural fit. Provides method-of-moments
beta fitting and quantile/mean estimation for downturn LGD.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def fit_beta_lgd(lgd_observations: np.ndarray) -> tuple[float, float]:
    """Fit a Beta(alpha, beta) to observed LGDs via method of moments.

    Given sample mean m and variance v:
        alpha = m * (m*(1-m)/v - 1)
        beta  = (1-m) * (m*(1-m)/v - 1)

    Args:
        lgd_observations: Observed LGD values in [0, 1].

    Returns:
        Tuple of (alpha, beta) Beta-distribution parameters.

    Raises:
        ValueError: If observations are empty or out of [0, 1].

    Reference:
        Gupton & Stein (2002).
    """
    lgd = np.asarray(lgd_observations, dtype=np.float64)
    if len(lgd) == 0:
        raise ValueError("lgd_observations must be non-empty")
    if np.any(lgd < 0) or np.any(lgd > 1):
        raise ValueError("LGD observations must be in [0, 1]")

    m = float(np.mean(lgd))
    v = float(np.var(lgd, ddof=1)) if len(lgd) > 1 else 0.0

    # Degenerate variance → near point mass; return a peaked beta
    if v <= 0 or v >= m * (1 - m):
        # Fall back to a concentration that reproduces the mean
        concentration = 100.0
        return m * concentration, (1 - m) * concentration

    common = m * (1 - m) / v - 1.0
    alpha = m * common
    beta = (1 - m) * common
    return alpha, beta


def beta_lgd_mean(alpha: float, beta: float) -> float:
    """Mean of a Beta(alpha, beta) distribution.

    Args:
        alpha: Beta alpha parameter (>0).
        beta: Beta beta parameter (>0).

    Returns:
        Mean = alpha / (alpha + beta).
    """
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be positive")
    return alpha / (alpha + beta)


def downturn_lgd_quantile(
    alpha: float,
    beta: float,
    confidence_level: float = 0.90,
) -> float:
    """Downturn LGD as a high quantile of the fitted beta distribution.

    A common downturn LGD proxy uses a high percentile (e.g., 90th) of
    the LGD distribution to capture economic-stress conditions.

    Args:
        alpha: Beta alpha parameter.
        beta: Beta beta parameter.
        confidence_level: Quantile level (e.g., 0.90).

    Returns:
        Downturn LGD (the confidence-level quantile of Beta).

    Raises:
        ValueError: If parameters or confidence level are invalid.

    Reference:
        EBA/GL/2017/16 (downturn LGD), CRE32.
    """
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    return float(stats.beta.ppf(confidence_level, alpha, beta))
