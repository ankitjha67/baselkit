"""
Vasicek ASRF model — theoretical foundation of Basel III IRB formulas.

Reference: Vasicek (2002), "The Distribution of Loan Portfolio Value".
This is the same model underlying BCBS IRB formulas (CRE31).

The key insight: in a perfectly granular portfolio where all exposures
share a single systematic factor, the conditional default rate at
the 99.9th percentile is:

    P(D|Z) = Phi( (Phi^-1(PD) + sqrt(rho) * Phi^-1(0.999)) / sqrt(1-rho) )

This is exactly the formula used in BCBS CRE31.4.
"""

import logging
import math

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)


def vasicek_conditional_default_rate(
    pd: float,
    rho: float,
    z: float,
) -> float:
    """Conditional default rate given systematic factor realization.

    Formula:
        P(D|Z=z) = Phi( (Phi^-1(PD) + sqrt(rho) * z) / sqrt(1-rho) )

    Args:
        pd: Unconditional probability of default.
        rho: Asset correlation (systematic factor loading).
        z: Systematic factor realization (standard normal).

    Returns:
        Conditional default rate.
    """
    if pd <= 0.0:
        return 0.0
    if pd >= 1.0:
        return 1.0

    g_pd = norm.ppf(pd)
    conditional = norm.cdf(
        (g_pd + math.sqrt(rho) * z) / math.sqrt(1.0 - rho)
    )
    return float(conditional)


def vasicek_loss_quantile(
    pd: float,
    rho: float,
    lgd: float,
    confidence: float = 0.999,
) -> float:
    """Loss quantile for an infinitely granular portfolio (ASRF).

    This is the Basel III IRB formula for capital requirement:
        VaR = LGD * Phi( (Phi^-1(PD) + sqrt(rho) * Phi^-1(q)) / sqrt(1-rho) )

    Args:
        pd: Probability of default.
        rho: Asset correlation.
        lgd: Loss given default.
        confidence: Confidence level (default 99.9%).

    Returns:
        Loss quantile (fraction of portfolio).
    """
    g_pd = norm.ppf(max(min(pd, 0.9999), 0.0001))
    g_q = norm.ppf(confidence)

    conditional_pd = norm.cdf(
        (g_pd + math.sqrt(rho) * g_q) / math.sqrt(1.0 - rho)
    )

    return lgd * float(conditional_pd)


def expected_loss(pd: float, lgd: float) -> float:
    """Expected loss for a single exposure.

    EL = PD * LGD

    Args:
        pd: Probability of default.
        lgd: Loss given default.

    Returns:
        Expected loss rate.
    """
    return pd * lgd


def unexpected_loss_asrf(
    pd: float,
    rho: float,
    lgd: float,
    confidence: float = 0.999,
) -> float:
    """Unexpected loss under ASRF model.

    UL = VaR(q) - EL

    Args:
        pd: Probability of default.
        rho: Asset correlation.
        lgd: Loss given default.
        confidence: Confidence level.

    Returns:
        Unexpected loss rate.
    """
    var = vasicek_loss_quantile(pd, rho, lgd, confidence)
    el = expected_loss(pd, lgd)
    return max(var - el, 0.0)


def economic_capital_asrf(
    pd: float,
    rho: float,
    lgd: float,
    ead: float,
    confidence: float = 0.999,
) -> dict[str, float]:
    """Economic capital calculation under ASRF model.

    Args:
        pd: Probability of default.
        rho: Asset correlation.
        lgd: Loss given default.
        ead: Exposure at default.
        confidence: Confidence level.

    Returns:
        Dict with el, ul, var, ec (all in currency units).
    """
    el_rate = expected_loss(pd, lgd)
    var_rate = vasicek_loss_quantile(pd, rho, lgd, confidence)
    ul_rate = max(var_rate - el_rate, 0.0)

    return {
        "expected_loss": el_rate * ead,
        "var": var_rate * ead,
        "unexpected_loss": ul_rate * ead,
        "economic_capital": ul_rate * ead,
        "el_rate": el_rate,
        "var_rate": var_rate,
        "ul_rate": ul_rate,
    }


def vasicek_portfolio_loss_distribution(
    pd: float,
    rho: float,
    lgd: float,
    n_points: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the Vasicek portfolio loss distribution.

    Computes the PDF of portfolio losses for an infinitely granular
    portfolio under the single-factor model.

    Args:
        pd: Probability of default.
        rho: Asset correlation.
        lgd: Loss given default.
        n_points: Number of points for the distribution.

    Returns:
        Tuple of (loss_values, probability_density).
    """
    # Generate systematic factor realizations
    z_values = np.linspace(-4.0, 4.0, n_points)

    # Conditional default rates
    g_pd = norm.ppf(max(min(pd, 0.9999), 0.0001))
    cond_pds = norm.cdf(
        (g_pd + np.sqrt(rho) * z_values) / np.sqrt(1.0 - rho)
    )

    # Loss values
    losses = lgd * cond_pds

    # PDF: standard normal density for Z
    density = norm.pdf(z_values)

    return losses, density
