"""
Credit VaR utilities.

Value-at-Risk calculations for credit portfolios.
"""

import logging

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)


def parametric_credit_var(
    el: float,
    ul_std: float,
    confidence: float = 0.999,
) -> float:
    """Parametric Credit VaR assuming normal loss distribution.

    VaR = EL + z_alpha * sigma

    Args:
        el: Expected loss.
        ul_std: Standard deviation of unexpected loss.
        confidence: Confidence level.

    Returns:
        Credit VaR.
    """
    z = norm.ppf(confidence)
    return el + z * ul_std


def marginal_var(
    portfolio_var: float,
    portfolio_std: float,
    exposure_contribution_to_std: float,
) -> float:
    """Marginal VaR contribution of a single exposure.

    Args:
        portfolio_var: Total portfolio VaR.
        portfolio_std: Portfolio loss standard deviation.
        exposure_contribution_to_std: Exposure's contribution to portfolio std.

    Returns:
        Marginal VaR contribution.
    """
    if portfolio_std < 1e-15:
        return 0.0
    return portfolio_var * (exposure_contribution_to_std / portfolio_std)
