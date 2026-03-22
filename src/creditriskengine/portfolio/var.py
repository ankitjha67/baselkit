"""
Credit VaR utilities.

Value-at-Risk calculations for credit portfolios.
"""

import logging

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
    z = float(norm.ppf(confidence))
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


def historical_simulation_var(
    loss_distribution: "np.ndarray",
    confidence: float = 0.999,
) -> float:
    """Historical simulation VaR from empirical loss distribution.

    Computes VaR as the quantile of observed/simulated losses.
    No distributional assumption required.

    Args:
        loss_distribution: Array of historical/simulated losses.
        confidence: Confidence level (e.g., 0.999 for 99.9%).

    Returns:
        VaR at given confidence level.
    """
    import numpy as np

    return float(np.percentile(loss_distribution, confidence * 100))


def cornish_fisher_var(
    el: float,
    ul_std: float,
    skewness: float,
    kurtosis: float,
    confidence: float = 0.999,
) -> float:
    """Cornish-Fisher VaR adjusting for skewness and excess kurtosis.

    Adjusts the normal quantile for non-normality in the loss distribution.

    Formula:
        z_cf = z + (z² - 1) * S/6 + (z³ - 3z) * K/24 - (2z³ - 5z) * S²/36

    Where S = skewness, K = excess kurtosis, z = normal quantile.

    Args:
        el: Expected loss.
        ul_std: Standard deviation of unexpected loss.
        skewness: Skewness of loss distribution.
        kurtosis: Excess kurtosis of loss distribution.
        confidence: Confidence level.

    Returns:
        Cornish-Fisher adjusted VaR.
    """
    z = float(norm.ppf(confidence))

    z_cf = (
        z
        + (z**2 - 1) * skewness / 6.0
        + (z**3 - 3 * z) * kurtosis / 24.0
        - (2 * z**3 - 5 * z) * skewness**2 / 36.0
    )

    return el + z_cf * ul_std


def incremental_var(
    portfolio_losses: "np.ndarray",
    portfolio_with_exposure_losses: "np.ndarray",
    confidence: float = 0.999,
) -> float:
    """Incremental VaR — change in portfolio VaR from adding an exposure.

    Args:
        portfolio_losses: Loss distribution without the new exposure.
        portfolio_with_exposure_losses: Loss distribution with the new exposure.
        confidence: Confidence level.

    Returns:
        Incremental VaR (positive means VaR increases).
    """
    var_before = historical_simulation_var(portfolio_losses, confidence)
    var_after = historical_simulation_var(portfolio_with_exposure_losses, confidence)
    return var_after - var_before


def component_var(
    portfolio_var: float,
    portfolio_std: float,
    exposure_stds: "np.ndarray",
    correlations_with_portfolio: "np.ndarray",
) -> "np.ndarray":
    """Component VaR — decompose portfolio VaR into per-exposure contributions.

    Component_VaR_i = (VaR / sigma_p) * sigma_i * rho(i, portfolio)

    The sum of component VaRs equals total portfolio VaR (Euler decomposition).

    Args:
        portfolio_var: Total portfolio VaR.
        portfolio_std: Portfolio loss standard deviation.
        exposure_stds: Per-exposure loss standard deviations.
        correlations_with_portfolio: Correlation of each exposure with portfolio.

    Returns:
        Array of component VaR contributions.
    """
    import numpy as np

    if portfolio_std < 1e-15:
        return np.zeros_like(exposure_stds)

    return (portfolio_var / portfolio_std) * exposure_stds * correlations_with_portfolio


def expected_shortfall(
    loss_distribution: "np.ndarray",
    confidence: float = 0.999,
) -> float:
    """Expected Shortfall (CVaR) — average loss beyond VaR.

    ES = E[Loss | Loss > VaR]

    More coherent risk measure than VaR; satisfies sub-additivity.

    Args:
        loss_distribution: Array of losses.
        confidence: Confidence level.

    Returns:
        Expected Shortfall.
    """
    import numpy as np

    var = historical_simulation_var(loss_distribution, confidence)
    tail = loss_distribution[loss_distribution >= var]
    if len(tail) == 0:
        return var
    return float(np.mean(tail))
