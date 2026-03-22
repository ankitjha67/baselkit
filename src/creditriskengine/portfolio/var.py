"""
Credit VaR utilities.

Value-at-Risk calculations for credit portfolios, including parametric,
historical simulation, and Cornish-Fisher approaches, as well as
risk decomposition tools (marginal, incremental, component VaR) and
Expected Shortfall.

Regulatory context:
    - Basel II/III Pillar 1: 99.9% confidence for credit risk capital
    - Basel III FRTB: Expected Shortfall replaces VaR for market risk
    - BCBS 128 (June 2006): International Convergence of Capital Measurement
    - SR 11-7 (Fed): Model Risk Management — VaR model validation
"""

import logging

import numpy as np
from scipy.stats import norm  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


def parametric_credit_var(
    el: float,
    ul_std: float,
    confidence: float = 0.999,
) -> float:
    """Parametric Credit VaR assuming normal loss distribution.

    VaR = EL + z_alpha * sigma

    Reference: BCBS 128 (IRB formula foundation), Pillar 1 at 99.9%.

    Args:
        el: Expected loss.
        ul_std: Standard deviation of unexpected loss.
        confidence: Confidence level (e.g. 0.999 for Basel II IRB).

    Returns:
        Credit VaR at the specified confidence level.

    Raises:
        ValueError: If confidence is not in (0, 1) or ul_std is negative.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if ul_std < 0.0:
        raise ValueError(f"ul_std must be non-negative, got {ul_std}")

    z = float(norm.ppf(confidence))
    result = el + z * ul_std

    logger.debug(
        "Parametric VaR: EL=%.4f, UL_std=%.4f, z=%.4f, VaR=%.4f",
        el, ul_std, z, result,
    )
    return result


def marginal_var(
    portfolio_var: float,
    portfolio_std: float,
    exposure_contribution_to_std: float,
) -> float:
    """Marginal VaR contribution of a single exposure.

    Marginal VaR measures the rate of change in portfolio VaR with
    respect to a small increase in the exposure size.

    MVaR_i = VaR_p * (sigma_i_contribution / sigma_p)

    Reference: Jorion (2007), "Value at Risk", Ch. 7.

    Args:
        portfolio_var: Total portfolio VaR.
        portfolio_std: Portfolio loss standard deviation.
        exposure_contribution_to_std: Exposure's contribution to portfolio std
            (i.e. d(sigma_p)/d(w_i) or cov(L_i, L_p)/sigma_p).

    Returns:
        Marginal VaR contribution.
    """
    if portfolio_std < 1e-15:
        logger.warning("Portfolio std near zero (%.2e); marginal VaR is zero.", portfolio_std)
        return 0.0
    return portfolio_var * (exposure_contribution_to_std / portfolio_std)


def historical_simulation_var(
    loss_distribution: np.ndarray,
    confidence: float = 0.999,
) -> float:
    """Historical simulation VaR from an empirical loss distribution.

    Computes VaR as the quantile of observed or simulated losses.
    No distributional assumption is required — the result is purely
    data-driven.

    Reference:
        - BCBS 128: Non-parametric approaches to credit VaR
        - SR 11-7: Outcomes analysis for VaR back-testing

    Args:
        loss_distribution: Array of historical or simulated portfolio losses.
            Positive values represent losses.
        confidence: Confidence level (e.g. 0.999 for 99.9%).

    Returns:
        VaR at the given confidence level.

    Raises:
        ValueError: If loss_distribution is empty or confidence is out of range.
    """
    loss_distribution = np.asarray(loss_distribution, dtype=np.float64)

    if loss_distribution.size == 0:
        raise ValueError("loss_distribution must be non-empty.")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    result = float(np.percentile(loss_distribution, confidence * 100))

    logger.debug(
        "Historical VaR (%.2f%%): n_scenarios=%d, VaR=%.4f",
        confidence * 100,
        loss_distribution.size,
        result,
    )
    return result


def cornish_fisher_var(
    el: float,
    ul_std: float,
    skewness: float,
    kurtosis: float,
    confidence: float = 0.999,
) -> float:
    """Cornish-Fisher VaR adjusting for skewness and excess kurtosis.

    Adjusts the normal quantile for non-normality in the loss distribution,
    which is critical for credit portfolios that are typically right-skewed
    with fat tails.

    Formula:
        z_cf = z + (z^2 - 1)*S/6 + (z^3 - 3z)*K/24 - (2z^3 - 5z)*S^2/36

    Where S = skewness, K = excess kurtosis, z = normal quantile.

    Reference:
        - Cornish & Fisher (1937)
        - Mina & Xiao (2001), "Return to RiskMetrics" — application to
          non-normal portfolio loss distributions

    Args:
        el: Expected loss.
        ul_std: Standard deviation of unexpected loss.
        skewness: Skewness of the loss distribution.
        kurtosis: Excess kurtosis of the loss distribution (kurtosis - 3
            for the normal distribution adjustment).
        confidence: Confidence level.

    Returns:
        Cornish-Fisher adjusted VaR.

    Raises:
        ValueError: If ul_std is negative or confidence out of range.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if ul_std < 0.0:
        raise ValueError(f"ul_std must be non-negative, got {ul_std}")

    z = float(norm.ppf(confidence))

    z_cf = (
        z
        + (z**2 - 1) * skewness / 6.0
        + (z**3 - 3 * z) * kurtosis / 24.0
        - (2 * z**3 - 5 * z) * skewness**2 / 36.0
    )

    result = el + z_cf * ul_std

    logger.debug(
        "Cornish-Fisher VaR: z_normal=%.4f, z_cf=%.4f, skew=%.4f, "
        "kurt=%.4f, VaR=%.4f",
        z, z_cf, skewness, kurtosis, result,
    )
    return result


def incremental_var(
    portfolio_losses: np.ndarray,
    portfolio_with_exposure_losses: np.ndarray,
    confidence: float = 0.999,
) -> float:
    """Incremental VaR -- change in portfolio VaR from adding an exposure.

    Measures the VaR impact of adding a new exposure to the portfolio,
    useful for portfolio construction and limit-setting decisions.

    IVaR = VaR(portfolio + exposure) - VaR(portfolio)

    A positive value indicates the new exposure increases portfolio risk.

    Reference:
        - Jorion (2007), "Value at Risk", Ch. 7
        - SR 11-7: Concentration risk measurement

    Args:
        portfolio_losses: Simulated loss distribution without the new exposure.
        portfolio_with_exposure_losses: Simulated loss distribution including
            the new exposure.
        confidence: Confidence level.

    Returns:
        Incremental VaR (positive means VaR increases).

    Raises:
        ValueError: If either loss distribution is empty.
    """
    portfolio_losses = np.asarray(portfolio_losses, dtype=np.float64)
    portfolio_with_exposure_losses = np.asarray(
        portfolio_with_exposure_losses, dtype=np.float64
    )

    if portfolio_losses.size == 0:
        raise ValueError("portfolio_losses must be non-empty.")
    if portfolio_with_exposure_losses.size == 0:
        raise ValueError("portfolio_with_exposure_losses must be non-empty.")

    var_before = historical_simulation_var(portfolio_losses, confidence)
    var_after = historical_simulation_var(portfolio_with_exposure_losses, confidence)
    result = var_after - var_before

    logger.debug(
        "Incremental VaR (%.2f%%): VaR_before=%.4f, VaR_after=%.4f, "
        "IVaR=%.4f",
        confidence * 100,
        var_before,
        var_after,
        result,
    )
    return result


def component_var(
    portfolio_var: float,
    portfolio_std: float,
    exposure_stds: np.ndarray,
    correlations_with_portfolio: np.ndarray,
) -> np.ndarray:
    """Component VaR -- decompose portfolio VaR into per-exposure contributions.

    Uses the Euler decomposition to attribute portfolio VaR to individual
    exposures. The key property is that component VaRs sum to total
    portfolio VaR:

        Component_VaR_i = (VaR_p / sigma_p) * sigma_i * rho(i, portfolio)
        Sum(Component_VaR_i) = VaR_p

    This is essential for risk-based capital allocation and concentration
    risk monitoring.

    Reference:
        - Tasche (2000), "Risk contributions and performance measurement"
        - BCBS 128: Pillar 2 concentration risk
        - Jorion (2007), Ch. 7: Euler allocation of VaR

    Args:
        portfolio_var: Total portfolio VaR.
        portfolio_std: Portfolio loss standard deviation.
        exposure_stds: Per-exposure loss standard deviations (n_exposures,).
        correlations_with_portfolio: Correlation of each exposure's loss
            with the total portfolio loss (n_exposures,).

    Returns:
        Array of component VaR contributions (n_exposures,).

    Raises:
        ValueError: If exposure_stds and correlations_with_portfolio have
            different lengths.
    """
    exposure_stds = np.asarray(exposure_stds, dtype=np.float64)
    correlations_with_portfolio = np.asarray(correlations_with_portfolio, dtype=np.float64)

    if exposure_stds.shape != correlations_with_portfolio.shape:
        raise ValueError(
            f"exposure_stds and correlations_with_portfolio must have the same shape, "
            f"got {exposure_stds.shape} and {correlations_with_portfolio.shape}."
        )

    if portfolio_std < 1e-15:
        logger.warning(
            "Portfolio std near zero (%.2e); all component VaRs are zero.",
            portfolio_std,
        )
        return np.zeros_like(exposure_stds)

    result = (portfolio_var / portfolio_std) * exposure_stds * correlations_with_portfolio

    logger.debug(
        "Component VaR: n_exposures=%d, portfolio_VaR=%.4f, "
        "sum(component_VaR)=%.4f",
        len(exposure_stds),
        portfolio_var,
        float(np.sum(result)),
    )
    return result  # type: ignore[no-any-return]


def expected_shortfall(
    loss_distribution: np.ndarray,
    confidence: float = 0.999,
) -> float:
    """Expected Shortfall (CVaR) -- average loss beyond VaR.

    ES = E[Loss | Loss >= VaR]

    Expected Shortfall is a coherent risk measure (satisfies sub-additivity),
    unlike VaR. It captures tail risk more comprehensively and is mandated
    by Basel III FRTB for market risk capital.

    Reference:
        - Basel III FRTB (BCBS d352/d457): ES replaces VaR for market risk
        - Acerbi & Tasche (2002): "On the coherence of Expected Shortfall"
        - BCBS 128: Pillar 2 supplementary risk measures

    Args:
        loss_distribution: Array of portfolio losses (positive = loss).
        confidence: Confidence level (e.g. 0.975 for FRTB, 0.999 for credit).

    Returns:
        Expected Shortfall at the given confidence level.

    Raises:
        ValueError: If loss_distribution is empty or confidence out of range.
    """
    loss_distribution = np.asarray(loss_distribution, dtype=np.float64)

    if loss_distribution.size == 0:
        raise ValueError("loss_distribution must be non-empty.")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    var = historical_simulation_var(loss_distribution, confidence)
    tail = loss_distribution[loss_distribution >= var]

    if tail.size == 0:
        # Edge case: no observations at or above VaR quantile
        # This can happen with very small samples; return VaR as lower bound
        logger.warning(
            "No observations at or above VaR (%.4f); returning VaR as ES.",
            var,
        )
        return var

    result = float(np.mean(tail))

    logger.debug(
        "Expected Shortfall (%.2f%%): VaR=%.4f, n_tail=%d, ES=%.4f",
        confidence * 100,
        var,
        tail.size,
        result,
    )
    return result
