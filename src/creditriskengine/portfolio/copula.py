"""
Gaussian copula Monte Carlo simulation for portfolio credit risk.

Implements single-factor and multi-factor models for credit portfolio
loss simulation.
"""

import logging

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)


def simulate_single_factor(
    pds: np.ndarray,
    lgds: np.ndarray,
    eads: np.ndarray,
    rho: float,
    n_simulations: int = 10_000,
    seed: int | None = None,
    antithetic: bool = True,
) -> np.ndarray:
    """Single-factor Gaussian copula Monte Carlo simulation.

    Maps to the Basel ASRF model but with finite portfolio granularity.

    For each simulation:
    1. Draw systematic factor Z ~ N(0,1)
    2. For each obligor i, draw idiosyncratic factor eps_i ~ N(0,1)
    3. Asset return: A_i = sqrt(rho)*Z + sqrt(1-rho)*eps_i
    4. Default if: A_i < Phi^-1(PD_i)
    5. Loss = sum(default_i * LGD_i * EAD_i)

    Args:
        pds: Array of PDs per obligor (N,).
        lgds: Array of LGDs per obligor (N,).
        eads: Array of EADs per obligor (N,).
        rho: Common asset correlation.
        n_simulations: Number of Monte Carlo simulations.
        seed: Random seed for reproducibility.
        antithetic: Use antithetic variates for variance reduction.

    Returns:
        Array of portfolio losses (n_simulations,).
    """
    if not 0.0 < rho < 1.0:
        raise ValueError(f"rho must be in (0, 1), got {rho}")

    rng = np.random.default_rng(seed)
    n_obligors = len(pds)

    # Default thresholds
    thresholds = norm.ppf(np.maximum(pds, 1e-10))

    if antithetic:
        half_sims = n_simulations // 2
        z_half = rng.standard_normal(half_sims)
        z = np.concatenate([z_half, -z_half])
        eps_half = rng.standard_normal((half_sims, n_obligors))
        eps = np.concatenate([eps_half, -eps_half], axis=0)
        n_simulations = len(z)
    else:
        z = rng.standard_normal(n_simulations)
        eps = rng.standard_normal((n_simulations, n_obligors))

    # Asset returns: (n_sims, n_obligors)
    sqrt_rho = np.sqrt(rho)
    sqrt_1_minus_rho = np.sqrt(1.0 - rho)
    asset_returns = sqrt_rho * z[:, np.newaxis] + sqrt_1_minus_rho * eps

    # Default indicator
    defaults = asset_returns < thresholds[np.newaxis, :]

    # Portfolio losses
    losses = defaults * lgds[np.newaxis, :] * eads[np.newaxis, :]
    portfolio_losses = np.sum(losses, axis=1)

    return np.asarray(portfolio_losses)


def simulate_multi_factor(
    pds: np.ndarray,
    lgds: np.ndarray,
    eads: np.ndarray,
    factor_loadings: np.ndarray,
    n_simulations: int = 10_000,
    seed: int | None = None,
) -> np.ndarray:
    """Multi-factor Gaussian copula simulation with sector correlations.

    Each obligor has loadings on K systematic factors:
        A_i = sum_k(w_ik * Z_k) + sqrt(1 - sum(w_ik^2)) * eps_i

    Args:
        pds: Array of PDs (N,).
        lgds: Array of LGDs (N,).
        eads: Array of EADs (N,).
        factor_loadings: Factor loading matrix (N, K).
        n_simulations: Number of simulations.
        seed: Random seed.

    Returns:
        Array of portfolio losses (n_simulations,).
    """
    rng = np.random.default_rng(seed)
    n_obligors, n_factors = factor_loadings.shape

    thresholds = norm.ppf(np.maximum(pds, 1e-10))

    # Systematic factors (n_sims, K)
    z = rng.standard_normal((n_simulations, n_factors))

    # Idiosyncratic factors (n_sims, N)
    eps = rng.standard_normal((n_simulations, n_obligors))

    # Systematic component: (n_sims, N)
    systematic = z @ factor_loadings.T

    # Idiosyncratic scaling
    r_squared = np.sum(factor_loadings ** 2, axis=1)
    idio_scale = np.sqrt(np.maximum(1.0 - r_squared, 0.0))

    asset_returns = systematic + idio_scale[np.newaxis, :] * eps

    defaults = asset_returns < thresholds[np.newaxis, :]
    losses = defaults * lgds[np.newaxis, :] * eads[np.newaxis, :]

    return np.asarray(np.sum(losses, axis=1))


def credit_var(
    losses: np.ndarray,
    confidence: float = 0.999,
) -> float:
    """Credit Value at Risk from simulated loss distribution.

    Args:
        losses: Simulated portfolio losses.
        confidence: Confidence level.

    Returns:
        Credit VaR at the specified confidence level.
    """
    return float(np.percentile(losses, confidence * 100))


def expected_shortfall(
    losses: np.ndarray,
    confidence: float = 0.999,
) -> float:
    """Expected Shortfall (CVaR) from simulated losses.

    ES = E[Loss | Loss > VaR(q)]

    Args:
        losses: Simulated portfolio losses.
        confidence: Confidence level.

    Returns:
        Expected shortfall.
    """
    var = credit_var(losses, confidence)
    tail = losses[losses >= var]
    return float(np.mean(tail)) if len(tail) > 0 else var


def loss_distribution_stats(
    losses: np.ndarray,
    total_ead: float,
) -> dict[str, float]:
    """Summary statistics of the loss distribution.

    Args:
        losses: Simulated portfolio losses.
        total_ead: Total portfolio EAD for percentage calculations.

    Returns:
        Dict with mean, std, var_999, es_999, skewness, kurtosis.
    """
    from scipy.stats import kurtosis, skew

    return {
        "mean_loss": float(np.mean(losses)),
        "std_loss": float(np.std(losses)),
        "var_999": credit_var(losses, 0.999),
        "var_995": credit_var(losses, 0.995),
        "es_999": expected_shortfall(losses, 0.999),
        "mean_loss_pct": float(np.mean(losses)) / total_ead * 100 if total_ead > 0 else 0,
        "var_999_pct": credit_var(losses, 0.999) / total_ead * 100 if total_ead > 0 else 0,
        "skewness": float(skew(losses)),
        "kurtosis": float(kurtosis(losses)),
    }
