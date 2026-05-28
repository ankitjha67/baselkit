"""
Portfolio capital allocation.

Reference:
    - Tasche (2008) — "Capital Allocation to Business Units and
      Sub-Portfolios: the Euler Principle".
    - BCBS 152 — range of practices in economic capital frameworks.

Allocates portfolio-level risk capital (VaR or Expected Shortfall) to
individual exposures using:
    - Marginal contribution: stand-alone vs portfolio-without-i.
    - Euler / VaR contribution: E[L_i | L = VaR] (covariance-based proxy).
    - Expected Shortfall contribution: E[L_i | L >= VaR].

Euler allocations are full (sum to total) and are the coherent
allocation for ES.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def marginal_contributions(
    loss_scenarios: np.ndarray,
    confidence_level: float = 0.99,
) -> np.ndarray:
    """Marginal VaR contribution per exposure.

    Marginal contribution_i = VaR(portfolio) - VaR(portfolio without i)

    Args:
        loss_scenarios: (n_scenarios, n_exposures) simulated losses.
        confidence_level: VaR quantile.

    Returns:
        Marginal contributions (n_exposures,).

    Raises:
        ValueError: If confidence_level not in (0, 1).
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    loss_scenarios = np.asarray(loss_scenarios, dtype=np.float64)
    total_loss = loss_scenarios.sum(axis=1)
    portfolio_var = np.quantile(total_loss, confidence_level)

    n_exposures = loss_scenarios.shape[1]
    contributions = np.zeros(n_exposures)
    for i in range(n_exposures):
        without_i = total_loss - loss_scenarios[:, i]
        var_without_i = np.quantile(without_i, confidence_level)
        contributions[i] = portfolio_var - var_without_i
    return contributions


def euler_var_contributions(
    loss_scenarios: np.ndarray,
    confidence_level: float = 0.99,
    band: float = 0.05,
) -> np.ndarray:
    """Euler (VaR) contributions: E[L_i | L ≈ VaR].

    Estimates each exposure's expected loss conditional on the total
    portfolio loss being in a neighbourhood of the VaR quantile. Euler
    contributions sum (approximately) to the portfolio VaR.

    Args:
        loss_scenarios: (n_scenarios, n_exposures) simulated losses.
        confidence_level: VaR quantile.
        band: Half-width (as a fraction of scenarios) of the
            conditioning band around the VaR.

    Returns:
        Euler VaR contributions (n_exposures,).

    Reference:
        Tasche (2008).
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    loss_scenarios = np.asarray(loss_scenarios, dtype=np.float64)
    total_loss = loss_scenarios.sum(axis=1)
    var = np.quantile(total_loss, confidence_level)

    # Scenarios in a band around the VaR
    tol = band * (np.max(total_loss) - np.min(total_loss) + 1e-12)
    mask = np.abs(total_loss - var) <= tol
    if not np.any(mask):
        # Fall back to the single nearest scenario
        idx = np.argmin(np.abs(total_loss - var))
        return loss_scenarios[idx]
    return loss_scenarios[mask].mean(axis=0)


def expected_shortfall_contributions(
    loss_scenarios: np.ndarray,
    confidence_level: float = 0.975,
) -> np.ndarray:
    """Expected Shortfall contributions: E[L_i | L >= VaR].

    ES contributions are coherent and sum exactly to the portfolio ES.

    Args:
        loss_scenarios: (n_scenarios, n_exposures) simulated losses.
        confidence_level: ES/VaR quantile (e.g., 0.975 per FRTB).

    Returns:
        ES contributions (n_exposures,).

    Raises:
        ValueError: If confidence_level not in (0, 1).

    Reference:
        Tasche (2008), FRTB Expected Shortfall.
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    loss_scenarios = np.asarray(loss_scenarios, dtype=np.float64)
    total_loss = loss_scenarios.sum(axis=1)
    var = np.quantile(total_loss, confidence_level)

    tail_mask = total_loss >= var
    if not np.any(tail_mask):
        return np.zeros(loss_scenarios.shape[1])
    return loss_scenarios[tail_mask].mean(axis=0)
