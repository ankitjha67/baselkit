"""
Economic capital estimation.

Combines portfolio models with regulatory requirements.
"""

import logging

import numpy as np

from creditriskengine.portfolio.copula import (
    credit_var,
    expected_shortfall,
    simulate_single_factor,
)
from creditriskengine.portfolio.vasicek import (
    economic_capital_asrf,
    expected_loss,
)

logger = logging.getLogger(__name__)


def ec_single_factor(
    pds: np.ndarray,
    lgds: np.ndarray,
    eads: np.ndarray,
    rho: float,
    confidence: float = 0.999,
    n_simulations: int = 50_000,
    seed: int | None = None,
) -> dict[str, float]:
    """Economic capital via single-factor Monte Carlo.

    Args:
        pds: PDs per obligor.
        lgds: LGDs per obligor.
        eads: EADs per obligor.
        rho: Common asset correlation.
        confidence: Confidence level.
        n_simulations: Number of simulations.
        seed: Random seed.

    Returns:
        Dict with el, var, ec, es.
    """
    losses = simulate_single_factor(pds, lgds, eads, rho, n_simulations, seed)

    el = float(np.sum(pds * lgds * eads))
    var = credit_var(losses, confidence)
    es = expected_shortfall(losses, confidence)

    return {
        "expected_loss": el,
        "var": var,
        "economic_capital": max(var - el, 0.0),
        "expected_shortfall": es,
        "total_ead": float(np.sum(eads)),
    }
