"""
US CECL (ASC 326) — Current Expected Credit Losses.

Key differences from IFRS 9:
1. NO staging — lifetime losses from Day 1 for all assets
2. Applies to amortized cost and certain off-balance-sheet commitments
3. No SICR assessment
4. Historical loss experience adjusted for current conditions
   and reasonable/supportable forecasts
5. Reversion to historical mean beyond forecast period

Reference: ASC 326-20 (Financial Instruments — Credit Losses).
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def cecl_pd_lgd(
    marginal_pds: np.ndarray,
    lgds: np.ndarray | float,
    eads: np.ndarray | float,
    discount_rate: float = 0.0,
) -> float:
    """CECL lifetime ECL using PD/LGD method.

    Similar to IFRS 9 lifetime ECL but applied from Day 1
    (no staging distinction).

    Formula:
        ECL = Sum(t=1..T) [Marginal_PD(t) * LGD(t) * EAD(t) * DF(t)]

    Args:
        marginal_pds: Marginal PD curve for remaining life.
        lgds: LGD values (scalar or per-period array).
        eads: EAD values (scalar or per-period array).
        discount_rate: Discount rate for present value.

    Returns:
        Lifetime ECL amount.
    """
    periods = len(marginal_pds)
    t = np.arange(1, periods + 1, dtype=np.float64)
    dfs = 1.0 / (1.0 + discount_rate) ** t if discount_rate > 0 else np.ones(periods)

    if isinstance(lgds, (int, float)):
        lgds = np.full(periods, lgds)
    if isinstance(eads, (int, float)):
        eads = np.full(periods, eads)

    ecl = float(np.sum(marginal_pds * lgds * eads * dfs))
    logger.debug("CECL PD/LGD ECL: periods=%d ECL=%.2f", periods, ecl)
    return ecl


def cecl_loss_rate(
    ead: float,
    historical_loss_rate: float,
    qualitative_adjustment: float = 0.0,
    forecast_adjustment: float = 0.0,
    remaining_life_years: float = 1.0,
) -> float:
    """CECL loss-rate method.

    ECL = EAD * adjusted_loss_rate * remaining_life

    The loss rate is adjusted for current conditions and
    reasonable/supportable forecasts.

    Args:
        ead: Exposure at default.
        historical_loss_rate: Historical annualized loss rate.
        qualitative_adjustment: Q-factor adjustment (additive).
        forecast_adjustment: Forecast-based adjustment (additive).
        remaining_life_years: Weighted average remaining maturity.

    Returns:
        ECL amount.
    """
    adjusted_rate = max(historical_loss_rate + qualitative_adjustment + forecast_adjustment, 0.0)
    ecl = ead * adjusted_rate * remaining_life_years
    return ecl
