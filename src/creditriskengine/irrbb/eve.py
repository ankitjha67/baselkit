"""
Economic Value of Equity (EVE) sensitivity for IRRBB.

Reference:
    - BCBS d368 (Interest Rate Risk in the Banking Book, April 2016).
    - EBA/GL/2018/02.

EVE measures the change in the present value of banking-book cash flows
under prescribed interest-rate shock scenarios. The six standardised
shocks (BCBS d368 Annex 2) are:
    1. Parallel up
    2. Parallel down
    3. Steepener (short down, long up)
    4. Flattener (short up, long down)
    5. Short rate up
    6. Short rate down
"""

from __future__ import annotations

import logging
from enum import StrEnum

import numpy as np

logger = logging.getLogger(__name__)


class InterestRateShock(StrEnum):
    """The six BCBS d368 standardised interest-rate shock scenarios."""

    PARALLEL_UP = "parallel_up"
    PARALLEL_DOWN = "parallel_down"
    STEEPENER = "steepener"
    FLATTENER = "flattener"
    SHORT_UP = "short_up"
    SHORT_DOWN = "short_down"


SHOCK_SCENARIOS: tuple[InterestRateShock, ...] = tuple(InterestRateShock)
"""All six standardised shock scenarios."""


def _shocked_curve(
    base_rates: np.ndarray,
    tenors_years: np.ndarray,
    shock: InterestRateShock,
    parallel_bps: float,
    short_bps: float,
    long_bps: float,
) -> np.ndarray:
    """Apply a BCBS d368 shock to the yield curve.

    The steepener/flattener use the BCBS scalar weighting:
        short scalar = exp(-t / 4)  (decays with tenor)
        long scalar  = 1 - exp(-t / 4)
    """
    base_rates = np.asarray(base_rates, dtype=np.float64)
    tenors_years = np.asarray(tenors_years, dtype=np.float64)

    short_scalar = np.exp(-tenors_years / 4.0)
    long_scalar = 1.0 - short_scalar

    if shock == InterestRateShock.PARALLEL_UP:
        delta = np.full_like(base_rates, parallel_bps / 10_000.0)
    elif shock == InterestRateShock.PARALLEL_DOWN:
        delta = np.full_like(base_rates, -parallel_bps / 10_000.0)
    elif shock == InterestRateShock.STEEPENER:
        # short down, long up
        delta = (-0.65 * short_bps * short_scalar + 0.9 * long_bps * long_scalar) / 10_000.0
    elif shock == InterestRateShock.FLATTENER:
        # short up, long down
        delta = (0.8 * short_bps * short_scalar - 0.6 * long_bps * long_scalar) / 10_000.0
    elif shock == InterestRateShock.SHORT_UP:
        delta = (short_bps * short_scalar) / 10_000.0
    else:  # SHORT_DOWN
        delta = (-short_bps * short_scalar) / 10_000.0

    return base_rates + delta


def repricing_gap(
    cashflows: np.ndarray,
    tenors_years: np.ndarray,
    base_rates: np.ndarray,
) -> float:
    """Present value of banking-book cash flows under the base curve.

    PV = sum_t CF(t) / (1 + r(t))^t

    Args:
        cashflows: Net cash flow per tenor bucket.
        tenors_years: Tenor (years) of each bucket.
        base_rates: Base discount rate per bucket.

    Returns:
        Present value of the cash-flow profile.
    """
    cashflows = np.asarray(cashflows, dtype=np.float64)
    tenors_years = np.asarray(tenors_years, dtype=np.float64)
    base_rates = np.asarray(base_rates, dtype=np.float64)
    dfs = 1.0 / (1.0 + base_rates) ** tenors_years
    return float(np.sum(cashflows * dfs))


def eve_sensitivity(
    cashflows: np.ndarray,
    tenors_years: np.ndarray,
    base_rates: np.ndarray,
    parallel_bps: float = 200.0,
    short_bps: float = 250.0,
    long_bps: float = 100.0,
    currency: str | None = None,
    apply_floor: bool = False,
) -> dict[InterestRateShock, float]:
    """Compute Delta-EVE for all six standardised shocks.

    Delta-EVE = PV(shocked) - PV(base) for each scenario. A negative
    value indicates a loss of economic value under that shock.

    Args:
        cashflows: Net banking-book cash flow per tenor bucket.
        tenors_years: Tenor (years) per bucket.
        base_rates: Base discount rates per bucket.
        parallel_bps: Parallel shock size (default 200bps, d368 generic).
        short_bps: Short-rate shock size (default 250bps).
        long_bps: Long-rate shock size (default 100bps).
        currency: If supplied, the per-currency d578 (July 2024,
            applicable 1 Jan 2026) shock calibration overrides the three
            ``*_bps`` arguments — see ``irrbb.shocks``.
        apply_floor: If True, apply the post-shock interest-rate floor
            (-100 bps at overnight, +5 bps/year) to each shocked curve.

    Returns:
        Mapping of each shock scenario to its Delta-EVE.

    Reference:
        BCBS d368 Annex 2; BCBS d578 (per-currency recalibration).
    """
    if currency is not None:
        from creditriskengine.irrbb.shocks import get_currency_shocks

        shocks = get_currency_shocks(currency)
        parallel_bps = shocks.parallel_bps
        short_bps = shocks.short_bps
        long_bps = shocks.long_bps

    base_pv = repricing_gap(cashflows, tenors_years, base_rates)
    cashflows = np.asarray(cashflows, dtype=np.float64)
    tenors_years = np.asarray(tenors_years, dtype=np.float64)

    result: dict[InterestRateShock, float] = {}
    for shock in SHOCK_SCENARIOS:
        shocked_rates = _shocked_curve(
            base_rates, tenors_years, shock, parallel_bps, short_bps, long_bps
        )
        if apply_floor:
            from creditriskengine.irrbb.shocks import apply_post_shock_floor

            shocked_rates = apply_post_shock_floor(shocked_rates, tenors_years)
        dfs = 1.0 / (1.0 + shocked_rates) ** tenors_years
        shocked_pv = float(np.sum(cashflows * dfs))
        result[shock] = shocked_pv - base_pv

    return result
