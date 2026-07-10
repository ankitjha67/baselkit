"""IRRBB standardised interest-rate shock calibrations (d368 / d578).

Reference:
    - BCBS d368 (April 2016) — original standardised shocks (SRP31/98).
    - BCBS d578 (16 July 2024) — "Recalibration of shocks for interest
      rate risk in the banking book", implementation 1 January 2026.

d578 methodology changes vs d368:
    * Local (per-currency) shock factors replace the global factors.
    * Absolute rate changes over a 24-year window (2000-2023), 99.9th
      percentile, rounded to 25 bps (previously 50 bps).
    * Shock caps: parallel 400 bps, short 500 bps, long 300 bps.
    * Post-shock interest-rate floor: -100 bps at the overnight tenor,
      rising 5 bps per year (0% from the 20-year tenor onward).

The exact per-currency shock sizes are published in the d578 annex. This
module ships the calibrations that could be verified (EUR, USD — from
secondary sources reproducing the annex; confirm against the primary text
for supervisory use) plus the d368 baseline as a fallback, and lets banks
register their supervisor-published values per currency via
:func:`register_currency_shocks`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# d578 shock caps (bps).
PARALLEL_CAP_BPS: float = 400.0
SHORT_CAP_BPS: float = 500.0
LONG_CAP_BPS: float = 300.0

# d578 rounding increment (bps).
ROUNDING_INCREMENT_BPS: float = 25.0

# Post-shock floor: -100 bps at t=0, +5 bps per year, 0% from 20 years.
FLOOR_START_BPS: float = -100.0
FLOOR_SLOPE_BPS_PER_YEAR: float = 5.0


@dataclass(frozen=True)
class CurrencyShocks:
    """Standardised shock sizes for one currency (bps).

    Attributes:
        parallel_bps: Parallel up/down shock size.
        short_bps: Short-rate shock size.
        long_bps: Long-rate shock size.
    """

    parallel_bps: float
    short_bps: float
    long_bps: float

    def __post_init__(self) -> None:
        if self.parallel_bps > PARALLEL_CAP_BPS:
            raise ValueError(f"parallel shock exceeds the {PARALLEL_CAP_BPS:.0f} bps cap")
        if self.short_bps > SHORT_CAP_BPS:
            raise ValueError(f"short shock exceeds the {SHORT_CAP_BPS:.0f} bps cap")
        if self.long_bps > LONG_CAP_BPS:
            raise ValueError(f"long shock exceeds the {LONG_CAP_BPS:.0f} bps cap")
        if min(self.parallel_bps, self.short_bps, self.long_bps) < 0.0:
            raise ValueError("shock sizes must be non-negative")


# d368 (2016) baseline calibration — retained as the pre-2026 fallback.
D368_BASELINE = CurrencyShocks(parallel_bps=200.0, short_bps=250.0, long_bps=100.0)

# d578 (2024) recalibrated values, applicable from 1 Jan 2026.
# EUR/USD reproduce the d578 annex via secondary sources — verify against
# the primary text before supervisory use; other currencies should be
# registered from the supervisor's published table.
_D578_SHOCKS: dict[str, CurrencyShocks] = {
    "EUR": CurrencyShocks(parallel_bps=225.0, short_bps=350.0, long_bps=200.0),
    "USD": CurrencyShocks(parallel_bps=200.0, short_bps=300.0, long_bps=225.0),
}


def register_currency_shocks(currency: str, shocks: CurrencyShocks) -> None:
    """Register (or override) the shock calibration for a currency.

    Use this to load the supervisor-published d578 annex values for
    currencies not shipped with the library.

    Args:
        currency: ISO currency code (e.g. "GBP").
        shocks: The shock sizes; validated against the d578 caps.
    """
    _D578_SHOCKS[currency.upper()] = shocks
    logger.info("Registered IRRBB shocks for %s: %s", currency.upper(), shocks)


def get_currency_shocks(currency: str, fallback_to_baseline: bool = True) -> CurrencyShocks:
    """Shock calibration for a currency.

    Args:
        currency: ISO currency code.
        fallback_to_baseline: If True (default), currencies without a
            registered d578 calibration fall back to the d368 baseline
            (200/250/100). If False, an unregistered currency raises.

    Returns:
        The :class:`CurrencyShocks` for the currency.

    Raises:
        KeyError: If the currency is unregistered and
            ``fallback_to_baseline`` is False.
    """
    key = currency.upper()
    if key in _D578_SHOCKS:
        return _D578_SHOCKS[key]
    if fallback_to_baseline:
        logger.warning(
            "No d578 calibration registered for %s; falling back to the "
            "d368 baseline (200/250/100)", key,
        )
        return D368_BASELINE
    raise KeyError(f"No IRRBB shock calibration registered for {key}")


def is_valid_shock_rounding(shock_bps: float) -> bool:
    """Whether a shock size respects the d578 25 bps rounding increment."""
    increments = shock_bps / ROUNDING_INCREMENT_BPS
    return abs(increments - round(increments)) < 1e-9


def post_shock_floor(tenor_years: float) -> float:
    """Post-shock interest-rate floor at a tenor (as a decimal rate).

    The floor starts at -100 bps for the overnight tenor and rises by
    5 bps per year of tenor, reaching 0% at 20 years (0% thereafter).

    Args:
        tenor_years: Tenor in years (>= 0).

    Returns:
        The floor as a decimal rate (e.g. -0.01 at t=0).

    Raises:
        ValueError: If ``tenor_years`` is negative.
    """
    if tenor_years < 0.0:
        raise ValueError("tenor_years must be non-negative")
    floor_bps = min(FLOOR_START_BPS + FLOOR_SLOPE_BPS_PER_YEAR * tenor_years, 0.0)
    return floor_bps / 10_000.0


def apply_post_shock_floor(
    shocked_rates: np.ndarray,
    tenors_years: np.ndarray,
) -> np.ndarray:
    """Apply the post-shock floor to a shocked rate curve.

    Args:
        shocked_rates: Shocked zero rates (decimals) per tenor.
        tenors_years: Tenor of each rate in years.

    Returns:
        The floored shocked curve: ``max(rate, floor(tenor))`` per point.

    Raises:
        ValueError: If the arrays have different shapes.
    """
    shocked_rates = np.asarray(shocked_rates, dtype=np.float64)
    tenors_years = np.asarray(tenors_years, dtype=np.float64)
    if shocked_rates.shape != tenors_years.shape:
        raise ValueError("shocked_rates and tenors_years must have the same shape")
    floors = np.minimum(
        (FLOOR_START_BPS + FLOOR_SLOPE_BPS_PER_YEAR * tenors_years) / 10_000.0, 0.0
    )
    return np.maximum(shocked_rates, floors)
