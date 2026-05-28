"""
CDS-implied probability of default.

Reference:
    - ISDA CDS Standard Model (credit triangle approximation).
    - Hull (2018) — Options, Futures and Other Derivatives, Ch. 25.
    - JPMorgan (2001) — "Par Credit Default Swap Spread Approximation".

Strips risk-neutral (Q-measure) default probabilities from CDS spreads
and provides an optional Q→P (real-world) conversion using a risk
premium adjustment. Used for sovereign and large-corporate PD where
market spreads are observable.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def cds_implied_hazard_rate(
    cds_spread_bps: float,
    recovery_rate: float = 0.40,
) -> float:
    """Strip the constant hazard rate from a CDS spread (credit triangle).

    Credit triangle approximation:
        hazard ≈ spread / (1 - recovery)

    where spread is expressed as a decimal.

    Args:
        cds_spread_bps: CDS par spread in basis points.
        recovery_rate: Assumed recovery rate (default 40%).

    Returns:
        Constant annual hazard rate (intensity lambda).

    Raises:
        ValueError: If recovery_rate is not in [0, 1).

    Reference:
        ISDA CDS Standard Model, Hull (2018).
    """
    if not 0.0 <= recovery_rate < 1.0:
        raise ValueError("recovery_rate must be in [0, 1)")
    spread = cds_spread_bps / 10_000.0
    return spread / (1.0 - recovery_rate)


def cds_implied_pd(
    cds_spread_bps: float,
    tenor_years: float,
    recovery_rate: float = 0.40,
) -> float:
    """Risk-neutral cumulative PD implied by a CDS spread.

    PD(T) = 1 - exp(-hazard * T)
          = 1 - exp(-(spread / (1 - R)) * T)

    Args:
        cds_spread_bps: CDS par spread in basis points.
        tenor_years: Tenor in years.
        recovery_rate: Assumed recovery rate (default 40%).

    Returns:
        Risk-neutral (Q-measure) cumulative PD over the tenor.

    Reference:
        ISDA CDS Standard Model.
    """
    hazard = cds_implied_hazard_rate(cds_spread_bps, recovery_rate)
    return float(1.0 - np.exp(-hazard * tenor_years))


def cds_pd_term_structure(
    cds_spreads_bps: dict[float, float],
    recovery_rate: float = 0.40,
) -> dict[float, float]:
    """Build a cumulative PD term structure from a CDS spread curve.

    Args:
        cds_spreads_bps: Mapping of tenor (years) → CDS spread (bps).
        recovery_rate: Assumed recovery rate.

    Returns:
        Mapping of tenor → risk-neutral cumulative PD.
    """
    return {
        tenor: cds_implied_pd(spread, tenor, recovery_rate)
        for tenor, spread in cds_spreads_bps.items()
    }


def risk_neutral_to_real_world(
    pd_q: float,
    sharpe_ratio: float = 0.40,
    asset_correlation: float = 0.20,
    horizon_years: float = 1.0,
) -> float:
    """Convert a risk-neutral PD to a real-world (P-measure) PD.

    Uses the standard one-factor adjustment (Vasicek / single-factor
    Merton) where the risk-neutral default threshold is shifted by the
    market risk premium:

        PD_P = N( N^{-1}(PD_Q) - sqrt(rho) * SR * sqrt(T) )

    The Q-measure PD embeds a risk premium that overstates the physical
    default probability; removing it yields the lower real-world PD.

    Args:
        pd_q: Risk-neutral (Q-measure) PD.
        sharpe_ratio: Market Sharpe ratio (risk premium per unit risk).
        asset_correlation: Asset correlation to the systematic factor.
        horizon_years: Horizon in years.

    Returns:
        Real-world (P-measure) PD.

    Raises:
        ValueError: If pd_q is not in (0, 1) or correlation invalid.

    Reference:
        Vasicek (2002), single-factor Merton risk-premium adjustment.
    """
    from scipy.stats import norm

    if not 0.0 < pd_q < 1.0:
        # Degenerate cases pass through
        return max(min(pd_q, 1.0), 0.0)
    if not 0.0 <= asset_correlation < 1.0:
        raise ValueError("asset_correlation must be in [0, 1)")

    shift = np.sqrt(asset_correlation) * sharpe_ratio * np.sqrt(horizon_years)
    pd_p = norm.cdf(norm.ppf(pd_q) - shift)
    return float(pd_p)
