"""
FRTB Internal Models Approach (IMA).

Reference:
    - BCBS d457 (Minimum capital requirements for market risk, Jan 2019).
    - CRR3 (EU implementation, effective January 1, 2026).
    - PRA PS9/24 (UK FRTB).

Implements the IMA capital components:
    - Expected Shortfall (ES) at 97.5% confidence
    - Liquidity-horizon scaling (10/20/40/60/120 days)
    - Stressed ES calibration
    - P&L Attribution Test (PLAT): Spearman correlation + KS test
    - Internal Default Risk Charge (DRC, 99.9% one-year)
    - Non-Modellable Risk Factor (NMRF) Stress Scenario charge
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


ES_CONFIDENCE_LEVEL: Final[float] = 0.975
"""Expected Shortfall confidence level under FRTB IMA (MAR33.4)."""

BASE_LIQUIDITY_HORIZON_DAYS: Final[int] = 10
"""Base liquidity horizon for ES (MAR33.12)."""

# Liquidity horizons by risk-factor category (MAR33.12, Table).
LIQUIDITY_HORIZONS: Final[dict[str, int]] = {
    "interest_rate": 10,
    "equity_price_large_cap": 10,
    "fx": 10,
    "equity_price_small_cap": 20,
    "credit_spread_ig": 20,
    "commodity_energy": 20,
    "credit_spread_hy": 40,
    "equity_volatility": 40,
    "commodity_other": 60,
    "credit_spread_structured": 120,
    "credit_spread_other": 120,
}
"""Regulatory liquidity horizons (days) per risk-factor category."""


class PLATZone(StrEnum):
    """P&L Attribution Test traffic-light zones (MAR33.45)."""

    GREEN = "green"
    """Test passed — IMA permitted."""

    AMBER = "amber"
    """Capital surcharge applies; monitoring required."""

    RED = "red"
    """Test failed — desk reverts to Standardised Approach."""


def expected_shortfall(
    pnl: np.ndarray,
    confidence_level: float = ES_CONFIDENCE_LEVEL,
) -> float:
    """Expected Shortfall (ES) of a P&L distribution.

    ES = -E[ P&L | P&L <= VaR ]   (loss is the negative tail)

    Args:
        pnl: Array of profit-and-loss outcomes (losses negative).
        confidence_level: ES confidence level (default 97.5%).

    Returns:
        Expected Shortfall (positive number = expected tail loss).

    Raises:
        ValueError: If confidence_level not in (0, 1).

    Reference:
        BCBS d457 MAR33.4.
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    pnl = np.asarray(pnl, dtype=np.float64)
    # VaR at the loss tail: the (1 - cl) quantile of P&L
    var_quantile = np.quantile(pnl, 1.0 - confidence_level)
    tail = pnl[pnl <= var_quantile]
    if len(tail) == 0:
        return float(-var_quantile)
    return float(-np.mean(tail))


def liquidity_scaled_es(
    base_es: float,
    liquidity_horizon_days: int,
    base_horizon_days: int = BASE_LIQUIDITY_HORIZON_DAYS,
) -> float:
    """Scale a base-horizon ES to a longer liquidity horizon.

    ES(LH) = ES(base) * sqrt(LH / base)

    per the square-root-of-time scaling in MAR33.13.

    Args:
        base_es: ES computed at the base (10-day) horizon.
        liquidity_horizon_days: Target liquidity horizon (days).
        base_horizon_days: Base horizon (default 10).

    Returns:
        Liquidity-horizon-scaled ES.

    Reference:
        BCBS d457 MAR33.13.
    """
    if base_horizon_days <= 0 or liquidity_horizon_days <= 0:
        raise ValueError("horizons must be positive")
    return base_es * np.sqrt(liquidity_horizon_days / base_horizon_days)


def internal_model_capital_charge(
    es_current: float,
    es_stressed: float,
    multiplier: float = 1.5,
) -> float:
    """IMA capital charge combining current and stressed ES.

    IMCC = multiplier * max(ES_stressed, current contribution)

    A common form (MAR33.5-33.9) uses the stressed ES scaled by the
    ratio of full to reduced-set ES; here we apply the regulatory
    multiplier (>= 1.5) to the stressed ES as the binding term.

    Args:
        es_current: Current-period ES.
        es_stressed: Stressed-period ES.
        multiplier: Regulatory multiplier (>= 1.5; raised by backtesting
            exceptions).

    Returns:
        Internally-modelled capital charge.

    Reference:
        BCBS d457 MAR33.5-33.9.
    """
    if multiplier < 1.5:
        raise ValueError("multiplier must be >= 1.5")
    return multiplier * max(es_stressed, es_current)


def plat_test(
    hypothetical_pnl: np.ndarray,
    risk_theoretical_pnl: np.ndarray,
) -> tuple[PLATZone, float, float]:
    """P&L Attribution Test — Spearman correlation + KS test.

    Compares the desk's hypothetical P&L (front-office) with the
    risk-theoretical P&L (risk-engine). Two metrics:
        - Spearman rank correlation (higher = better alignment)
        - Kolmogorov-Smirnov statistic (lower = better alignment)

    Traffic light (MAR33.45):
        GREEN: Spearman >= 0.80 AND KS <= 0.09
        RED:   Spearman <  0.70 OR  KS >  0.12
        AMBER: otherwise

    Args:
        hypothetical_pnl: Front-office hypothetical P&L series.
        risk_theoretical_pnl: Risk-engine theoretical P&L series.

    Returns:
        Tuple of (zone, spearman_correlation, ks_statistic).

    Reference:
        BCBS d457 MAR33.45.
    """
    hypothetical_pnl = np.asarray(hypothetical_pnl, dtype=np.float64)
    risk_theoretical_pnl = np.asarray(risk_theoretical_pnl, dtype=np.float64)

    spearman = float(stats.spearmanr(hypothetical_pnl, risk_theoretical_pnl).statistic)
    if np.isnan(spearman):
        spearman = 0.0
    ks = float(stats.ks_2samp(hypothetical_pnl, risk_theoretical_pnl).statistic)

    if spearman >= 0.80 and ks <= 0.09:
        zone = PLATZone.GREEN
    elif spearman < 0.70 or ks > 0.12:
        zone = PLATZone.RED
    else:
        zone = PLATZone.AMBER

    return zone, spearman, ks


# MAR22.24 default risk weights by credit-quality grade.
DRC_DEFAULT_RISK_WEIGHTS: Final[dict[str, float]] = {
    "AAA": 0.005,
    "AA": 0.02,
    "A": 0.03,
    "BBB": 0.06,
    "BB": 0.15,
    "B": 0.30,
    "CCC": 0.50,
    "DEFAULT": 1.00,
    "UNRATED": 0.15,
}


def drc_default_risk_weight(rating: str) -> float:
    """Default risk weight for a credit-quality grade (MAR22.24)."""
    return DRC_DEFAULT_RISK_WEIGHTS.get(rating.upper(), DRC_DEFAULT_RISK_WEIGHTS["UNRATED"])


@dataclass(frozen=True)
class DRCPosition:
    """A default-risk position for the DRC.

    Attributes:
        obligor: Issuer/obligor identifier (JTD nets within an obligor).
        jtd: Signed jump-to-default loss — positive for long (loses on
            the obligor's default), negative for short (gains on default).
        risk_weight: Default risk weight for the obligor (MAR22.24).
        bucket: DRC bucket — "corporates", "sovereigns" or "local_gov".
            JTD offsetting and the hedge-benefit ratio apply within a
            bucket; there is no cross-bucket diversification (MAR22.30-31).
    """

    obligor: str
    jtd: float
    risk_weight: float
    bucket: str = "corporates"


def default_risk_charge(positions: Sequence[DRCPosition]) -> float:
    """Full Default Risk Charge per MAR22.18-22.33.

    Implements the standardised DRC methodology (also used by the IMA at
    99.9 % one-year, MAR33):

    1. Net the signed JTD of positions sharing an obligor.
    2. Within each bucket, split obligors into net-long and net-short.
    3. Apply a book-wide hedge-benefit ratio (WtS) recognising partial
       offset of short hedges (MAR22.30)::

           WtS = sum(net long JTD) / (sum(net long JTD) + sum(net short JTD))

    4. Per bucket (MAR22.31)::

           DRC_b = max(sum(RW * net long JTD) - WtS * sum(RW * net short JTD), 0)

    5. Total DRC is the simple sum across buckets (no cross-bucket
       diversification).

    Args:
        positions: Default-risk positions.

    Returns:
        Total Default Risk Charge (non-negative).
    """
    if not positions:
        return 0.0

    # 1. Net JTD per (bucket, obligor); keep the obligor's risk weight.
    net_jtd: dict[tuple[str, str], float] = {}
    obligor_rw: dict[tuple[str, str], float] = {}
    for pos in positions:
        key = (pos.bucket, pos.obligor)
        net_jtd[key] = net_jtd.get(key, 0.0) + pos.jtd
        obligor_rw[key] = pos.risk_weight

    # 2-3. Book-wide WtS from net long/short JTD (notional basis).
    gross_long = sum(j for j in net_jtd.values() if j > 0.0)
    gross_short = -sum(j for j in net_jtd.values() if j < 0.0)
    if gross_long + gross_short == 0.0:
        return 0.0
    wts = gross_long / (gross_long + gross_short)

    # 4. Per-bucket DRC with the WtS-scaled short offset.
    bucket_long: dict[str, float] = {}
    bucket_short: dict[str, float] = {}
    for (bucket, _obligor), jtd in net_jtd.items():
        rw = obligor_rw[(bucket, _obligor)]
        if jtd > 0.0:
            bucket_long[bucket] = bucket_long.get(bucket, 0.0) + rw * jtd
        else:
            bucket_short[bucket] = bucket_short.get(bucket, 0.0) + rw * (-jtd)

    total = 0.0
    for bucket in set(bucket_long) | set(bucket_short):
        long_rw = bucket_long.get(bucket, 0.0)
        short_rw = bucket_short.get(bucket, 0.0)
        total += max(long_rw - wts * short_rw, 0.0)
    return total


def default_risk_charge_ima(
    jtd_long: np.ndarray,
    jtd_short: np.ndarray,
) -> float:
    """Internal DRC via net jump-to-default at 99.9% one-year.

    Convenience wrapper over :func:`default_risk_charge` for a single
    bucket of unit-risk-weight positions, where each array element is a
    distinct obligor. Long elements are positive JTD, short elements are
    positive magnitudes converted to negative JTD.

    Args:
        jtd_long: Jump-to-default amounts for long positions.
        jtd_short: Jump-to-default amounts for short positions
            (as positive magnitudes).

    Returns:
        Default Risk Charge (non-negative).

    Reference:
        BCBS d457 MAR22 (DRC), MAR33 (IMA DRC at 99.9%).
    """
    positions: list[DRCPosition] = []
    for i, jtd in enumerate(np.atleast_1d(jtd_long)):
        positions.append(DRCPosition(f"L{i}", float(jtd), 1.0))
    for i, jtd in enumerate(np.atleast_1d(jtd_short)):
        positions.append(DRCPosition(f"S{i}", -float(jtd), 1.0))
    return default_risk_charge(positions)


@dataclass(frozen=True)
class NMRFCharge:
    """Non-Modellable Risk Factor stress-scenario charge.

    Attributes:
        idiosyncratic_charge: Aggregated idiosyncratic NMRF charge.
        non_idiosyncratic_charge: Aggregated non-idiosyncratic charge.
        total: Total NMRF capital charge.
    """

    idiosyncratic_charge: float
    non_idiosyncratic_charge: float
    total: float


def nmrf_stress_charge(
    idiosyncratic_stress_losses: np.ndarray,
    non_idiosyncratic_stress_losses: np.ndarray,
    rho: float = 0.6,
) -> NMRFCharge:
    """Aggregate Non-Modellable Risk Factor stress charges.

    Idiosyncratic NMRFs aggregate by simple sum-of-squares (zero
    correlation); non-idiosyncratic NMRFs aggregate with correlation
    rho. Total combines the two:
        SES = sqrt( ISES^2 + NISES^2 )
    where NISES uses rho-correlated aggregation.

    Args:
        idiosyncratic_stress_losses: Per-NMRF idiosyncratic stress losses.
        non_idiosyncratic_stress_losses: Per-NMRF non-idiosyncratic
            stress losses.
        rho: Correlation for non-idiosyncratic aggregation (MAR33.16).

    Returns:
        :class:`NMRFCharge`.

    Reference:
        BCBS d457 MAR33.16-33.18.
    """
    idio = np.asarray(idiosyncratic_stress_losses, dtype=np.float64)
    non_idio = np.asarray(non_idiosyncratic_stress_losses, dtype=np.float64)

    # Idiosyncratic: zero correlation → sqrt of sum of squares
    ises = float(np.sqrt(np.sum(idio**2)))

    # Non-idiosyncratic: rho-correlated aggregation
    sum_sq = float(np.sum(non_idio**2))
    sum_total = float(np.sum(non_idio))
    nises = float(np.sqrt(rho * sum_total**2 + (1.0 - rho) * sum_sq))

    total = float(np.sqrt(ises**2 + nises**2))
    return NMRFCharge(
        idiosyncratic_charge=ises,
        non_idiosyncratic_charge=nises,
        total=total,
    )
