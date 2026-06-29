"""Equity investments in funds (BCBS CRE60).

Reference:
    - BCBS CRE60 "Equity investments in funds" (consolidated framework),
      based on the December 2013 standard.

A bank's equity investment in a fund is risk-weighted using a hierarchy of
three approaches:

1. **Look-Through Approach (LTA)** — risk-weight the underlying exposures
   of the fund as if held directly. The average risk weight of the fund is
   scaled by the fund's leverage and applied to the investment.
2. **Mandate-Based Approach (MBA)** — where the LTA cannot be used, infer
   the maximum risk using the fund's mandate / prospectus limits.
3. **Fall-Back Approach (FBA)** — where neither applies, a 1250% risk
   weight on the investment.

Under the LTA and MBA, the product of the average risk weight and the
fund's leverage is capped at 1250% (CRE60.7).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

logger = logging.getLogger(__name__)

# 1250% expressed as a risk weight multiple.
_MAX_RISK_WEIGHT: float = 12.50


class FundApproach(StrEnum):
    """CRE60 risk-weighting approaches for fund investments."""

    LOOK_THROUGH = "look_through"
    MANDATE_BASED = "mandate_based"
    FALL_BACK = "fall_back"


def fund_average_risk_weight(underlying_rwa: float, fund_total_assets: float) -> float:
    """Average risk weight of a fund (CRE60.7).

        avg_RW = total RWA of underlying exposures / total assets of fund

    Args:
        underlying_rwa: Total risk-weighted assets of the fund's
            underlying exposures.
        fund_total_assets: Total assets of the fund.

    Returns:
        Average risk weight as a decimal multiple (e.g. 0.75 = 75%).

    Raises:
        ValueError: If ``fund_total_assets`` is not positive or
            ``underlying_rwa`` is negative.
    """
    if fund_total_assets <= 0.0:
        raise ValueError("fund_total_assets must be positive")
    if underlying_rwa < 0.0:
        raise ValueError("underlying_rwa must be non-negative")
    return underlying_rwa / fund_total_assets


def fund_leverage(total_assets: float, total_equity: float) -> float:
    """Financial leverage of a fund (CRE60.7).

        leverage = total assets / total equity

    Args:
        total_assets: Total assets of the fund.
        total_equity: Total equity of the fund.

    Returns:
        Leverage ratio (>= 1 for a typical fund).

    Raises:
        ValueError: If ``total_equity`` is not positive or assets are
            negative.
    """
    if total_equity <= 0.0:
        raise ValueError("total_equity must be positive")
    if total_assets < 0.0:
        raise ValueError("total_assets must be non-negative")
    return total_assets / total_equity


@dataclass(frozen=True)
class FundRWAResult:
    """RWA result for an equity investment in a fund.

    Attributes:
        approach: The approach applied.
        investment: The bank's equity investment amount.
        effective_risk_weight: Applied risk weight (avg_RW x leverage,
            capped at 1250%), as a decimal multiple.
        rwa: Resulting risk-weighted assets.
        capped: True if the 1250% cap was binding.
    """

    approach: FundApproach
    investment: float
    effective_risk_weight: float
    rwa: float
    capped: bool


def _apply(
    approach: FundApproach,
    investment: float,
    average_risk_weight: float,
    leverage: float,
) -> FundRWAResult:
    raw_rw = average_risk_weight * leverage
    effective_rw = min(raw_rw, _MAX_RISK_WEIGHT)
    return FundRWAResult(
        approach=approach,
        investment=round(investment, 6),
        effective_risk_weight=round(effective_rw, 6),
        rwa=round(effective_rw * investment, 6),
        capped=raw_rw > _MAX_RISK_WEIGHT,
    )


def look_through_rwa(
    investment: float,
    average_risk_weight: float,
    leverage: float,
) -> FundRWAResult:
    """RWA under the Look-Through Approach (CRE60.5-60.7).

    Args:
        investment: The bank's equity investment amount.
        average_risk_weight: Average risk weight of the fund (see
            :func:`fund_average_risk_weight`).
        leverage: Fund leverage (see :func:`fund_leverage`).

    Returns:
        A :class:`FundRWAResult`.

    Raises:
        ValueError: If any input is negative.
    """
    if min(investment, average_risk_weight, leverage) < 0.0:
        raise ValueError("inputs must be non-negative")
    return _apply(FundApproach.LOOK_THROUGH, investment, average_risk_weight, leverage)


def mandate_based_rwa(
    investment: float,
    average_risk_weight: float,
    leverage: float,
) -> FundRWAResult:
    """RWA under the Mandate-Based Approach (CRE60.8-60.10).

    Uses the maximum risk weight implied by the fund's mandate; the
    mechanics mirror the LTA but with mandate-derived inputs.

    Args:
        investment: The bank's equity investment amount.
        average_risk_weight: Average risk weight implied by the mandate.
        leverage: Maximum leverage permitted by the mandate.

    Returns:
        A :class:`FundRWAResult`.

    Raises:
        ValueError: If any input is negative.
    """
    if min(investment, average_risk_weight, leverage) < 0.0:
        raise ValueError("inputs must be non-negative")
    return _apply(FundApproach.MANDATE_BASED, investment, average_risk_weight, leverage)


def fall_back_rwa(investment: float) -> FundRWAResult:
    """RWA under the Fall-Back Approach — 1250% risk weight (CRE60.11).

    Args:
        investment: The bank's equity investment amount.

    Returns:
        A :class:`FundRWAResult` with a 1250% risk weight.

    Raises:
        ValueError: If ``investment`` is negative.
    """
    if investment < 0.0:
        raise ValueError("investment must be non-negative")
    return FundRWAResult(
        approach=FundApproach.FALL_BACK,
        investment=round(investment, 6),
        effective_risk_weight=_MAX_RISK_WEIGHT,
        rwa=round(_MAX_RISK_WEIGHT * investment, 6),
        capped=True,
    )
