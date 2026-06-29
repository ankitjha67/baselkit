"""Minimum haircut floors for securities financing transactions (CRE56).

Reference:
    - BCBS CRE56 "Minimum haircut floors for securities financing
      transactions" (consolidated framework).
    - FSB "Regulatory framework for haircuts on non-centrally cleared
      securities financing transactions" (2015).

In-scope SFTs (lending against non-government collateral to non-banks /
non-supervised entities) are subject to minimum collateral haircut floors.
If the actual collateral haircut is below the floor, the transaction is
treated as uncollateralised for the lender — no credit-risk-mitigation
benefit is recognised.

Haircut floors (CRE56.2), by collateral type and residual maturity:

    Corporate / other debt:  <=1y 0.5%, <=5y 1.5%, <=10y 3%, >10y 4%
    Securitisations:         <=1y 1%,   <=5y 4%,   <=10y 6%, >10y 7%
    Main-index equities:     6% (maturity-independent)
    Other in-scope assets:   10%
    Cash and government debt: out of scope (0%).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

logger = logging.getLogger(__name__)


class SFTCollateralType(StrEnum):
    """Collateral types for the CRE56 haircut-floor table."""

    CASH = "cash"  # out of scope -> 0
    SOVEREIGN_DEBT = "sovereign_debt"  # out of scope -> 0
    CORPORATE_DEBT = "corporate_debt"
    SECURITISATION = "securitisation"
    MAIN_INDEX_EQUITY = "main_index_equity"
    OTHER_ASSET = "other_asset"


# Maturity-bucketed floors for debt (inclusive upper bound in years, floor).
_CORPORATE_DEBT_FLOORS: tuple[tuple[float, float], ...] = (
    (1.0, 0.005),
    (5.0, 0.015),
    (10.0, 0.03),
)
_CORPORATE_DEBT_FLOOR_LONG: float = 0.04  # > 10 years

_SECURITISATION_FLOORS: tuple[tuple[float, float], ...] = (
    (1.0, 0.01),
    (5.0, 0.04),
    (10.0, 0.06),
)
_SECURITISATION_FLOOR_LONG: float = 0.07  # > 10 years

_MAIN_INDEX_EQUITY_FLOOR: float = 0.06
_OTHER_ASSET_FLOOR: float = 0.10


def _bucketed_floor(
    buckets: tuple[tuple[float, float], ...],
    long_floor: float,
    residual_maturity_years: float,
) -> float:
    for upper, floor in buckets:
        if residual_maturity_years <= upper:
            return floor
    return long_floor


def minimum_haircut_floor(
    collateral_type: SFTCollateralType,
    residual_maturity_years: float = 0.0,
) -> float:
    """Minimum collateral haircut floor for an SFT (CRE56.2).

    Args:
        collateral_type: Type of collateral.
        residual_maturity_years: Residual maturity of debt collateral in
            years (ignored for equities/other/cash). Must be non-negative.

    Returns:
        The haircut floor as a decimal (e.g. 0.015 = 1.5%). Cash and
        government debt are out of scope and return 0.

    Raises:
        ValueError: If ``residual_maturity_years`` is negative.
    """
    if residual_maturity_years < 0.0:
        raise ValueError("residual_maturity_years must be non-negative")

    if collateral_type in (SFTCollateralType.CASH, SFTCollateralType.SOVEREIGN_DEBT):
        return 0.0
    if collateral_type == SFTCollateralType.CORPORATE_DEBT:
        return _bucketed_floor(
            _CORPORATE_DEBT_FLOORS, _CORPORATE_DEBT_FLOOR_LONG, residual_maturity_years
        )
    if collateral_type == SFTCollateralType.SECURITISATION:
        return _bucketed_floor(
            _SECURITISATION_FLOORS, _SECURITISATION_FLOOR_LONG, residual_maturity_years
        )
    if collateral_type == SFTCollateralType.MAIN_INDEX_EQUITY:
        return _MAIN_INDEX_EQUITY_FLOOR
    return _OTHER_ASSET_FLOOR


def sft_haircut(exposure: float, collateral_value: float) -> float:
    """Collateral haircut (over-collateralisation) of an SFT.

        haircut = collateral_value / exposure - 1

    Args:
        exposure: Amount lent (the exposure the collateral secures).
        collateral_value: Market value of collateral received.

    Returns:
        The haircut as a decimal (e.g. 0.02 = 2% over-collateralisation).

    Raises:
        ValueError: If ``exposure`` is not positive or ``collateral_value``
            is negative.
    """
    if exposure <= 0.0:
        raise ValueError("exposure must be positive")
    if collateral_value < 0.0:
        raise ValueError("collateral_value must be non-negative")
    return collateral_value / exposure - 1.0


@dataclass(frozen=True)
class SFTFloorResult:
    """Single-SFT haircut-floor compliance result.

    Attributes:
        haircut: Actual collateral haircut.
        floor: Applicable minimum haircut floor.
        meets_floor: True if the haircut is at or above the floor.
        collateral_recognised: True if CRM is recognised (i.e. the floor is
            met); False means the SFT is treated as uncollateralised.
    """

    haircut: float
    floor: float
    meets_floor: bool
    collateral_recognised: bool


def assess_sft_floor(
    exposure: float,
    collateral_value: float,
    collateral_type: SFTCollateralType,
    residual_maturity_years: float = 0.0,
) -> SFTFloorResult:
    """Assess a single SFT against its minimum haircut floor (CRE56.4).

    Args:
        exposure: Amount lent.
        collateral_value: Market value of collateral received.
        collateral_type: Collateral type.
        residual_maturity_years: Residual maturity of debt collateral.

    Returns:
        An :class:`SFTFloorResult`. When the floor is not met the
        collateral is not recognised and the SFT is treated as unsecured.

    Raises:
        ValueError: On invalid exposure / collateral inputs.
    """
    haircut = sft_haircut(exposure, collateral_value)
    floor = minimum_haircut_floor(collateral_type, residual_maturity_years)
    meets = haircut >= floor
    return SFTFloorResult(
        haircut=round(haircut, 6),
        floor=floor,
        meets_floor=meets,
        collateral_recognised=meets,
    )


@dataclass(frozen=True)
class SFTLeg:
    """A single SFT within a netting agreement for the portfolio floor test."""

    exposure: float
    collateral_value: float
    collateral_type: SFTCollateralType
    residual_maturity_years: float = 0.0


def portfolio_floor_compliant(legs: Sequence[SFTLeg]) -> bool:
    """Netting-agreement haircut-floor test (CRE56.5).

    A portfolio of SFTs under a netting agreement meets the floor when its
    actual portfolio haircut is at least the exposure-weighted average of
    the individual floors::

        portfolio_haircut = (sum C - sum E) / sum E
        portfolio_floor   = sum(E_i * f_i) / sum E
        compliant         = portfolio_haircut >= portfolio_floor

    Args:
        legs: SFTs under the netting agreement.

    Returns:
        True if the portfolio meets the floor.

    Raises:
        ValueError: If ``legs`` is empty or total exposure is not positive.
    """
    if not legs:
        raise ValueError("at least one SFT leg is required")
    total_exposure = sum(leg.exposure for leg in legs)
    if total_exposure <= 0.0:
        raise ValueError("total exposure must be positive")

    total_collateral = sum(leg.collateral_value for leg in legs)
    portfolio_haircut = (total_collateral - total_exposure) / total_exposure
    weighted_floor = (
        sum(
            leg.exposure
            * minimum_haircut_floor(leg.collateral_type, leg.residual_maturity_years)
            for leg in legs
        )
        / total_exposure
    )
    return portfolio_haircut >= weighted_floor
