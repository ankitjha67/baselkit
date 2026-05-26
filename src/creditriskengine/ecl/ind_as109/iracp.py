"""
IRACP standard-asset provisioning rates for Scheduled Commercial Banks.

Reference: Master Circular DOR.STR.REC.9/21.04.048/2025-26 dated
April 1, 2025 and Reserve Bank of India (Project Finance) Directions,
2025 (effective October 1, 2025).

In force through March 31, 2027 (repealed by the RBI ECL Directions 2026
on commencement April 1, 2027).
"""

from __future__ import annotations

import logging
from enum import StrEnum

logger = logging.getLogger(__name__)


class StandardAssetSector(StrEnum):
    """Standard-asset sector for IRACP provisioning.

    Reference: DOR.STR.REC.9/21.04.048/2025-26, paragraph 5.
    """

    AGRICULTURE_DIRECT = "agriculture_direct"
    """Direct agriculture / farm credit: 0.25%."""

    SME_DIRECT = "sme_direct"
    """Direct SME (small and micro enterprises): 0.25%."""

    HOUSING_INDIVIDUAL = "housing_individual"
    """Housing loans to individuals: 0.25%."""

    HOUSING_TEASER = "housing_teaser"
    """Housing loans at teaser rates: 2.00% (reverts to 0.40% one year
    after rate reset per DOR.STR.REC.9/21.04.048/2025-26)."""

    CRE = "cre"
    """Commercial Real Estate: 1.00%."""

    CRE_RH = "cre_rh"
    """CRE — Residential Housing: 0.75%."""

    PROJECT_UNDER_CONSTRUCTION = "project_under_construction"
    """Project finance under construction (non-CRE): 1.00%.
    Reference: Project Finance Directions 2025."""

    PROJECT_UNDER_CONSTRUCTION_CRE = "project_under_construction_cre"
    """Project finance under construction (CRE): 1.25%.
    Reference: Project Finance Directions 2025."""

    PROJECT_OPERATIONAL_CRE = "project_operational_cre"
    """Project finance operational (CRE): 1.00%."""

    PROJECT_OPERATIONAL_CRE_RH = "project_operational_cre_rh"
    """Project finance operational (CRE-RH): 0.75%."""

    PROJECT_OPERATIONAL_OTHER = "project_operational_other"
    """Project finance operational (other): 0.40%."""

    OTHER = "other"
    """All other standard assets: 0.40%."""


IRACP_STANDARD_RATES: dict[StandardAssetSector, float] = {
    StandardAssetSector.AGRICULTURE_DIRECT: 0.0025,
    StandardAssetSector.SME_DIRECT: 0.0025,
    StandardAssetSector.HOUSING_INDIVIDUAL: 0.0025,
    StandardAssetSector.HOUSING_TEASER: 0.0200,
    StandardAssetSector.CRE: 0.0100,
    StandardAssetSector.CRE_RH: 0.0075,
    StandardAssetSector.PROJECT_UNDER_CONSTRUCTION: 0.0100,
    StandardAssetSector.PROJECT_UNDER_CONSTRUCTION_CRE: 0.0125,
    StandardAssetSector.PROJECT_OPERATIONAL_CRE: 0.0100,
    StandardAssetSector.PROJECT_OPERATIONAL_CRE_RH: 0.0075,
    StandardAssetSector.PROJECT_OPERATIONAL_OTHER: 0.0040,
    StandardAssetSector.OTHER: 0.0040,
}
"""Standard-asset provisioning rates per DOR.STR.REC.9/21.04.048/2025-26
and Project Finance Directions 2025."""


def standard_asset_provision(
    funded_outstanding: float,
    sector: StandardAssetSector = StandardAssetSector.OTHER,
    teaser_one_year_post_reset: bool = False,
) -> float:
    """Calculate IRACP standard-asset provision.

    Reference: DOR.STR.REC.9/21.04.048/2025-26, paragraph 5.

    Args:
        funded_outstanding: Funded outstanding amount.
        sector: Standard-asset sector determining the provisioning rate.
        teaser_one_year_post_reset: For teaser-rate housing, whether
            one year has elapsed since rate reset (reverts to 0.40%).

    Returns:
        Provision amount.
    """
    if funded_outstanding <= 0:
        return 0.0
    if sector == StandardAssetSector.HOUSING_TEASER and teaser_one_year_post_reset:
        return funded_outstanding * 0.0040
    return funded_outstanding * IRACP_STANDARD_RATES[sector]


# ---------------------------------------------------------------------------
# Resolution Framework add-ons (RF 1.0 / 2.0)
# ---------------------------------------------------------------------------

RF_RESTRUCTURED_ADDON_RATE: float = 0.10
"""10% additional provision on residual debt upon restructuring.

Reference: DOR.No.BP.BC/3/21.04.048/2020-21 (RF 1.0, August 6, 2020)
and DOR.STR.REC.12/21.04.048/2021-22 (RF 2.0, May 5, 2021).
"""

RF_SLIPPAGE_ADDON_RATE: float = 0.05
"""5% additional provision if restructured account slips.

Reference: DOR.STR.REC.12/21.04.048/2021-22 (RF 2.0).
"""


def resolution_framework_addon(
    residual_debt: float,
    has_slipped: bool = False,
) -> float:
    """Calculate Resolution Framework 1.0/2.0 provisioning add-on.

    Upon implementation of a restructuring plan, lending institutions
    must keep a provision of 10% of the borrower's residual debt.
    If the account subsequently deteriorates, an additional 5% applies.

    Reference:
        DOR.No.BP.BC/3/21.04.048/2020-21 (August 6, 2020) — RF 1.0.
        DOR.STR.REC.12/21.04.048/2021-22 (May 5, 2021) — RF 2.0.

    Args:
        residual_debt: Residual debt of the borrower after restructuring.
        has_slipped: Whether the account has subsequently deteriorated.

    Returns:
        Add-on provision amount.
    """
    if residual_debt <= 0:
        return 0.0
    addon = residual_debt * RF_RESTRUCTURED_ADDON_RATE
    if has_slipped:
        addon += residual_debt * RF_SLIPPAGE_ADDON_RATE
    return addon


# ---------------------------------------------------------------------------
# Out-of-order CC/OD classification
# ---------------------------------------------------------------------------


def is_out_of_order(
    outstanding: float,
    sanctioned_limit: float,
    drawing_power: float | None = None,
    days_continuously_over_limit: int = 0,
    no_credits_for_days: int = 0,
    credits_less_than_interest_debited: bool = False,
) -> bool:
    """Determine if a CC/OD account is 'out of order' per RBI norms.

    Per DOR.STR.REC.68/21.04.048/2021-22 (November 12, 2021), a
    CC/OD account is out of order if any of three conditions hold:

    1. Outstanding balance remains continuously in excess of the
       sanctioned limit or drawing power (whichever is lower) for
       more than 90 days.
    2. There are no credits continuously for 90 days as on the date
       of classification.
    3. Credits in the account are not enough to cover the interest
       debited during the same period.

    Reference:
        DOR.STR.REC.68/21.04.048/2021-22, November 12, 2021.
        DOR.STR.REC.85/21.04.048/2021-22, February 15, 2022.

    Args:
        outstanding: Current outstanding balance.
        sanctioned_limit: Sanctioned credit limit.
        drawing_power: Drawing power (if lower than sanctioned limit).
        days_continuously_over_limit: Days the outstanding has
            continuously exceeded the limit/DP.
        no_credits_for_days: Consecutive days with zero credits.
        credits_less_than_interest_debited: Whether total credits
            during the period are less than interest debited.

    Returns:
        ``True`` if the account is out of order (NPA trigger).
    """
    effective_limit = min(sanctioned_limit, drawing_power or sanctioned_limit)
    if outstanding > effective_limit and days_continuously_over_limit > 90:
        return True
    if no_credits_for_days > 90:
        return True
    return credits_less_than_interest_debited
