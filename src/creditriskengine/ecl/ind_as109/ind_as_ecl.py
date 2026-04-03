"""
Ind AS 109 ECL calculation — Indian Accounting Standard 109.

Ind AS 109 is fully converged with IFRS 9 for impairment, but RBI
overlays additional requirements through IRAC norms and Master
Directions on Provisioning.

Key India-specific elements:
    - Default definition: 90 DPD for ALL loan categories (RBI IRAC norms).
    - Agricultural loans: special treatment (1/2 crop seasons).
    - NPA classification: Standard → SMA-0/1/2 → Sub-Standard → Doubtful →
      Loss per RBI Master Circular DBOD.No.BP.BC.94/21.06.001.
    - Minimum provisioning percentages per asset classification.
    - Restructured account treatment per RBI/2019-20/170.
    - Ind AS provisioning floors per RBI Draft Directions Oct 2025.

References:
    - RBI Master Circular on IRAC Norms (DBOD.No.BP.BC.94/21.06.001)
    - RBI Circular on Prudential Norms on Income Recognition, Asset
      Classification and Provisioning (IRACP)
    - RBI Master Direction — Classification, Valuation and Operation
      of Investment Portfolio (DoR.MRG.46/21.04.141/2023-24)
    - RBI/2019-20/170 — Ind AS Provisioning
    - RBI Draft Directions October 2025 — Provision floors
    - Ind AS 109 (= IFRS 9) Financial Instruments
"""

from __future__ import annotations

import logging
from enum import StrEnum

import numpy as np

from creditriskengine.core.types import IFRS9Stage
from creditriskengine.ecl.ifrs9.ecl_calc import calculate_ecl
from creditriskengine.ecl.ifrs9.staging import assign_stage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RBI Constants
# ---------------------------------------------------------------------------

RBI_DEFAULT_DPD_THRESHOLD: int = 90
"""RBI default threshold: 90 DPD for all categories (IRAC para 2.1.2)."""

RBI_SICR_DPD_BACKSTOP: int = 30
"""RBI SICR backstop: 30 DPD (aligned with IFRS 9)."""

RBI_AGRI_SHORT_CROP_DPD: int = 60
"""Agricultural short-duration crops: 2 crop seasons (approx. 60 days)."""

RBI_AGRI_LONG_CROP_DPD: int = 90
"""Agricultural long-duration crops: 1 crop season (approx. 90 days)."""


# ---------------------------------------------------------------------------
# RBI IRAC Asset Classification (IRAC para 2.1)
# ---------------------------------------------------------------------------


class IRACAssetClass(StrEnum):
    """RBI Income Recognition and Asset Classification categories.

    Reference: RBI Master Circular IRAC Norms para 2.1-2.5.
    """

    STANDARD = "standard"
    """Performing, not overdue beyond threshold."""

    SMA_0 = "sma_0"
    """Special Mention Account: 1-30 DPD."""

    SMA_1 = "sma_1"
    """Special Mention Account: 31-60 DPD."""

    SMA_2 = "sma_2"
    """Special Mention Account: 61-90 DPD."""

    SUBSTANDARD = "substandard"
    """NPA for ≤ 12 months (IRAC para 2.2)."""

    DOUBTFUL_1 = "doubtful_1"
    """NPA for 12-24 months (IRAC para 2.3.1)."""

    DOUBTFUL_2 = "doubtful_2"
    """NPA for 24-36 months (IRAC para 2.3.2)."""

    DOUBTFUL_3 = "doubtful_3"
    """NPA for > 36 months (IRAC para 2.3.3)."""

    LOSS = "loss"
    """Identified as loss by bank or auditor (IRAC para 2.4)."""


# ---------------------------------------------------------------------------
# RBI Minimum Provisioning Percentages (IRAC para 4)
# ---------------------------------------------------------------------------

RBI_PROVISION_RATES: dict[str, float] = {
    "standard_commercial": 0.0040,     # 0.40% (commercial credit)
    "standard_cre": 0.0100,            # 1.00% (CRE)
    "standard_cre_rre": 0.0075,        # 0.75% (CRE - residential housing)
    "standard_sme": 0.0025,            # 0.25% (SME)
    "standard_agri": 0.0025,           # 0.25% (agriculture / micro)
    "standard_other": 0.0040,          # 0.40% (other)
    "substandard_secured": 0.15,       # 15% (secured substandard)
    "substandard_unsecured": 0.25,     # 25% (unsecured substandard)
    "doubtful_1_secured": 0.25,        # 25% of secured + 100% unsecured
    "doubtful_1_unsecured": 1.00,
    "doubtful_2_secured": 0.40,        # 40% of secured + 100% unsecured
    "doubtful_2_unsecured": 1.00,
    "doubtful_3": 1.00,                # 100%
    "loss": 1.00,                      # 100%
}
"""Minimum provisioning rates per RBI IRAC Norms para 4.2-4.5.

For Standard assets, rates vary by sector (commercial, CRE, SME, etc.).
For NPAs, rates depend on the NPA sub-category and whether the exposure
is secured or unsecured.

Reference: RBI Master Circular DBOD.No.BP.BC.94/21.06.001, para 4.
"""


def classify_irac(
    days_past_due: int,
    months_as_npa: int = 0,
    is_loss: bool = False,
    is_agricultural: bool = False,
    is_short_duration_crop: bool = False,
) -> IRACAssetClass:
    """Classify a loan per RBI IRAC norms.

    Reference: RBI Master Circular IRAC Norms para 2.1-2.5.

    Args:
        days_past_due: Days past due count.
        months_as_npa: Number of months the account has been classified
            as NPA.  Only relevant for distinguishing Substandard from
            Doubtful sub-categories.
        is_loss: Whether the account has been identified as loss by
            the bank, internal/external auditor, or RBI inspection.
        is_agricultural: Whether the exposure is an agricultural loan.
        is_short_duration_crop: If agricultural, whether the crop is
            short-duration (affects DPD threshold).

    Returns:
        :class:`IRACAssetClass` category.
    """
    if is_loss:
        return IRACAssetClass.LOSS

    # Determine NPA threshold
    if is_agricultural and is_short_duration_crop:
        npa_threshold = RBI_AGRI_SHORT_CROP_DPD
    else:
        npa_threshold = RBI_DEFAULT_DPD_THRESHOLD

    if days_past_due < npa_threshold:
        # Standard or SMA
        if days_past_due <= 0:
            return IRACAssetClass.STANDARD
        if days_past_due <= 30:
            return IRACAssetClass.SMA_0
        if days_past_due <= 60:
            return IRACAssetClass.SMA_1
        return IRACAssetClass.SMA_2

    # NPA: determine sub-category by time as NPA
    if months_as_npa <= 12:
        return IRACAssetClass.SUBSTANDARD
    if months_as_npa <= 24:
        return IRACAssetClass.DOUBTFUL_1
    if months_as_npa <= 36:
        return IRACAssetClass.DOUBTFUL_2
    return IRACAssetClass.DOUBTFUL_3


def irac_to_ifrs9_stage(irac_class: IRACAssetClass) -> IFRS9Stage:
    """Map RBI IRAC classification to IFRS 9 stage.

    SMA-0 is considered performing (Stage 1). SMA-1/2 indicate
    deterioration and map to Stage 2 (SICR). NPAs map to Stage 3.

    Reference: RBI/2019-20/170 — Ind AS Provisioning mapping.

    Args:
        irac_class: RBI IRAC asset classification.

    Returns:
        Corresponding IFRS 9 stage.
    """
    mapping: dict[IRACAssetClass, IFRS9Stage] = {
        IRACAssetClass.STANDARD: IFRS9Stage.STAGE_1,
        IRACAssetClass.SMA_0: IFRS9Stage.STAGE_1,
        IRACAssetClass.SMA_1: IFRS9Stage.STAGE_2,
        IRACAssetClass.SMA_2: IFRS9Stage.STAGE_2,
        IRACAssetClass.SUBSTANDARD: IFRS9Stage.STAGE_3,
        IRACAssetClass.DOUBTFUL_1: IFRS9Stage.STAGE_3,
        IRACAssetClass.DOUBTFUL_2: IFRS9Stage.STAGE_3,
        IRACAssetClass.DOUBTFUL_3: IFRS9Stage.STAGE_3,
        IRACAssetClass.LOSS: IFRS9Stage.STAGE_3,
    }
    return mapping[irac_class]


def rbi_minimum_provision(
    ead: float,
    irac_class: IRACAssetClass,
    is_secured: bool = True,
    sector: str = "commercial",
) -> float:
    """Calculate RBI minimum provisioning requirement.

    Returns the regulatory minimum provision amount based on the IRAC
    asset classification and sector.

    Reference: RBI Master Circular IRAC Norms para 4.2-4.5.

    Args:
        ead: Exposure at default.
        irac_class: RBI IRAC classification.
        is_secured: Whether the exposure is secured by collateral.
        sector: Sector for Standard asset provisioning
            (``"commercial"``, ``"cre"``, ``"cre_rre"``, ``"sme"``,
            ``"agri"``, ``"other"``).

    Returns:
        Minimum provision amount per RBI norms.
    """
    if irac_class == IRACAssetClass.STANDARD or irac_class in (
        IRACAssetClass.SMA_0, IRACAssetClass.SMA_1, IRACAssetClass.SMA_2
    ):
        key = f"standard_{sector}"
        rate = RBI_PROVISION_RATES.get(key, RBI_PROVISION_RATES["standard_other"])
        return ead * rate

    if irac_class == IRACAssetClass.SUBSTANDARD:
        rate = (
            RBI_PROVISION_RATES["substandard_secured"]
            if is_secured
            else RBI_PROVISION_RATES["substandard_unsecured"]
        )
        return ead * rate

    if irac_class == IRACAssetClass.DOUBTFUL_1:
        key = "doubtful_1_secured" if is_secured else "doubtful_1_unsecured"
        return ead * RBI_PROVISION_RATES[key]

    if irac_class == IRACAssetClass.DOUBTFUL_2:
        key = "doubtful_2_secured" if is_secured else "doubtful_2_unsecured"
        return ead * RBI_PROVISION_RATES[key]

    # Doubtful 3 or Loss
    return ead * RBI_PROVISION_RATES["loss"]


# ---------------------------------------------------------------------------
# Restructured account handling (IRAC para 12-14)
# ---------------------------------------------------------------------------


def restructured_account_stage(
    days_past_due_post_restructure: int,
    months_since_restructure: int,
    satisfactory_performance_months: int = 12,
) -> IFRS9Stage:
    """Determine IFRS 9 stage for a restructured account.

    Per RBI norms, restructured accounts are classified as NPA (Stage 3)
    at the point of restructuring.  An upgraded classification to Stage 2
    requires satisfactory performance for a minimum period.

    Reference:
        RBI Master Circular IRAC para 12-14 — restructured advances.
        RBI/2019-20/170 — restructured account treatment under Ind AS.

    Args:
        days_past_due_post_restructure: DPD since restructuring.
        months_since_restructure: Months elapsed since restructuring.
        satisfactory_performance_months: Required months of satisfactory
            performance for upgrade (default 12 per RBI norms).

    Returns:
        IFRS 9 stage.
    """
    # If overdue post-restructure, remains Stage 3
    if days_past_due_post_restructure >= RBI_DEFAULT_DPD_THRESHOLD:
        return IFRS9Stage.STAGE_3

    # If satisfactory performance period not yet completed, Stage 2
    if months_since_restructure < satisfactory_performance_months:
        return IFRS9Stage.STAGE_2

    # If >30 DPD but performing, Stage 2 (SICR triggered)
    if days_past_due_post_restructure > RBI_SICR_DPD_BACKSTOP:
        return IFRS9Stage.STAGE_2

    return IFRS9Stage.STAGE_1


# ---------------------------------------------------------------------------
# Core ECL functions (wrappers with RBI-specific defaults)
# ---------------------------------------------------------------------------


def assign_stage_ind_as(
    days_past_due: int,
    is_npa: bool = False,
    is_poci: bool = False,
    sicr_triggered: bool = False,
) -> IFRS9Stage:
    """Assign Ind AS 109 impairment stage with RBI-specific defaults.

    RBI classifies assets as NPA (Non-Performing Asset) at 90 DPD.
    NPA maps to Stage 3 (credit-impaired).

    Args:
        days_past_due: Days past due.
        is_npa: Whether classified as NPA per RBI IRAC norms.
        is_poci: Whether purchased/originated credit-impaired.
        sicr_triggered: Whether SICR has been triggered.

    Returns:
        IFRS9Stage.
    """
    is_defaulted = is_npa or days_past_due >= RBI_DEFAULT_DPD_THRESHOLD
    return assign_stage(
        days_past_due=days_past_due,
        is_credit_impaired=is_defaulted,
        is_defaulted=is_defaulted,
        is_poci=is_poci,
        sicr_triggered=sicr_triggered,
        dpd_backstop=RBI_SICR_DPD_BACKSTOP,
    )


def calculate_ecl_ind_as(
    stage: IFRS9Stage,
    pd_12m: float,
    lgd: float,
    ead: float,
    eir: float = 0.0,
    marginal_pds: np.ndarray | None = None,
    lgd_curve: np.ndarray | None = None,
    ead_curve: np.ndarray | None = None,
    irac_class: IRACAssetClass | None = None,
    is_secured: bool = True,
    sector: str = "commercial",
) -> float:
    """Calculate ECL per Ind AS 109 with RBI provisioning floor.

    Computes IFRS 9-based ECL and then applies the RBI minimum
    provisioning requirement as a floor, ensuring the reported
    provision is at least the regulatory minimum.

    Reference:
        Ind AS 109 (= IFRS 9) — ECL measurement.
        RBI/2019-20/170 — higher of Ind AS ECL and IRAC provision.

    Args:
        stage: Ind AS 109 impairment stage (same as IFRS 9).
        pd_12m: 12-month PD.
        lgd: Loss given default.
        ead: Exposure at default.
        eir: Effective interest rate.
        marginal_pds: Marginal PD curve for lifetime ECL.
        lgd_curve: Optional LGD term structure.
        ead_curve: Optional EAD term structure.
        irac_class: Optional IRAC classification for provisioning floor.
        is_secured: Whether the exposure is secured (for IRAC floor).
        sector: Sector for Standard asset provisioning.

    Returns:
        ECL amount (higher of model ECL and RBI minimum provision).
    """
    model_ecl = calculate_ecl(
        stage=stage,
        pd_12m=pd_12m,
        lgd=lgd,
        ead=ead,
        eir=eir,
        marginal_pds=marginal_pds,
        lgd_curve=lgd_curve,
        ead_curve=ead_curve,
    )

    if irac_class is not None:
        rbi_floor = rbi_minimum_provision(ead, irac_class, is_secured, sector)
        return max(model_ecl, rbi_floor)

    return model_ecl
