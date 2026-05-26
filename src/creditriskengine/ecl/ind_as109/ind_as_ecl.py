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
from datetime import date
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

    # At point of restructuring (month 0), account is NPA → Stage 3
    if months_since_restructure == 0:
        return IFRS9Stage.STAGE_3

    # Probation period: performing but not yet upgraded → Stage 2
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


# ---------------------------------------------------------------------------
# RBI ECL Master Direction 2026 (RBI/DOR/2026-27/398) — effective April 1, 2027
# ---------------------------------------------------------------------------

# Re-export 2026 framework symbols at module level for convenience.
from creditriskengine.ecl.ind_as109.borrower_classification import (  # noqa: E402
    apply_borrower_level_staging,
)
from creditriskengine.ecl.ind_as109.collateral_valuation import (  # noqa: E402
    validate_collateral_revaluation,
)
from creditriskengine.ecl.ind_as109.dlg import DLGAdjustment, ecl_with_dlg  # noqa: E402
from creditriskengine.ecl.ind_as109.pd_lgd_floors import (  # noqa: E402
    RBI_LGD_BACKSTOP_SECURED,
    RBI_LGD_BACKSTOP_UNSECURED,
    RBI_LGD_ELIGIBLE_COLLATERAL,
    RBI_PD_FLOOR,
    apply_rbi_lgd_backstop,
    apply_rbi_pd_floor,
)
from creditriskengine.ecl.ind_as109.provision_floors_2026 import (  # noqa: E402
    classify_rbi_exposure_category,
    dcco_additional_provision,
    rbi_ecl_floor_2026,
)
from creditriskengine.ecl.ind_as109.transition import (  # noqa: E402
    RBI_ECL_EFFECTIVE_DATE,
    capital_add_back_factor,
    is_ecl_framework_effective,
)
from creditriskengine.ecl.ind_as109.types import (  # noqa: E402
    RBICollateralCategory,
    RBIExposureCategory,
)

RBI_REVOLVING_SICR_OVERLIMIT_DAYS: int = 60
"""Revolving facility SICR backstop per Paragraph 33.

When the outstanding balance of a revolving facility remains
continuously in excess of the sanctioned limit or drawing power
for the specified period, SICR is presumed.

Reference: RBI/DOR/2026-27/398 Paragraph 33.
"""


def assess_sicr_rbi(
    days_past_due: int = 0,
    is_revolving: bool = False,
    days_over_limit: int = 0,
    rebuttal_applied: bool = False,
    sicr_triggered_quantitative: bool = False,
) -> bool:
    """Assess SICR under RBI ECL Master Direction 2026 rules.

    Combines:
        - The IFRS 9 30-day DPD backstop (Paragraph 33).
        - The 60-day revolving overlimit backstop (Paragraph 33).
        - Bank's own quantitative SICR triggers (relative/absolute PD
          changes, rating migrations, etc.) passed via
          *sicr_triggered_quantitative*.

    Either backstop is rebuttable with reasonable and supportable
    information per Paragraph 33; set *rebuttal_applied=True* to
    suppress the DPD/overlimit-driven SICR.

    Args:
        days_past_due: Current days past due.
        is_revolving: Whether the facility is revolving.
        days_over_limit: For revolving facilities, the number of
            consecutive days the outstanding balance has been in
            excess of the sanctioned limit.
        rebuttal_applied: Whether the bank has documented a rebuttal
            of the DPD/overlimit presumption with reasonable and
            supportable information.
        sicr_triggered_quantitative: Whether the bank's quantitative
            SICR criteria are independently triggered.

    Returns:
        ``True`` if SICR is identified per Paragraph 33.

    Reference:
        RBI/DOR/2026-27/398 Paragraph 33.
    """
    if sicr_triggered_quantitative:
        return True
    if rebuttal_applied:
        return False
    if days_past_due > RBI_SICR_DPD_BACKSTOP:
        return True
    return is_revolving and days_over_limit > RBI_REVOLVING_SICR_OVERLIMIT_DAYS


def determine_upgrade_eligibility(
    current_stage: IFRS9Stage,
    is_restructured: bool,
    all_arrears_repaid: bool = False,
    sicr_triggered: bool = False,
    satisfactory_performance: bool = False,
    resolution_directions_met: bool = False,
) -> IFRS9Stage:
    """Determine the eligible stage post-upgrade per RBI 2026 criteria.

    Upgrade paths per Paragraphs 77-79 and Paragraph 10(1):

        Stage 3 (non-restructured) -> Stage 1:
            All arrears (interest + principal) repaid AND no SICR.

        Stage 3 (restructured, non-MSME) -> Stage 2:
            Resolution Directions conditions met AND satisfactory
            performance during specified period AND no SICR.

        Stage 2 -> Stage 1:
            Satisfactory performance AND no SICR.

    If none of the upgrade conditions are met, the current stage is
    retained.

    Args:
        current_stage: Current stage of the exposure.
        is_restructured: Whether the account is restructured.
        all_arrears_repaid: All interest and principal arrears across
            all facilities of the borrower have been repaid.
        sicr_triggered: Whether SICR is currently identified.
        satisfactory_performance: Whether the satisfactory performance
            period has been completed.
        resolution_directions_met: Whether the relevant Resolution
            Directions conditions are met (Paragraph 58/59/60 of the
            RBI Resolution Directions 2025).

    Returns:
        The stage the exposure is eligible to be classified at.

    Reference:
        RBI/DOR/2026-27/398 Paragraphs 10, 77, 78, 79.
    """
    if current_stage == IFRS9Stage.STAGE_3:
        if not is_restructured:
            if all_arrears_repaid and not sicr_triggered:
                return IFRS9Stage.STAGE_1
            return IFRS9Stage.STAGE_3
        # Restructured Stage 3: upgrade to Stage 2 only
        if (
            resolution_directions_met
            and satisfactory_performance
            and not sicr_triggered
        ):
            return IFRS9Stage.STAGE_2
        return IFRS9Stage.STAGE_3

    if current_stage == IFRS9Stage.STAGE_2:
        if satisfactory_performance and not sicr_triggered:
            return IFRS9Stage.STAGE_1
        return IFRS9Stage.STAGE_2

    return current_stage


def calculate_ecl_ind_as_2026(
    stage: IFRS9Stage,
    pd_12m: float,
    lgd: float,
    ead: float,
    eir: float = 0.0,
    marginal_pds: np.ndarray | None = None,
    lgd_curve: np.ndarray | None = None,
    ead_curve: np.ndarray | None = None,
    category: RBIExposureCategory = RBIExposureCategory.OTHER,
    is_secured: bool = True,
    has_eligible_collateral: bool = False,
    years_in_stage3: float = 0.0,
    dlg_remaining_capacity: float = 0.0,
    is_wilful_defaulter: bool = False,
    is_sovereign_slr: bool = False,
    apply_pd_lgd_floors: bool = True,
) -> float:
    """Calculate ECL per RBI ECL Master Direction 2026 (RBI/DOR/2026-27/398).

    Pipeline:
        1. Apply PD floor of 0.03% (Paragraph 96).
        2. Apply LGD backstop per Paragraphs 97-98 (65%/70%/30%).
        3. Compute model ECL using IFRS 9 mechanics.
        4. Apply DLG adjustment if applicable (Paragraph 88).
        5. Apply prudential floor per Paragraph 82 (category- and
           stage-specific; Stage 3 duration-dependent).
        6. Return the higher of net model ECL and regulatory floor.

    Args:
        stage: Ind AS 109 / IFRS 9 stage.
        pd_12m: 12-month PD (model estimate).
        lgd: Loss given default (model estimate).
        ead: Exposure at default.
        eir: Effective interest rate for discounting.
        marginal_pds: Marginal PD curve for lifetime ECL.
        lgd_curve: Optional LGD term structure.
        ead_curve: Optional EAD term structure.
        category: RBI exposure category per Paragraph 82.
        is_secured: Whether the exposure is secured.
        has_eligible_collateral: Whether the exposure has eligible
            collateral per Paragraph 98 (cash, gold, govt sec, LIC,
            KVP, NSC).
        years_in_stage3: Years elapsed since Stage 3 classification
            (used for Stage 3 duration-dependent floors).
        dlg_remaining_capacity: Remaining DLG cover available to
            absorb ECL per Paragraph 88. Set to 0.0 to disable.
        is_wilful_defaulter: Whether the borrower is classified as
            a wilful defaulter — attracts +5% surcharge on the
            prudential floor per Paragraph 101(4).
        is_sovereign_slr: Whether the exposure is SLR-eligible,
            direct Central Government, Central-Government-guaranteed,
            or zero-risk-weight foreign sovereign/MDB/BIS/IMF.
            Per Paragraphs 37-38: no SICR test and no Stage 1 ECL.
        apply_pd_lgd_floors: Whether to apply the PD floor and LGD
            backstops to the inputs (default ``True``).

    Returns:
        ECL amount: max(net model ECL, regulatory floor). Returns
        0.0 for sovereign/SLR carve-out exposures per Paragraphs 37-38.

    Reference:
        RBI/DOR/2026-27/398; DOR.STR.REC.No.6/21.06.011/2026-27,
        Paragraphs 37-38, 82, 88, 96-98, 101(4).
    """
    # Sovereign / SLR carve-out per Paragraphs 37-38
    if is_sovereign_slr:
        return 0.0

    if apply_pd_lgd_floors:
        pd_12m = apply_rbi_pd_floor(pd_12m)
        lgd = apply_rbi_lgd_backstop(lgd, is_secured, has_eligible_collateral)

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

    # DLG absorption reduces the model ECL before applying the regulatory floor
    if dlg_remaining_capacity > 0:
        adj = ecl_with_dlg(model_ecl, dlg_remaining_capacity)
        net_model_ecl = adj.net_ecl
    else:
        net_model_ecl = model_ecl

    regulatory_floor = rbi_ecl_floor_2026(
        ead=ead,
        stage=stage,
        category=category,
        is_secured=is_secured,
        years_in_stage3=years_in_stage3,
        is_wilful_defaulter=is_wilful_defaulter,
    )

    return max(net_model_ecl, regulatory_floor)


def calculate_ecl_ind_as_auto(
    reporting_date: date,
    stage: IFRS9Stage,
    pd_12m: float,
    lgd: float,
    ead: float,
    eir: float = 0.0,
    marginal_pds: np.ndarray | None = None,
    lgd_curve: np.ndarray | None = None,
    ead_curve: np.ndarray | None = None,
    # Legacy IRAC pathway
    irac_class: IRACAssetClass | None = None,
    sector: str = "commercial",
    # 2026 framework pathway
    category: RBIExposureCategory = RBIExposureCategory.OTHER,
    is_secured: bool = True,
    has_eligible_collateral: bool = False,
    years_in_stage3: float = 0.0,
    dlg_remaining_capacity: float = 0.0,
) -> float:
    """Dispatch to the legacy or 2026 ECL framework based on reporting date.

    Reports dated before April 1, 2027 use :func:`calculate_ecl_ind_as`
    with the legacy IRAC-based provisioning floor. Reports on or after
    April 1, 2027 use :func:`calculate_ecl_ind_as_2026` with the new
    Master Direction floors.

    Args:
        reporting_date: Date of the reporting period.
        stage: IFRS 9 / Ind AS 109 stage.
        pd_12m: 12-month PD.
        lgd: Loss given default.
        ead: Exposure at default.
        eir: Effective interest rate.
        marginal_pds: Marginal PD curve (lifetime ECL).
        lgd_curve: LGD term structure.
        ead_curve: EAD term structure.
        irac_class: Legacy IRAC classification (pre-2027).
        sector: Sector for legacy Standard provisioning.
        category: RBI exposure category (post-2027).
        is_secured: Secured/unsecured flag.
        has_eligible_collateral: Eligible collateral flag (post-2027).
        years_in_stage3: Time in Stage 3 (post-2027).
        dlg_remaining_capacity: DLG cover capacity (post-2027).

    Returns:
        ECL amount per the applicable framework.

    Reference:
        RBI/DOR/2026-27/398 Paragraph 2 (effective date dispatch).
    """
    if is_ecl_framework_effective(reporting_date):
        return calculate_ecl_ind_as_2026(
            stage=stage,
            pd_12m=pd_12m,
            lgd=lgd,
            ead=ead,
            eir=eir,
            marginal_pds=marginal_pds,
            lgd_curve=lgd_curve,
            ead_curve=ead_curve,
            category=category,
            is_secured=is_secured,
            has_eligible_collateral=has_eligible_collateral,
            years_in_stage3=years_in_stage3,
            dlg_remaining_capacity=dlg_remaining_capacity,
        )
    return calculate_ecl_ind_as(
        stage=stage,
        pd_12m=pd_12m,
        lgd=lgd,
        ead=ead,
        eir=eir,
        marginal_pds=marginal_pds,
        lgd_curve=lgd_curve,
        ead_curve=ead_curve,
        irac_class=irac_class,
        is_secured=is_secured,
        sector=sector,
    )


__all__ = [
    # Legacy IRAC framework
    "IRACAssetClass",
    "RBI_AGRI_SHORT_CROP_DPD",
    "RBI_DEFAULT_DPD_THRESHOLD",
    "RBI_SICR_DPD_BACKSTOP",
    "RBI_PROVISION_RATES",
    "assign_stage_ind_as",
    "calculate_ecl_ind_as",
    "classify_irac",
    "irac_to_ifrs9_stage",
    "rbi_minimum_provision",
    "restructured_account_stage",
    # RBI ECL Master Direction 2026 framework
    "RBIExposureCategory",
    "RBICollateralCategory",
    "RBI_ECL_EFFECTIVE_DATE",
    "RBI_PD_FLOOR",
    "RBI_LGD_BACKSTOP_SECURED",
    "RBI_LGD_BACKSTOP_UNSECURED",
    "RBI_LGD_ELIGIBLE_COLLATERAL",
    "RBI_REVOLVING_SICR_OVERLIMIT_DAYS",
    "apply_rbi_pd_floor",
    "apply_rbi_lgd_backstop",
    "rbi_ecl_floor_2026",
    "classify_rbi_exposure_category",
    "dcco_additional_provision",
    "apply_borrower_level_staging",
    "validate_collateral_revaluation",
    "DLGAdjustment",
    "ecl_with_dlg",
    "capital_add_back_factor",
    "is_ecl_framework_effective",
    "assess_sicr_rbi",
    "determine_upgrade_eligibility",
    "calculate_ecl_ind_as_2026",
    "calculate_ecl_ind_as_auto",
]
