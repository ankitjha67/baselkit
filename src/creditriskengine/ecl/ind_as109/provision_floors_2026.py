"""
RBI ECL Master Direction 2026 — Provisioning floor tables.

Reference: RBI Master Direction on Asset Classification, Provisioning
and Income Recognition (Commercial Banks), April 27, 2026.
Effective: April 1, 2027. Document: RBI/2026-27/34.

Implements Paragraph 82 prudential floor tables across all 20 exposure
categories with duration-dependent Stage 3 floors per Paragraphs 82(1)-(4).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from creditriskengine.core.types import IFRS9Stage
from creditriskengine.ecl.ind_as109.types import (
    RBICollateralCategory,
    RBIExposureCategory,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 1 and Stage 2 floor table per Paragraph 82
# ---------------------------------------------------------------------------

RBI_ECL_FLOOR_STAGE_1_2: dict[RBIExposureCategory, tuple[float, float]] = {
    # (Stage 1 floor, Stage 2 floor)
    RBIExposureCategory.SECURED_RETAIL: (0.0040, 0.05),
    RBIExposureCategory.CORPORATE: (0.0040, 0.05),
    RBIExposureCategory.SMALL_MICRO_ENTERPRISE: (0.0025, 0.05),
    RBIExposureCategory.MEDIUM_ENTERPRISE: (0.0040, 0.05),
    RBIExposureCategory.FARM_CREDIT_AGRICULTURAL: (0.0025, 0.05),
    RBIExposureCategory.BANKS_NBFCS_REGULATED_FIS: (0.0040, 0.05),
    RBIExposureCategory.LOANS_AGAINST_DEPOSITS_LIC_KVP: (0.0040, 0.0040),
    RBIExposureCategory.GOLD_LOANS: (0.0040, 0.015),
    RBIExposureCategory.STATE_GOVT_GUARANTEED: (0.0040, 0.025),
    RBIExposureCategory.UNSECURED_RETAIL: (0.01, 0.05),
    RBIExposureCategory.HOUSING_LOANS_INDIVIDUALS: (0.0025, 0.015),
    RBIExposureCategory.CRE_ADC_150: (0.0125, 0.05),
    RBIExposureCategory.CRE_RH_ADC: (0.01, 0.05),
    RBIExposureCategory.OTHER_RESIDENTIAL_RE: (0.0040, 0.015),
    RBIExposureCategory.OTHER_COMMERCIAL_RE: (0.0040, 0.025),
    RBIExposureCategory.PROJECT_FINANCE_PRE_OPERATIONAL: (0.01, 0.05),
    RBIExposureCategory.PROJECT_FINANCE_OPERATIONAL: (0.0040, 0.05),
    RBIExposureCategory.CENTRAL_GOVT_GUARANTEED: (0.0025, 0.0025),
    RBIExposureCategory.NATURAL_CALAMITY_RESTRUCTURED: (0.05, 0.10),
    RBIExposureCategory.OTHER: (0.0040, 0.05),
}
"""Stage 1 and Stage 2 provisioning floors per Paragraph 82.

Each tuple is (stage_1_floor, stage_2_floor) as a decimal fraction
of EAD. All 20 RBI exposure categories are covered.
"""


# ---------------------------------------------------------------------------
# Stage 3 duration-dependent floor schedules per Paragraph 82
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Stage3FloorBracket:
    """One bracket of a Stage 3 duration-dependent floor schedule.

    Attributes:
        max_years: Upper bound of the duration bracket (exclusive).
            Use ``float('inf')`` for the open-ended >4 year bracket.
        secured_floor: Floor for the secured portion.
        unsecured_floor: Floor for the unsecured portion.
    """

    max_years: float
    secured_floor: float
    unsecured_floor: float


# Standard Stage 3 schedule (most categories i-vi, x, xv default to this
# pattern for secured; unsecured retail uses a separate flat schedule).
# Per Paragraph 82(1): 25/40/55/75/100 secured; 25/100/100/100/100 unsecured.
STAGE3_STANDARD: tuple[Stage3FloorBracket, ...] = (
    Stage3FloorBracket(max_years=1.0, secured_floor=0.25, unsecured_floor=0.25),
    Stage3FloorBracket(max_years=2.0, secured_floor=0.40, unsecured_floor=1.00),
    Stage3FloorBracket(max_years=3.0, secured_floor=0.55, unsecured_floor=1.00),
    Stage3FloorBracket(max_years=4.0, secured_floor=0.75, unsecured_floor=1.00),
    Stage3FloorBracket(max_years=float("inf"), secured_floor=1.00, unsecured_floor=1.00),
)
"""Standard Stage 3 floors per Paragraph 82(1)."""

# Lower-floor Stage 3 schedule for deposits/LIC/KVP/gold/state govt
# (Categories vii, viii, ix). Per Paragraph 82(2).
STAGE3_DEPOSITS_GOLD_STATE: tuple[Stage3FloorBracket, ...] = (
    Stage3FloorBracket(max_years=1.0, secured_floor=0.10, unsecured_floor=0.25),
    Stage3FloorBracket(max_years=2.0, secured_floor=0.20, unsecured_floor=1.00),
    Stage3FloorBracket(max_years=3.0, secured_floor=0.30, unsecured_floor=1.00),
    Stage3FloorBracket(max_years=4.0, secured_floor=0.40, unsecured_floor=1.00),
    Stage3FloorBracket(max_years=float("inf"), secured_floor=1.00, unsecured_floor=1.00),
)
"""Lower Stage 3 floors per Paragraph 82(2) for deposits/LIC/gold/state."""

# Unsecured retail Stage 3 schedule per Paragraph 82(3).
STAGE3_UNSECURED_RETAIL: tuple[Stage3FloorBracket, ...] = (
    Stage3FloorBracket(max_years=1.0, secured_floor=0.25, unsecured_floor=0.25),
    Stage3FloorBracket(max_years=float("inf"), secured_floor=1.00, unsecured_floor=1.00),
)
"""Unsecured retail Stage 3 floors per Paragraph 82(3): 25% first year, 100% after."""

# Housing / residential real estate Stage 3 schedule per Paragraph 82(4).
STAGE3_HOUSING_RESIDENTIAL_RE: tuple[Stage3FloorBracket, ...] = (
    Stage3FloorBracket(max_years=1.0, secured_floor=0.10, unsecured_floor=0.25),
    Stage3FloorBracket(max_years=2.0, secured_floor=0.20, unsecured_floor=1.00),
    Stage3FloorBracket(max_years=3.0, secured_floor=0.30, unsecured_floor=1.00),
    Stage3FloorBracket(max_years=4.0, secured_floor=0.40, unsecured_floor=1.00),
    Stage3FloorBracket(max_years=float("inf"), secured_floor=1.00, unsecured_floor=1.00),
)
"""Housing and residential RE Stage 3 floors per Paragraph 82(4)."""


# Mapping from exposure category to its applicable Stage 3 schedule
_STAGE3_SCHEDULE_MAP: dict[RBIExposureCategory, tuple[Stage3FloorBracket, ...]] = {
    RBIExposureCategory.SECURED_RETAIL: STAGE3_STANDARD,
    RBIExposureCategory.CORPORATE: STAGE3_STANDARD,
    RBIExposureCategory.SMALL_MICRO_ENTERPRISE: STAGE3_STANDARD,
    RBIExposureCategory.MEDIUM_ENTERPRISE: STAGE3_STANDARD,
    RBIExposureCategory.FARM_CREDIT_AGRICULTURAL: STAGE3_STANDARD,
    RBIExposureCategory.BANKS_NBFCS_REGULATED_FIS: STAGE3_STANDARD,
    RBIExposureCategory.LOANS_AGAINST_DEPOSITS_LIC_KVP: STAGE3_DEPOSITS_GOLD_STATE,
    RBIExposureCategory.GOLD_LOANS: STAGE3_DEPOSITS_GOLD_STATE,
    RBIExposureCategory.STATE_GOVT_GUARANTEED: STAGE3_DEPOSITS_GOLD_STATE,
    RBIExposureCategory.UNSECURED_RETAIL: STAGE3_UNSECURED_RETAIL,
    RBIExposureCategory.HOUSING_LOANS_INDIVIDUALS: STAGE3_HOUSING_RESIDENTIAL_RE,
    RBIExposureCategory.CRE_ADC_150: STAGE3_STANDARD,
    RBIExposureCategory.CRE_RH_ADC: STAGE3_STANDARD,
    RBIExposureCategory.OTHER_RESIDENTIAL_RE: STAGE3_HOUSING_RESIDENTIAL_RE,
    RBIExposureCategory.OTHER_COMMERCIAL_RE: STAGE3_HOUSING_RESIDENTIAL_RE,
    RBIExposureCategory.PROJECT_FINANCE_PRE_OPERATIONAL: STAGE3_STANDARD,
    RBIExposureCategory.PROJECT_FINANCE_OPERATIONAL: STAGE3_STANDARD,
    RBIExposureCategory.CENTRAL_GOVT_GUARANTEED: STAGE3_HOUSING_RESIDENTIAL_RE,
    RBIExposureCategory.NATURAL_CALAMITY_RESTRUCTURED: STAGE3_STANDARD,
    RBIExposureCategory.OTHER: STAGE3_STANDARD,
}


# ---------------------------------------------------------------------------
# DCCO (Date of Commencement of Commercial Operations) additional provisioning
# ---------------------------------------------------------------------------

RBI_DCCO_INFRA_QUARTERLY_RATE: float = 0.00375
"""0.375% per quarter of deferment for infrastructure project finance.

Reference: Paragraph 82(4) Note 1, RBI/2026-27/34.
"""

RBI_DCCO_NON_INFRA_QUARTERLY_RATE: float = 0.005625
"""0.5625% per quarter of deferment for non-infrastructure project finance
and CRE(ADC)/CRE-RH(ADC) accounts.

Reference: Paragraph 82(4) Note 1, RBI/2026-27/34.
"""


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def _stage3_floor_for_duration(
    schedule: tuple[Stage3FloorBracket, ...],
    years_in_stage3: float,
    is_secured: bool,
) -> float:
    """Look up the Stage 3 floor for a given duration and security."""
    for bracket in schedule:
        if years_in_stage3 < bracket.max_years:
            return bracket.secured_floor if is_secured else bracket.unsecured_floor
    # Fallthrough (should be unreachable due to inf-capped final bracket)
    return 1.00


def rbi_ecl_floor_2026(
    ead: float,
    stage: IFRS9Stage,
    category: RBIExposureCategory,
    is_secured: bool = True,
    years_in_stage3: float = 0.0,
) -> float:
    """Calculate the RBI ECL Master Direction 2026 prudential floor.

    Returns the minimum provision amount per Paragraph 82 of
    RBI/2026-27/34, applied as a regulatory backstop to the
    model-computed ECL.

    Args:
        ead: Exposure at default (in same currency unit as the return).
        stage: IFRS 9 / Ind AS 109 impairment stage.
        category: RBI exposure category per Paragraph 82.
        is_secured: Whether the exposure is secured (Stage 3 only).
        years_in_stage3: Years elapsed since Stage 3 classification.
            Only used when stage is Stage 3.

    Returns:
        Minimum provision amount as a decimal amount of EAD.

    Reference:
        RBI/2026-27/34 Paragraph 82(1)-(4).
    """
    if ead <= 0:
        return 0.0

    if stage == IFRS9Stage.STAGE_1:
        floor_rate, _ = RBI_ECL_FLOOR_STAGE_1_2[category]
        return ead * floor_rate

    if stage == IFRS9Stage.STAGE_2:
        _, floor_rate = RBI_ECL_FLOOR_STAGE_1_2[category]
        return ead * floor_rate

    # Stage 3 or POCI: duration-dependent floor
    schedule = _STAGE3_SCHEDULE_MAP[category]
    floor_rate = _stage3_floor_for_duration(schedule, years_in_stage3, is_secured)
    return ead * floor_rate


def dcco_additional_provision(
    ead: float,
    quarters_of_deferment: int,
    is_infrastructure: bool = True,
) -> float:
    """Additional provision for project finance with DCCO deferment.

    Per Paragraph 82(4) Note 1, projects that have availed DCCO
    deferment require account-wise specific additional provisions
    over and above the Stage 1 floor.

    Formula:
        provision = quarters * rate * EAD

    Args:
        ead: Exposure at default.
        quarters_of_deferment: Number of quarters of DCCO deferment.
        is_infrastructure: Whether the project is infrastructure
            (lower rate) or non-infrastructure (higher rate).

    Returns:
        Additional provision amount.

    Reference:
        RBI/2026-27/34 Paragraph 82(4) Note 1.
    """
    if ead <= 0 or quarters_of_deferment <= 0:
        return 0.0
    rate = (
        RBI_DCCO_INFRA_QUARTERLY_RATE
        if is_infrastructure
        else RBI_DCCO_NON_INFRA_QUARTERLY_RATE
    )
    return ead * rate * quarters_of_deferment


def classify_rbi_exposure_category(
    sector: str = "other",
    is_secured: bool = True,
    is_retail: bool = False,
    is_housing_individual: bool = False,
    is_cre: bool = False,
    is_cre_adc: bool = False,
    is_cre_rh_adc: bool = False,
    is_project_finance: bool = False,
    project_phase: str | None = None,
    is_central_govt_guaranteed: bool = False,
    is_state_govt_guaranteed: bool = False,
    is_natural_calamity_restructured: bool = False,
    is_gold_loan: bool = False,
    is_loan_against_deposit: bool = False,
    is_msme: bool = False,
    msme_size: str | None = None,
    is_agricultural: bool = False,
    is_bank_nbfc_fi: bool = False,
) -> RBIExposureCategory:
    """Classify an exposure into the appropriate RBI 2026 category.

    Resolution order (most specific to most general):
        1. Government guarantees
        2. Specific collateral types (gold, deposits)
        3. Real estate (housing individual, CRE-ADC, other RRE/CRE)
        4. Project finance (pre-op vs operational)
        5. Bank/NBFC/FI exposures
        6. Sector-specific (agricultural, MSME)
        7. Retail (secured/unsecured)
        8. Corporate (default for businesses)
        9. Other (fallback)

    Args:
        sector: Free-text sector hint (e.g., "agricultural", "corporate").
        is_secured: Whether the exposure has eligible collateral.
        is_retail: Retail exposure flag.
        is_housing_individual: Housing loan to an individual borrower.
        is_cre: Commercial real estate exposure.
        is_cre_adc: CRE Acquisition, Development, Construction (150% RW).
        is_cre_rh_adc: CRE Residential Housing ADC.
        is_project_finance: Project finance exposure.
        project_phase: ``"pre_operational"`` or ``"operational"``.
        is_central_govt_guaranteed: Covered by CGTMSE/CRGFTLIH/NCGTC.
        is_state_govt_guaranteed: State government direct/guaranteed.
        is_natural_calamity_restructured: Restructured under natural
            calamity dispensation and currently classified standard.
        is_gold_loan: Loan secured by gold ornaments.
        is_loan_against_deposit: Loan against term deposit, LIC, KVP.
        is_msme: Micro/Small/Medium Enterprise.
        msme_size: ``"micro"``, ``"small"``, or ``"medium"``.
        is_agricultural: Farm credit / agricultural activity.
        is_bank_nbfc_fi: Counterparty is a bank, NBFC, or regulated FI.

    Returns:
        Most specific applicable :class:`RBIExposureCategory`.
    """
    if is_natural_calamity_restructured:
        return RBIExposureCategory.NATURAL_CALAMITY_RESTRUCTURED
    if is_central_govt_guaranteed:
        return RBIExposureCategory.CENTRAL_GOVT_GUARANTEED
    if is_state_govt_guaranteed:
        return RBIExposureCategory.STATE_GOVT_GUARANTEED
    if is_loan_against_deposit:
        return RBIExposureCategory.LOANS_AGAINST_DEPOSITS_LIC_KVP
    if is_gold_loan:
        return RBIExposureCategory.GOLD_LOANS
    if is_housing_individual:
        return RBIExposureCategory.HOUSING_LOANS_INDIVIDUALS
    if is_cre_adc:
        return RBIExposureCategory.CRE_ADC_150
    if is_cre_rh_adc:
        return RBIExposureCategory.CRE_RH_ADC
    if is_cre:
        return RBIExposureCategory.OTHER_COMMERCIAL_RE
    # Residential RE not already captured as housing-individual
    if sector.lower() in ("residential_re", "residential_real_estate"):
        return RBIExposureCategory.OTHER_RESIDENTIAL_RE
    if is_project_finance:
        if project_phase == "operational":
            return RBIExposureCategory.PROJECT_FINANCE_OPERATIONAL
        return RBIExposureCategory.PROJECT_FINANCE_PRE_OPERATIONAL
    if is_bank_nbfc_fi:
        return RBIExposureCategory.BANKS_NBFCS_REGULATED_FIS
    if is_agricultural:
        return RBIExposureCategory.FARM_CREDIT_AGRICULTURAL
    if is_msme:
        if msme_size == "medium":
            return RBIExposureCategory.MEDIUM_ENTERPRISE
        return RBIExposureCategory.SMALL_MICRO_ENTERPRISE
    if is_retail:
        return (
            RBIExposureCategory.SECURED_RETAIL
            if is_secured
            else RBIExposureCategory.UNSECURED_RETAIL
        )
    if sector.lower() == "corporate":
        return RBIExposureCategory.CORPORATE
    return RBIExposureCategory.OTHER


def collateral_category_for(
    category: RBIExposureCategory,
    is_secured: bool,
) -> RBICollateralCategory:
    """Map an exposure category to its collateral category for floor lookup.

    Args:
        category: RBI exposure category.
        is_secured: Whether the exposure is secured.

    Returns:
        Corresponding :class:`RBICollateralCategory`.
    """
    if not is_secured:
        return RBICollateralCategory.UNSECURED
    if category in (
        RBIExposureCategory.LOANS_AGAINST_DEPOSITS_LIC_KVP,
        RBIExposureCategory.GOLD_LOANS,
        RBIExposureCategory.STATE_GOVT_GUARANTEED,
    ):
        return RBICollateralCategory.DEPOSITS_LIC_GOLD_STATE_GOVT
    if category in (
        RBIExposureCategory.HOUSING_LOANS_INDIVIDUALS,
        RBIExposureCategory.OTHER_RESIDENTIAL_RE,
        RBIExposureCategory.OTHER_COMMERCIAL_RE,
        RBIExposureCategory.CENTRAL_GOVT_GUARANTEED,
    ):
        return RBICollateralCategory.HOUSING_RESIDENTIAL_RE
    return RBICollateralCategory.STANDARD_SECURED
