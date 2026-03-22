"""
Standardized Approach (SA) for Credit Risk — BCBS d424 CRE20.

Risk weight assignment logic for all SA exposure classes.
Supports jurisdiction-specific overrides via YAML config.
"""

import logging
from typing import Any

from creditriskengine.core.types import (
    CreditQualityStep,
    Jurisdiction,
    SAExposureClass,
)

logger = logging.getLogger(__name__)


# ============================================================
# BCBS BASE RISK WEIGHT TABLES — CRE20
# ============================================================

# Sovereigns — CRE20.7, Table 1
SOVEREIGN_RW: dict[int, float] = {
    CreditQualityStep.CQS_1: 0.0,
    CreditQualityStep.CQS_2: 20.0,
    CreditQualityStep.CQS_3: 50.0,
    CreditQualityStep.CQS_4: 100.0,
    CreditQualityStep.CQS_5: 100.0,
    CreditQualityStep.CQS_6: 150.0,
    CreditQualityStep.UNRATED: 100.0,
}

# Banks (ECRA) — CRE20.15-20.18, Table 4
BANK_ECRA_RW: dict[int, float] = {
    CreditQualityStep.CQS_1: 20.0,
    CreditQualityStep.CQS_2: 30.0,
    CreditQualityStep.CQS_3: 50.0,
    CreditQualityStep.CQS_4: 100.0,
    CreditQualityStep.CQS_5: 100.0,
    CreditQualityStep.CQS_6: 150.0,
    CreditQualityStep.UNRATED: 50.0,
}

# Banks (ECRA) short-term claims — CRE20.17, Table 5
BANK_ECRA_SHORT_TERM_RW: dict[int, float] = {
    CreditQualityStep.CQS_1: 20.0,
    CreditQualityStep.CQS_2: 20.0,
    CreditQualityStep.CQS_3: 20.0,
    CreditQualityStep.CQS_4: 50.0,
    CreditQualityStep.CQS_5: 50.0,
    CreditQualityStep.CQS_6: 150.0,
    CreditQualityStep.UNRATED: 20.0,
}

# Banks (SCRA) — CRE20.19-20.21 (for jurisdictions not using external ratings)
BANK_SCRA_RW: dict[str, float] = {
    "A": 40.0,
    "B": 75.0,
    "C": 150.0,
}

# Corporates — CRE20.28-20.32, Table 7
CORPORATE_RW: dict[int, float] = {
    CreditQualityStep.CQS_1: 20.0,
    CreditQualityStep.CQS_2: 50.0,
    CreditQualityStep.CQS_3: 75.0,
    CreditQualityStep.CQS_4: 100.0,
    CreditQualityStep.CQS_5: 150.0,
    CreditQualityStep.CQS_6: 150.0,
    CreditQualityStep.UNRATED: 100.0,
}


# ============================================================
# RESIDENTIAL REAL ESTATE — CRE20.71-20.86
# ============================================================

# BCBS whole-loan approach, CRE20.73 Table 12
# General RRE NOT dependent on cashflows from property
RRE_WHOLE_LOAN_RW: list[tuple[float, float, float]] = [
    # (ltv_lower, ltv_upper, risk_weight)
    (0.0, 0.50, 20.0),
    (0.50, 0.60, 25.0),
    (0.60, 0.70, 30.0),
    (0.70, 0.80, 35.0),
    (0.80, 0.90, 40.0),
    (0.90, 1.00, 50.0),
    (1.00, float("inf"), 70.0),
]

# RRE dependent on cashflows — CRE20.78 Table 13
RRE_CASHFLOW_DEPENDENT_RW: list[tuple[float, float, float]] = [
    (0.0, 0.50, 30.0),
    (0.50, 0.60, 35.0),
    (0.60, 0.70, 45.0),
    (0.70, 0.80, 50.0),
    (0.80, 0.90, 60.0),
    (0.90, 1.00, 75.0),
    (1.00, float("inf"), 105.0),
]


# ============================================================
# COMMERCIAL REAL ESTATE — CRE20.87-20.98
# ============================================================

# CRE NOT dependent on cashflows — CRE20.89 Table 14
CRE_NOT_CASHFLOW_RW: list[tuple[float, float, float]] = [
    (0.0, 0.60, 60.0),  # min(60%, counterparty_rw) — simplified to 60%
    (0.60, 0.80, 75.0),
    (0.80, float("inf"), -1.0),  # counterparty risk weight
]

# CRE (IPRE) dependent on cashflows — CRE20.89 Table 15
CRE_IPRE_RW: list[tuple[float, float, float]] = [
    (0.0, 0.60, 70.0),
    (0.60, 0.80, 90.0),
    (0.80, float("inf"), 110.0),
]

# Land ADC — CRE20.97
LAND_ADC_RW: float = 150.0
LAND_ADC_PRESOLD_RW: float = 100.0


def get_sovereign_risk_weight(
    cqs: CreditQualityStep,
    jurisdiction: Jurisdiction = Jurisdiction.BCBS,
    is_domestic_own_currency: bool = False,
) -> float:
    """Risk weight for sovereign exposures.

    Reference: BCBS CRE20.7, Table 1.

    Args:
        cqs: Credit quality step.
        jurisdiction: Regulatory jurisdiction.
        is_domestic_own_currency: If True, domestic sovereign in own currency.

    Returns:
        Risk weight as percentage.
    """
    if is_domestic_own_currency:
        # Most jurisdictions assign 0% to own sovereign in domestic currency
        return 0.0
    return SOVEREIGN_RW.get(cqs.value, 100.0)


def get_bank_risk_weight(
    cqs: CreditQualityStep | None = None,
    jurisdiction: Jurisdiction = Jurisdiction.BCBS,
    scra_grade: str | None = None,
    is_short_term: bool = False,
) -> float:
    """Risk weight for bank exposures.

    Uses ECRA (External Credit Risk Assessment) if CQS provided,
    otherwise SCRA (Standardized Credit Risk Assessment).

    Reference: BCBS CRE20.15-20.21.

    Args:
        cqs: Credit quality step (for ECRA).
        jurisdiction: Regulatory jurisdiction.
        scra_grade: SCRA grade A/B/C (when ECRA not used).
        is_short_term: If True, use short-term claim risk weights.

    Returns:
        Risk weight as percentage.
    """
    if cqs is not None:
        if is_short_term:
            return BANK_ECRA_SHORT_TERM_RW.get(cqs.value, 20.0)
        return BANK_ECRA_RW.get(cqs.value, 50.0)
    if scra_grade is not None:
        rw = BANK_SCRA_RW.get(scra_grade.upper())
        if rw is None:
            raise ValueError(f"Invalid SCRA grade: {scra_grade}. Must be A, B, or C.")
        return rw
    # Default unrated
    return 50.0


def get_corporate_risk_weight(
    cqs: CreditQualityStep,
    jurisdiction: Jurisdiction = Jurisdiction.BCBS,
    is_investment_grade: bool | None = None,
    is_sme: bool = False,
) -> float:
    """Risk weight for corporate exposures.

    Reference: BCBS CRE20.28-20.32, Table 7.

    UK PRA divergence (PS9/24, para 3.17):
        Unrated investment-grade corporates = 65%.

    EU CRR3 Art. 501 — SME supporting factor:
        Exposures <= EUR 2.5M: multiply RW by 0.7619
        Exposures > EUR 2.5M: 0.7619 for first EUR 2.5M, 0.85 for remainder

    Args:
        cqs: Credit quality step.
        jurisdiction: Regulatory jurisdiction.
        is_investment_grade: For UK PRA unrated corporate treatment.
        is_sme: If True and jurisdiction supports it, apply SME factor.

    Returns:
        Risk weight as percentage.
    """
    if cqs == CreditQualityStep.UNRATED:
        if jurisdiction == Jurisdiction.UK and is_investment_grade:
            return 65.0
        rw = 100.0
    else:
        rw = CORPORATE_RW.get(cqs.value, 100.0)

    # EU SME supporting factor (CRR3 Art. 501)
    if is_sme and jurisdiction == Jurisdiction.EU:
        rw *= 0.7619

    return rw


def get_residential_re_risk_weight(
    ltv: float,
    jurisdiction: Jurisdiction = Jurisdiction.BCBS,
    is_cashflow_dependent: bool = False,
    is_income_producing: bool = False,
) -> float:
    """Risk weight for residential real estate exposures.

    Reference: BCBS CRE20.71-20.86, Tables 12-13.
    Whole-loan approach (BCBS/EU CRR3).

    Args:
        ltv: Loan-to-value ratio (e.g., 0.75 for 75%).
        jurisdiction: Regulatory jurisdiction.
        is_cashflow_dependent: If True, use cashflow-dependent table.
        is_income_producing: If True, treated as income-producing.

    Returns:
        Risk weight as percentage.
    """
    if is_cashflow_dependent or is_income_producing:
        table = RRE_CASHFLOW_DEPENDENT_RW
    else:
        table = RRE_WHOLE_LOAN_RW

    # India (RBI) specific treatment
    if jurisdiction == Jurisdiction.INDIA:
        if ltv <= 0.80:
            return 20.0
        return 35.0

    for ltv_lower, ltv_upper, rw in table:
        if ltv_lower < ltv <= ltv_upper:
            return rw
    # LTV exactly 0 case
    if ltv <= 0:
        return table[0][2]
    return table[-1][2]


def uk_pra_loan_splitting_rre(
    loan_amount: float,
    property_value: float,
    counterparty_rw: float = 100.0,
    is_cashflow_dependent: bool = False,
) -> dict[str, float]:
    """UK PRA loan-splitting for residential real estate (PS9/24).

    The UK PRA diverges from the EU/BCBS whole-loan approach by splitting
    each residential mortgage into two tranches:

    - **Secured tranche**: The portion of the loan up to the LTV threshold
      (55% of property value). This tranche receives the lower LTV-based
      risk weight from the RRE table.
    - **Unsecured tranche**: The remainder of the loan above the threshold.
      This tranche receives the counterparty risk weight (typically 100%
      for unrated corporates/retail).

    The blended risk weight is the EAD-weighted average of both tranches.

    Reference: PRA PS9/24, Chapter 4 (Real Estate).

    Args:
        loan_amount: Outstanding loan amount.
        property_value: Current property value.
        counterparty_rw: Risk weight for the unsecured tranche (default 100%).
        is_cashflow_dependent: If True, use cashflow-dependent RRE table.

    Returns:
        Dict with:
            - 'secured_amount': Amount in the secured tranche.
            - 'unsecured_amount': Amount in the unsecured tranche.
            - 'secured_rw': Risk weight for the secured tranche (%).
            - 'unsecured_rw': Risk weight for the unsecured tranche (%).
            - 'blended_rw': EAD-weighted blended risk weight (%).
            - 'ltv': Loan-to-value ratio.
    """
    if property_value <= 0 or loan_amount <= 0:
        return {
            "secured_amount": 0.0,
            "unsecured_amount": loan_amount,
            "secured_rw": 0.0,
            "unsecured_rw": counterparty_rw,
            "blended_rw": counterparty_rw,
            "ltv": 0.0,
        }

    ltv = loan_amount / property_value

    # PRA splitting threshold: 55% of property value
    split_threshold = 0.55 * property_value
    secured_amount = min(loan_amount, split_threshold)
    unsecured_amount = max(loan_amount - split_threshold, 0.0)

    # Secured tranche risk weight based on the LTV of the secured portion
    secured_ltv = secured_amount / property_value
    table = RRE_CASHFLOW_DEPENDENT_RW if is_cashflow_dependent else RRE_WHOLE_LOAN_RW
    secured_rw = table[0][2]  # default to first bucket
    for ltv_lower, ltv_upper, rw in table:
        if ltv_lower < secured_ltv <= ltv_upper:
            secured_rw = rw
            break

    # Unsecured tranche gets counterparty risk weight
    unsecured_rw = counterparty_rw

    # Blended risk weight (loan_amount > 0 guaranteed by early return above)
    blended_rw = (
        secured_amount * secured_rw + unsecured_amount * unsecured_rw
    ) / loan_amount

    return {
        "secured_amount": secured_amount,
        "unsecured_amount": unsecured_amount,
        "secured_rw": secured_rw,
        "unsecured_rw": unsecured_rw,
        "blended_rw": blended_rw,
        "ltv": ltv,
    }


def get_commercial_re_risk_weight(
    ltv: float,
    counterparty_rw: float = 100.0,
    is_cashflow_dependent: bool = False,
    is_adc: bool = False,
    is_presold_residential: bool = False,
) -> float:
    """Risk weight for commercial real estate exposures.

    Reference: BCBS CRE20.87-20.98, Tables 14-15.

    Args:
        ltv: Loan-to-value ratio.
        counterparty_rw: Risk weight of the counterparty.
        is_cashflow_dependent: If True, use IPRE table (Table 15).
        is_adc: If True, Land ADC treatment (150%).
        is_presold_residential: If True and ADC, use 100%.

    Returns:
        Risk weight as percentage.
    """
    if is_adc:
        if is_presold_residential:
            return LAND_ADC_PRESOLD_RW
        return LAND_ADC_RW

    if is_cashflow_dependent:
        for ltv_lower, ltv_upper, rw in CRE_IPRE_RW:
            if ltv_lower < ltv <= ltv_upper:
                return rw
        if ltv <= 0:
            return CRE_IPRE_RW[0][2]
        return CRE_IPRE_RW[-1][2]

    # Not cashflow dependent
    if ltv <= 0.60:
        return min(60.0, counterparty_rw)
    elif ltv <= 0.80:
        return 75.0
    else:
        return counterparty_rw


def get_defaulted_risk_weight(
    specific_provisions_pct: float,
    is_rre_secured: bool = False,
) -> float:
    """Risk weight for defaulted exposures.

    Reference: BCBS CRE20.99-20.101.

    Args:
        specific_provisions_pct: Specific provisions as % of outstanding.
        is_rre_secured: If secured by residential real estate.

    Returns:
        Risk weight as percentage.
    """
    # CRE20.101: RRE-secured defaulted exposures always get 100%
    if is_rre_secured:
        return 100.0
    if specific_provisions_pct >= 0.20:
        return 100.0
    return 150.0


def get_retail_risk_weight(is_regulatory_retail: bool = True) -> float:
    """Risk weight for retail exposures.

    Reference: BCBS CRE20.65.

    Args:
        is_regulatory_retail: If True, meets regulatory retail criteria.

    Returns:
        Risk weight as percentage (75% or 100%).
    """
    return 75.0 if is_regulatory_retail else 100.0


def get_equity_risk_weight(
    is_listed: bool = True,
    is_speculative: bool = False,
) -> float:
    """Risk weight for equity exposures.

    Reference: BCBS CRE20.49-20.58.

    Args:
        is_listed: If True, listed equity.
        is_speculative: If True, speculative unlisted equity.

    Returns:
        Risk weight as percentage.
    """
    if is_speculative:
        return 400.0
    if is_listed:
        return 250.0
    return 400.0


def get_subordinated_debt_risk_weight() -> float:
    """Risk weight for subordinated debt.

    Reference: BCBS CRE20.49.

    Returns:
        150% risk weight.
    """
    return 150.0


def assign_sa_risk_weight(
    exposure_class: SAExposureClass,
    cqs: CreditQualityStep = CreditQualityStep.UNRATED,
    jurisdiction: Jurisdiction = Jurisdiction.BCBS,
    ltv: float | None = None,
    counterparty_rw: float = 100.0,
    is_investment_grade: bool | None = None,
    is_sme: bool = False,
    is_cashflow_dependent: bool = False,
    is_income_producing: bool = False,
    is_adc: bool = False,
    is_presold_residential: bool = False,
    is_domestic_own_currency: bool = False,
    specific_provisions_pct: float = 0.0,
    is_rre_secured: bool = False,
    is_listed: bool = True,
    is_speculative: bool = False,
    is_regulatory_retail: bool = True,
    scra_grade: str | None = None,
    is_short_term: bool = False,
    config: dict[str, Any] | None = None,
) -> float:
    """Assign SA risk weight based on exposure class and parameters.

    Master dispatcher that routes to the appropriate risk weight function.

    Args:
        exposure_class: SA exposure class per BCBS CRE20.
        cqs: Credit quality step from external rating.
        jurisdiction: Regulatory jurisdiction.
        ltv: Loan-to-value ratio for real estate.
        counterparty_rw: Counterparty risk weight for CRE.
        is_investment_grade: For UK unrated corporate treatment.
        is_sme: For EU SME supporting factor.
        is_cashflow_dependent: For real estate cashflow dependency.
        is_income_producing: For income-producing property.
        is_adc: For land ADC exposures.
        is_presold_residential: For pre-sold ADC.
        is_domestic_own_currency: For domestic sovereign treatment.
        specific_provisions_pct: For defaulted exposures.
        is_rre_secured: For defaulted RRE-secured exposures.
        is_listed: For equity classification.
        is_speculative: For speculative equity.
        is_regulatory_retail: For retail qualification.
        scra_grade: SCRA grade for banks (A/B/C).
        is_short_term: For short-term bank claims (CRE20.17).
        config: Optional jurisdiction config dict.

    Returns:
        Risk weight as percentage.
    """
    if exposure_class == SAExposureClass.SOVEREIGN:
        return get_sovereign_risk_weight(cqs, jurisdiction, is_domestic_own_currency)

    if exposure_class in (SAExposureClass.BANK, SAExposureClass.SECURITIES_FIRM):
        # When SCRA grade is provided, use SCRA path (cqs=None)
        bank_cqs = None if scra_grade is not None else cqs
        return get_bank_risk_weight(bank_cqs, jurisdiction, scra_grade, is_short_term)

    if exposure_class in (SAExposureClass.CORPORATE, SAExposureClass.CORPORATE_SME):
        is_sme_flag = is_sme or exposure_class == SAExposureClass.CORPORATE_SME
        return get_corporate_risk_weight(cqs, jurisdiction, is_investment_grade, is_sme_flag)

    if exposure_class == SAExposureClass.RESIDENTIAL_MORTGAGE:
        if ltv is None:
            raise ValueError("LTV required for residential mortgage risk weight")
        return get_residential_re_risk_weight(
            ltv, jurisdiction, is_cashflow_dependent, is_income_producing
        )

    if exposure_class == SAExposureClass.COMMERCIAL_REAL_ESTATE:
        if ltv is None:
            raise ValueError("LTV required for commercial real estate risk weight")
        return get_commercial_re_risk_weight(
            ltv, counterparty_rw, is_cashflow_dependent, is_adc, is_presold_residential
        )

    if exposure_class == SAExposureClass.LAND_ADC:
        return get_commercial_re_risk_weight(
            ltv=1.0, is_adc=True, is_presold_residential=is_presold_residential
        )

    if exposure_class == SAExposureClass.RETAIL:
        return get_retail_risk_weight(is_regulatory_retail)

    if exposure_class == SAExposureClass.RETAIL_REGULATORY:
        return 75.0

    if exposure_class == SAExposureClass.DEFAULTED:
        return get_defaulted_risk_weight(specific_provisions_pct, is_rre_secured)

    if exposure_class == SAExposureClass.EQUITY:
        return get_equity_risk_weight(is_listed, is_speculative)

    if exposure_class == SAExposureClass.SUBORDINATED_DEBT:
        return get_subordinated_debt_risk_weight()

    if exposure_class == SAExposureClass.PSE:
        # PSEs Option A: use bank risk weight table per CRE20.10
        return get_bank_risk_weight(cqs, jurisdiction, scra_grade)

    if exposure_class == SAExposureClass.MDB:
        # Qualifying MDBs: 0%; others: per bank table
        return 0.0

    if exposure_class == SAExposureClass.OTHER:
        return 100.0

    return 100.0
