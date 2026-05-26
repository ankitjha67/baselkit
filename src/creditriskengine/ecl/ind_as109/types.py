"""
Type definitions for RBI ECL Master Direction 2026 (RBI/DOR/2026-27/398).

Reference: RBI Master Direction on Asset Classification, Provisioning
and Income Recognition (Commercial Banks), April 27, 2026.
Effective: April 1, 2027.

Defines exposure categories and collateral categories used by the
RBI ECL provisioning floor framework (Paragraph 82 of the Master
Direction).
"""

from __future__ import annotations

from enum import StrEnum


class RBIExposureCategory(StrEnum):
    """RBI ECL Master Direction 2026 exposure categories.

    These 20 categories drive the Stage 1 and Stage 2 prudential
    provisioning floors per Paragraph 82.

    Reference: RBI/DOR/2026-27/398 Paragraph 82.
    """

    SECURED_RETAIL = "secured_retail"
    """Category (i): Secured retail loans (excluding housing, gold)."""

    CORPORATE = "corporate"
    """Category (ii): Corporate loans."""

    SMALL_MICRO_ENTERPRISE = "small_micro_enterprise"
    """Category (iii): Small and micro enterprises."""

    MEDIUM_ENTERPRISE = "medium_enterprise"
    """Category (iv): Medium enterprises."""

    FARM_CREDIT_AGRICULTURAL = "farm_credit_agricultural"
    """Category (v): Farm credit and agricultural activities."""

    BANKS_NBFCS_REGULATED_FIS = "banks_nbfcs_regulated_fis"
    """Category (vi): Loans to banks, NBFCs, regulated financial institutions."""

    LOANS_AGAINST_DEPOSITS_LIC_KVP = "loans_against_deposits_lic_kvp"
    """Category (vii): Loans against term deposits, LIC policy, KVP."""

    GOLD_LOANS = "gold_loans"
    """Category (viii): Gold loans."""

    STATE_GOVT_GUARANTEED = "state_govt_guaranteed"
    """Category (ix): State government direct/guaranteed exposures."""

    UNSECURED_RETAIL = "unsecured_retail"
    """Category (x): Unsecured retail loans (incl. credit cards, personal loans)."""

    HOUSING_LOANS_INDIVIDUALS = "housing_loans_individuals"
    """Category (xi.a): Housing loans to individuals."""

    CRE_ADC_150 = "cre_adc_150"
    """Category (xi.b.i): Commercial Real Estate ADC with 150% risk weight."""

    CRE_RH_ADC = "cre_rh_adc"
    """Category (xi.b.ii): Commercial Real Estate Residential Housing ADC."""

    OTHER_RESIDENTIAL_RE = "other_residential_re"
    """Category (xi.c): Other residential real estate claims."""

    OTHER_COMMERCIAL_RE = "other_commercial_re"
    """Category (xi.d): Other commercial real estate claims."""

    PROJECT_FINANCE_PRE_OPERATIONAL = "project_finance_pre_operational"
    """Category (xii.a): Project finance in pre-operational phase."""

    PROJECT_FINANCE_OPERATIONAL = "project_finance_operational"
    """Category (xii.b): Project finance in operational phase."""

    CENTRAL_GOVT_GUARANTEED = "central_govt_guaranteed"
    """Category (xiii): Central government guaranteed (CGTMSE, CRGFTLIH, NCGTC)."""

    NATURAL_CALAMITY_RESTRUCTURED = "natural_calamity_restructured"
    """Category (xiv): Natural calamity restructured (standard)."""

    OTHER = "other"
    """Category (xv): All other loan products."""


class RBICollateralCategory(StrEnum):
    """RBI ECL Master Direction 2026 collateral categories.

    Used for Stage 3 duration-dependent floor lookup and LGD backstop
    assignment per Paragraphs 82, 97-98.
    """

    ELIGIBLE_COLLATERAL = "eligible_collateral"
    """Cash, gold, government securities, LIC policy, KVP, NSC.
    Attracts 30% LGD backstop (Paragraph 98)."""

    DEPOSITS_LIC_GOLD_STATE_GOVT = "deposits_lic_gold_state_govt"
    """Lower Stage 3 floors apply (10/25, 20/100, ...).
    Covers Categories (vii), (viii), (ix)."""

    HOUSING_RESIDENTIAL_RE = "housing_residential_re"
    """Housing and residential real estate Stage 3 floors."""

    STANDARD_SECURED = "standard_secured"
    """Standard Stage 3 secured floors (25/40/55/75/100)."""

    UNSECURED = "unsecured"
    """Unsecured exposures — Stage 3 floors 25% first year, 100% after."""
