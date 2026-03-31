"""Revolving credit ECL types and enumerations.

Defines product types, CCF methodology choices, and behavioral life
estimation methods specific to revolving credit facilities under IFRS 9.

References:
    - IFRS 9 paragraphs 5.5.20, B5.5.39-40
    - BCBS d424 (Basel III final reforms, December 2017)
    - CRR3 (EU Regulation 2024/1623)
"""

from __future__ import annotations

from enum import StrEnum


class RevolvingProductType(StrEnum):
    """Revolving credit product classification.

    Determines default behavioral life, CCF ranges, and whether
    the facility falls under IFRS 9 paragraph 5.5.20 (collectively
    managed) or 5.5.19 (individually managed).
    """

    CREDIT_CARD = "credit_card"
    OVERDRAFT = "overdraft"
    HELOC = "heloc"
    CORPORATE_REVOLVER = "corporate_revolver"
    WORKING_CAPITAL = "working_capital"
    MARGIN_LENDING = "margin_lending"


class CCFMethod(StrEnum):
    """Credit Conversion Factor estimation methodology.

    Regulatory approaches use prescribed values; behavioral approaches
    use bank-estimated PIT CCFs from internal data.

    References:
        - BCBS d424 CRE32.29-32.32 (supervisory CCFs)
        - CRR3 Art. 166(8b) (A-IRB own-estimate restriction)
    """

    REGULATORY_SA = "regulatory_sa"
    REGULATORY_FIRB = "regulatory_firb"
    BEHAVIORAL = "behavioral"
    EADF = "eadf"


class BehavioralLifeMethod(StrEnum):
    """Method for determining the exposure period per B5.5.40.

    SURVIVAL_ANALYSIS: Uses account-level survival curves.
    HISTORICAL_AVERAGE: Uses average observed life on similar instruments.
    FIXED: Uses a predetermined fixed period (e.g., from product config).
    """

    SURVIVAL_ANALYSIS = "survival_analysis"
    HISTORICAL_AVERAGE = "historical_average"
    FIXED = "fixed"
