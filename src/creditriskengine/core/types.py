"""
Core type definitions for the credit risk engine.

All enums map directly to regulatory classifications from
BCBS d424 (Basel III final reforms, December 2017).
"""

from enum import Enum, StrEnum


class Jurisdiction(StrEnum):
    """Supported regulatory jurisdictions."""
    BCBS = "bcbs"
    EU = "eu"
    UK = "uk"
    US = "us"
    INDIA = "india"
    SINGAPORE = "singapore"
    HONG_KONG = "hong_kong"
    JAPAN = "japan"
    AUSTRALIA = "australia"
    CANADA = "canada"
    CHINA = "china"
    SOUTH_KOREA = "south_korea"
    UAE = "uae"
    SAUDI_ARABIA = "saudi_arabia"
    SOUTH_AFRICA = "south_africa"
    BRAZIL = "brazil"
    MALAYSIA = "malaysia"


class CreditRiskApproach(StrEnum):
    """Credit risk capital calculation approach.

    Reference: BCBS d424, CRE20 (SA), CRE30-36 (IRB).
    """
    SA = "standardized"
    FIRB = "foundation_irb"
    AIRB = "advanced_irb"


class IRBAssetClass(StrEnum):
    """IRB asset classes per BCBS CRE30.4."""
    CORPORATE = "corporate"
    SOVEREIGN = "sovereign"
    BANK = "bank"
    RETAIL = "retail"
    EQUITY = "equity"


class IRBCorporateSubClass(StrEnum):
    """Corporate sub-classes per BCBS CRE30.5-30.9."""
    GENERAL_CORPORATE = "general_corporate"
    SME_CORPORATE = "sme_corporate"
    SPECIALISED_LENDING = "specialised_lending"


class IRBSpecialisedLendingType(StrEnum):
    """Specialised lending categories per BCBS CRE30.7."""
    PROJECT_FINANCE = "project_finance"
    OBJECT_FINANCE = "object_finance"
    COMMODITIES_FINANCE = "commodities_finance"
    INCOME_PRODUCING_REAL_ESTATE = "ipre"
    HIGH_VOLATILITY_CRE = "hvcre"


class IRBRetailSubClass(StrEnum):
    """Retail sub-classes per BCBS CRE30.11-30.15."""
    RESIDENTIAL_MORTGAGE = "residential_mortgage"
    QRRE = "qualifying_revolving_retail"
    OTHER_RETAIL = "other_retail"
    SME_RETAIL = "sme_retail"


class SAExposureClass(StrEnum):
    """Standardized approach exposure classes per BCBS CRE20."""
    SOVEREIGN = "sovereign"
    PSE = "public_sector_entity"
    MDB = "multilateral_development_bank"
    BANK = "bank"
    SECURITIES_FIRM = "securities_firm"
    CORPORATE = "corporate"
    CORPORATE_SME = "corporate_sme"
    SUBORDINATED_DEBT = "subordinated_debt"
    EQUITY = "equity"
    RETAIL = "retail"
    RETAIL_REGULATORY = "retail_regulatory"
    RESIDENTIAL_MORTGAGE = "residential_mortgage"
    COMMERCIAL_REAL_ESTATE = "commercial_real_estate"
    LAND_ADC = "land_acquisition_development_construction"
    DEFAULTED = "defaulted"
    OTHER = "other"


class CreditQualityStep(int, Enum):
    """Credit quality steps for SA risk weight mapping.

    Maps external ratings to standardized risk weight buckets.
    CQS 1 = AAA/AA-, CQS 2 = A+/A-, etc.
    Reference: BCBS d424, CRE20 Table 1-10.
    """
    CQS_1 = 1
    CQS_2 = 2
    CQS_3 = 3
    CQS_4 = 4
    CQS_5 = 5
    CQS_6 = 6
    UNRATED = 0


class IFRS9Stage(int, Enum):
    """IFRS 9 impairment stages per IFRS 9.5.5.1-5.5.20."""
    STAGE_1 = 1
    STAGE_2 = 2
    STAGE_3 = 3
    POCI = 4


class DefaultDefinition(StrEnum):
    """Default definition frameworks."""
    BCBS = "bcbs_default"
    EBA = "eba_default"
    PRA = "pra_default"
    RBI = "rbi_default"
    APRA = "apra_default"
    MAS = "mas_default"


class CollateralType(StrEnum):
    """Eligible collateral types per BCBS CRE22."""
    CASH = "cash"
    GOLD = "gold"
    DEBT_SECURITIES = "debt_securities"
    EQUITIES = "equities"
    MUTUAL_FUNDS = "mutual_funds"
    RESIDENTIAL_REAL_ESTATE = "rre"
    COMMERCIAL_REAL_ESTATE = "cre_collateral"
    RECEIVABLES = "receivables"
    OTHER_PHYSICAL = "other_physical"
    NETTING = "netting"
    GUARANTEE = "guarantee"
    CREDIT_DERIVATIVE = "credit_derivative"


class CRMApproach(StrEnum):
    """Credit risk mitigation approaches per BCBS CRE22."""
    SIMPLE = "simple"
    COMPREHENSIVE = "comprehensive"
    SUPERVISORY_HAIRCUTS = "supervisory_haircuts"
    OWN_ESTIMATE_HAIRCUTS = "own_estimate_haircuts"
