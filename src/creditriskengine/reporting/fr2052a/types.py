"""FR 2052a enumeration types and constants.

Every enum maps directly to the field value lists defined in the
FR 2052a Instructions (April 2022) and Appendices I--V.

References:
    - FR 2052a Instructions, Field Definitions (pp. 16--32)
    - Appendix III: Asset Category Table (pp. 98--100)
    - Appendix IV-a: Maturity Bucket Value List (pp. 101--102)
"""

from __future__ import annotations

from enum import StrEnum

# ---------------------------------------------------------------------------
# Reporter / Submission
# ---------------------------------------------------------------------------

class ReporterCategory(StrEnum):
    """Banking organisation category per 12 CFR 252.5 / 238.10.

    Determines reporting frequency (daily vs. monthly) and timing
    (T+2 vs. T+10).
    """

    CATEGORY_I = "category_i"
    CATEGORY_II = "category_ii"
    CATEGORY_III_DAILY = "category_iii_daily"
    CATEGORY_III_MONTHLY = "category_iii_monthly"
    CATEGORY_IV = "category_iv"
    FBO_CATEGORY_II = "fbo_category_ii"
    FBO_CATEGORY_III_DAILY = "fbo_category_iii_daily"
    FBO_CATEGORY_III_MONTHLY = "fbo_category_iii_monthly"
    FBO_CATEGORY_IV = "fbo_category_iv"


# ---------------------------------------------------------------------------
# FR 2052a Tables (top-level schedule groupings)
# ---------------------------------------------------------------------------

class FR2052aTable(StrEnum):
    """Top-level schedule tables in the FR 2052a report."""

    INFLOWS_ASSETS = "I.A"
    INFLOWS_UNSECURED = "I.U"
    INFLOWS_SECURED = "I.S"
    INFLOWS_OTHER = "I.O"
    OUTFLOWS_WHOLESALE = "O.W"
    OUTFLOWS_SECURED = "O.S"
    OUTFLOWS_DEPOSITS = "O.D"
    OUTFLOWS_OTHER = "O.O"
    SUPPLEMENTAL_DC = "S.DC"
    SUPPLEMENTAL_LRM = "S.L"
    SUPPLEMENTAL_BALANCE_SHEET = "S.B"
    SUPPLEMENTAL_INFORMATIONAL = "S.I"
    SUPPLEMENTAL_FX = "S.FX"


# ---------------------------------------------------------------------------
# Currency
# ---------------------------------------------------------------------------

class FR2052aCurrency(StrEnum):
    """Reportable currency codes per FR 2052a instructions (p. 16)."""

    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    CHF = "CHF"
    JPY = "JPY"
    AUD = "AUD"
    CAD = "CAD"


# ---------------------------------------------------------------------------
# Counterparty Types (pp. 17--21)
# ---------------------------------------------------------------------------

class CounterpartyType(StrEnum):
    """Counterparty classification per FR 2052a instructions.

    Definitions align with the legal counterparty to a given exposure
    (not the counterparty's ultimate parent), per FR 2052a p. 17.
    """

    RETAIL = "Retail"
    SMALL_BUSINESS = "Small Business"
    NON_FINANCIAL_CORPORATE = "Non-Financial Corporate"
    SOVEREIGN = "Sovereign"
    CENTRAL_BANK = "Central Bank"
    GSE = "Government Sponsored Entity"
    PSE = "Public Sector Entity"
    MDB = "Multilateral Development Bank"
    OTHER_SUPRANATIONAL = "Other Supranational"
    PENSION_FUND = "Pension Fund"
    BANK = "Bank"
    BROKER_DEALER = "Broker-Dealer"
    INVESTMENT_COMPANY_OR_ADVISOR = "Investment Company or Advisor"
    FINANCIAL_MARKET_UTILITY = "Financial Market Utility"
    OTHER_SUPERVISED_NON_BANK = "Other Supervised Non-Bank Financial Entity"
    DEBT_ISSUING_SPE = "Debt Issuing Special Purpose Entity"
    NON_REGULATED_FUND = "Non-Regulated Fund"
    MUNICIPALITIES_VRDN = "Municipalities for VRDN Structures"
    OTHER = "Other"


# ---------------------------------------------------------------------------
# Asset Category / Collateral Class (Appendix III, pp. 98--100)
# ---------------------------------------------------------------------------

class AssetCategory(StrEnum):
    """Asset category codes from FR 2052a Appendix III.

    The ``-Q`` suffix indicates assets meeting all asset-specific tests
    in section 20 of Regulation WW.
    """

    # HQLA Level 1
    A_0_Q = "A-0-Q"    # Cash
    A_1_Q = "A-1-Q"    # US Treasury debt
    A_2_Q = "A-2-Q"    # US Govt Agency debt (excl. Treasury) with guarantee
    A_3_Q = "A-3-Q"    # Vanilla debt (incl. pass-through MBS) guaranteed
    A_4_Q = "A-4-Q"    # Structured debt (excl. pass-through MBS) guaranteed
    A_5_Q = "A-5-Q"    # Other debt with US Govt guarantee
    S_1_Q = "S-1-Q"    # Non-US Sovereign debt 0% RW
    S_2_Q = "S-2-Q"    # MDB/supranational debt 0% RW
    S_3_Q = "S-3-Q"    # Debt guaranteed by 0% RW sovereign/MDB
    S_4_Q = "S-4-Q"    # Non-US sovereign debt (home currency/jurisdiction)
    CB_1_Q = "CB-1-Q"  # Central bank securities 0% RW
    CB_2_Q = "CB-2-Q"  # Non-US central bank (home currency/jurisdiction)

    # HQLA Level 2a
    G_1_Q = "G-1-Q"    # Senior/preferred GSE debt
    G_2_Q = "G-2-Q"    # Vanilla GSE-guaranteed debt (incl. pass-through MBS)
    G_3_Q = "G-3-Q"    # Structured GSE-guaranteed debt (excl. pass-through)
    S_5_Q = "S-5-Q"    # Non-US sovereign 20% RW
    S_6_Q = "S-6-Q"    # MDB/supranational 20% RW
    S_7_Q = "S-7-Q"    # Debt guaranteed by 20% RW sovereign/MDB
    CB_3_Q = "CB-3-Q"  # Central bank securities 20% RW

    # HQLA Level 2b
    E_1_Q = "E-1-Q"    # US equities (Russell 1000)
    E_2_Q = "E-2-Q"    # Non-US qualifying equities
    IG_1_Q = "IG-1-Q"  # Investment grade corporate debt
    IG_2_Q = "IG-2-Q"  # Investment grade municipal obligations

    # Non-HQLA (same asset types without -Q qualification)
    A_2 = "A-2"
    A_3 = "A-3"
    A_4 = "A-4"
    A_5 = "A-5"
    S_1 = "S-1"
    S_2 = "S-2"
    S_3 = "S-3"
    S_4 = "S-4"
    CB_1 = "CB-1"
    CB_2 = "CB-2"
    G_1 = "G-1"
    G_2 = "G-2"
    G_3 = "G-3"
    S_5 = "S-5"
    S_6 = "S-6"
    S_7 = "S-7"
    CB_3 = "CB-3"
    E_1 = "E-1"
    E_2 = "E-2"
    IG_1 = "IG-1"
    IG_2 = "IG-2"

    # Non-HQLA other
    S_8 = "S-8"     # All other sovereign/supranational debt
    CB_4 = "CB-4"   # All other central bank securities
    G_4 = "G-4"     # Non-senior/preferred GSE debt
    E_3 = "E-3"     # All other US common equity
    E_4 = "E-4"     # All other non-US common equity
    E_5 = "E-5"     # ETFs (US exchanges)
    E_6 = "E-6"     # ETFs (non-US exchanges)
    E_7 = "E-7"     # US mutual fund shares
    E_8 = "E-8"     # Non-US mutual fund shares
    E_9 = "E-9"     # All other US equity investments
    E_10 = "E-10"   # All other non-US equity investments
    IG_3 = "IG-3"   # IG Vanilla ABS
    IG_4 = "IG-4"   # IG Structured ABS
    IG_5 = "IG-5"   # IG Private label pass-through CMBS/RMBS
    IG_6 = "IG-6"   # IG Private label structured CMBS/RMBS
    IG_7 = "IG-7"   # IG Covered bonds
    IG_8 = "IG-8"   # IG Municipal/PSE obligations
    N_1 = "N-1"     # Non-IG US municipal general obligations
    N_2 = "N-2"     # Non-IG corporate debt
    N_3 = "N-3"     # Non-IG Vanilla ABS
    N_4 = "N-4"     # Non-IG Structured ABS
    N_5 = "N-5"     # Non-IG Private label pass-through CMBS/RMBS
    N_6 = "N-6"     # Non-IG Private label structured CMBS/RMBS
    N_7 = "N-7"     # Non-IG covered bonds
    N_8 = "N-8"     # Non-IG municipal/PSE obligations
    L_1 = "L-1"     # GSE-eligible conforming residential mortgages
    L_2 = "L-2"     # Other GSE-eligible loans
    L_3 = "L-3"     # Other 1-4 family residential mortgages
    L_4 = "L-4"     # Other multi-family residential mortgages
    L_5 = "L-5"     # Home equity loans
    L_6 = "L-6"     # Credit card loans
    L_7 = "L-7"     # Auto loans and leases
    L_8 = "L-8"     # Other consumer loans and leases
    L_9 = "L-9"     # Commercial real estate loans
    L_10 = "L-10"   # Commercial and industrial loans
    L_11 = "L-11"   # All other loans (excl. govt-guaranteed)
    L_12 = "L-12"   # Loans guaranteed by US government agencies
    Y_1 = "Y-1"     # Debt issued by reporting firm -- parent
    Y_2 = "Y-2"     # Debt issued by reporting firm -- bank
    Y_3 = "Y-3"     # Debt issued by reporting firm -- all other
    Y_4 = "Y-4"     # Equity investment in affiliates
    C_1 = "C-1"     # Commodities
    P_1 = "P-1"     # Residential property
    P_2 = "P-2"     # All other physical property
    LC_1 = "LC-1"   # Letters of credit issued by a GSE
    LC_2 = "LC-2"   # All other letters of credit / bankers' acceptances
    Z_1 = "Z-1"     # All other assets


# Convenience sets for HQLA classification
HQLA_LEVEL_1: frozenset[AssetCategory] = frozenset({
    AssetCategory.A_0_Q, AssetCategory.A_1_Q, AssetCategory.A_2_Q,
    AssetCategory.A_3_Q, AssetCategory.A_4_Q, AssetCategory.A_5_Q,
    AssetCategory.S_1_Q, AssetCategory.S_2_Q, AssetCategory.S_3_Q,
    AssetCategory.S_4_Q, AssetCategory.CB_1_Q, AssetCategory.CB_2_Q,
})

HQLA_LEVEL_2A: frozenset[AssetCategory] = frozenset({
    AssetCategory.G_1_Q, AssetCategory.G_2_Q, AssetCategory.G_3_Q,
    AssetCategory.S_5_Q, AssetCategory.S_6_Q, AssetCategory.S_7_Q,
    AssetCategory.CB_3_Q,
})

HQLA_LEVEL_2B: frozenset[AssetCategory] = frozenset({
    AssetCategory.E_1_Q, AssetCategory.E_2_Q,
    AssetCategory.IG_1_Q, AssetCategory.IG_2_Q,
})


def hqla_level(category: AssetCategory) -> str | None:
    """Return the HQLA level ('1', '2a', '2b') or None if non-HQLA."""
    if category in HQLA_LEVEL_1:
        return "1"
    if category in HQLA_LEVEL_2A:
        return "2a"
    if category in HQLA_LEVEL_2B:
        return "2b"
    return None


# ---------------------------------------------------------------------------
# Maturity Buckets (Appendix IV-a, pp. 101--102)
# ---------------------------------------------------------------------------

class MaturityBucket(StrEnum):
    """Maturity bucket values per FR 2052a Appendix IV-a.

    Day 1--60 are individual daily buckets.  Beyond 60 days the buckets
    widen to weekly, monthly, and annual ranges.
    """

    OPEN = "Open"
    DAY_1 = "Day 1"
    DAY_2 = "Day 2"
    DAY_3 = "Day 3"
    DAY_4 = "Day 4"
    DAY_5 = "Day 5"
    DAY_6 = "Day 6"
    DAY_7 = "Day 7"
    DAY_8 = "Day 8"
    DAY_9 = "Day 9"
    DAY_10 = "Day 10"
    DAY_11 = "Day 11"
    DAY_12 = "Day 12"
    DAY_13 = "Day 13"
    DAY_14 = "Day 14"
    DAY_15 = "Day 15"
    DAY_16 = "Day 16"
    DAY_17 = "Day 17"
    DAY_18 = "Day 18"
    DAY_19 = "Day 19"
    DAY_20 = "Day 20"
    DAY_21 = "Day 21"
    DAY_22 = "Day 22"
    DAY_23 = "Day 23"
    DAY_24 = "Day 24"
    DAY_25 = "Day 25"
    DAY_26 = "Day 26"
    DAY_27 = "Day 27"
    DAY_28 = "Day 28"
    DAY_29 = "Day 29"
    DAY_30 = "Day 30"
    DAY_31 = "Day 31"
    DAY_32 = "Day 32"
    DAY_33 = "Day 33"
    DAY_34 = "Day 34"
    DAY_35 = "Day 35"
    DAY_36 = "Day 36"
    DAY_37 = "Day 37"
    DAY_38 = "Day 38"
    DAY_39 = "Day 39"
    DAY_40 = "Day 40"
    DAY_41 = "Day 41"
    DAY_42 = "Day 42"
    DAY_43 = "Day 43"
    DAY_44 = "Day 44"
    DAY_45 = "Day 45"
    DAY_46 = "Day 46"
    DAY_47 = "Day 47"
    DAY_48 = "Day 48"
    DAY_49 = "Day 49"
    DAY_50 = "Day 50"
    DAY_51 = "Day 51"
    DAY_52 = "Day 52"
    DAY_53 = "Day 53"
    DAY_54 = "Day 54"
    DAY_55 = "Day 55"
    DAY_56 = "Day 56"
    DAY_57 = "Day 57"
    DAY_58 = "Day 58"
    DAY_59 = "Day 59"
    DAY_60 = "Day 60"
    DAYS_61_67 = "61 - 67 Days"
    DAYS_68_74 = "68 - 74 Days"
    DAYS_75_82 = "75 - 82 Days"
    DAYS_83_90 = "83 - 90 Days"
    DAYS_91_120 = "91 - 120 Days"
    DAYS_121_150 = "121 - 150 Days"
    DAYS_151_179 = "151 - 179 Days"
    DAYS_180_270 = "180 - 270 Days"
    DAYS_271_364 = "271 - 364 Days"
    YR_1_2 = ">= 1 Yr <= 2 Yr"
    YR_2_3 = ">2 Yr <= 3 Yr"
    YR_3_4 = ">3 Yr <= 4 Yr"
    YR_4_5 = ">4 Yr <= 5 Yr"
    YR_GT_5 = ">5 Yr"
    PERPETUAL = "Perpetual"


def maturity_bucket_midpoint_days(bucket: MaturityBucket) -> float | None:
    """Return approximate midpoint in calendar days for a maturity bucket.

    Returns ``None`` for Open and Perpetual (no finite horizon).
    """
    special_map: dict[MaturityBucket, float] = {
        MaturityBucket.OPEN: 0.0,
        MaturityBucket.PERPETUAL: float("inf"),
    }
    if bucket in special_map:
        return None
    name = bucket.value
    if name.startswith("Day "):
        return float(name.split()[-1])
    range_map: dict[str, float] = {
        "61 - 67 Days": 64.0,
        "68 - 74 Days": 71.0,
        "75 - 82 Days": 78.5,
        "83 - 90 Days": 86.5,
        "91 - 120 Days": 105.5,
        "121 - 150 Days": 135.5,
        "151 - 179 Days": 165.0,
        "180 - 270 Days": 225.0,
        "271 - 364 Days": 317.5,
        ">= 1 Yr <= 2 Yr": 547.5,
        ">2 Yr <= 3 Yr": 912.5,
        ">3 Yr <= 4 Yr": 1277.5,
        ">4 Yr <= 5 Yr": 1642.5,
        ">5 Yr": 2190.0,
    }
    return range_map.get(name)


# ---------------------------------------------------------------------------
# Settlement Types
# ---------------------------------------------------------------------------

class SecuredSettlement(StrEnum):
    """Settlement type for Inflows-Secured / Outflows-Secured (p. 26)."""

    FICC = "FICC"
    TRIPARTY = "Triparty"
    OTHER = "Other"
    BILATERAL = "Bilateral"


class FXSettlement(StrEnum):
    """Settlement type for Supplemental-Foreign Exchange (p. 26)."""

    CLS = "CLS"
    OTHER = "Other"
    BILATERAL = "Bilateral"


# ---------------------------------------------------------------------------
# Insured Type (p. 27)
# ---------------------------------------------------------------------------

class InsuredType(StrEnum):
    """Deposit insurance status for Outflows-Deposits."""

    FDIC = "FDIC"
    OTHER = "Other"
    UNINSURED = "Uninsured"


# ---------------------------------------------------------------------------
# Encumbrance Type (pp. 28--29)
# ---------------------------------------------------------------------------

class EncumbranceType(StrEnum):
    """Encumbrance type for assets."""

    SECURITIES_FINANCING = "Securities Financing Transaction"
    DERIVATIVE_VM = "Derivative VM"
    DERIVATIVE_IM_DFC = "Derivative IM and DFC"
    OTHER_IM_DFC = "Other IM and DFC"
    SEGREGATED_CUSTOMER = "Segregated for Customer Protection"
    COVERED_FRB_FACILITY = "Covered Federal Reserve Facility Funding"
    OTHER = "Other"


# ---------------------------------------------------------------------------
# Collateral Level (p. 29)
# ---------------------------------------------------------------------------

class CollateralLevel(StrEnum):
    """Collateral level for Supplemental-Derivatives & Collateral."""

    UNCOLLATERALIZED = "Uncollateralized"
    UNDERCOLLATERALIZED = "Undercollateralized"
    FULLY_COLLATERALIZED = "Fully Collateralized"
    OVERCOLLATERALIZED = "Overcollateralized"


# ---------------------------------------------------------------------------
# Accounting Designation (p. 30)
# ---------------------------------------------------------------------------

class AccountingDesignation(StrEnum):
    """Accounting designation for Inflows-Assets."""

    AFS = "Available-for-Sale"
    HTM = "Held-to-Maturity"
    TRADING = "Trading Asset"
    NOT_APPLICABLE = "Not Applicable"


# ---------------------------------------------------------------------------
# Loss Absorbency (p. 30)
# ---------------------------------------------------------------------------

class LossAbsorbency(StrEnum):
    """Loss absorbency designation for Outflows-Wholesale."""

    CAPITAL = "Capital"
    TLAC = "TLAC"


# ---------------------------------------------------------------------------
# Maturity Optionality (pp. 31--32)
# ---------------------------------------------------------------------------

class MaturityOptionality(StrEnum):
    """Embedded optionality type for maturity."""

    EVERGREEN = "Evergreen"
    EXTENDIBLE = "Extendible"
    ACCELERATED_COUNTERPARTY = "Accelerated-Counterparty"
    ACCELERATED_FIRM = "Accelerated-Firm"
    NOT_ACCELERATED = "Not Accelerated"


# ---------------------------------------------------------------------------
# Sub-product values for specific products
# ---------------------------------------------------------------------------

class CapacitySubProduct(StrEnum):
    """Sub-product for I.A.2: Capacity."""

    FRB = "FRB"
    SNB = "SNB"
    BOE = "BOE"
    ECB = "ECB"
    BOJ = "BOJ"
    RBA = "RBA"
    BOC = "BOC"
    OCB = "OCB"
    FHLB = "FHLB"
    OTHER_GSE = "Other GSE"


class ReserveSubProduct(StrEnum):
    """Sub-product for I.A.3/I.A.4: Reserve Balances."""

    FRB = "FRB"
    SNB = "SNB"
    BOE = "BOE"
    ECB = "ECB"
    BOJ = "BOJ"
    RBA = "RBA"
    BOC = "BOC"
    OCB = "OCB"
    CURRENCY_AND_COIN = "Currency and Coin"


class CollateralSwapSubProduct(StrEnum):
    """Sub-product for I.S.4 / O.S.4: Collateral Swaps."""

    # Inflow side (pledged)
    LEVEL_1_PLEDGED = "Level 1 Pledged"
    LEVEL_2A_PLEDGED = "Level 2a Pledged"
    LEVEL_2B_PLEDGED = "Level 2b Pledged"
    NON_HQLA_PLEDGED = "Non-HQLA Pledged"
    NO_COLLATERAL_PLEDGED = "No Collateral Pledged"
    # Outflow side (received)
    LEVEL_1_RECEIVED = "Level 1 Received"
    LEVEL_2A_RECEIVED = "Level 2a Received"
    LEVEL_2B_RECEIVED = "Level 2b Received"
    NON_HQLA_RECEIVED = "Non-HQLA Received"
    NO_COLLATERAL_RECEIVED = "No Collateral Received"


class ExceptionalCBSubProduct(StrEnum):
    """Sub-product for O.S.6: Exceptional Central Bank Operations."""

    FRB = "FRB"
    SNB = "SNB"
    BOE = "BOE"
    ECB = "ECB"
    BOJ = "BOJ"
    RBA = "RBA"
    BOC = "BOC"
    OCB = "OCB"
    COVERED_FRB_FACILITY = "Covered Federal Reserve Facility Funding"


class CustomerShortsSubProduct(StrEnum):
    """Sub-product for O.S.7: Customer Shorts."""

    EXTERNAL_CASH = "External Cash Transactions"
    EXTERNAL_NON_CASH = "External Non-Cash Transactions"


class FirmShortsSubProduct(StrEnum):
    """Sub-product for O.S.8: Firm Shorts."""

    FIRM_LONGS = "Firm Longs"
    CUSTOMER_LONGS = "Customer Longs"
    UNSETTLED_REGULAR = "Unsettled - Regular Way"
    UNSETTLED_FORWARD = "Unsettled - Forward"


class SyntheticSubProduct(StrEnum):
    """Sub-product for I.S.9/10 and O.S.9/10: Synthetic positions."""

    PHYSICAL_LONG = "Physical Long Position"
    FIRM_SHORT = "Firm Short"
    SYNTHETIC_CUSTOMER_LONG = "Synthetic Customer Long"
    SYNTHETIC_CUSTOMER_SHORT = "Synthetic Customer Short"
    SYNTHETIC_FIRM_FINANCING = "Synthetic Firm Financing"
    SYNTHETIC_FIRM_SOURCING = "Synthetic Firm Sourcing"
    FUTURES = "Futures"
    OTHER = "Other"
    UNHEDGED = "Unhedged"


class DCSubProduct(StrEnum):
    """Sub-product for S.DC: Derivatives & Collateral."""

    REHYP_UNENCUMBERED = "Rehypothecatable Collateral Unencumbered"
    REHYP_ENCUMBERED = "Rehypothecatable Collateral Encumbered"
    NON_REHYP = "Non-Rehypothecatable Collateral"
    SEGREGATED_CASH = "Segregated Cash"
    NON_SEGREGATED_CASH = "Non-Segregated Cash"


class DCSubProduct2(StrEnum):
    """Sub-product 2 for S.DC: Derivatives & Collateral clearing type."""

    OTC_BILATERAL = "OTC - Bilateral"
    OTC_CENTRALIZED_PRINCIPAL = "OTC - Centralized (Principal)"
    OTC_CENTRALIZED_AGENT = "OTC - Centralized (Agent)"
    EXCHANGE_TRADED_PRINCIPAL = "Exchange-traded (Principal)"
    EXCHANGE_TRADED_AGENT = "Exchange-traded (Agent)"
