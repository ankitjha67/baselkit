"""FR 2052a Pydantic data models for each schedule table.

Each model maps to one row in its respective FR 2052a schedule.
Fields marked as ``None``-able are optional depending on the
specific product being reported (see :mod:`products` for
per-product required field rules).

References:
    - FR 2052a Instructions, Appendix I: Data Format, Tables, and Fields
    - FR 2052a Instructions, Field Definitions (pp. 16--32)
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from creditriskengine.reporting.fr2052a.types import (
    AccountingDesignation,
    AssetCategory,
    CollateralLevel,
    CounterpartyType,
    EncumbranceType,
    FR2052aCurrency,
    FR2052aTable,
    FXSettlement,
    InsuredType,
    LossAbsorbency,
    MaturityBucket,
    MaturityOptionality,
    SecuredSettlement,
)


class FR2052aRecord(BaseModel):
    """Base record shared by all FR 2052a schedule rows.

    Every row in every table shares these common fields.
    Product-specific fields are added in the table subclasses.
    """

    reporting_entity: str = Field(
        ..., min_length=1, description="Legal entity name."
    )
    as_of_date: str = Field(
        ..., description="As-of date in ISO format (YYYY-MM-DD)."
    )
    table: FR2052aTable = Field(
        ..., description="Schedule table code (e.g. I.A, O.W)."
    )
    product_id: int = Field(
        ..., ge=1, description="Product identifier within the table."
    )
    product_name: str = Field(
        default="", description="Product name (informational)."
    )
    currency: FR2052aCurrency | None = Field(
        default=None,
        description="Currency code.  May be None for USD-only reporters.",
    )
    converted: bool = Field(
        default=False,
        description="True if values have been converted to USD-equivalent.",
    )
    maturity_bucket: MaturityBucket | None = Field(
        default=None, description="Contractual maturity bucket."
    )
    maturity_amount: float = Field(
        default=0.0,
        description="Notional amount at maturity (in millions).",
    )
    forward_start_bucket: MaturityBucket | None = Field(
        default=None, description="Forward start settlement bucket."
    )
    forward_start_amount: float | None = Field(
        default=None,
        description="Notional on forward settlement date (in millions).",
    )
    internal: bool = Field(
        default=False,
        description="True for inter-affiliate transactions.",
    )
    internal_counterparty: str | None = Field(
        default=None, description="Internal counterparty entity name."
    )
    business_line: str | None = Field(
        default=None, description="Business line responsible."
    )
    gsib: str | None = Field(
        default=None,
        description="G-SIB name per FSB list, if counterparty is a G-SIB.",
    )

    model_config = {"extra": "forbid"}


# -----------------------------------------------------------------------
# Inflows -- Assets (I.A)
# -----------------------------------------------------------------------

class InflowAssetRecord(FR2052aRecord):
    """Row in the Inflows-Assets table (I.A).

    Products: I.A.1 (Unencumbered Assets) through I.A.7 (Encumbered Assets).
    """

    table: FR2052aTable = FR2052aTable.INFLOWS_ASSETS
    collateral_class: AssetCategory | None = Field(
        default=None, description="Asset category per Appendix III."
    )
    market_value: float = Field(
        default=0.0, description="Fair value (GAAP) in millions."
    )
    lendable_value: float | None = Field(
        default=None,
        description="Value obtainable in secured funding markets.",
    )
    treasury_control: bool = Field(
        default=False,
        description="True if under liquidity management control (HQLA).",
    )
    effective_maturity_bucket: MaturityBucket | None = Field(
        default=None, description="Remaining encumbrance period."
    )
    accounting_designation: AccountingDesignation | None = Field(
        default=None, description="AFS / HTM / Trading / Not Applicable."
    )
    encumbrance_type: EncumbranceType | None = Field(
        default=None, description="Type of encumbrance, if encumbered."
    )
    sub_product: str | None = Field(
        default=None, description="Sub-product value."
    )
    collateral_value: float | None = Field(
        default=None, description="Fair value of collateral."
    )


# -----------------------------------------------------------------------
# Inflows -- Unsecured (I.U)
# -----------------------------------------------------------------------

class InflowUnsecuredRecord(FR2052aRecord):
    """Row in the Inflows-Unsecured table (I.U).

    Products: I.U.1 (Onshore Placements) through I.U.8 (Short-Term
    Investments).
    """

    table: FR2052aTable = FR2052aTable.INFLOWS_UNSECURED
    counterparty: CounterpartyType | None = Field(
        default=None, description="Borrower type."
    )
    risk_weight: float | None = Field(
        default=None, ge=0.0,
        description="Standardized risk weight per 12 CFR 217 subpart D.",
    )
    effective_maturity_bucket: MaturityBucket | None = Field(
        default=None, description="Remaining encumbrance period."
    )
    encumbrance_type: EncumbranceType | None = Field(
        default=None, description="Type of encumbrance."
    )
    maturity_optionality: MaturityOptionality | None = Field(
        default=None, description="Embedded optionality type."
    )


# -----------------------------------------------------------------------
# Inflows -- Secured (I.S)
# -----------------------------------------------------------------------

class InflowSecuredRecord(FR2052aRecord):
    """Row in the Inflows-Secured table (I.S).

    Products: I.S.1 (Reverse Repo) through I.S.10 (Synthetic Firm Sourcing).
    """

    table: FR2052aTable = FR2052aTable.INFLOWS_SECURED
    counterparty: CounterpartyType | None = Field(
        default=None, description="Borrower/counterparty type."
    )
    collateral_class: AssetCategory | None = Field(
        default=None, description="Asset category of collateral received."
    )
    collateral_value: float | None = Field(
        default=None, description="Fair value of collateral (in millions)."
    )
    settlement: SecuredSettlement | None = Field(
        default=None, description="Settlement type."
    )
    unencumbered: bool = Field(
        default=False,
        description="True if collateral held unencumbered.",
    )
    treasury_control: bool = Field(
        default=False,
        description="True if under liquidity management control.",
    )
    effective_maturity_bucket: MaturityBucket | None = Field(
        default=None, description="Remaining encumbrance period."
    )
    risk_weight: float | None = Field(
        default=None, ge=0.0,
        description="Standardized risk weight.",
    )
    encumbrance_type: EncumbranceType | None = Field(
        default=None, description="Type of encumbrance."
    )
    sub_product: str | None = Field(
        default=None, description="Sub-product value."
    )
    maturity_optionality: MaturityOptionality | None = Field(
        default=None, description="Embedded optionality type."
    )


# -----------------------------------------------------------------------
# Inflows -- Other (I.O)
# -----------------------------------------------------------------------

class InflowOtherRecord(FR2052aRecord):
    """Row in the Inflows-Other table (I.O).

    Products: I.O.1 (Derivative Receivables) through I.O.9 (Other Cash
    Inflows).
    """

    table: FR2052aTable = FR2052aTable.INFLOWS_OTHER
    counterparty: CounterpartyType | None = Field(
        default=None, description="Counterparty type."
    )
    collateral_class: AssetCategory | None = Field(
        default=None, description="Collateral asset category."
    )
    collateral_value: float | None = Field(
        default=None, description="Fair value of collateral."
    )
    treasury_control: bool = Field(
        default=False, description="True if under liquidity management."
    )


# -----------------------------------------------------------------------
# Outflows -- Wholesale (O.W)
# -----------------------------------------------------------------------

class OutflowWholesaleRecord(FR2052aRecord):
    """Row in the Outflows-Wholesale table (O.W).

    Products: O.W.1 (ABCP Single-Seller) through O.W.19 (Other Unsecured
    Financing).
    """

    table: FR2052aTable = FR2052aTable.OUTFLOWS_WHOLESALE
    counterparty: CounterpartyType | None = Field(
        default=None, description="Counterparty type."
    )
    collateral_class: AssetCategory | None = Field(
        default=None, description="Collateral securing the obligation."
    )
    collateral_value: float | None = Field(
        default=None, description="Fair value of collateral."
    )
    loss_absorbency: LossAbsorbency | None = Field(
        default=None,
        description="Capital or TLAC per 12 CFR 217/252.",
    )
    maturity_optionality: MaturityOptionality | None = Field(
        default=None, description="Embedded optionality type."
    )


# -----------------------------------------------------------------------
# Outflows -- Secured (O.S)
# -----------------------------------------------------------------------

class OutflowSecuredRecord(FR2052aRecord):
    """Row in the Outflows-Secured table (O.S).

    Products: O.S.1 (Repo) through O.S.11 (Other Secured Financing).
    """

    table: FR2052aTable = FR2052aTable.OUTFLOWS_SECURED
    counterparty: CounterpartyType | None = Field(
        default=None, description="Counterparty type."
    )
    collateral_class: AssetCategory | None = Field(
        default=None, description="Asset category of collateral."
    )
    collateral_value: float | None = Field(
        default=None, description="Fair value of collateral (in millions)."
    )
    settlement: SecuredSettlement | None = Field(
        default=None, description="Settlement type."
    )
    rehypothecated: bool = Field(
        default=False,
        description="True if secured by rehypothecated collateral.",
    )
    treasury_control: bool = Field(
        default=False, description="True if under liquidity management."
    )
    effective_maturity_bucket: MaturityBucket | None = Field(
        default=None, description="Remaining encumbrance period."
    )
    sub_product: str | None = Field(
        default=None, description="Sub-product value."
    )
    maturity_optionality: MaturityOptionality | None = Field(
        default=None, description="Embedded optionality type."
    )


# -----------------------------------------------------------------------
# Outflows -- Deposits (O.D)
# -----------------------------------------------------------------------

class OutflowDepositRecord(FR2052aRecord):
    """Row in the Outflows-Deposits table (O.D).

    Products: O.D.1 (Transactional) through O.D.15 (Other Accounts).
    """

    table: FR2052aTable = FR2052aTable.OUTFLOWS_DEPOSITS
    counterparty: CounterpartyType | None = Field(
        default=None, description="Depositor type."
    )
    insured: InsuredType | None = Field(
        default=None, description="Deposit insurance status."
    )
    trigger: bool = Field(
        default=False,
        description="True if deposit has a trigger provision.",
    )
    rehypothecated: bool = Field(
        default=False, description="True if rehypothecated collateral."
    )
    collateral_class: AssetCategory | None = Field(
        default=None, description="Collateral class, if applicable."
    )
    collateral_value: float | None = Field(
        default=None, description="Fair value of collateral."
    )
    maturity_optionality: MaturityOptionality | None = Field(
        default=None, description="Embedded optionality type."
    )


# -----------------------------------------------------------------------
# Outflows -- Other (O.O)
# -----------------------------------------------------------------------

class OutflowOtherRecord(FR2052aRecord):
    """Row in the Outflows-Other table (O.O).

    Products: O.O.1 (Derivative Payables) through O.O.22 (Other Cash
    Outflows).
    """

    table: FR2052aTable = FR2052aTable.OUTFLOWS_OTHER
    counterparty: CounterpartyType | None = Field(
        default=None, description="Counterparty type."
    )
    collateral_class: AssetCategory | None = Field(
        default=None, description="Collateral class."
    )
    collateral_value: float | None = Field(
        default=None, description="Fair value of collateral."
    )


# -----------------------------------------------------------------------
# Supplemental -- Derivatives & Collateral (S.DC)
# -----------------------------------------------------------------------

class SupplementalDCRecord(FR2052aRecord):
    """Row in the Supplemental-Derivatives & Collateral table (S.DC).

    Products: S.DC.1 through S.DC.21.
    """

    table: FR2052aTable = FR2052aTable.SUPPLEMENTAL_DC
    counterparty: CounterpartyType | None = Field(
        default=None, description="Counterparty type."
    )
    collateral_class: AssetCategory | None = Field(
        default=None, description="Asset category."
    )
    collateral_value: float | None = Field(
        default=None, description="Fair value of collateral."
    )
    market_value: float | None = Field(
        default=None, description="Fair value under GAAP."
    )
    settlement: SecuredSettlement | None = Field(
        default=None, description="Settlement type."
    )
    treasury_control: bool = Field(
        default=False, description="True if under liquidity management."
    )
    effective_maturity_bucket: MaturityBucket | None = Field(
        default=None, description="Remaining encumbrance period."
    )
    encumbrance_type: EncumbranceType | None = Field(
        default=None, description="Type of encumbrance."
    )
    collateral_level: CollateralLevel | None = Field(
        default=None,
        description="Collateralization level for derivative values/VM.",
    )
    netting_eligible: bool | None = Field(
        default=None,
        description="True if VM eligible for netting per LRM sec 107(f)(1).",
    )
    sub_product: str | None = Field(
        default=None, description="Sub-product value."
    )
    sub_product_2: str | None = Field(
        default=None, description="Second sub-product (clearing type)."
    )


# -----------------------------------------------------------------------
# Supplemental -- LRM (S.L)
# -----------------------------------------------------------------------

class SupplementalLRMRecord(FR2052aRecord):
    """Row in the Supplemental-LRM table (S.L).

    Products: S.L.1 through S.L.10.
    """

    table: FR2052aTable = FR2052aTable.SUPPLEMENTAL_LRM
    market_value: float | None = Field(
        default=None, description="Market value."
    )
    collateral_class: AssetCategory | None = Field(
        default=None, description="Asset category."
    )


# -----------------------------------------------------------------------
# Supplemental -- Balance Sheet (S.B)
# -----------------------------------------------------------------------

class SupplementalBalanceSheetRecord(FR2052aRecord):
    """Row in the Supplemental-Balance Sheet table (S.B).

    Products: S.B.1 through S.B.6.
    """

    table: FR2052aTable = FR2052aTable.SUPPLEMENTAL_BALANCE_SHEET
    counterparty: CounterpartyType | None = Field(
        default=None, description="Counterparty type."
    )
    market_value: float | None = Field(
        default=None, description="Fair value."
    )
    risk_weight: float | None = Field(
        default=None, ge=0.0, description="Standardized risk weight."
    )
    collateral_class: AssetCategory | None = Field(
        default=None, description="Asset category."
    )
    effective_maturity_bucket: MaturityBucket | None = Field(
        default=None, description="Remaining encumbrance period."
    )
    collection_reference: str | None = Field(
        default=None,
        description="Table designation for the reported adjustment.",
    )
    product_reference: int | None = Field(
        default=None,
        description="Product ID for the reported adjustment.",
    )
    sub_product_reference: str | None = Field(
        default=None,
        description="Sub-product for the reported adjustment.",
    )


# -----------------------------------------------------------------------
# Supplemental -- Informational (S.I)
# -----------------------------------------------------------------------

class SupplementalInfoRecord(FR2052aRecord):
    """Row in the Supplemental-Informational table (S.I).

    Products: S.I.1 through S.I.6.
    """

    table: FR2052aTable = FR2052aTable.SUPPLEMENTAL_INFORMATIONAL
    market_value: float | None = Field(
        default=None, description="Market value."
    )


# -----------------------------------------------------------------------
# Supplemental -- Foreign Exchange (S.FX)
# -----------------------------------------------------------------------

class SupplementalFXRecord(FR2052aRecord):
    """Row in the Supplemental-Foreign Exchange table (S.FX).

    Products: S.FX.1 (Spot), S.FX.2 (Forwards/Futures), S.FX.3 (Swaps).
    """

    table: FR2052aTable = FR2052aTable.SUPPLEMENTAL_FX
    counterparty: CounterpartyType | None = Field(
        default=None, description="Counterparty type."
    )
    settlement: FXSettlement | None = Field(
        default=None, description="FX settlement type."
    )
    currency_buy: FR2052aCurrency | None = Field(
        default=None, description="Currency to be received."
    )
    currency_sell: FR2052aCurrency | None = Field(
        default=None, description="Currency to be delivered."
    )
    buy_amount: float | None = Field(
        default=None, description="Amount to be received (in millions)."
    )
    sell_amount: float | None = Field(
        default=None, description="Amount to be delivered (in millions)."
    )


# Table code -> Schema model mapping
TABLE_SCHEMA_MAP: dict[FR2052aTable, type[FR2052aRecord]] = {
    FR2052aTable.INFLOWS_ASSETS: InflowAssetRecord,
    FR2052aTable.INFLOWS_UNSECURED: InflowUnsecuredRecord,
    FR2052aTable.INFLOWS_SECURED: InflowSecuredRecord,
    FR2052aTable.INFLOWS_OTHER: InflowOtherRecord,
    FR2052aTable.OUTFLOWS_WHOLESALE: OutflowWholesaleRecord,
    FR2052aTable.OUTFLOWS_SECURED: OutflowSecuredRecord,
    FR2052aTable.OUTFLOWS_DEPOSITS: OutflowDepositRecord,
    FR2052aTable.OUTFLOWS_OTHER: OutflowOtherRecord,
    FR2052aTable.SUPPLEMENTAL_DC: SupplementalDCRecord,
    FR2052aTable.SUPPLEMENTAL_LRM: SupplementalLRMRecord,
    FR2052aTable.SUPPLEMENTAL_BALANCE_SHEET: SupplementalBalanceSheetRecord,
    FR2052aTable.SUPPLEMENTAL_INFORMATIONAL: SupplementalInfoRecord,
    FR2052aTable.SUPPLEMENTAL_FX: SupplementalFXRecord,
}


def record_for_table(table: FR2052aTable, **kwargs: object) -> FR2052aRecord:
    """Construct the appropriate record subclass for a given table.

    Args:
        table: The FR 2052a table code.
        **kwargs: Field values forwarded to the record constructor.

    Returns:
        A validated record instance for the specified table.
    """
    cls = TABLE_SCHEMA_MAP[table]
    return cls(table=table, **kwargs)
