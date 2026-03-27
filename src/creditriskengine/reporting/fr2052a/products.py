"""FR 2052a product definitions catalog.

Defines all 100+ products across the 13 FR 2052a tables with their
product IDs, names, required/optional fields, sub-product values,
and applicable counterparty types.

References:
    - FR 2052a Instructions, Product Definitions (pp. 33--79)
    - Appendix II-a: Product/Sub-Product Requirements (pp. 86--87)
    - Appendix II-b: Counterparty Requirements (pp. 88--90)
    - Appendix II-c: Collateral Class Requirements (pp. 91--94)
    - Appendix II-d: Forward Start Exclusions (pp. 95--97)
"""

from __future__ import annotations

from dataclasses import dataclass

from creditriskengine.reporting.fr2052a.types import FR2052aTable


@dataclass(frozen=True)
class FR2052aProduct:
    """Metadata for one FR 2052a product.

    Attributes:
        table: The schedule table this product belongs to.
        product_id: Numeric product identifier within the table.
        name: Human-readable product name.
        description: Brief regulatory description.
        sub_products: Allowed sub-product values (empty = no sub-product).
        collateral_required: Whether the collateral class field is required.
        collateral_dependent: Whether collateral is conditionally required.
        counterparty_required: Whether the counterparty field is required.
        forward_start_excluded: Whether forward start fields are excluded.
        regulatory_ref: Regulatory paragraph reference.
    """

    table: FR2052aTable
    product_id: int
    name: str
    description: str = ""
    sub_products: tuple[str, ...] = ()
    collateral_required: bool = False
    collateral_dependent: bool = False
    counterparty_required: bool = True
    forward_start_excluded: bool = False
    regulatory_ref: str = ""

    @property
    def code(self) -> str:
        """Full product code, e.g. 'I.A.1'."""
        return f"{self.table.value}.{self.product_id}"


def _build_products() -> dict[str, FR2052aProduct]:
    """Build the complete product catalog."""
    products: list[FR2052aProduct] = []

    # =================================================================
    # I.A: Inflows -- Assets
    # =================================================================
    tbl_ia = FR2052aTable.INFLOWS_ASSETS

    products.append(FR2052aProduct(
        table=tbl_ia, product_id=1, name="Unencumbered Assets",
        description="Assets owned outright, free of restrictions, not pledged.",
        collateral_required=True, counterparty_required=False,
        forward_start_excluded=True,
        regulatory_ref="FR 2052a p.33",
    ))
    products.append(FR2052aProduct(
        table=tbl_ia, product_id=2, name="Capacity",
        description="Available credit from central banks or GSEs secured by collateral.",
        sub_products=(
            "FRB", "SNB", "BOE", "ECB", "BOJ", "RBA", "BOC", "OCB",
            "FHLB", "Other GSE",
        ),
        collateral_required=True, counterparty_required=False,
        forward_start_excluded=True,
        regulatory_ref="FR 2052a p.33",
    ))
    products.append(FR2052aProduct(
        table=tbl_ia, product_id=3, name="Unrestricted Reserve Balances",
        description="Reserve balances at Federal Reserve / foreign central banks.",
        sub_products=(
            "FRB", "SNB", "BOE", "ECB", "BOJ", "RBA", "BOC", "OCB",
            "Currency and Coin",
        ),
        collateral_required=True, counterparty_required=False,
        forward_start_excluded=True,
        regulatory_ref="FR 2052a p.34",
    ))
    products.append(FR2052aProduct(
        table=tbl_ia, product_id=4, name="Restricted Reserve Balances",
        description="Balances at central banks not immediately withdrawable.",
        sub_products=(
            "FRB", "SNB", "BOE", "ECB", "BOJ", "RBA", "BOC", "OCB",
            "Currency and Coin",
        ),
        collateral_required=True, counterparty_required=False,
        forward_start_excluded=True,
        regulatory_ref="FR 2052a p.35",
    ))
    products.append(FR2052aProduct(
        table=tbl_ia, product_id=5, name="Unsettled Asset Purchases",
        description="Executed but unsettled regular-way security purchases.",
        collateral_required=True, counterparty_required=False,
        regulatory_ref="FR 2052a p.35",
    ))
    products.append(FR2052aProduct(
        table=tbl_ia, product_id=6, name="Forward Asset Purchases",
        description="Executed but unsettled non-regular-way security purchases.",
        collateral_required=True, counterparty_required=False,
        regulatory_ref="FR 2052a p.35",
    ))
    products.append(FR2052aProduct(
        table=tbl_ia, product_id=7, name="Encumbered Assets",
        description="Encumbered assets not captured elsewhere in I.A/I.U/I.S.",
        collateral_required=True, counterparty_required=False,
        forward_start_excluded=True,
        regulatory_ref="FR 2052a p.36",
    ))

    # =================================================================
    # I.U: Inflows -- Unsecured
    # =================================================================
    tbl_iu = FR2052aTable.INFLOWS_UNSECURED

    products.append(FR2052aProduct(
        table=tbl_iu, product_id=1, name="Onshore Placements",
        description="Unsecured domestic-currency inter-bank placements.",
        forward_start_excluded=False,
        regulatory_ref="FR 2052a p.37",
    ))
    products.append(FR2052aProduct(
        table=tbl_iu, product_id=2, name="Offshore Placements",
        description="Unsecured domestic-currency placements outside onshore market.",
        regulatory_ref="FR 2052a p.37",
    ))
    products.append(FR2052aProduct(
        table=tbl_iu, product_id=3, name="Required Operational Balances",
        description="Minimum balances at other FIs for clearing/settlement.",
        forward_start_excluded=True,
        regulatory_ref="FR 2052a p.37",
    ))
    products.append(FR2052aProduct(
        table=tbl_iu, product_id=4, name="Excess Operational Balances",
        description="Balances at FIs above required operational balances.",
        forward_start_excluded=True,
        regulatory_ref="FR 2052a p.37",
    ))
    products.append(FR2052aProduct(
        table=tbl_iu, product_id=5, name="Outstanding Draws on Unsecured Revolving Facilities",
        description="Drawn portion of unsecured revolving credit facilities.",
        regulatory_ref="FR 2052a p.37",
    ))
    products.append(FR2052aProduct(
        table=tbl_iu, product_id=6, name="Other Loans",
        description="All other unsecured loans not included in other I.U products.",
        regulatory_ref="FR 2052a p.37",
    ))
    products.append(FR2052aProduct(
        table=tbl_iu, product_id=7, name="Cash Items in the Process of Collection",
        description="Checks/drafts in process of collection.",
        forward_start_excluded=True,
        regulatory_ref="FR 2052a p.38",
    ))
    products.append(FR2052aProduct(
        table=tbl_iu, product_id=8, name="Short-Term Investments",
        description="Time deposits and other short-term investments at external FIs.",
        forward_start_excluded=True,
        regulatory_ref="FR 2052a p.38",
    ))

    # =================================================================
    # I.S: Inflows -- Secured
    # =================================================================
    tbl_is = FR2052aTable.INFLOWS_SECURED

    products.append(FR2052aProduct(
        table=tbl_is, product_id=1, name="Reverse Repo",
        description="Reverse repurchase agreements.",
        collateral_required=True,
        regulatory_ref="FR 2052a p.39",
    ))
    products.append(FR2052aProduct(
        table=tbl_is, product_id=2, name="Securities Borrowing",
        description="All securities borrowing transactions.",
        collateral_required=True,
        regulatory_ref="FR 2052a p.39",
    ))
    products.append(FR2052aProduct(
        table=tbl_is, product_id=3, name="Dollar Rolls",
        description="TBA-based financing for specific securities/pools.",
        collateral_required=True,
        regulatory_ref="FR 2052a p.39",
    ))
    products.append(FR2052aProduct(
        table=tbl_is, product_id=4, name="Collateral Swaps",
        description="Non-cash asset exchanges (collateral upgrade/downgrade).",
        sub_products=(
            "Level 1 Pledged", "Level 2a Pledged", "Level 2b Pledged",
            "Non-HQLA Pledged", "No Collateral Pledged",
        ),
        collateral_required=True,
        regulatory_ref="FR 2052a p.39",
    ))
    products.append(FR2052aProduct(
        table=tbl_is, product_id=5, name="Margin Loans",
        description="Credit to fund trading positions, collateralized by holdings.",
        collateral_required=True,
        regulatory_ref="FR 2052a p.40",
    ))
    products.append(FR2052aProduct(
        table=tbl_is, product_id=6, name="Other Secured Loans - Rehypothecatable",
        description="Other secured lending with rehypothecatable collateral.",
        collateral_required=True,
        regulatory_ref="FR 2052a p.40",
    ))
    products.append(FR2052aProduct(
        table=tbl_is, product_id=7, name="Outstanding Draws on Secured Revolving Facilities",
        description="Drawn portion of secured revolving facilities.",
        collateral_required=True,
        regulatory_ref="FR 2052a p.40",
    ))
    products.append(FR2052aProduct(
        table=tbl_is, product_id=8, name="Other Secured Loans - Non-Rehypothecatable",
        description="Other secured lending with non-rehypothecatable collateral.",
        collateral_required=True,
        regulatory_ref="FR 2052a p.41",
    ))
    products.append(FR2052aProduct(
        table=tbl_is, product_id=9, name="Synthetic Customer Longs",
        description="TRS where firm is short reference asset, client is long.",
        sub_products=(
            "Physical Long Position", "Synthetic Customer Short",
            "Synthetic Firm Financing", "Futures", "Other", "Unhedged",
        ),
        collateral_required=True,
        regulatory_ref="FR 2052a p.41",
    ))
    products.append(FR2052aProduct(
        table=tbl_is, product_id=10, name="Synthetic Firm Sourcing",
        description="TRS where firm sources assets synthetically.",
        sub_products=(
            "Synthetic Customer Short", "Synthetic Firm Financing",
            "Futures", "Other", "Unhedged",
        ),
        collateral_required=True,
        regulatory_ref="FR 2052a p.41",
    ))

    # =================================================================
    # I.O: Inflows -- Other
    # =================================================================
    tbl_io = FR2052aTable.INFLOWS_OTHER

    products.append(FR2052aProduct(
        table=tbl_io, product_id=1, name="Derivative Receivables",
        description="Contractual derivative cash inflows.",
        collateral_dependent=True, counterparty_required=False,
        forward_start_excluded=True,
        regulatory_ref="FR 2052a p.42",
    ))
    products.append(FR2052aProduct(
        table=tbl_io, product_id=2, name="Collateral Called for Receipt",
        description="Collateral called but not yet received.",
        collateral_required=True, counterparty_required=False,
        forward_start_excluded=True,
        regulatory_ref="FR 2052a p.42",
    ))
    products.append(FR2052aProduct(
        table=tbl_io, product_id=3, name="TBA Sales",
        description="To-Be-Announced MBS sales.",
        collateral_required=True, counterparty_required=False,
        forward_start_excluded=True,
        regulatory_ref="FR 2052a p.43",
    ))
    products.append(FR2052aProduct(
        table=tbl_io, product_id=4, name="Undrawn Committed Facilities Purchased",
        description="Undrawn committed credit/liquidity facilities available.",
        counterparty_required=False,
        forward_start_excluded=True,
        regulatory_ref="FR 2052a p.43",
    ))
    products.append(FR2052aProduct(
        table=tbl_io, product_id=5, name="Lock-up Balance",
        description="Cash balances subject to lock-up provisions.",
        counterparty_required=False,
        forward_start_excluded=True,
        regulatory_ref="FR 2052a p.43",
    ))
    products.append(FR2052aProduct(
        table=tbl_io, product_id=6, name="Interest and Dividends Receivable",
        description="Expected interest and dividend cash inflows.",
        counterparty_required=False,
        forward_start_excluded=True,
        regulatory_ref="FR 2052a p.43",
    ))
    products.append(FR2052aProduct(
        table=tbl_io, product_id=7, name="Net 30-Day Derivative Receivables",
        description="Net positive 30-day derivative cash flow.",
        counterparty_required=False,
        forward_start_excluded=True,
        regulatory_ref="FR 2052a p.43",
    ))
    products.append(FR2052aProduct(
        table=tbl_io, product_id=8,
        name="Principal Payments Receivable on Unencumbered Investment Securities",
        description="Principal payments on unencumbered investment securities.",
        counterparty_required=False,
        forward_start_excluded=True,
        regulatory_ref="FR 2052a p.44",
    ))
    products.append(FR2052aProduct(
        table=tbl_io, product_id=9, name="Other Cash Inflows",
        description="All other cash inflows not captured elsewhere.",
        counterparty_required=False,
        forward_start_excluded=True,
        regulatory_ref="FR 2052a p.44",
    ))

    # =================================================================
    # O.W: Outflows -- Wholesale
    # =================================================================
    tbl_ow = FR2052aTable.OUTFLOWS_WHOLESALE

    _ow_products = [
        (1, "Asset-Backed Commercial Paper (ABCP) Single-Seller", True),
        (2, "Asset-Backed Commercial Paper (ABCP) Multi-Seller", True),
        (3, "Collateralized Commercial Paper", True),
        (4, "Asset-Backed Securities (ABS)", True),
        (5, "Covered Bonds", True),
        (6, "Tender Option Bonds", True),
        (7, "Other Asset-Backed Financing", True),
        (8, "Commercial Paper", False),
        (9, "Onshore Borrowing", False),
        (10, "Offshore Borrowing", False),
        (11, "Unstructured Long Term Debt", False),
        (12, "Structured Long Term Debt", True),
        (13, "Government Supported Debt", False),
        (14, "Unsecured Notes", False),
        (15, "Structured Notes", True),
        (16, "Wholesale CDs", False),
        (17, "Draws on Committed Lines", False),
        (18, "Free Credits", False),
        (19, "Other Unsecured Financing", False),
    ]
    for pid, pname, has_collateral in _ow_products:
        products.append(FR2052aProduct(
            table=tbl_ow, product_id=pid, name=pname,
            collateral_required=has_collateral if pid <= 7 else False,
            collateral_dependent=has_collateral if pid > 7 else False,
            forward_start_excluded=(pid == 18),
            regulatory_ref="FR 2052a p.46-48",
        ))

    # =================================================================
    # O.S: Outflows -- Secured
    # =================================================================
    tbl_os = FR2052aTable.OUTFLOWS_SECURED

    products.append(FR2052aProduct(
        table=tbl_os, product_id=1, name="Repo",
        description="Repurchase agreements.",
        collateral_required=True,
        regulatory_ref="FR 2052a p.49",
    ))
    products.append(FR2052aProduct(
        table=tbl_os, product_id=2, name="Securities Lending",
        description="Securities lending transactions.",
        collateral_required=True,
        regulatory_ref="FR 2052a p.49",
    ))
    products.append(FR2052aProduct(
        table=tbl_os, product_id=3, name="Dollar Rolls",
        description="TBA-based financing (outflow leg).",
        collateral_required=True,
        regulatory_ref="FR 2052a p.49",
    ))
    products.append(FR2052aProduct(
        table=tbl_os, product_id=4, name="Collateral Swaps",
        description="Non-cash asset exchange (outflow leg).",
        sub_products=(
            "Level 1 Received", "Level 2a Received", "Level 2b Received",
            "Non-HQLA Received", "No Collateral Received",
        ),
        collateral_required=True,
        regulatory_ref="FR 2052a p.49",
    ))
    products.append(FR2052aProduct(
        table=tbl_os, product_id=5, name="FHLB Advances",
        description="Federal Home Loan Bank advances.",
        collateral_required=True,
        regulatory_ref="FR 2052a p.50",
    ))
    products.append(FR2052aProduct(
        table=tbl_os, product_id=6, name="Exceptional Central Bank Operations",
        description="Central bank funding operations.",
        sub_products=(
            "FRB", "SNB", "BOE", "ECB", "BOJ", "RBA", "BOC", "OCB",
            "Covered Federal Reserve Facility Funding",
        ),
        collateral_required=True,
        regulatory_ref="FR 2052a p.50",
    ))
    products.append(FR2052aProduct(
        table=tbl_os, product_id=7, name="Customer Shorts",
        description="Customer short positions requiring collateral delivery.",
        sub_products=("External Cash Transactions", "External Non-Cash Transactions"),
        collateral_required=True,
        regulatory_ref="FR 2052a p.51",
    ))
    products.append(FR2052aProduct(
        table=tbl_os, product_id=8, name="Firm Shorts",
        description="Firm proprietary short positions.",
        sub_products=(
            "Firm Longs", "Customer Longs",
            "Unsettled - Regular Way", "Unsettled - Forward",
        ),
        collateral_required=True,
        regulatory_ref="FR 2052a p.52",
    ))
    products.append(FR2052aProduct(
        table=tbl_os, product_id=9, name="Synthetic Customer Shorts",
        description="TRS where client is economically short.",
        sub_products=(
            "Firm Short", "Synthetic Customer Long",
            "Synthetic Firm Sourcing", "Futures", "Other", "Unhedged",
        ),
        collateral_required=True,
        regulatory_ref="FR 2052a p.53",
    ))
    products.append(FR2052aProduct(
        table=tbl_os, product_id=10, name="Synthetic Firm Financing",
        description="TRS for firm proprietary financing.",
        sub_products=(
            "Synthetic Customer Long", "Synthetic Firm Sourcing",
            "Futures", "Other", "Unhedged",
        ),
        collateral_required=True,
        regulatory_ref="FR 2052a p.53",
    ))
    products.append(FR2052aProduct(
        table=tbl_os, product_id=11, name="Other Secured Financing Transactions",
        description="All other secured financing not elsewhere classified.",
        collateral_required=True,
        regulatory_ref="FR 2052a p.54",
    ))

    # =================================================================
    # O.D: Outflows -- Deposits
    # =================================================================
    tbl_od = FR2052aTable.OUTFLOWS_DEPOSITS

    _od_products = [
        (1, "Transactional Accounts"),
        (2, "Non-Transactional Relationship Accounts"),
        (3, "Non-Transactional Non-Relationship Accounts"),
        (4, "Operational Account Balances"),
        (5, "Excess Balances in Operational Accounts"),
        (6, "Non-Operational Account Balances"),
        (7, "Operational Escrow Accounts"),
        (8, "Non-Reciprocal Brokered Deposits"),
        (9, "Stable Affiliated Sweep Account Balances"),
        (10, "Less Stable Affiliated Sweep Account Balances"),
        (11, "Non-Affiliated Sweep Accounts"),
        (12, "Other Product Sweep Accounts"),
        (13, "Reciprocal Accounts"),
        (14, "Other Third-Party Deposits"),
        (15, "Other Accounts"),
    ]
    for pid, pname in _od_products:
        products.append(FR2052aProduct(
            table=tbl_od, product_id=pid, name=pname,
            counterparty_required=True,
            forward_start_excluded=True,
            regulatory_ref="FR 2052a p.55-58",
        ))

    # =================================================================
    # O.O: Outflows -- Other
    # =================================================================
    tbl_oo = FR2052aTable.OUTFLOWS_OTHER

    _oo_products = [
        (1, "Derivative Payables"),
        (2, "Collateral Called for Delivery"),
        (3, "TBA Purchases"),
        (4, "Credit Facilities"),
        (5, "Liquidity Facilities"),
        (6, "Retail Mortgage Commitments"),
        (7, "Trade Finance Instruments"),
        (8, "MTM Impact on Derivative Positions"),
        (9, "Loss of Rehypothecation Rights Due to a 1 Notch Downgrade"),
        (10, "Loss of Rehypothecation Rights Due to a 2 Notch Downgrade"),
        (11, "Loss of Rehypothecation Rights Due to a 3 Notch Downgrade"),
        (12, "Loss of Rehypothecation Rights Due to a Change in Financial Condition"),
        (13, "Total Collateral Required Due to a 1 Notch Downgrade"),
        (14, "Total Collateral Required Due to a 2 Notch Downgrade"),
        (15, "Total Collateral Required Due to a 3 Notch Downgrade"),
        (16, "Total Collateral Required Due to a Change in Financial Condition"),
        (17, "Excess Margin"),
        (18, "Unfunded Term Margin"),
        (19, "Interest and Dividends Payable"),
        (20, "Net 30-Day Derivative Payables"),
        (21, "Other Outflows Related to Structured Transactions"),
        (22, "Other Cash Outflows"),
    ]
    for pid, pname in _oo_products:
        products.append(FR2052aProduct(
            table=tbl_oo, product_id=pid, name=pname,
            counterparty_required=(pid in {4, 5}),
            collateral_dependent=(pid in {2, 3, 9, 10, 11, 12, 13, 14, 15, 16, 17}),
            forward_start_excluded=True,
            regulatory_ref="FR 2052a p.59-64",
        ))

    # =================================================================
    # S.DC: Supplemental -- Derivatives & Collateral
    # =================================================================
    tbl_sdc = FR2052aTable.SUPPLEMENTAL_DC

    _sdc_products = [
        (1, "Gross Derivative Asset Values"),
        (2, "Gross Derivative Liability Values"),
        (3, "Derivative Settlement Payments Delivered"),
        (4, "Derivative Settlement Payments Received"),
        (5, "Initial Margin Posted - House"),
        (6, "Initial Margin Posted - Customer"),
        (7, "Initial Margin Received"),
        (8, "Variation Margin Posted - House"),
        (9, "Variation Margin Posted - Customer"),
        (10, "Variation Margin Received"),
        (11, "Derivative CCP Default Fund Contribution"),
        (12, "Other CCP Pledges and Contributions"),
        (13, "Collateral Disputes Deliverables"),
        (14, "Collateral Disputes Receivables"),
        (15, "Sleeper Collateral Deliverables"),
        (16, "Required Collateral Deliverables"),
        (17, "Sleeper Collateral Receivables"),
        (18, "Derivative Collateral Substitution Risk"),
        (19, "Derivative Collateral Substitution Capacity"),
        (20, "Other Collateral Substitution Risk"),
        (21, "Other Collateral Substitution Capacity"),
    ]
    for pid, pname in _sdc_products:
        products.append(FR2052aProduct(
            table=tbl_sdc, product_id=pid, name=pname,
            counterparty_required=(pid in {1, 2, 5, 6, 7, 8, 9, 10}),
            collateral_required=(pid >= 5),
            collateral_dependent=(pid in {1, 2, 3, 4}),
            forward_start_excluded=True,
            regulatory_ref="FR 2052a p.65-69",
        ))

    # =================================================================
    # S.L: Supplemental -- LRM
    # =================================================================
    tbl_sl = FR2052aTable.SUPPLEMENTAL_LRM

    _sl_products = [
        (1, "Subsidiary Liquidity That Cannot Be Transferred"),
        (2, "Subsidiary Liquidity Available for Transfer"),
        (3, "Unencumbered Asset Hedges - Early Termination Outflows"),
        (4, "Non-Structured Debt Maturing in Greater than 30-days - Primary Market Maker"),
        (5, "Structured Debt Maturing in Greater than 30-days - Primary Market Maker"),
        (6, "Liquidity Coverage Ratio"),
        (7, "Subsidiary Funding That Cannot Be Transferred"),
        (8, "Subsidiary Funding Available for Transfer"),
        (9, "Additional Funding Requirement for Off-Balance Sheet Rehypothecated Assets"),
        (10, "Net Stable Funding Ratio"),
    ]
    for pid, pname in _sl_products:
        products.append(FR2052aProduct(
            table=tbl_sl, product_id=pid, name=pname,
            counterparty_required=False,
            forward_start_excluded=True,
            regulatory_ref="FR 2052a p.70-72",
        ))

    # =================================================================
    # S.B: Supplemental -- Balance Sheet
    # =================================================================
    tbl_sb = FR2052aTable.SUPPLEMENTAL_BALANCE_SHEET

    _sb_products = [
        (1, "Regulatory Capital Element"),
        (2, "Other Liabilities"),
        (3, "Non-Performing Assets"),
        (4, "Other Assets"),
        (5, "Counterparty Netting"),
        (6, "Carrying Value Adjustment"),
    ]
    for pid, pname in _sb_products:
        products.append(FR2052aProduct(
            table=tbl_sb, product_id=pid, name=pname,
            counterparty_required=(pid == 5),
            forward_start_excluded=True,
            regulatory_ref="FR 2052a p.72-73",
        ))

    # =================================================================
    # S.I: Supplemental -- Informational
    # =================================================================
    tbl_si = FR2052aTable.SUPPLEMENTAL_INFORMATIONAL

    _si_products = [
        (1, "Long Market Value Client Assets"),
        (2, "Short Market Value Client Assets"),
        (3, "Gross Client Wires Received"),
        (4, "Gross Client Wires Paid"),
        (5, "FRB 23A Capacity"),
        (6, "Subsidiary Liquidity Not Transferable"),
    ]
    for pid, pname in _si_products:
        products.append(FR2052aProduct(
            table=tbl_si, product_id=pid, name=pname,
            counterparty_required=False,
            forward_start_excluded=True,
            regulatory_ref="FR 2052a p.74-75",
        ))

    # =================================================================
    # S.FX: Supplemental -- Foreign Exchange
    # =================================================================
    tbl_sfx = FR2052aTable.SUPPLEMENTAL_FX

    products.append(FR2052aProduct(
        table=tbl_sfx, product_id=1, name="Spot",
        description="FX spot transactions.",
        regulatory_ref="FR 2052a p.78",
    ))
    products.append(FR2052aProduct(
        table=tbl_sfx, product_id=2, name="Forwards and Futures",
        description="FX forwards and futures.",
        regulatory_ref="FR 2052a p.78",
    ))
    products.append(FR2052aProduct(
        table=tbl_sfx, product_id=3, name="Swaps",
        description="FX swaps.",
        regulatory_ref="FR 2052a p.78",
    ))

    # Build lookup dict
    return {p.code: p for p in products}


ALL_PRODUCTS: dict[str, FR2052aProduct] = _build_products()
"""Complete product catalog keyed by product code (e.g. ``'I.A.1'``)."""


def get_product(code: str) -> FR2052aProduct:
    """Look up a product by its code.

    Args:
        code: Product code, e.g. ``'I.A.1'`` or ``'O.W.8'``.

    Returns:
        The matching :class:`FR2052aProduct`.

    Raises:
        KeyError: If the product code is not found.
    """
    if code not in ALL_PRODUCTS:
        raise KeyError(f"Unknown FR 2052a product code: {code!r}")
    return ALL_PRODUCTS[code]


def get_products_for_table(table: FR2052aTable) -> list[FR2052aProduct]:
    """Return all products for a given table, sorted by product_id.

    Args:
        table: The FR 2052a table code.

    Returns:
        List of products in the specified table.
    """
    return sorted(
        [p for p in ALL_PRODUCTS.values() if p.table == table],
        key=lambda p: p.product_id,
    )
