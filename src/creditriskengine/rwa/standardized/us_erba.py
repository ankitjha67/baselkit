"""US Basel III Endgame — Expanded Risk-Based Approach (ERBA).

STATUS: PROPOSED — NOT FINAL LAW. This module implements the July 2023
NPR as published (88 FR 64028, 18 September 2023; FRB R-1813 / OCC
2023-0008 / FDIC RIN 3064-AF29), with the March 2026 reproposal's
structural changes (FR docs 2026-05959/60/61, comments closed June 2026)
available via explicit options. All figures were extracted from the
operative Federal Register rule text (proposed §§ __.111/__.112/__.141).
Do not treat any value here as currently binding US law.

Key structural difference between the two proposals:

* 2023 NPR — **dual stack**: every Category I-IV bank computes both
  Standardized and Expanded total RWA and binds to the HIGHER (lower
  ratio), with the expanded RWA phased in 80/85/90/100% (Table 9).
* 2026 reproposal — **single stack**: Category I & II (>= $700bn) use
  ERBA only; everyone else uses the (revised) Standardized Approach only.
  The dual-stack and its market-risk-only 72.5% floor are removed.

The US ERBA deliberately does NOT use external ratings (Dodd-Frank
s.939A): corporates use an investment-grade + public-security test, and
banks use Grade A/B/C classifications.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

logger = logging.getLogger(__name__)

# ============================================================
# Residential mortgages — proposed § __.111(f), Tables 5 & 6
# (88 FR 64190-64191)
# ============================================================

# (LTV upper bound inclusive, risk weight %) — NOT cash-flow dependent.
_ERBA_RRE_NOT_DEPENDENT: tuple[tuple[float, float], ...] = (
    (0.50, 40.0),
    (0.60, 45.0),
    (0.80, 50.0),
    (0.90, 60.0),
    (1.00, 70.0),
)
_ERBA_RRE_NOT_DEPENDENT_ABOVE_100: float = 90.0

# Cash-flow dependent (repayment relies on property cash flows).
_ERBA_RRE_DEPENDENT: tuple[tuple[float, float], ...] = (
    (0.50, 50.0),
    (0.60, 55.0),
    (0.80, 65.0),
    (0.90, 80.0),
    (1.00, 95.0),
)
_ERBA_RRE_DEPENDENT_ABOVE_100: float = 125.0

# Non-regulatory ("other") residential — § __.111(f)(7).
_ERBA_RRE_OTHER_NOT_DEPENDENT: float = 100.0
_ERBA_RRE_OTHER_DEPENDENT: float = 150.0


def erba_residential_mortgage_rw(
    ltv: float,
    is_cashflow_dependent: bool = False,
    is_regulatory_residential: bool = True,
    is_defaulted: bool = False,
    is_currency_mismatched: bool = False,
) -> float:
    """ERBA residential-mortgage risk weight (2023 NPR Tables 5-6).

    PROPOSED figures; the 2026 reproposal lowers these grids (removing
    the ~20pp "gold-plating" over Basel).

    Args:
        ltv: Loan-to-value ratio (0.75 = 75%). PMI is NOT recognised.
        is_cashflow_dependent: Repayment depends on property cash flows.
        is_regulatory_residential: Meets the regulatory residential
            criteria; otherwise the "other real estate" treatment applies
            (100% not-dependent / 150% dependent).
        is_defaulted: Defaulted exposure — 150%, except a non-cashflow-
            dependent residential mortgage which stays at 100%.
        is_currency_mismatched: Apply the 1.5x multiplier capped at 150%
            (proposed s.__.111(f)(9)).

    Returns:
        Risk weight as a percentage.

    Raises:
        ValueError: If ``ltv`` is negative.
    """
    if ltv < 0.0:
        raise ValueError("ltv must be non-negative")

    if is_defaulted:
        rw = 100.0 if not is_cashflow_dependent else 150.0
    elif not is_regulatory_residential:
        rw = (
            _ERBA_RRE_OTHER_DEPENDENT
            if is_cashflow_dependent
            else _ERBA_RRE_OTHER_NOT_DEPENDENT
        )
    else:
        table = _ERBA_RRE_DEPENDENT if is_cashflow_dependent else _ERBA_RRE_NOT_DEPENDENT
        above = (
            _ERBA_RRE_DEPENDENT_ABOVE_100
            if is_cashflow_dependent
            else _ERBA_RRE_NOT_DEPENDENT_ABOVE_100
        )
        rw = above
        for upper, band_rw in table:
            if ltv <= upper:
                rw = band_rw
                break

    if is_currency_mismatched:
        rw = min(rw * 1.5, 150.0)
    return rw


# ============================================================
# Retail — proposed § __.111(g) (88 FR 64191)
# ============================================================

class ERBARetailCategory(StrEnum):
    """ERBA retail exposure categories."""

    TRANSACTOR = "transactor"  # 55%
    REGULATORY_RETAIL = "regulatory_retail"  # non-transactor, 85%
    OTHER_RETAIL = "other_retail"  # 110%


_ERBA_RETAIL_RW: dict[ERBARetailCategory, float] = {
    ERBARetailCategory.TRANSACTOR: 55.0,
    ERBARetailCategory.REGULATORY_RETAIL: 85.0,
    ERBARetailCategory.OTHER_RETAIL: 110.0,
}


def erba_retail_rw(
    category: ERBARetailCategory,
    is_currency_mismatched: bool = False,
) -> float:
    """ERBA retail risk weight (2023 NPR): 55/85/110%.

    Args:
        category: Retail category.
        is_currency_mismatched: Apply the 1.5x multiplier capped at 150%.

    Returns:
        Risk weight as a percentage.
    """
    rw = _ERBA_RETAIL_RW[category]
    if is_currency_mismatched:
        rw = min(rw * 1.5, 150.0)
    return rw


# ============================================================
# Corporates — proposed § __.111(h) (88 FR 64192)
# ============================================================

def erba_corporate_rw(
    is_investment_grade: bool = False,
    has_public_security: bool = False,
    is_subordinated_debt: bool = False,
    is_project_finance_preoperational: bool = False,
) -> float:
    """ERBA corporate risk weight (2023 NPR).

    The 65% preferential weight requires BOTH investment grade AND a
    publicly traded security outstanding (obligor or controlling parent)
    — no external ratings are used (Dodd-Frank s.939A). The 2026
    reproposal drops the public-security condition and moves the base
    corporate weight to 95%.

    Args:
        is_investment_grade: Obligor is investment grade (defined test).
        has_public_security: Obligor (or controlling parent) has a
            publicly traded security outstanding.
        is_subordinated_debt: Subordinated or covered debt instrument
            (150%).
        is_project_finance_preoperational: Project finance in its
            non-operational phase (130%).

    Returns:
        Risk weight as a percentage.
    """
    if is_subordinated_debt:
        return 150.0
    if is_project_finance_preoperational:
        return 130.0
    if is_investment_grade and has_public_security:
        return 65.0
    return 100.0


# ============================================================
# Banks — proposed § __.111(d): Grade A/B/C
# ============================================================

class ERBABankGrade(StrEnum):
    """ERBA bank counterparty grades (no external ratings)."""

    GRADE_A = "grade_a"  # 40% (short-term <= 3m trade: 20%)
    GRADE_B = "grade_b"  # 75% (short-term: 50%)
    GRADE_C = "grade_c"  # 150%


_ERBA_BANK_RW: dict[ERBABankGrade, float] = {
    ERBABankGrade.GRADE_A: 40.0,
    ERBABankGrade.GRADE_B: 75.0,
    ERBABankGrade.GRADE_C: 150.0,
}
_ERBA_BANK_SHORT_TERM_RW: dict[ERBABankGrade, float] = {
    ERBABankGrade.GRADE_A: 20.0,
    ERBABankGrade.GRADE_B: 50.0,
    ERBABankGrade.GRADE_C: 150.0,
}


def erba_bank_rw(grade: ERBABankGrade, is_short_term: bool = False) -> float:
    """ERBA bank risk weight (2023 NPR): Grade A/B/C = 40/75/150%.

    Args:
        grade: Bank counterparty grade.
        is_short_term: Original maturity <= 3 months (trade-related) —
            20%/50% for Grades A/B.

    Returns:
        Risk weight as a percentage.
    """
    table = _ERBA_BANK_SHORT_TERM_RW if is_short_term else _ERBA_BANK_RW
    return table[grade]


# ============================================================
# Off-balance-sheet CCFs — proposed § __.112(b) (88 FR 64192-93)
# ============================================================

class ERBAOffBalanceItem(StrEnum):
    """ERBA off-balance-sheet item categories."""

    UNCONDITIONALLY_CANCELLABLE = "unconditionally_cancellable"  # 10%
    TRADE_SHORT_TERM = "trade_short_term"  # 20%
    COMMITMENT = "commitment"  # 40% regardless of maturity
    TRANSACTION_RELATED = "transaction_related"  # 50% (perf. bonds, NIF/RUF)
    DIRECT_CREDIT_SUBSTITUTE = "direct_credit_substitute"  # 100%


ERBA_CCF: dict[ERBAOffBalanceItem, float] = {
    ERBAOffBalanceItem.UNCONDITIONALLY_CANCELLABLE: 0.10,
    ERBAOffBalanceItem.TRADE_SHORT_TERM: 0.20,
    ERBAOffBalanceItem.COMMITMENT: 0.40,
    ERBAOffBalanceItem.TRANSACTION_RELATED: 0.50,
    ERBAOffBalanceItem.DIRECT_CREDIT_SUBSTITUTE: 1.00,
}


def erba_ccf(item: ERBAOffBalanceItem) -> float:
    """ERBA credit conversion factor (2023 NPR).

    Key changes vs the current US standardized approach: unconditionally
    cancellable commitments move 0% -> 10%, and the 20%/50% maturity split
    for commitments is replaced by a flat 40%.
    """
    return ERBA_CCF[item]


# ============================================================
# Equity — proposed § __.141(b) (88 FR 64213-14)
# ============================================================

class ERBAEquityCategory(StrEnum):
    """ERBA equity exposure categories."""

    SOVEREIGN = "sovereign"  # 0%
    PSE_FHLB_FARMER_MAC = "pse_fhlb_farmer_mac"  # 20%
    COMMUNITY_DEVELOPMENT = "community_development"  # 100% (incl. SBIC)
    PUBLICLY_TRADED = "publicly_traded"  # 250%
    NON_PUBLICLY_TRADED = "non_publicly_traded"  # 400%
    INVESTMENT_FIRM = "investment_firm"  # 1250%


_ERBA_EQUITY_RW: dict[ERBAEquityCategory, float] = {
    ERBAEquityCategory.SOVEREIGN: 0.0,
    ERBAEquityCategory.PSE_FHLB_FARMER_MAC: 20.0,
    ERBAEquityCategory.COMMUNITY_DEVELOPMENT: 100.0,
    ERBAEquityCategory.PUBLICLY_TRADED: 250.0,
    ERBAEquityCategory.NON_PUBLICLY_TRADED: 400.0,
    ERBAEquityCategory.INVESTMENT_FIRM: 1250.0,
}


def erba_equity_rw(category: ERBAEquityCategory) -> float:
    """ERBA equity risk weight (2023 NPR § __.141(b))."""
    return _ERBA_EQUITY_RW[category]


# ============================================================
# Framework structure: dual stack (2023) vs single stack (2026)
# ============================================================

@dataclass(frozen=True)
class BindingRWAResult:
    """Binding total RWA under a US Endgame framework variant.

    Attributes:
        binding_rwa: The RWA that drives the capital ratios.
        binding_stack: "standardized" or "expanded".
        framework: "dual_stack_2023_npr" or "single_stack_2026_reproposal".
    """

    binding_rwa: float
    binding_stack: str
    framework: str


def dual_stack_binding_rwa(
    standardized_total_rwa: float,
    expanded_total_rwa: float,
) -> BindingRWAResult:
    """2023 NPR dual-stack: bind to the HIGHER total RWA (lower ratio).

    Args:
        standardized_total_rwa: Subpart D standardized total RWA (no
            operational-risk or CVA components).
        expanded_total_rwa: Subpart E expanded total RWA (credit + equity
            + operational risk + CVA + market risk).

    Returns:
        A :class:`BindingRWAResult`.

    Raises:
        ValueError: If either RWA is negative.
    """
    if min(standardized_total_rwa, expanded_total_rwa) < 0.0:
        raise ValueError("RWA amounts must be non-negative")
    if expanded_total_rwa >= standardized_total_rwa:
        return BindingRWAResult(expanded_total_rwa, "expanded", "dual_stack_2023_npr")
    return BindingRWAResult(standardized_total_rwa, "standardized", "dual_stack_2023_npr")


def single_stack_rwa(
    standardized_total_rwa: float,
    expanded_total_rwa: float,
    total_assets_usd: float,
) -> BindingRWAResult:
    """2026 reproposal single-stack: ERBA for >= $700bn, SA otherwise.

    The March 2026 reproposal eliminates the dual-stack: Category I & II
    firms (>= $700bn total assets) use ERBA only; all others use the
    (revised) Standardized Approach only.

    Args:
        standardized_total_rwa: Standardized total RWA.
        expanded_total_rwa: ERBA total RWA.
        total_assets_usd: Total consolidated assets in USD.

    Returns:
        A :class:`BindingRWAResult`.

    Raises:
        ValueError: If any input is negative.
    """
    if min(standardized_total_rwa, expanded_total_rwa, total_assets_usd) < 0.0:
        raise ValueError("inputs must be non-negative")
    if total_assets_usd >= 700_000_000_000.0:
        return BindingRWAResult(
            expanded_total_rwa, "expanded", "single_stack_2026_reproposal"
        )
    return BindingRWAResult(
        standardized_total_rwa, "standardized", "single_stack_2026_reproposal"
    )


# ============================================================
# Transitions — 2023 NPR Tables 9 & 10 (88 FR 64166-67)
# ============================================================

# Table 9: % of expanded total RWA recognised. NOTE: no 95% step —
# the schedule is 80 -> 85 -> 90 -> 100.
_EXPANDED_RWA_TRANSITION: tuple[tuple[date, float], ...] = (
    (date(2025, 7, 1), 0.80),
    (date(2026, 7, 1), 0.85),
    (date(2027, 7, 1), 0.90),
    (date(2028, 7, 1), 1.00),
)


def expanded_rwa_transition_pct(reporting_date: date) -> float:
    """2023 NPR Table 9 expanded-RWA recognition percentage.

    80% (from 1 Jul 2025), 85%, 90%, then 100% from 1 Jul 2028 — there is
    no 95% step. Before the proposed effective date the expanded stack
    does not apply (0%).
    """
    pct = 0.0
    for start, p in _EXPANDED_RWA_TRANSITION:
        if reporting_date >= start:
            pct = p
    return pct


# Table 10: % of the AOCI opt-out adjustment still applied (Cat III/IV).
_AOCI_TRANSITION_2023: tuple[tuple[date, float], ...] = (
    (date(2025, 7, 1), 0.75),
    (date(2026, 7, 1), 0.50),
    (date(2027, 7, 1), 0.25),
    (date(2028, 7, 1), 0.0),
)


def aoci_optout_remaining_pct(reporting_date: date) -> float:
    """2023 NPR Table 10: remaining AOCI opt-out adjustment (Cat III/IV).

    75% -> 50% -> 25% -> 0% over July 2025 - July 2028; as the percentage
    declines, more AOCI flows into CET1. Before 1 Jul 2025 the opt-out is
    fully in force (100%).

    (The 2026 reproposal instead phases AOCI IN at 20%/year over five
    years from 1 Jan 2027 — see :func:`aoci_included_pct_reproposal`.)
    """
    pct = 1.0
    for start, p in _AOCI_TRANSITION_2023:
        if reporting_date >= start:
            pct = p
    return pct


def aoci_included_pct_reproposal(reporting_date: date) -> float:
    """2026 reproposal: share of AOCI included in CET1 for Cat III/IV.

    Phases in at 20% per year over five years starting 1 January 2027
    (fully included from 1 January 2031).
    """
    start_year = 2027
    if reporting_date < date(start_year, 1, 1):
        return 0.0
    years_elapsed = reporting_date.year - start_year
    return min(0.20 * (years_elapsed + 1), 1.0)
