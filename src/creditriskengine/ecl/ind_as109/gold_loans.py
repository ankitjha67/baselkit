"""RBI (Lending Against Gold and Silver Collateral) Directions, 2025.

Reference:
    - RBI (Lending Against Gold and Silver Collateral) Directions, 2025 —
      notified 6 June 2025, compliance from 1 April 2026. Unified
      framework across banks, NBFCs, co-operative banks and HFCs.

Implements the tiered loan-to-value (LTV) ceilings for consumption loans
against gold/silver collateral:

    loan amount <= INR 2.5 lakh   -> LTV up to 85%
    INR 2.5 lakh < amount <= 5 lakh -> LTV up to 80%
    loan amount > INR 5 lakh      -> LTV up to 75%
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

INR_LAKH: float = 100_000.0

# Tiered LTV ceilings: (inclusive upper loan-amount bound in INR, max LTV).
GOLD_LOAN_LTV_TIERS: tuple[tuple[float, float], ...] = (
    (2.5 * INR_LAKH, 0.85),
    (5.0 * INR_LAKH, 0.80),
)
GOLD_LOAN_LTV_ABOVE_5_LAKH: float = 0.75


def gold_loan_max_ltv(loan_amount_inr: float) -> float:
    """Maximum permitted LTV for a gold/silver-collateral loan.

    Args:
        loan_amount_inr: Sanctioned loan amount in INR.

    Returns:
        The LTV ceiling as a decimal (0.85 / 0.80 / 0.75 by tier).

    Raises:
        ValueError: If ``loan_amount_inr`` is not positive.
    """
    if loan_amount_inr <= 0.0:
        raise ValueError("loan_amount_inr must be positive")
    for upper, ltv in GOLD_LOAN_LTV_TIERS:
        if loan_amount_inr <= upper:
            return ltv
    return GOLD_LOAN_LTV_ABOVE_5_LAKH


@dataclass(frozen=True)
class GoldLoanLTVResult:
    """Gold-loan LTV compliance assessment.

    Attributes:
        loan_amount_inr: Sanctioned loan amount.
        collateral_value_inr: Value of the gold/silver collateral.
        ltv: Actual LTV (loan / collateral value).
        max_ltv: Applicable tier ceiling.
        is_compliant: True if the actual LTV is at or below the ceiling.
        max_permissible_loan: Largest loan the collateral supports at the
            applicable ceiling.
    """

    loan_amount_inr: float
    collateral_value_inr: float
    ltv: float
    max_ltv: float
    is_compliant: bool
    max_permissible_loan: float


def assess_gold_loan_ltv(
    loan_amount_inr: float,
    collateral_value_inr: float,
) -> GoldLoanLTVResult:
    """Assess a gold/silver-collateral loan against its tiered LTV ceiling.

    Args:
        loan_amount_inr: Sanctioned loan amount in INR.
        collateral_value_inr: Assessed collateral value in INR.

    Returns:
        A :class:`GoldLoanLTVResult`.

    Raises:
        ValueError: If either amount is not positive.
    """
    if collateral_value_inr <= 0.0:
        raise ValueError("collateral_value_inr must be positive")

    max_ltv = gold_loan_max_ltv(loan_amount_inr)
    ltv = loan_amount_inr / collateral_value_inr

    return GoldLoanLTVResult(
        loan_amount_inr=loan_amount_inr,
        collateral_value_inr=collateral_value_inr,
        ltv=round(ltv, 6),
        max_ltv=max_ltv,
        is_compliant=ltv <= max_ltv,
        max_permissible_loan=round(max_ltv * collateral_value_inr, 2),
    )
