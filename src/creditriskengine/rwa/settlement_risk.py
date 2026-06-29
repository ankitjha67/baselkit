"""Settlement and failed-trade risk capital (BCBS CRE70).

Reference:
    - BCBS CRE70 "Capital treatment of unsettled transactions and failed
      trades" (consolidated Basel framework).

Covers two cases:

* **Delivery-versus-payment (DvP)** transactions unsettled after their due
  date attract a capital charge on the positive current exposure (the
  loss the bank would face if it had to replace the transaction), scaled
  by a multiplier that rises with the number of business days the trade
  remains unsettled (CRE70.5):

      5-15 business days  -> 8%
      16-30 business days -> 50%
      31-45 business days -> 75%
      46+ business days   -> 100%

* **Non-DvP / free deliveries** (the bank delivers but has not yet received
  the counter-value): from the first contractual leg until 4 business days
  after the second leg the delivered value is treated as a normal exposure
  (counterparty risk weight); 5+ business days after the second leg the
  full delivered value is deducted (1250% risk weight) (CRE70.7).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# CRE70.5 — DvP capital-charge multipliers by business days unsettled.
# Each tuple is (inclusive_lower_days, multiplier); evaluated in order.
_DVP_MULTIPLIERS: tuple[tuple[int, float], ...] = (
    (46, 1.00),
    (31, 0.75),
    (16, 0.50),
    (5, 0.08),
)

# 1250% risk weight (full deduction equivalent) for non-DvP 5+ days late.
_FULL_DEDUCTION_RW: float = 12.50


def dvp_settlement_multiplier(business_days_late: int) -> float:
    """Capital-charge multiplier for an unsettled DvP transaction (CRE70.5).

    Args:
        business_days_late: Business days elapsed since the settlement
            date (0 on the due date).

    Returns:
        The capital-charge multiplier as a fraction (0.0 below 5 days).

    Raises:
        ValueError: If ``business_days_late`` is negative.
    """
    if business_days_late < 0:
        raise ValueError("business_days_late must be non-negative")
    for lower, mult in _DVP_MULTIPLIERS:
        if business_days_late >= lower:
            return mult
    return 0.0


@dataclass(frozen=True)
class SettlementRiskResult:
    """Settlement-risk capital result.

    Attributes:
        positive_exposure: Positive current exposure at risk.
        business_days_late: Business days the trade is unsettled.
        multiplier: Applied CRE70.5 multiplier.
        capital_charge: Capital charge (positive_exposure * multiplier).
        rwa_equivalent: RWA equivalent (capital_charge * 12.5).
    """

    positive_exposure: float
    business_days_late: int
    multiplier: float
    capital_charge: float
    rwa_equivalent: float


def dvp_settlement_capital(
    positive_current_exposure: float,
    business_days_late: int,
) -> SettlementRiskResult:
    """Capital charge for an unsettled DvP transaction (CRE70.4-70.5).

    Args:
        positive_current_exposure: The positive difference between the
            agreed settlement value and the current market value (the loss
            on counterparty failure). Must be non-negative.
        business_days_late: Business days since the settlement date.

    Returns:
        A :class:`SettlementRiskResult`. Below 5 business days the charge
        is zero (only normal counterparty treatment applies).

    Raises:
        ValueError: If ``positive_current_exposure`` is negative.
    """
    if positive_current_exposure < 0.0:
        raise ValueError("positive_current_exposure must be non-negative")

    mult = dvp_settlement_multiplier(business_days_late)
    charge = positive_current_exposure * mult
    return SettlementRiskResult(
        positive_exposure=round(positive_current_exposure, 6),
        business_days_late=business_days_late,
        multiplier=mult,
        capital_charge=round(charge, 6),
        rwa_equivalent=round(charge * 12.5, 6),
    )


def non_dvp_risk_weight(
    business_days_after_second_leg: int,
    counterparty_risk_weight: float,
) -> float:
    """Risk weight for a non-DvP (free-delivery) exposure (CRE70.7).

    From the first contractual payment/delivery leg until 4 business days
    after the second leg, the delivered value carries the counterparty
    risk weight. Five or more business days after the second leg, the full
    value is deducted (treated as 1250%).

    Args:
        business_days_after_second_leg: Business days since the second
            contractual leg's due date (negative/zero before it falls due).
        counterparty_risk_weight: The counterparty's risk weight as a
            decimal multiple (e.g. 1.0 for 100%).

    Returns:
        Applicable risk weight as a decimal multiple.

    Raises:
        ValueError: If ``counterparty_risk_weight`` is negative.
    """
    if counterparty_risk_weight < 0.0:
        raise ValueError("counterparty_risk_weight must be non-negative")
    if business_days_after_second_leg >= 5:
        return _FULL_DEDUCTION_RW
    return counterparty_risk_weight
