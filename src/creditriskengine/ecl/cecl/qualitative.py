"""
Qualitative factor (Q-factor) adjustments for CECL.

Reference: ASC 326-20-55-4, Interagency Policy Statement on
Allowances for Credit Losses (2019).

Q-factors adjust historical loss rates for conditions not
fully captured in quantitative models.
"""

import logging

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class QualitativeFactor(BaseModel):
    """A single qualitative adjustment factor."""
    name: str
    description: str = ""
    adjustment_bps: float = Field(
        default=0.0, description="Adjustment in basis points (can be negative)"
    )
    category: str = "general"


# Standard Q-factor categories per interagency guidance
STANDARD_CATEGORIES: list[str] = [
    "lending_policies",
    "economic_conditions",
    "portfolio_trends",
    "credit_concentration",
    "staff_experience",
    "collateral_values",
    "regulatory_environment",
    "external_factors",
]


def total_q_factor_adjustment(factors: list[QualitativeFactor]) -> float:
    """Sum all Q-factor adjustments.

    Args:
        factors: List of qualitative factors.

    Returns:
        Total adjustment in decimal (e.g., 0.005 for 50 bps).
    """
    total_bps = sum(f.adjustment_bps for f in factors)
    return total_bps / 10_000.0


def apply_q_factors(
    base_loss_rate: float,
    factors: list[QualitativeFactor],
    floor: float = 0.0,
) -> float:
    """Apply Q-factor adjustments to base loss rate.

    Args:
        base_loss_rate: Historical/quantitative loss rate.
        factors: List of qualitative adjustment factors.
        floor: Minimum adjusted loss rate.

    Returns:
        Adjusted loss rate.
    """
    adjustment = total_q_factor_adjustment(factors)
    return max(base_loss_rate + adjustment, floor)
