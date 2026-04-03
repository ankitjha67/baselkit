"""
Qualitative factor (Q-factor) adjustments for CECL.

Reference: ASC 326-20-55-4, Interagency Policy Statement on
Allowances for Credit Losses (2019), OCC Bulletin 2020-49.

Q-factors adjust historical loss rates for conditions not
fully captured in quantitative models.  This module provides
both the core calculation functions and a governance framework
for tracking approvals, caps, and effectiveness reviews.
"""

from __future__ import annotations

import logging
from datetime import datetime

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class QualitativeFactor(BaseModel):
    """A single qualitative adjustment factor with governance metadata.

    Reference: Interagency Policy Statement (2019), OCC 2020-49.

    Attributes:
        name: Descriptive name of the Q-factor.
        description: Explanation of the condition being addressed.
        adjustment_bps: Adjustment in basis points (can be negative).
        category: Standard Q-factor category per interagency guidance.
        rationale: Detailed justification for the adjustment.
        approved_by: Approving authority (e.g., "ALLL Committee").
        approval_date: Date of most recent approval.
        effective_date: Date from which the factor takes effect.
        expiry_date: Date by which the factor must be reviewed.
        data_source: Supporting data or analysis reference.
        is_active: Whether the factor is currently in force.
    """

    name: str
    description: str = ""
    adjustment_bps: float = Field(
        default=0.0, description="Adjustment in basis points (can be negative)"
    )
    category: str = "general"
    rationale: str = ""
    approved_by: str = ""
    approval_date: datetime | None = None
    effective_date: datetime | None = None
    expiry_date: datetime | None = None
    data_source: str = ""
    is_active: bool = True


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

# Maximum adjustment per category (in bps) — governance guardrail
DEFAULT_CATEGORY_CAPS_BPS: dict[str, float] = {
    "lending_policies": 100.0,
    "economic_conditions": 150.0,
    "portfolio_trends": 100.0,
    "credit_concentration": 100.0,
    "staff_experience": 50.0,
    "collateral_values": 150.0,
    "regulatory_environment": 75.0,
    "external_factors": 100.0,
    "general": 100.0,
}
"""Default per-category caps in basis points.

Reference: OCC Bulletin 2020-49 — expectations on Q-factor governance.
These are illustrative defaults; institutions should set their own caps
based on their risk appetite framework.
"""


def total_q_factor_adjustment(factors: list[QualitativeFactor]) -> float:
    """Sum all Q-factor adjustments.

    Only includes active factors.

    Args:
        factors: List of qualitative factors.

    Returns:
        Total adjustment in decimal (e.g., 0.005 for 50 bps).
    """
    total_bps = sum(f.adjustment_bps for f in factors if f.is_active)
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


def apply_q_factors_with_caps(
    base_loss_rate: float,
    factors: list[QualitativeFactor],
    category_caps_bps: dict[str, float] | None = None,
    floor: float = 0.0,
) -> tuple[float, list[str]]:
    """Apply Q-factor adjustments with per-category caps.

    Enforces governance guardrails by capping the total adjustment
    within each category.  Returns both the adjusted rate and any
    warnings about factors that were capped.

    Reference: OCC Bulletin 2020-49 — Q-factor governance expectations.

    Args:
        base_loss_rate: Historical/quantitative loss rate.
        factors: List of qualitative adjustment factors.
        category_caps_bps: Per-category caps in basis points.
            Defaults to :data:`DEFAULT_CATEGORY_CAPS_BPS`.
        floor: Minimum adjusted loss rate.

    Returns:
        Tuple of (adjusted_loss_rate, list_of_cap_warnings).
    """
    caps = category_caps_bps or DEFAULT_CATEGORY_CAPS_BPS
    warnings: list[str] = []

    # Aggregate by category
    category_totals: dict[str, float] = {}
    for f in factors:
        if not f.is_active:
            continue
        cat = f.category
        category_totals[cat] = category_totals.get(cat, 0.0) + f.adjustment_bps

    # Apply caps
    total_bps = 0.0
    for cat, raw_bps in category_totals.items():
        cap = caps.get(cat, caps.get("general", 200.0))
        if abs(raw_bps) > cap:
            capped = cap if raw_bps > 0 else -cap
            warnings.append(
                f"Category '{cat}' capped: {raw_bps:.0f} bps -> {capped:.0f} bps "
                f"(cap: {cap:.0f} bps)"
            )
            total_bps += capped
        else:
            total_bps += raw_bps

    adjusted = max(base_loss_rate + total_bps / 10_000.0, floor)
    return adjusted, warnings


def validate_q_factors(
    factors: list[QualitativeFactor],
) -> list[str]:
    """Validate governance completeness of Q-factor set.

    Returns warnings for any governance gaps that auditors would flag.

    Reference:
        Interagency Policy Statement (2019) — Q-factor documentation.
        OCC Bulletin 2020-49 — governance expectations.

    Args:
        factors: List of Q-factors to validate.

    Returns:
        List of governance warning messages.
    """
    warnings: list[str] = []

    for f in factors:
        if not f.is_active:
            continue
        prefix = f"Q-factor '{f.name}'"
        if not f.rationale:
            warnings.append(f"{prefix}: missing rationale")
        if not f.approved_by:
            warnings.append(f"{prefix}: missing approval authority")
        if f.approval_date is None:
            warnings.append(f"{prefix}: missing approval date")
        if f.adjustment_bps == 0.0:
            warnings.append(f"{prefix}: zero adjustment — consider removing")
        if f.category not in STANDARD_CATEGORIES:
            warnings.append(
                f"{prefix}: non-standard category '{f.category}' — "
                "verify against interagency guidance"
            )

    # Check category coverage
    active_cats = {f.category for f in factors if f.is_active}
    missing = set(STANDARD_CATEGORIES) - active_cats
    if missing and len(factors) > 0:
        warnings.append(
            f"Categories not addressed: {sorted(missing)} — "
            "interagency guidance expects all 8 categories to be evaluated"
        )

    return warnings


def q_factor_summary(
    factors: list[QualitativeFactor],
    base_loss_rate: float,
) -> dict[str, float | int | str]:
    """Generate a summary of Q-factor adjustments for reporting.

    Reference: ASC 326-20-50-13 — disclosure of ECL methodology.

    Args:
        factors: List of qualitative factors.
        base_loss_rate: Base historical loss rate.

    Returns:
        Dict with summary metrics for governance reporting.
    """
    active = [f for f in factors if f.is_active]
    total_adj = total_q_factor_adjustment(active)
    adjusted_rate = max(base_loss_rate + total_adj, 0.0)

    by_category: dict[str, float] = {}
    for f in active:
        by_category[f.category] = by_category.get(f.category, 0.0) + f.adjustment_bps

    return {
        "base_loss_rate": base_loss_rate,
        "adjusted_loss_rate": adjusted_rate,
        "total_adjustment_bps": sum(f.adjustment_bps for f in active),
        "n_active_factors": len(active),
        "n_inactive_factors": len(factors) - len(active),
        "adjustment_by_category": str(by_category),
    }
