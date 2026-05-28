"""
RBI ECL Master Direction 2026 — Collateral valuation compliance checks.

Reference: RBI/DOR/2026-27/398 Paragraph 55.

For Stage 3 exposures of ₹7.5 crore or more, collateral must be valued
at the time of classification and at least once every two years
thereafter. For stock collateral, the valuation must be undertaken at
least annually.
"""

from __future__ import annotations

import logging
from datetime import date

from creditriskengine.core.types import IFRS9Stage

logger = logging.getLogger(__name__)


RBI_COLLATERAL_REVALUATION_THRESHOLD_INR_CRORE: float = 7.5
"""Stage 3 exposure threshold (₹ crore) above which biennial revaluation
applies. Reference: RBI/DOR/2026-27/398 Paragraph 55."""

RBI_STAGE3_REVALUATION_YEARS: float = 2.0
"""Maximum interval (years) between revaluations for large Stage 3
exposures. Reference: RBI/DOR/2026-27/398 Paragraph 55."""

RBI_STOCK_REVALUATION_YEARS: float = 1.0
"""Maximum interval (years) between revaluations for stock collateral.
Reference: RBI/DOR/2026-27/398 Paragraph 55."""


def _years_between(d1: date, d2: date) -> float:
    """Return the number of years between two dates (positive if d2 > d1)."""
    return (d2 - d1).days / 365.25


def validate_collateral_revaluation(
    stage: IFRS9Stage,
    exposure_inr_crore: float,
    last_revaluation_date: date,
    reporting_date: date,
    collateral_type: str = "real_estate",
) -> list[str]:
    """Validate collateral revaluation compliance under RBI Paragraph 55.

    Returns warnings for cases where revaluation is overdue.

    Args:
        stage: IFRS 9 / Ind AS 109 impairment stage.
        exposure_inr_crore: Exposure size in INR crore.
        last_revaluation_date: Date of the most recent collateral
            valuation.
        reporting_date: Reference date for the compliance check.
        collateral_type: ``"stock"`` triggers annual revaluation;
            anything else uses the Stage 3 biennial rule.

    Returns:
        List of warning strings (empty if compliant).

    Reference:
        RBI/DOR/2026-27/398 Paragraph 55.
    """
    warnings: list[str] = []
    years_since = _years_between(last_revaluation_date, reporting_date)

    if collateral_type.lower() == "stock":
        if years_since > RBI_STOCK_REVALUATION_YEARS:
            warnings.append(
                f"Stock collateral last valued {years_since:.2f} years ago; "
                f"RBI requires annual revaluation (Paragraph 55)."
            )
        return warnings

    # Stage 3 large-exposure rule
    if (
        stage == IFRS9Stage.STAGE_3
        and exposure_inr_crore >= RBI_COLLATERAL_REVALUATION_THRESHOLD_INR_CRORE
        and years_since > RBI_STAGE3_REVALUATION_YEARS
    ):
        warnings.append(
            f"Stage 3 exposure ₹{exposure_inr_crore:.2f} crore "
            f">= ₹{RBI_COLLATERAL_REVALUATION_THRESHOLD_INR_CRORE:.1f} crore "
            f"not revalued for {years_since:.2f} years; RBI requires "
            f"revaluation every {RBI_STAGE3_REVALUATION_YEARS:.0f} years "
            f"(Paragraph 55)."
        )

    return warnings
