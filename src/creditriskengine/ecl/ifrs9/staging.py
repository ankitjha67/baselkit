"""
IFRS 9 three-stage impairment model.

Reference: IFRS 9.5.5.1-5.5.20.

Stage 1: 12-month ECL (performing, no SICR)
Stage 2: Lifetime ECL (performing, SICR identified)
Stage 3: Lifetime ECL (credit-impaired / defaulted)
POCI: Purchased or originated credit-impaired
"""

import logging
from typing import Optional

from creditriskengine.core.types import IFRS9Stage

logger = logging.getLogger(__name__)


def assign_stage(
    days_past_due: int,
    is_credit_impaired: bool = False,
    is_defaulted: bool = False,
    is_poci: bool = False,
    sicr_triggered: bool = False,
    dpd_backstop: int = 30,
) -> IFRS9Stage:
    """Assign IFRS 9 impairment stage.

    Logic per IFRS 9.5.5.1-5.5.20:
    - Stage 3: Credit-impaired or defaulted
    - Stage 2: SICR triggered or DPD > backstop (rebuttable, IFRS 9.B5.5.19)
    - Stage 1: All other performing exposures
    - POCI: Purchased/originated credit-impaired (separate treatment)

    Args:
        days_past_due: Days past due count.
        is_credit_impaired: Whether exposure is credit-impaired.
        is_defaulted: Whether exposure is in default.
        is_poci: Whether this is a POCI asset.
        sicr_triggered: Whether SICR assessment triggered Stage 2.
        dpd_backstop: DPD backstop for Stage 2 (default 30, per IFRS 9.5.5.11).

    Returns:
        IFRS9Stage enum value.
    """
    if is_poci:
        return IFRS9Stage.POCI

    if is_credit_impaired or is_defaulted:
        return IFRS9Stage.STAGE_3

    if sicr_triggered or days_past_due > dpd_backstop:
        return IFRS9Stage.STAGE_2

    return IFRS9Stage.STAGE_1


def stage_allocation_summary(
    stages: list[IFRS9Stage],
    eads: list[float],
) -> dict[str, dict[str, float]]:
    """Summarize stage allocation by count and EAD.

    Args:
        stages: List of stage assignments.
        eads: Corresponding EAD values.

    Returns:
        Dict with count and ead totals per stage.
    """
    summary: dict[str, dict[str, float]] = {}
    for stage, ead in zip(stages, eads):
        key = stage.name
        if key not in summary:
            summary[key] = {"count": 0.0, "ead": 0.0}
        summary[key]["count"] += 1
        summary[key]["ead"] += ead
    return summary
