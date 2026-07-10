"""Brazil CMN 4.966 three-stage expected-loss classification.

Reference:
    - Resolucao CMN no. 4.966/21 (+ Res. BCB 352-356), effective
      1 January 2025 — replaces the Res. CMN 2.682/99 rating-grade
      provisioning table with an IFRS-9-aligned three-stage ECL model.

Staging:
    Stage 1 — no significant increase in credit risk -> 12-month ECL.
    Stage 2 — SICR, including the **>30 days-past-due** backstop ->
              lifetime ECL.
    Stage 3 — default: **>90 DPD** or objective evidence of loss ->
              lifetime ECL on the credit-impaired asset.

BCB retained a simplified expected-loss approach for smaller prudential
segments (S3-S5); the full model applies to S1/S2 institutions.
"""

from __future__ import annotations

import logging

from creditriskengine.core.types import IFRS9Stage

logger = logging.getLogger(__name__)

# CMN 4.966 staging backstops.
CMN_4966_SICR_DPD_BACKSTOP: int = 30
CMN_4966_DEFAULT_DPD: int = 90

# Prudential segments using the BCB simplified expected-loss approach.
SIMPLIFIED_MODEL_SEGMENTS: frozenset[str] = frozenset({"S3", "S4", "S5"})


def classify_cmn_4966_stage(
    days_past_due: int,
    has_sicr: bool = False,
    has_objective_loss_evidence: bool = False,
) -> IFRS9Stage:
    """Assign the CMN 4.966 ECL stage for an exposure.

    Args:
        days_past_due: Current days past due.
        has_sicr: True if the institution's SICR assessment (beyond the
            DPD backstop) has triggered — e.g. significant PD
            deterioration, restructuring indicators.
        has_objective_loss_evidence: True on objective evidence of loss
            (bankruptcy, unlikeliness to pay) regardless of DPD.

    Returns:
        The :class:`IFRS9Stage`.

    Raises:
        ValueError: If ``days_past_due`` is negative.
    """
    if days_past_due < 0:
        raise ValueError("days_past_due must be non-negative")

    if has_objective_loss_evidence or days_past_due > CMN_4966_DEFAULT_DPD:
        return IFRS9Stage.STAGE_3
    if has_sicr or days_past_due > CMN_4966_SICR_DPD_BACKSTOP:
        return IFRS9Stage.STAGE_2
    return IFRS9Stage.STAGE_1


def uses_simplified_model(prudential_segment: str) -> bool:
    """Whether a BCB prudential segment uses the simplified EL approach.

    Segments S3-S5 apply the BCB simplified expected-loss methodology;
    S1/S2 institutions apply the full three-stage ECL model.

    Args:
        prudential_segment: BCB segment label ("S1" ... "S5").

    Returns:
        True for S3/S4/S5.

    Raises:
        ValueError: If the segment label is not S1-S5.
    """
    seg = prudential_segment.upper()
    if seg not in {"S1", "S2", "S3", "S4", "S5"}:
        raise ValueError(f"Unknown BCB prudential segment: {prudential_segment}")
    return seg in SIMPLIFIED_MODEL_SEGMENTS
