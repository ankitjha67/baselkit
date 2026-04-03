"""Management overlays (post-model adjustments) for IFRS 9 ECL.

Provides a structured framework for applying expert-judgment adjustments
on top of modeled ECL outputs.  Management overlays capture model
limitations, emerging risks, data gaps, and other factors not reflected
in the quantitative models.

References:
    - IFRS 9.B5.5.1 — reasonable and supportable information.
    - IFRS 9.B5.5.52 — adjustments for current conditions and forecasts.
    - IFRS 9.5.5.17(c) — forward-looking information requirement.
    - ECB "Letter to banks on IFRS 9" (Dec 2020) — COVID-19 overlay guidance.
    - EBA/GL/2020/06 — guidelines on COVID-19 reporting and disclosure.
    - EBA/GL/2017/06 — credit risk management practices and ECL accounting.
    - IASB "IFRS 9 and COVID-19" (Mar 2020) — overlay vs. model recalibration.
    - BCBS "Measures to reflect the impact of COVID-19" (Apr 2020).
    - PRA Dear CFO letter (Jul 2020) — expectations on overlay governance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

logger = logging.getLogger(__name__)


class OverlayType(StrEnum):
    """Classification of management overlay purpose.

    Reference: EBA/GL/2020/06 para 25-28, ECB guidance on COVID-19
    overlays (Dec 2020).
    """

    MODEL_LIMITATION = "model_limitation"
    """Model does not capture a known risk factor (e.g., new product)."""

    EMERGING_RISK = "emerging_risk"
    """Risk not yet observable in historical data (e.g., geopolitical)."""

    DATA_GAP = "data_gap"
    """Insufficient data for reliable model estimation."""

    ECONOMIC_UNCERTAINTY = "economic_uncertainty"
    """Macro-economic uncertainty beyond scenario range."""

    SECTOR_SPECIFIC = "sector_specific"
    """Concentrated exposure to a stressed sector."""

    REGULATORY = "regulatory"
    """Regulator-directed adjustment (e.g., supervisory add-on)."""

    TEMPORARY_EVENT = "temporary_event"
    """Short-lived event not reflected in model calibration."""


@dataclass(frozen=True)
class ManagementOverlay:
    """A single post-model adjustment with full governance metadata.

    Every overlay must include a rationale, regulatory basis, and approval
    record so that auditors and validators can trace the adjustment back
    to its justification.

    Attributes:
        name: Short descriptive name (e.g., "CRE sector stress overlay").
        overlay_type: Classification per :class:`OverlayType`.
        adjustment_amount: Absolute ECL adjustment (positive = increase).
            Use *adjustment_rate* for percentage-based overlays.
        adjustment_rate: Proportional adjustment applied to model ECL
            (e.g., 0.10 for +10%).  Ignored when *adjustment_amount*
            is non-zero.
        rationale: Detailed explanation of why the overlay is needed.
            This is the primary audit evidence for the adjustment.
        regulatory_basis: IFRS paragraph, regulator letter, or guidance
            document that supports the overlay (e.g., "IFRS 9.B5.5.52").
        approved_by: Name or identifier of the approving authority
            (e.g., "Credit Risk Committee", "CRO").
        approval_date: Date the overlay was approved.
        effective_date: Date from which the overlay takes effect.
        expiry_date: Date on which the overlay must be reviewed or
            removed.  Overlays without an expiry are permanent until
            explicitly revoked.
        portfolio_scope: Optional filter description (e.g., "UK CRE
            Stage 2 exposures").
        is_active: Whether the overlay is currently in force.
    """

    name: str
    overlay_type: OverlayType
    adjustment_amount: float = 0.0
    adjustment_rate: float = 0.0
    rationale: str = ""
    regulatory_basis: str = "IFRS 9.B5.5.52"
    approved_by: str = ""
    approval_date: datetime | None = None
    effective_date: datetime | None = None
    expiry_date: datetime | None = None
    portfolio_scope: str = ""
    is_active: bool = True


@dataclass
class OverlayResult:
    """Result of applying overlays to a model ECL.

    Attributes:
        model_ecl: Original model-computed ECL before overlays.
        overlay_ecl: Total ECL after all overlays applied.
        total_adjustment: Aggregate overlay impact (overlay_ecl − model_ecl).
        applied_overlays: List of overlays that were applied.
        skipped_overlays: List of overlays that were skipped (inactive
            or expired).
        timestamp: When the overlay application was performed.
    """

    model_ecl: float
    overlay_ecl: float
    total_adjustment: float
    applied_overlays: list[ManagementOverlay] = field(default_factory=list)
    skipped_overlays: list[ManagementOverlay] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


def _is_overlay_effective(
    overlay: ManagementOverlay,
    as_of: datetime | None = None,
) -> bool:
    """Check whether an overlay is currently in force.

    An overlay is effective if it is active, past its effective_date,
    and has not expired as of *as_of*.
    """
    if not overlay.is_active:
        return False
    ref = as_of or datetime.now(UTC)
    if overlay.effective_date is not None and ref < overlay.effective_date:
        return False
    return not (overlay.expiry_date is not None and ref >= overlay.expiry_date)


def apply_overlays(
    model_ecl: float,
    overlays: list[ManagementOverlay],
    floor: float = 0.0,
    as_of: datetime | None = None,
) -> OverlayResult:
    """Apply management overlays to a model-computed ECL.

    Processing order:
        1. Proportional (rate-based) overlays are applied first, each
           multiplicatively on the running ECL.
        2. Absolute overlays are then added.
        3. The result is floored at *floor* (default 0).

    This mirrors common bank practice where percentage uplifts are
    applied before absolute add-ons.

    Reference:
        IFRS 9.B5.5.52 — adjustments for current conditions.
        EBA/GL/2020/06 para 25-28 — overlay governance expectations.

    Args:
        model_ecl: Model-computed ECL before overlays.
        overlays: List of :class:`ManagementOverlay` adjustments.
        floor: Minimum ECL after overlays (default 0).
        as_of: Reference date for effectiveness checks.  Defaults to
            now (UTC).

    Returns:
        :class:`OverlayResult` with full lineage of applied adjustments.
    """
    applied: list[ManagementOverlay] = []
    skipped: list[ManagementOverlay] = []

    ecl = model_ecl

    # Pass 1: rate-based overlays
    for ov in overlays:
        if not _is_overlay_effective(ov, as_of):
            skipped.append(ov)
            continue
        if ov.adjustment_amount != 0.0:
            continue  # handled in pass 2
        if ov.adjustment_rate != 0.0:
            ecl *= (1.0 + ov.adjustment_rate)
            applied.append(ov)
            logger.debug(
                "Applied rate overlay '%s': %+.2f%% -> ECL=%.2f",
                ov.name, ov.adjustment_rate * 100, ecl,
            )

    # Pass 2: absolute overlays
    for ov in overlays:
        if not _is_overlay_effective(ov, as_of):
            if ov not in skipped:
                skipped.append(ov)
            continue
        if ov.adjustment_amount != 0.0:
            ecl += ov.adjustment_amount
            applied.append(ov)
            logger.debug(
                "Applied absolute overlay '%s': %+.2f -> ECL=%.2f",
                ov.name, ov.adjustment_amount, ecl,
            )

    ecl = max(ecl, floor)

    return OverlayResult(
        model_ecl=model_ecl,
        overlay_ecl=ecl,
        total_adjustment=ecl - model_ecl,
        applied_overlays=applied,
        skipped_overlays=skipped,
    )


def overlay_impact_summary(
    result: OverlayResult,
) -> dict[str, float | int | str]:
    """Generate a governance summary of overlay impacts.

    Designed for inclusion in Pillar 3 disclosures, board reporting,
    and audit committee packs.

    Reference:
        IFRS 7.35F-35L — disclosure of ECL measurement.
        EBA/GL/2020/06 para 31 — disclosure of overlay impacts.

    Args:
        result: An :class:`OverlayResult` from :func:`apply_overlays`.

    Returns:
        Dict with summary metrics suitable for reporting.
    """
    pct_impact = (
        (result.total_adjustment / result.model_ecl * 100.0)
        if result.model_ecl != 0
        else 0.0
    )

    by_type: dict[str, float] = {}
    for ov in result.applied_overlays:
        key = ov.overlay_type.value
        impact = ov.adjustment_amount if ov.adjustment_amount != 0.0 else (
            result.model_ecl * ov.adjustment_rate
        )
        by_type[key] = by_type.get(key, 0.0) + impact

    return {
        "model_ecl": result.model_ecl,
        "overlay_ecl": result.overlay_ecl,
        "total_adjustment": result.total_adjustment,
        "adjustment_pct": round(pct_impact, 2),
        "overlays_applied": len(result.applied_overlays),
        "overlays_skipped": len(result.skipped_overlays),
        "impact_by_type": str(by_type),
        "timestamp": result.timestamp.isoformat(),
    }


def validate_overlay(overlay: ManagementOverlay) -> list[str]:
    """Validate governance completeness of a management overlay.

    Returns a list of warnings for any governance gaps that auditors
    would flag.  An empty list means the overlay passes all checks.

    Reference:
        PRA Dear CFO letter (Jul 2020) — overlay governance expectations.
        EBA/GL/2017/06 — credit risk management and ECL accounting.

    Args:
        overlay: The overlay to validate.

    Returns:
        List of governance warning messages (empty if fully compliant).
    """
    warnings: list[str] = []

    if not overlay.rationale:
        warnings.append("Missing rationale — required per IFRS 9.B5.5.52")
    if not overlay.approved_by:
        warnings.append("Missing approval authority — required per EBA/GL/2017/06")
    if overlay.approval_date is None:
        warnings.append("Missing approval date")
    if overlay.expiry_date is None:
        warnings.append(
            "No expiry date — overlays should be time-bound per PRA guidance"
        )
    if overlay.adjustment_amount == 0.0 and overlay.adjustment_rate == 0.0:
        warnings.append("Overlay has zero impact — consider removing")
    if not overlay.portfolio_scope:
        warnings.append(
            "No portfolio scope defined — consider narrowing applicability"
        )

    return warnings
