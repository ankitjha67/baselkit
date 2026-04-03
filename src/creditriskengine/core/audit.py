"""Audit trail and lineage tracking for capital calculations.

Provides immutable calculation records and an audit trail for governance
and regulatory compliance.  Every RWA or ECL calculation can be recorded
with its full input/output context so that supervisory reviews, model
validation teams, and internal audit can trace any number back to its
regulatory basis.

Also provides :class:`OverlayAuditRecord` for tracking management
overlay / post-model adjustment (PMA) lifecycle events, supporting
the governance requirements of IFRS 9.B5.5.52, EBA/GL/2020/06,
and PRA Dear CFO letter (Jul 2020).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd

import creditriskengine


@dataclass(frozen=True)
class CalculationRecord:
    """Immutable record of a single capital calculation."""

    exposure_id: str
    timestamp: datetime
    approach: str
    jurisdiction: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    regulatory_reference: str
    engine_version: str
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class OverlayAuditRecord:
    """Immutable record of a management overlay application event.

    Captures the full lifecycle of a post-model adjustment for
    regulatory and audit purposes.

    Reference:
        IFRS 9.B5.5.52 — adjustments for conditions and forecasts.
        EBA/GL/2020/06 para 25-31 — overlay governance and disclosure.
        PRA Dear CFO letter (Jul 2020) — overlay documentation expectations.

    Attributes:
        overlay_name: Descriptive name of the overlay.
        overlay_type: Classification (e.g., "emerging_risk").
        event: Lifecycle event (e.g., "applied", "approved", "expired",
            "revoked", "reviewed").
        model_ecl: ECL before overlay application.
        overlay_ecl: ECL after overlay application.
        adjustment: The overlay impact (overlay_ecl - model_ecl).
        portfolio_scope: Description of affected exposures.
        rationale: Why the overlay was applied/changed.
        regulatory_basis: IFRS paragraph or regulator guidance.
        approved_by: Approving authority.
        timestamp: When this event occurred.
        engine_version: Library version at time of recording.
    """

    overlay_name: str
    overlay_type: str
    event: str
    model_ecl: float
    overlay_ecl: float
    adjustment: float
    portfolio_scope: str = ""
    rationale: str = ""
    regulatory_basis: str = ""
    approved_by: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    engine_version: str = field(default_factory=lambda: creditriskengine.__version__)


class AuditTrail:
    """Stores and queries :class:`CalculationRecord` entries.

    Parameters
    ----------
    records : list[CalculationRecord] | None
        Optional pre-existing records to initialise the trail with.
    """

    def __init__(self, records: list[CalculationRecord] | None = None) -> None:
        self._records: list[CalculationRecord] = list(records) if records else []
        self._overlay_records: list[OverlayAuditRecord] = []

    # -- mutators -------------------------------------------------------------

    def record(
        self,
        exposure_id: str,
        approach: str,
        jurisdiction: str,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        regulatory_reference: str,
        warnings: list[str] | None = None,
    ) -> CalculationRecord:
        """Create and store a new :class:`CalculationRecord`.

        ``engine_version`` and ``timestamp`` are captured automatically.

        Returns
        -------
        CalculationRecord
            The newly created record.
        """
        rec = CalculationRecord(
            exposure_id=exposure_id,
            timestamp=datetime.now(UTC),
            approach=approach,
            jurisdiction=jurisdiction,
            inputs=dict(inputs),
            outputs=dict(outputs),
            regulatory_reference=regulatory_reference,
            engine_version=creditriskengine.__version__,
            warnings=list(warnings) if warnings else [],
        )
        self._records.append(rec)
        return rec

    # -- queries --------------------------------------------------------------

    @property
    def records(self) -> list[CalculationRecord]:
        """Return a shallow copy of the internal record list."""
        return list(self._records)

    def get_records(
        self,
        exposure_id: str | None = None,
        approach: str | None = None,
    ) -> list[CalculationRecord]:
        """Return records matching the given filters.

        Parameters
        ----------
        exposure_id : str | None
            If provided, only records for this exposure are returned.
        approach : str | None
            If provided, only records using this approach are returned.
        """
        result = self._records
        if exposure_id is not None:
            result = [r for r in result if r.exposure_id == exposure_id]
        if approach is not None:
            result = [r for r in result if r.approach == approach]
        return result

    # -- overlay audit --------------------------------------------------------

    def record_overlay(
        self,
        overlay_name: str,
        overlay_type: str,
        event: str,
        model_ecl: float,
        overlay_ecl: float,
        portfolio_scope: str = "",
        rationale: str = "",
        regulatory_basis: str = "",
        approved_by: str = "",
    ) -> OverlayAuditRecord:
        """Record a management overlay lifecycle event.

        Args:
            overlay_name: Descriptive name of the overlay.
            overlay_type: Classification (e.g., ``"emerging_risk"``).
            event: Lifecycle event (``"applied"``, ``"approved"``,
                ``"expired"``, ``"revoked"``, ``"reviewed"``).
            model_ecl: ECL before overlay.
            overlay_ecl: ECL after overlay.
            portfolio_scope: Affected exposures description.
            rationale: Justification text.
            regulatory_basis: Supporting regulatory reference.
            approved_by: Approving authority identifier.

        Returns:
            The newly created :class:`OverlayAuditRecord`.
        """
        rec = OverlayAuditRecord(
            overlay_name=overlay_name,
            overlay_type=overlay_type,
            event=event,
            model_ecl=model_ecl,
            overlay_ecl=overlay_ecl,
            adjustment=overlay_ecl - model_ecl,
            portfolio_scope=portfolio_scope,
            rationale=rationale,
            regulatory_basis=regulatory_basis,
            approved_by=approved_by,
        )
        self._overlay_records.append(rec)
        return rec

    @property
    def overlay_records(self) -> list[OverlayAuditRecord]:
        """Return a shallow copy of overlay audit records."""
        return list(self._overlay_records)

    def get_overlay_records(
        self,
        overlay_name: str | None = None,
        event: str | None = None,
    ) -> list[OverlayAuditRecord]:
        """Return overlay records matching the given filters.

        Args:
            overlay_name: Filter by overlay name.
            event: Filter by lifecycle event.
        """
        result = self._overlay_records
        if overlay_name is not None:
            result = [r for r in result if r.overlay_name == overlay_name]
        if event is not None:
            result = [r for r in result if r.event == event]
        return result

    def overlay_summary(self) -> dict[str, Any]:
        """Aggregate overlay audit statistics.

        Returns:
            ``{"by_type": {<type>: <count>}, "by_event": {<event>: <count>},
            "total_adjustment": <float>}``
        """
        by_type: Counter[str] = Counter()
        by_event: Counter[str] = Counter()
        total_adj = 0.0
        for rec in self._overlay_records:
            by_type[rec.overlay_type] += 1
            by_event[rec.event] += 1
            if rec.event == "applied":
                total_adj += rec.adjustment
        return {
            "by_type": dict(by_type),
            "by_event": dict(by_event),
            "total_adjustment": total_adj,
        }

    # -- export / reporting ---------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the audit trail to a :class:`pandas.DataFrame`."""
        if not self._records:
            return pd.DataFrame(
                columns=[
                    "exposure_id",
                    "timestamp",
                    "approach",
                    "jurisdiction",
                    "inputs",
                    "outputs",
                    "regulatory_reference",
                    "engine_version",
                    "warnings",
                ],
            )
        rows = []
        for r in self._records:
            rows.append(
                {
                    "exposure_id": r.exposure_id,
                    "timestamp": r.timestamp,
                    "approach": r.approach,
                    "jurisdiction": r.jurisdiction,
                    "inputs": r.inputs,
                    "outputs": r.outputs,
                    "regulatory_reference": r.regulatory_reference,
                    "engine_version": r.engine_version,
                    "warnings": r.warnings,
                }
            )
        return pd.DataFrame(rows)

    def export_json(self, filepath: str) -> None:
        """Write the audit trail to a JSON file.

        Parameters
        ----------
        filepath : str
            Destination path (will be overwritten if it exists).
        """

        data = []
        for r in self._records:
            data.append(
                {
                    "exposure_id": r.exposure_id,
                    "timestamp": r.timestamp.isoformat(),
                    "approach": r.approach,
                    "jurisdiction": r.jurisdiction,
                    "inputs": r.inputs,
                    "outputs": r.outputs,
                    "regulatory_reference": r.regulatory_reference,
                    "engine_version": r.engine_version,
                    "warnings": r.warnings,
                }
            )
        overlay_data = []
        for ov in self._overlay_records:
            overlay_data.append(
                {
                    "overlay_name": ov.overlay_name,
                    "overlay_type": ov.overlay_type,
                    "event": ov.event,
                    "model_ecl": ov.model_ecl,
                    "overlay_ecl": ov.overlay_ecl,
                    "adjustment": ov.adjustment,
                    "portfolio_scope": ov.portfolio_scope,
                    "rationale": ov.rationale,
                    "regulatory_basis": ov.regulatory_basis,
                    "approved_by": ov.approved_by,
                    "timestamp": ov.timestamp.isoformat(),
                    "engine_version": ov.engine_version,
                }
            )

        export = {"calculations": data, "overlays": overlay_data}
        with open(filepath, "w") as fh:
            json.dump(export, fh, indent=2)

    def summary(self) -> dict[str, dict[str, int]]:
        """Aggregate statistics for the audit trail.

        Returns
        -------
        dict
            ``{"by_approach": {<approach>: <count>}, "by_jurisdiction": {<jurisdiction>: <count>}}``
        """
        by_approach: Counter[str] = Counter()
        by_jurisdiction: Counter[str] = Counter()
        for r in self._records:
            by_approach[r.approach] += 1
            by_jurisdiction[r.jurisdiction] += 1
        return {
            "by_approach": dict(by_approach),
            "by_jurisdiction": dict(by_jurisdiction),
        }
