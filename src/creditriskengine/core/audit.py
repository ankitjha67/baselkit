"""Audit trail and lineage tracking for capital calculations.

Provides immutable calculation records and an audit trail for governance
and regulatory compliance.  Every RWA or ECL calculation can be recorded
with its full input/output context so that supervisory reviews, model
validation teams, and internal audit can trace any number back to its
regulatory basis.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
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


class AuditTrail:
    """Stores and queries :class:`CalculationRecord` entries.

    Parameters
    ----------
    records : list[CalculationRecord] | None
        Optional pre-existing records to initialise the trail with.
    """

    def __init__(self, records: list[CalculationRecord] | None = None) -> None:
        self._records: list[CalculationRecord] = list(records) if records else []

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
            timestamp=datetime.now(timezone.utc),
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
        with open(filepath, "w") as fh:
            json.dump(data, fh, indent=2)

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
