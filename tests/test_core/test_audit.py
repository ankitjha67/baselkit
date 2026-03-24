"""Tests for creditriskengine.core.audit module."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import pytest

import creditriskengine
from creditriskengine.core.audit import AuditTrail, CalculationRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(**overrides) -> CalculationRecord:
    """Build a CalculationRecord with sensible defaults."""
    defaults = {
        "exposure_id": "EXP-001",
        "timestamp": datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
        "approach": "standardized",
        "jurisdiction": "EU",
        "inputs": {"pd": 0.01, "lgd": 0.45},
        "outputs": {"rwa": 100_000.0},
        "regulatory_reference": "CRE20",
        "engine_version": "0.1.0",
        "warnings": [],
    }
    defaults.update(overrides)
    return CalculationRecord(**defaults)


# ---------------------------------------------------------------------------
# CalculationRecord
# ---------------------------------------------------------------------------


class TestCalculationRecord:
    def test_creation(self) -> None:
        rec = _make_record()
        assert rec.exposure_id == "EXP-001"
        assert rec.approach == "standardized"
        assert rec.jurisdiction == "EU"
        assert rec.inputs == {"pd": 0.01, "lgd": 0.45}
        assert rec.outputs == {"rwa": 100_000.0}
        assert rec.regulatory_reference == "CRE20"
        assert rec.engine_version == "0.1.0"
        assert rec.warnings == []

    def test_frozen(self) -> None:
        rec = _make_record()
        with pytest.raises(AttributeError):
            rec.exposure_id = "EXP-999"  # type: ignore[misc]

    def test_default_warnings_empty(self) -> None:
        rec = CalculationRecord(
            exposure_id="X",
            timestamp=datetime.now(timezone.utc),
            approach="SA",
            jurisdiction="UK",
            inputs={},
            outputs={},
            regulatory_reference="CRE20",
            engine_version="0.1.0",
        )
        assert rec.warnings == []

    def test_warnings_stored(self) -> None:
        rec = _make_record(warnings=["PD clipped"])
        assert rec.warnings == ["PD clipped"]


# ---------------------------------------------------------------------------
# AuditTrail — init and record
# ---------------------------------------------------------------------------


class TestAuditTrailInit:
    def test_empty_init(self) -> None:
        trail = AuditTrail()
        assert trail.records == []

    def test_init_with_records(self) -> None:
        rec = _make_record()
        trail = AuditTrail(records=[rec])
        assert len(trail.records) == 1
        assert trail.records[0] is rec

    def test_records_returns_copy(self) -> None:
        trail = AuditTrail()
        trail.record("E1", "SA", "EU", {}, {}, "CRE20")
        records = trail.records
        records.clear()
        assert len(trail.records) == 1


class TestAuditTrailRecord:
    def test_record_creates_entry(self) -> None:
        trail = AuditTrail()
        rec = trail.record(
            exposure_id="EXP-100",
            approach="foundation_irb",
            jurisdiction="UK",
            inputs={"pd": 0.02},
            outputs={"rwa": 50_000},
            regulatory_reference="CRE31",
        )
        assert isinstance(rec, CalculationRecord)
        assert rec.exposure_id == "EXP-100"
        assert rec.approach == "foundation_irb"
        assert rec.jurisdiction == "UK"
        assert rec.inputs == {"pd": 0.02}
        assert rec.outputs == {"rwa": 50_000}
        assert rec.regulatory_reference == "CRE31"

    def test_auto_engine_version(self) -> None:
        trail = AuditTrail()
        rec = trail.record("E1", "SA", "EU", {}, {}, "CRE20")
        assert rec.engine_version == creditriskengine.__version__

    def test_auto_timestamp_utc(self) -> None:
        trail = AuditTrail()
        before = datetime.now(timezone.utc)
        rec = trail.record("E1", "SA", "EU", {}, {}, "CRE20")
        after = datetime.now(timezone.utc)
        assert before <= rec.timestamp <= after
        assert rec.timestamp.tzinfo is not None

    def test_record_with_warnings(self) -> None:
        trail = AuditTrail()
        rec = trail.record(
            "E1", "SA", "EU", {}, {}, "CRE20",
            warnings=["PD below floor"],
        )
        assert rec.warnings == ["PD below floor"]

    def test_record_without_warnings(self) -> None:
        trail = AuditTrail()
        rec = trail.record("E1", "SA", "EU", {}, {}, "CRE20")
        assert rec.warnings == []

    def test_inputs_outputs_are_copies(self) -> None:
        trail = AuditTrail()
        inp = {"pd": 0.01}
        out = {"rwa": 100}
        rec = trail.record("E1", "SA", "EU", inp, out, "CRE20")
        inp["pd"] = 999
        out["rwa"] = 999
        assert rec.inputs == {"pd": 0.01}
        assert rec.outputs == {"rwa": 100}

    def test_warnings_are_copies(self) -> None:
        trail = AuditTrail()
        warns = ["w1"]
        rec = trail.record("E1", "SA", "EU", {}, {}, "CRE20", warnings=warns)
        warns.append("w2")
        assert rec.warnings == ["w1"]


# ---------------------------------------------------------------------------
# AuditTrail — get_records
# ---------------------------------------------------------------------------


class TestAuditTrailGetRecords:
    def _populated_trail(self) -> AuditTrail:
        trail = AuditTrail()
        trail.record("E1", "SA", "EU", {}, {}, "CRE20")
        trail.record("E2", "SA", "UK", {}, {}, "CRE20")
        trail.record("E1", "FIRB", "EU", {}, {}, "CRE31")
        trail.record("E3", "AIRB", "US", {}, {}, "CRE32")
        return trail

    def test_no_filter(self) -> None:
        trail = self._populated_trail()
        assert len(trail.get_records()) == 4

    def test_filter_by_exposure_id(self) -> None:
        trail = self._populated_trail()
        recs = trail.get_records(exposure_id="E1")
        assert len(recs) == 2
        assert all(r.exposure_id == "E1" for r in recs)

    def test_filter_by_approach(self) -> None:
        trail = self._populated_trail()
        recs = trail.get_records(approach="SA")
        assert len(recs) == 2
        assert all(r.approach == "SA" for r in recs)

    def test_filter_by_both(self) -> None:
        trail = self._populated_trail()
        recs = trail.get_records(exposure_id="E1", approach="SA")
        assert len(recs) == 1
        assert recs[0].exposure_id == "E1"
        assert recs[0].approach == "SA"

    def test_filter_no_match(self) -> None:
        trail = self._populated_trail()
        recs = trail.get_records(exposure_id="NONEXISTENT")
        assert recs == []

    def test_filter_approach_no_match(self) -> None:
        trail = self._populated_trail()
        recs = trail.get_records(approach="NONEXISTENT")
        assert recs == []


# ---------------------------------------------------------------------------
# AuditTrail — to_dataframe
# ---------------------------------------------------------------------------


class TestAuditTrailToDataframe:
    def test_empty_trail(self) -> None:
        trail = AuditTrail()
        df = trail.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert "exposure_id" in df.columns
        assert "timestamp" in df.columns
        assert "approach" in df.columns
        assert "jurisdiction" in df.columns
        assert "inputs" in df.columns
        assert "outputs" in df.columns
        assert "regulatory_reference" in df.columns
        assert "engine_version" in df.columns
        assert "warnings" in df.columns

    def test_populated_trail(self) -> None:
        trail = AuditTrail()
        trail.record("E1", "SA", "EU", {"pd": 0.01}, {"rwa": 100}, "CRE20")
        trail.record("E2", "FIRB", "UK", {"pd": 0.02}, {"rwa": 200}, "CRE31")
        df = trail.to_dataframe()
        assert len(df) == 2
        assert list(df["exposure_id"]) == ["E1", "E2"]
        assert list(df["approach"]) == ["SA", "FIRB"]


# ---------------------------------------------------------------------------
# AuditTrail — export_json
# ---------------------------------------------------------------------------


class TestAuditTrailExportJson:
    def test_export_json_creates_file(self, tmp_path) -> None:
        trail = AuditTrail()
        trail.record("E1", "SA", "EU", {"pd": 0.01}, {"rwa": 100}, "CRE20")
        filepath = str(tmp_path / "audit.json")
        trail.export_json(filepath)

        with open(filepath) as fh:
            data = json.load(fh)

        assert len(data) == 1
        assert data[0]["exposure_id"] == "E1"
        assert data[0]["approach"] == "SA"
        assert data[0]["jurisdiction"] == "EU"
        assert data[0]["inputs"] == {"pd": 0.01}
        assert data[0]["outputs"] == {"rwa": 100}
        assert data[0]["regulatory_reference"] == "CRE20"
        assert data[0]["engine_version"] == creditriskengine.__version__
        assert data[0]["warnings"] == []
        # Timestamp should be an ISO string
        datetime.fromisoformat(data[0]["timestamp"])

    def test_export_json_empty(self, tmp_path) -> None:
        trail = AuditTrail()
        filepath = str(tmp_path / "empty.json")
        trail.export_json(filepath)
        with open(filepath) as fh:
            data = json.load(fh)
        assert data == []

    def test_export_json_multiple_records(self, tmp_path) -> None:
        trail = AuditTrail()
        trail.record("E1", "SA", "EU", {}, {}, "CRE20", warnings=["w1"])
        trail.record("E2", "FIRB", "UK", {}, {}, "CRE31")
        filepath = str(tmp_path / "multi.json")
        trail.export_json(filepath)
        with open(filepath) as fh:
            data = json.load(fh)
        assert len(data) == 2
        assert data[0]["warnings"] == ["w1"]
        assert data[1]["warnings"] == []


# ---------------------------------------------------------------------------
# AuditTrail — summary
# ---------------------------------------------------------------------------


class TestAuditTrailSummary:
    def test_empty_summary(self) -> None:
        trail = AuditTrail()
        s = trail.summary()
        assert s == {"by_approach": {}, "by_jurisdiction": {}}

    def test_populated_summary(self) -> None:
        trail = AuditTrail()
        trail.record("E1", "SA", "EU", {}, {}, "CRE20")
        trail.record("E2", "SA", "UK", {}, {}, "CRE20")
        trail.record("E3", "FIRB", "EU", {}, {}, "CRE31")
        s = trail.summary()
        assert s["by_approach"] == {"SA": 2, "FIRB": 1}
        assert s["by_jurisdiction"] == {"EU": 2, "UK": 1}

    def test_single_record_summary(self) -> None:
        trail = AuditTrail()
        trail.record("E1", "AIRB", "US", {}, {}, "CRE32")
        s = trail.summary()
        assert s["by_approach"] == {"AIRB": 1}
        assert s["by_jurisdiction"] == {"US": 1}
