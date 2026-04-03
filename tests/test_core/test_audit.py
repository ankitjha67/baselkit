"""Tests for creditriskengine.core.audit module."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd
import pytest

import creditriskengine
from creditriskengine.core.audit import AuditTrail, CalculationRecord, OverlayAuditRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(**overrides) -> CalculationRecord:
    """Build a CalculationRecord with sensible defaults."""
    defaults = {
        "exposure_id": "EXP-001",
        "timestamp": datetime(2025, 6, 15, 12, 0, 0, tzinfo=UTC),
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
            timestamp=datetime.now(UTC),
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
        before = datetime.now(UTC)
        rec = trail.record("E1", "SA", "EU", {}, {}, "CRE20")
        after = datetime.now(UTC)
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
            export = json.load(fh)

        data = export["calculations"]
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
        # Overlay section should be empty
        assert export["overlays"] == []

    def test_export_json_empty(self, tmp_path) -> None:
        trail = AuditTrail()
        filepath = str(tmp_path / "empty.json")
        trail.export_json(filepath)
        with open(filepath) as fh:
            export = json.load(fh)
        assert export == {"calculations": [], "overlays": []}

    def test_export_json_multiple_records(self, tmp_path) -> None:
        trail = AuditTrail()
        trail.record("E1", "SA", "EU", {}, {}, "CRE20", warnings=["w1"])
        trail.record("E2", "FIRB", "UK", {}, {}, "CRE31")
        filepath = str(tmp_path / "multi.json")
        trail.export_json(filepath)
        with open(filepath) as fh:
            export = json.load(fh)
        data = export["calculations"]
        assert len(data) == 2
        assert data[0]["warnings"] == ["w1"]
        assert data[1]["warnings"] == []

    def test_export_json_with_overlays(self, tmp_path) -> None:
        trail = AuditTrail()
        trail.record("E1", "SA", "EU", {}, {}, "CRE20")
        trail.record_overlay(
            overlay_name="CRE stress",
            overlay_type="sector_specific",
            event="applied",
            model_ecl=100.0,
            overlay_ecl=150.0,
            rationale="CRE downturn",
            regulatory_basis="IFRS 9.B5.5.52",
            approved_by="CRO",
        )
        filepath = str(tmp_path / "with_overlays.json")
        trail.export_json(filepath)
        with open(filepath) as fh:
            export = json.load(fh)
        assert len(export["calculations"]) == 1
        assert len(export["overlays"]) == 1
        ov = export["overlays"][0]
        assert ov["overlay_name"] == "CRE stress"
        assert ov["overlay_type"] == "sector_specific"
        assert ov["event"] == "applied"
        assert ov["model_ecl"] == 100.0
        assert ov["overlay_ecl"] == 150.0
        assert ov["adjustment"] == 50.0


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


# ---------------------------------------------------------------------------
# OverlayAuditRecord
# ---------------------------------------------------------------------------


class TestOverlayAuditRecord:
    def test_creation(self) -> None:
        rec = OverlayAuditRecord(
            overlay_name="CRE stress",
            overlay_type="sector_specific",
            event="applied",
            model_ecl=100.0,
            overlay_ecl=150.0,
            adjustment=50.0,
            rationale="CRE downturn",
            regulatory_basis="IFRS 9.B5.5.52",
            approved_by="CRO",
        )
        assert rec.overlay_name == "CRE stress"
        assert rec.adjustment == 50.0
        assert rec.event == "applied"

    def test_frozen(self) -> None:
        rec = OverlayAuditRecord(
            overlay_name="Test",
            overlay_type="emerging_risk",
            event="applied",
            model_ecl=100.0,
            overlay_ecl=120.0,
            adjustment=20.0,
        )
        with pytest.raises(AttributeError):
            rec.overlay_name = "Changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AuditTrail — overlay audit methods
# ---------------------------------------------------------------------------


class TestAuditTrailOverlays:
    def test_record_overlay(self) -> None:
        trail = AuditTrail()
        rec = trail.record_overlay(
            overlay_name="Geopolitical risk",
            overlay_type="emerging_risk",
            event="applied",
            model_ecl=100.0,
            overlay_ecl=125.0,
            rationale="Ukraine conflict impact",
            approved_by="CRC",
        )
        assert isinstance(rec, OverlayAuditRecord)
        assert rec.overlay_name == "Geopolitical risk"
        assert rec.adjustment == 25.0
        assert len(trail.overlay_records) == 1

    def test_overlay_records_returns_copy(self) -> None:
        trail = AuditTrail()
        trail.record_overlay("Test", "emerging_risk", "applied", 100.0, 110.0)
        records = trail.overlay_records
        records.clear()
        assert len(trail.overlay_records) == 1

    def test_get_overlay_records_by_name(self) -> None:
        trail = AuditTrail()
        trail.record_overlay("A", "model_limitation", "applied", 100.0, 110.0)
        trail.record_overlay("B", "sector_specific", "applied", 200.0, 250.0)
        trail.record_overlay("A", "model_limitation", "reviewed", 110.0, 110.0)
        recs = trail.get_overlay_records(overlay_name="A")
        assert len(recs) == 2
        assert all(r.overlay_name == "A" for r in recs)

    def test_get_overlay_records_by_event(self) -> None:
        trail = AuditTrail()
        trail.record_overlay("A", "emerging_risk", "applied", 100.0, 120.0)
        trail.record_overlay("A", "emerging_risk", "reviewed", 120.0, 120.0)
        trail.record_overlay("A", "emerging_risk", "revoked", 120.0, 100.0)
        recs = trail.get_overlay_records(event="applied")
        assert len(recs) == 1

    def test_overlay_summary(self) -> None:
        trail = AuditTrail()
        trail.record_overlay("A", "emerging_risk", "applied", 100.0, 125.0)
        trail.record_overlay("B", "sector_specific", "applied", 200.0, 250.0)
        trail.record_overlay("A", "emerging_risk", "reviewed", 125.0, 125.0)
        s = trail.overlay_summary()
        assert s["by_type"] == {"emerging_risk": 2, "sector_specific": 1}
        assert s["by_event"] == {"applied": 2, "reviewed": 1}
        # total_adjustment counts only "applied" events
        assert s["total_adjustment"] == pytest.approx(75.0)

    def test_overlay_summary_empty(self) -> None:
        trail = AuditTrail()
        s = trail.overlay_summary()
        assert s["by_type"] == {}
        assert s["by_event"] == {}
        assert s["total_adjustment"] == 0.0
