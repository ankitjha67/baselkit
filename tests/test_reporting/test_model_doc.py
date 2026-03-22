"""Tests for model documentation generator."""

import pytest
from datetime import date

from creditriskengine.reporting.model_doc import (
    ModelDocumentation,
    generate_model_card,
    generate_model_doc_report,
    generate_validation_report,
    model_inventory_entry,
)


def _make_doc(**overrides) -> ModelDocumentation:
    """Helper to create a sample ModelDocumentation."""
    defaults = dict(
        model_name="Corporate PD Model",
        model_id="PD-CORP-001",
        model_type="PD",
        model_owner="Risk Analytics",
        version="2.1",
        effective_date=date(2025, 1, 1),
        purpose="IRB PD estimation for corporate exposures",
        scope="EU wholesale corporate portfolio",
        regulatory_use="CRR3 IRB Foundation",
        methodology="Logistic regression with macro overlays",
        data_description="10-year internal default history",
        sample_period="2015-2024",
        sample_size=50_000,
        discrimination_metrics={"gini": 0.55, "auroc": 0.775},
        calibration_metrics={"p_value": 0.12, "hl_stat": 8.5},
        stability_metrics={"psi": 0.08},
        validation_status="approved",
        validation_date=date(2025, 6, 15),
        findings=["Minor calibration drift in SME sub-segment"],
        limitations=["Limited data for low-default portfolios"],
    )
    defaults.update(overrides)
    return ModelDocumentation(**defaults)


class TestGenerateModelCard:
    """Model card generation."""

    def test_returns_dict(self):
        card = generate_model_card(_make_doc())
        assert isinstance(card, dict)

    def test_required_top_level_keys(self):
        card = generate_model_card(_make_doc())
        assert "model_details" in card
        assert "intended_use" in card
        assert "development" in card
        assert "performance" in card
        assert "validation" in card
        assert "limitations" in card

    def test_model_details_populated(self):
        card = generate_model_card(_make_doc())
        details = card["model_details"]
        assert details["name"] == "Corporate PD Model"
        assert details["id"] == "PD-CORP-001"
        assert details["type"] == "PD"
        assert details["version"] == "2.1"

    def test_performance_rag_green(self):
        card = generate_model_card(_make_doc())
        perf = card["performance"]
        assert perf["discrimination"]["rag"] == "green"
        assert perf["stability"]["rag"] == "green"
        assert perf["calibration"]["rag"] == "green"
        assert perf["overall_rag"] == "green"

    def test_poor_gini_yellow_rag(self):
        doc = _make_doc(discrimination_metrics={"gini": 0.30, "auroc": 0.65})
        card = generate_model_card(doc)
        assert card["performance"]["discrimination"]["rag"] == "yellow"

    def test_poor_gini_red_rag(self):
        doc = _make_doc(discrimination_metrics={"gini": 0.15, "auroc": 0.575})
        card = generate_model_card(doc)
        assert card["performance"]["discrimination"]["rag"] == "red"

    def test_high_psi_yellow_rag(self):
        doc = _make_doc(stability_metrics={"psi": 0.15})
        card = generate_model_card(doc)
        assert card["performance"]["stability"]["rag"] == "yellow"

    def test_high_psi_red_rag(self):
        doc = _make_doc(stability_metrics={"psi": 0.30})
        card = generate_model_card(doc)
        assert card["performance"]["stability"]["rag"] == "red"

    def test_overall_rag_worst_of_all(self):
        doc = _make_doc(
            discrimination_metrics={"gini": 0.55, "auroc": 0.775},
            stability_metrics={"psi": 0.30},
        )
        card = generate_model_card(doc)
        assert card["performance"]["overall_rag"] == "red"

    def test_validation_finding_count(self):
        card = generate_model_card(_make_doc())
        assert card["validation"]["finding_count"] == 1

    def test_effective_date_none(self):
        doc = _make_doc(effective_date=None)
        card = generate_model_card(doc)
        assert card["model_details"]["effective_date"] is None


class TestGenerateModelDocReport:
    """Full model documentation report."""

    def test_returns_nonempty_string(self):
        report = generate_model_doc_report(_make_doc())
        assert isinstance(report, str)
        assert len(report) > 100

    def test_contains_model_name(self):
        report = generate_model_doc_report(_make_doc())
        assert "Corporate PD Model" in report

    def test_contains_methodology_section(self):
        report = generate_model_doc_report(_make_doc())
        assert "METHODOLOGY" in report or "methodology" in report.lower()


class TestGenerateValidationReport:
    """Validation report generation."""

    def test_returns_nonempty_string(self):
        report = generate_validation_report(_make_doc())
        assert isinstance(report, str)
        assert len(report) > 100

    def test_contains_validation_header(self):
        report = generate_validation_report(_make_doc())
        assert "VALIDATION REPORT" in report

    def test_overall_assessment(self):
        report = generate_validation_report(_make_doc())
        assert "GREEN" in report

    def test_override_metrics(self):
        report = generate_validation_report(
            _make_doc(),
            discrimination={"gini": 0.10, "auroc": 0.55},
        )
        assert "RED" in report


class TestModelInventoryEntry:
    """Model inventory row."""

    def test_returns_dict(self):
        entry = model_inventory_entry(_make_doc())
        assert isinstance(entry, dict)

    def test_required_keys(self):
        entry = model_inventory_entry(_make_doc())
        for key in [
            "model_name",
            "model_id",
            "model_type",
            "version",
            "overall_rag",
            "validation_status",
            "finding_count",
            "limitation_count",
        ]:
            assert key in entry

    def test_overall_rag_value(self):
        entry = model_inventory_entry(_make_doc())
        assert entry["overall_rag"] == "green"

    def test_discrimination_sub_dict(self):
        entry = model_inventory_entry(_make_doc())
        assert entry["discrimination"]["gini"] == pytest.approx(0.55)
        assert entry["discrimination"]["rag"] == "green"
