"""Tests for model documentation generator."""

from datetime import date

import pytest

from creditriskengine.reporting.model_doc import (
    ModelDocumentation,
    _generate_plain_text_report,
    _rag_calibration,
    _rag_discrimination,
    _rag_stability,
    _worst_rag,
    generate_model_card,
    generate_model_doc_report,
    generate_validation_report,
    model_inventory_entry,
)


def _make_doc(**overrides: object) -> ModelDocumentation:
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

    def test_returns_dict(self) -> None:
        card = generate_model_card(_make_doc())
        assert isinstance(card, dict)

    def test_required_top_level_keys(self) -> None:
        card = generate_model_card(_make_doc())
        assert "model_details" in card
        assert "intended_use" in card
        assert "development" in card
        assert "performance" in card
        assert "validation" in card
        assert "limitations" in card

    def test_model_details_populated(self) -> None:
        card = generate_model_card(_make_doc())
        details = card["model_details"]
        assert details["name"] == "Corporate PD Model"
        assert details["id"] == "PD-CORP-001"
        assert details["type"] == "PD"
        assert details["version"] == "2.1"

    def test_performance_rag_green(self) -> None:
        card = generate_model_card(_make_doc())
        perf = card["performance"]
        assert perf["discrimination"]["rag"] == "green"
        assert perf["stability"]["rag"] == "green"
        assert perf["calibration"]["rag"] == "green"
        assert perf["overall_rag"] == "green"

    def test_poor_gini_yellow_rag(self) -> None:
        doc = _make_doc(discrimination_metrics={"gini": 0.30, "auroc": 0.65})
        card = generate_model_card(doc)
        assert card["performance"]["discrimination"]["rag"] == "yellow"

    def test_poor_gini_red_rag(self) -> None:
        doc = _make_doc(discrimination_metrics={"gini": 0.15, "auroc": 0.575})
        card = generate_model_card(doc)
        assert card["performance"]["discrimination"]["rag"] == "red"

    def test_high_psi_yellow_rag(self) -> None:
        doc = _make_doc(stability_metrics={"psi": 0.15})
        card = generate_model_card(doc)
        assert card["performance"]["stability"]["rag"] == "yellow"

    def test_high_psi_red_rag(self) -> None:
        doc = _make_doc(stability_metrics={"psi": 0.30})
        card = generate_model_card(doc)
        assert card["performance"]["stability"]["rag"] == "red"

    def test_overall_rag_worst_of_all(self) -> None:
        doc = _make_doc(
            discrimination_metrics={"gini": 0.55, "auroc": 0.775},
            stability_metrics={"psi": 0.30},
        )
        card = generate_model_card(doc)
        assert card["performance"]["overall_rag"] == "red"

    def test_validation_finding_count(self) -> None:
        card = generate_model_card(_make_doc())
        assert card["validation"]["finding_count"] == 1

    def test_effective_date_none(self) -> None:
        doc = _make_doc(effective_date=None)
        card = generate_model_card(doc)
        assert card["model_details"]["effective_date"] is None


class TestGenerateModelDocReport:
    """Full model documentation report."""

    def test_returns_nonempty_string(self) -> None:
        report = generate_model_doc_report(_make_doc())
        assert isinstance(report, str)
        assert len(report) > 100

    def test_contains_model_name(self) -> None:
        report = generate_model_doc_report(_make_doc())
        assert "Corporate PD Model" in report

    def test_contains_methodology_section(self) -> None:
        report = generate_model_doc_report(_make_doc())
        assert "METHODOLOGY" in report or "methodology" in report.lower()


class TestGenerateValidationReport:
    """Validation report generation."""

    def test_returns_nonempty_string(self) -> None:
        report = generate_validation_report(_make_doc())
        assert isinstance(report, str)
        assert len(report) > 100

    def test_contains_validation_header(self) -> None:
        report = generate_validation_report(_make_doc())
        assert "VALIDATION REPORT" in report

    def test_overall_assessment(self) -> None:
        report = generate_validation_report(_make_doc())
        assert "GREEN" in report

    def test_override_metrics(self) -> None:
        report = generate_validation_report(
            _make_doc(),
            discrimination={"gini": 0.10, "auroc": 0.55},
        )
        assert "RED" in report


class TestModelInventoryEntry:
    """Model inventory row."""

    def test_returns_dict(self) -> None:
        entry = model_inventory_entry(_make_doc())
        assert isinstance(entry, dict)

    def test_required_keys(self) -> None:
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

    def test_overall_rag_value(self) -> None:
        entry = model_inventory_entry(_make_doc())
        assert entry["overall_rag"] == "green"

    def test_discrimination_sub_dict(self) -> None:
        entry = model_inventory_entry(_make_doc())
        assert entry["discrimination"]["gini"] == pytest.approx(0.55)
        assert entry["discrimination"]["rag"] == "green"

    def test_stability_sub_dict(self) -> None:
        entry = model_inventory_entry(_make_doc())
        assert entry["stability"]["psi"] == pytest.approx(0.08)
        assert entry["stability"]["rag"] == "green"

    def test_calibration_sub_dict(self) -> None:
        entry = model_inventory_entry(_make_doc())
        assert entry["calibration"]["p_value"] == pytest.approx(0.12)
        assert entry["calibration"]["rag"] == "green"

    def test_effective_date_none(self) -> None:
        entry = model_inventory_entry(_make_doc(effective_date=None))
        assert entry["effective_date"] is None

    def test_validation_date_none(self) -> None:
        entry = model_inventory_entry(_make_doc(validation_date=None))
        assert entry["validation_date"] is None


class TestRAGHelpers:
    """Test internal RAG status functions."""

    def test_rag_discrimination_green(self) -> None:
        assert _rag_discrimination(0.50) == "green"

    def test_rag_discrimination_yellow(self) -> None:
        assert _rag_discrimination(0.30) == "yellow"

    def test_rag_discrimination_red(self) -> None:
        assert _rag_discrimination(0.15) == "red"

    def test_rag_discrimination_boundary_green(self) -> None:
        assert _rag_discrimination(0.40) == "green"

    def test_rag_discrimination_boundary_yellow(self) -> None:
        assert _rag_discrimination(0.25) == "yellow"

    def test_rag_stability_green(self) -> None:
        assert _rag_stability(0.05) == "green"

    def test_rag_stability_yellow(self) -> None:
        assert _rag_stability(0.15) == "yellow"

    def test_rag_stability_red(self) -> None:
        assert _rag_stability(0.30) == "red"

    def test_rag_stability_boundary(self) -> None:
        assert _rag_stability(0.10) == "yellow"

    def test_rag_calibration_green(self) -> None:
        assert _rag_calibration(0.10) == "green"

    def test_rag_calibration_yellow(self) -> None:
        assert _rag_calibration(0.02) == "yellow"

    def test_rag_calibration_red(self) -> None:
        assert _rag_calibration(0.005) == "red"

    def test_rag_calibration_boundary(self) -> None:
        assert _rag_calibration(0.05) == "green"

    def test_worst_rag_all_green(self) -> None:
        assert _worst_rag("green", "green", "green") == "green"

    def test_worst_rag_one_yellow(self) -> None:
        assert _worst_rag("green", "yellow", "green") == "yellow"

    def test_worst_rag_one_red(self) -> None:
        assert _worst_rag("green", "yellow", "red") == "red"


class TestPlainTextReport:
    """Test plain-text report generation directly."""

    def test_contains_all_sections(self) -> None:
        report = _generate_plain_text_report(_make_doc())
        assert "1. MODEL OVERVIEW" in report
        assert "2. INTENDED USE" in report
        assert "3. METHODOLOGY" in report
        assert "4. DATA" in report
        assert "5. PERFORMANCE METRICS" in report
        assert "6. VALIDATION" in report
        assert "7. LIMITATIONS" in report

    def test_no_metrics(self) -> None:
        doc = _make_doc(
            discrimination_metrics={},
            calibration_metrics={},
            stability_metrics={},
        )
        report = _generate_plain_text_report(doc)
        assert "No discrimination metrics" in report
        assert "No calibration metrics" in report
        assert "No stability metrics" in report

    def test_no_findings(self) -> None:
        doc = _make_doc(findings=[])
        report = _generate_plain_text_report(doc)
        # Should not contain numbered findings
        assert "Findings:" not in report

    def test_no_limitations(self) -> None:
        doc = _make_doc(limitations=[])
        report = _generate_plain_text_report(doc)
        assert "No known limitations" in report

    def test_effective_date_none(self) -> None:
        doc = _make_doc(effective_date=None)
        report = _generate_plain_text_report(doc)
        assert "N/A" in report


class TestValidationReportExtended:
    """Additional validation report tests."""

    def test_red_overall_has_urgent(self) -> None:
        doc = _make_doc(
            discrimination_metrics={"gini": 0.10, "auroc": 0.55},
            stability_metrics={"psi": 0.30},
        )
        report = generate_validation_report(doc)
        assert "URGENT" in report

    def test_yellow_overall_has_action(self) -> None:
        doc = _make_doc(
            discrimination_metrics={"gini": 0.30, "auroc": 0.65},
            stability_metrics={"psi": 0.05},
        )
        report = generate_validation_report(doc)
        assert "ACTION" in report

    def test_override_all_metrics(self) -> None:
        report = generate_validation_report(
            _make_doc(),
            discrimination={"gini": 0.50, "auroc": 0.75},
            calibration={"p_value": 0.10},
            stability={"psi": 0.05},
        )
        assert "GREEN" in report

    def test_findings_in_report(self) -> None:
        doc = _make_doc(findings=["Finding A", "Finding B"])
        report = generate_validation_report(doc)
        assert "Finding A" in report
        assert "Finding B" in report

    def test_no_findings_in_report(self) -> None:
        doc = _make_doc(findings=[])
        report = generate_validation_report(doc)
        assert "No material findings" in report

    def test_status_text_in_conclusion(self) -> None:
        for status in ["approved", "conditional", "rejected", "pending"]:
            doc = _make_doc(validation_status=status)
            report = generate_validation_report(doc)
            assert status.upper() in report

    def test_validation_date_none(self) -> None:
        doc = _make_doc(validation_date=None)
        report = generate_validation_report(doc)
        assert "N/A" in report

    def test_additional_disc_metrics(self) -> None:
        report = generate_validation_report(
            _make_doc(),
            discrimination={"gini": 0.50, "auroc": 0.75, "ks": 0.40},
        )
        assert "ks" in report
