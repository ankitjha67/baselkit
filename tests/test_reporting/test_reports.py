"""Tests for regulatory reporting templates."""

from datetime import date

import pytest

from creditriskengine.reporting.reports import (
    generate_corep_credit_risk_summary,
    generate_model_inventory_entry,
    generate_pillar3_credit_risk,
)


class TestCOREPSummary:
    def test_basic(self) -> None:
        report = generate_corep_credit_risk_summary(
            jurisdiction="eu",
            reporting_date=date(2026, 3, 31),
            sa_rwa=10000.0,
            irb_rwa=8000.0,
            floored_rwa=8500.0,
            total_ead=50000.0,
            total_ecl=500.0,
        )
        assert report["report_type"] == "COREP_CR"
        assert report["capital_requirement_8pct"] == pytest.approx(680.0)
        assert report["output_floor_binding"] is True
        assert report["ecl_to_ead_ratio"] == pytest.approx(0.01)

    def test_floor_not_binding(self) -> None:
        report = generate_corep_credit_risk_summary(
            "bcbs", date(2026, 1, 1), 5000.0, 8000.0, 8000.0, 40000.0, 400.0,
        )
        assert report["output_floor_binding"] is False


class TestPillar3:
    def test_basic(self) -> None:
        classes = [
            {"class_name": "Corporate", "ead": 10000.0, "rwa": 8000.0, "expected_loss": 100.0},
            {"class_name": "Retail", "ead": 5000.0, "rwa": 3000.0, "expected_loss": 50.0},
        ]
        report = generate_pillar3_credit_risk(date(2026, 3, 31), classes)
        assert report["total_ead"] == pytest.approx(15000.0)
        assert report["total_rwa"] == pytest.approx(11000.0)
        assert report["avg_risk_weight"] == pytest.approx(11000.0 / 15000.0)

    def test_empty_classes(self) -> None:
        report = generate_pillar3_credit_risk(date(2026, 1, 1), [])
        assert report["total_ead"] == 0
        assert report["avg_risk_weight"] == 0.0


class TestModelInventory:
    def test_green(self) -> None:
        entry = generate_model_inventory_entry(
            "PD_Corporate", "PD", "Corporate", date(2026, 1, 15),
            auroc=0.80, gini=0.60, psi=0.05, calibration_result="green",
        )
        assert entry["overall_rag"] == "green"
        assert entry["discrimination"]["rag"] == "green"

    def test_yellow_low_gini(self) -> None:
        entry = generate_model_inventory_entry(
            "PD_Retail", "PD", "Retail", date(2026, 1, 15),
            auroc=0.60, gini=0.20, psi=0.05, calibration_result="green",
        )
        assert entry["overall_rag"] == "yellow"

    def test_red_high_psi(self) -> None:
        entry = generate_model_inventory_entry(
            "LGD_Corp", "LGD", "Corporate", date(2026, 1, 15),
            auroc=0.80, gini=0.60, psi=0.30, calibration_result="green",
        )
        assert entry["overall_rag"] == "red"
        assert entry["stability"]["rag"] == "red"

    def test_red_low_gini(self) -> None:
        entry = generate_model_inventory_entry(
            "PD_Low", "PD", "Retail", date(2026, 1, 15),
            auroc=0.55, gini=0.10, psi=0.05, calibration_result="green",
        )
        assert entry["discrimination"]["rag"] == "red"

    def test_yellow_moderate_psi(self) -> None:
        entry = generate_model_inventory_entry(
            "PD_ModPSI", "PD", "Corporate", date(2026, 1, 15),
            auroc=0.80, gini=0.60, psi=0.15, calibration_result="green",
        )
        assert entry["stability"]["rag"] == "yellow"
