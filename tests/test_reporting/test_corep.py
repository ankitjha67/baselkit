"""Tests for COREP (Common Reporting) template generation."""

import pytest

from creditriskengine.reporting.corep import (
    COREPTemplate,
    corep_summary,
    corep_to_dict,
    generate_c0700_sa,
    generate_c0801_irb,
    generate_c0802_airb,
)


def _sample_sa_data() -> dict[str, dict[str, float]]:
    return {
        "Corporates": {
            "original_exposure": 1_000_000,
            "credit_risk_mitigation": 200_000,
            "risk_weight_pct": 100.0,
            "provisions": 5_000,
        },
        "Retail": {
            "original_exposure": 500_000,
            "credit_risk_mitigation": 0,
            "risk_weight_pct": 75.0,
        },
    }


def _sample_irb_data() -> dict[str, dict[str, float]]:
    return {
        "Corporates — Other": {
            "original_exposure": 2_000_000,
            "credit_risk_mitigation": 100_000,
            "ead_post_crm": 1_900_000,
            "rwa": 1_200_000,
            "expected_loss": 15_000,
        },
        "Institutions": {
            "original_exposure": 500_000,
            "credit_risk_mitigation": 0,
            "ead_post_crm": 500_000,
            "rwa": 100_000,
            "expected_loss": 2_000,
        },
    }


class TestGenerateC0700SA:
    """C 07.00 -- Credit risk SA template."""

    def test_produces_corep_template(self) -> None:
        tpl = generate_c0700_sa(_sample_sa_data(), "2025-12-31", "Test Bank")
        assert isinstance(tpl, COREPTemplate)
        assert tpl.template_id == "C 07.00"

    def test_correct_row_count(self) -> None:
        tpl = generate_c0700_sa(_sample_sa_data(), "2025-12-31")
        assert len(tpl.rows) == 2

    def test_rwa_calculated(self) -> None:
        tpl = generate_c0700_sa(_sample_sa_data(), "2025-12-31")
        corp_row = [r for r in tpl.rows if r.exposure_class == "Corporates"][0]
        # RWA = (1M - 200k) * 100% = 800k
        assert corp_row.rwa == pytest.approx(800_000, rel=1e-6)

    def test_total_rwa(self) -> None:
        tpl = generate_c0700_sa(_sample_sa_data(), "2025-12-31")
        # Corporates: 800k, Retail: 500k * 75% = 375k
        assert tpl.total_rwa == pytest.approx(1_175_000, rel=1e-6)

    def test_metadata_populated(self) -> None:
        tpl = generate_c0700_sa(_sample_sa_data(), "2025-12-31", "Test Bank")
        assert tpl.metadata["approach"] == "SA"
        assert tpl.metadata["total_rwa"] == pytest.approx(1_175_000, rel=1e-6)

    def test_empty_input(self) -> None:
        tpl = generate_c0700_sa({}, "2025-12-31")
        assert len(tpl.rows) == 0
        assert tpl.total_rwa == 0.0

    def test_non_standard_class_appended(self) -> None:
        data = {"CustomClass": {"original_exposure": 100, "risk_weight_pct": 50.0}}
        tpl = generate_c0700_sa(data, "2025-12-31")
        assert len(tpl.rows) == 1
        assert tpl.rows[0].exposure_class == "CustomClass"


class TestGenerateC0801IRB:
    """C 08.01 -- F-IRB template."""

    def test_produces_valid_template(self) -> None:
        tpl = generate_c0801_irb(_sample_irb_data(), "2025-12-31", "Test Bank")
        assert isinstance(tpl, COREPTemplate)
        assert tpl.template_id == "C 08.01"
        assert tpl.metadata["approach"] == "F-IRB"

    def test_correct_row_count(self) -> None:
        tpl = generate_c0801_irb(_sample_irb_data(), "2025-12-31")
        assert len(tpl.rows) == 2

    def test_total_expected_loss(self) -> None:
        tpl = generate_c0801_irb(_sample_irb_data(), "2025-12-31")
        assert tpl.total_expected_loss == pytest.approx(17_000, rel=1e-6)

    def test_total_rwa(self) -> None:
        tpl = generate_c0801_irb(_sample_irb_data(), "2025-12-31")
        assert tpl.total_rwa == pytest.approx(1_300_000, rel=1e-6)

    def test_non_standard_class_appended(self) -> None:
        data = {
            "CustomIRBClass": {
                "original_exposure": 100_000,
                "credit_risk_mitigation": 0,
                "ead_post_crm": 100_000,
                "rwa": 50_000,
                "expected_loss": 500,
            },
        }
        tpl = generate_c0801_irb(data, "2025-12-31")
        assert len(tpl.rows) == 1
        assert tpl.rows[0].exposure_class == "CustomIRBClass"


class TestGenerateC0802AIRB:
    """C 08.02 -- A-IRB template."""

    def test_produces_valid_template(self) -> None:
        tpl = generate_c0802_airb(_sample_irb_data(), "2025-12-31", "Test Bank")
        assert isinstance(tpl, COREPTemplate)
        assert tpl.template_id == "C 08.02"
        assert tpl.metadata["approach"] == "A-IRB"

    def test_non_standard_class_appended(self) -> None:
        data = {
            "CustomAIRBClass": {
                "original_exposure": 200_000,
                "credit_risk_mitigation": 10_000,
                "ead_post_crm": 190_000,
                "rwa": 80_000,
                "expected_loss": 1_000,
            },
        }
        tpl = generate_c0802_airb(data, "2025-12-31")
        assert len(tpl.rows) == 1
        assert tpl.rows[0].exposure_class == "CustomAIRBClass"


class TestCOREPToDict:
    """Serialization."""

    def test_contains_required_keys(self) -> None:
        tpl = generate_c0700_sa(_sample_sa_data(), "2025-12-31", "Bank X")
        d = corep_to_dict(tpl)
        assert "template_id" in d
        assert "rows" in d
        assert "summary" in d
        assert d["summary"]["row_count"] == 2

    def test_rows_are_dicts(self) -> None:
        tpl = generate_c0700_sa(_sample_sa_data(), "2025-12-31")
        d = corep_to_dict(tpl)
        assert all(isinstance(r, dict) for r in d["rows"])

    def test_institution_name_preserved(self) -> None:
        tpl = generate_c0700_sa(_sample_sa_data(), "2025-12-31", "Mega Bank")
        d = corep_to_dict(tpl)
        assert d["institution_name"] == "Mega Bank"


class TestCOREPSummary:
    """Aggregation across multiple templates."""

    def test_summary_aggregation(self) -> None:
        sa = generate_c0700_sa(_sample_sa_data(), "2025-12-31", "Bank A")
        irb = generate_c0801_irb(_sample_irb_data(), "2025-12-31", "Bank A")
        summary = corep_summary([sa, irb])
        assert summary["template_count"] == 2
        assert summary["totals"]["total_rwa"] == pytest.approx(
            sa.total_rwa + irb.total_rwa, rel=1e-6
        )

    def test_empty_list(self) -> None:
        summary = corep_summary([])
        assert summary["templates"] == []
        assert summary["totals"] == {}

    def test_capital_requirement(self) -> None:
        sa = generate_c0700_sa(_sample_sa_data(), "2025-12-31")
        summary = corep_summary([sa])
        assert summary["totals"]["capital_requirement_8pct"] == pytest.approx(
            sa.total_rwa * 0.08, rel=1e-6
        )
