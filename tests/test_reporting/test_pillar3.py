"""Tests for Pillar 3 disclosure templates."""

import pytest

from creditriskengine.reporting.pillar3 import (
    generate_cr1_template,
    generate_cr3_crm_overview,
    generate_cr4_sa_overview,
    generate_cr6_irb_overview,
)


class TestCR1Template:
    """CR1: Credit quality of assets."""

    def test_basic(self) -> None:
        result = generate_cr1_template(
            total_defaulted=500.0,
            total_non_defaulted=9500.0,
            specific_provisions=200.0,
            general_provisions=100.0,
        )
        assert result["template"] == "CR1"
        assert result["defaulted_exposures"]["gross_carrying_amount"] == 500.0
        assert result["defaulted_exposures"]["specific_provisions"] == 200.0
        assert result["defaulted_exposures"]["net_carrying_amount"] == pytest.approx(300.0)
        assert result["non_defaulted_exposures"]["gross_carrying_amount"] == 9500.0
        assert result["non_defaulted_exposures"]["general_provisions"] == 100.0
        assert result["non_defaulted_exposures"]["net_carrying_amount"] == pytest.approx(9400.0)
        assert result["total"]["gross_carrying_amount"] == pytest.approx(10000.0)
        assert result["total"]["total_provisions"] == pytest.approx(300.0)
        assert result["total"]["net_carrying_amount"] == pytest.approx(9700.0)

    def test_zero_provisions(self) -> None:
        result = generate_cr1_template(100.0, 900.0, 0.0, 0.0)
        assert result["total"]["total_provisions"] == 0.0
        assert result["total"]["net_carrying_amount"] == pytest.approx(1000.0)

    def test_all_defaulted(self) -> None:
        result = generate_cr1_template(1000.0, 0.0, 500.0, 0.0)
        assert result["total"]["gross_carrying_amount"] == 1000.0
        assert result["defaulted_exposures"]["net_carrying_amount"] == pytest.approx(500.0)


class TestCR3CRMOverview:
    """CR3: CRM techniques overview."""

    def test_basic(self) -> None:
        result = generate_cr3_crm_overview(
            exposures_unsecured=5000.0,
            exposures_secured_collateral=3000.0,
            exposures_secured_guarantees=1500.0,
            exposures_secured_credit_derivatives=500.0,
        )
        assert result["template"] == "CR3"
        assert result["exposures_unsecured"] == 5000.0
        assert result["exposures_secured"]["total_secured"] == pytest.approx(5000.0)
        assert result["total_exposures"] == pytest.approx(10000.0)
        assert result["secured_pct"] == pytest.approx(0.50)

    def test_all_unsecured(self) -> None:
        result = generate_cr3_crm_overview(10000.0, 0.0, 0.0, 0.0)
        assert result["secured_pct"] == pytest.approx(0.0)
        assert result["exposures_secured"]["total_secured"] == 0.0

    def test_all_secured(self) -> None:
        result = generate_cr3_crm_overview(0.0, 5000.0, 3000.0, 2000.0)
        assert result["secured_pct"] == pytest.approx(1.0)

    def test_zero_total_exposures(self) -> None:
        result = generate_cr3_crm_overview(0.0, 0.0, 0.0, 0.0)
        assert result["secured_pct"] == 0.0


class TestCR4SAOverview:
    """CR4: Standardised approach overview."""

    def test_basic(self) -> None:
        data = [
            {
                "exposure_class": "Corporates",
                "gross_exposure": 5000.0,
                "crm_adjustments": -500.0,
                "net_exposure": 4500.0,
                "rwa": 4500.0,
            },
            {
                "exposure_class": "Retail",
                "gross_exposure": 3000.0,
                "crm_adjustments": -300.0,
                "net_exposure": 2700.0,
                "rwa": 2025.0,
            },
        ]
        result = generate_cr4_sa_overview(data)
        assert result["template"] == "CR4"
        assert len(result["exposure_classes"]) == 2
        assert result["totals"]["gross_exposure"] == pytest.approx(8000.0)
        assert result["totals"]["net_exposure"] == pytest.approx(7200.0)
        assert result["totals"]["rwa"] == pytest.approx(6525.0)
        assert result["totals"]["avg_risk_weight"] == pytest.approx(6525.0 / 7200.0)

    def test_risk_weight_per_class(self) -> None:
        data = [
            {
                "exposure_class": "Sovereign",
                "gross_exposure": 10000.0,
                "crm_adjustments": 0.0,
                "net_exposure": 10000.0,
                "rwa": 0.0,
            },
        ]
        result = generate_cr4_sa_overview(data)
        assert result["exposure_classes"][0]["risk_weight"] == pytest.approx(0.0)

    def test_empty_classes(self) -> None:
        result = generate_cr4_sa_overview([])
        assert result["totals"]["gross_exposure"] == 0.0
        assert result["totals"]["avg_risk_weight"] == 0.0

    def test_zero_net_exposure_class(self) -> None:
        data = [
            {
                "exposure_class": "Test",
                "gross_exposure": 100.0,
                "crm_adjustments": -100.0,
                "net_exposure": 0.0,
                "rwa": 0.0,
            },
        ]
        result = generate_cr4_sa_overview(data)
        assert result["exposure_classes"][0]["risk_weight"] == 0.0


class TestCR6IRBOverview:
    """CR6: IRB credit risk by PD grade."""

    def test_basic(self) -> None:
        data = [
            {
                "pd_range": "0.00-0.15%",
                "n_obligors": 100,
                "ead": 5000.0,
                "avg_pd": 0.001,
                "avg_lgd": 0.40,
                "avg_rw": 0.30,
                "rwa": 1500.0,
                "el": 2.0,
            },
            {
                "pd_range": "0.15-0.50%",
                "n_obligors": 200,
                "ead": 8000.0,
                "avg_pd": 0.003,
                "avg_lgd": 0.42,
                "avg_rw": 0.55,
                "rwa": 4400.0,
                "el": 10.08,
            },
        ]
        result = generate_cr6_irb_overview(data)
        assert result["template"] == "CR6"
        assert len(result["pd_grades"]) == 2
        assert result["totals"]["n_obligors"] == 300
        assert result["totals"]["ead"] == pytest.approx(13000.0)
        assert result["totals"]["rwa"] == pytest.approx(5900.0)
        assert result["totals"]["el"] == pytest.approx(12.08)

    def test_weighted_avg_pd(self) -> None:
        data = [
            {"pd_range": "low", "n_obligors": 50, "ead": 1000.0, "avg_pd": 0.01,
             "avg_lgd": 0.40, "avg_rw": 0.50, "rwa": 500.0, "el": 4.0},
            {"pd_range": "high", "n_obligors": 50, "ead": 1000.0, "avg_pd": 0.05,
             "avg_lgd": 0.40, "avg_rw": 0.80, "rwa": 800.0, "el": 20.0},
        ]
        result = generate_cr6_irb_overview(data)
        expected_wavg_pd = (0.01 * 1000.0 + 0.05 * 1000.0) / 2000.0
        assert result["totals"]["weighted_avg_pd"] == pytest.approx(expected_wavg_pd)

    def test_empty_data(self) -> None:
        result = generate_cr6_irb_overview([])
        assert result["totals"]["n_obligors"] == 0
        assert result["totals"]["ead"] == 0.0
        assert result["totals"]["weighted_avg_pd"] == 0.0

    def test_single_grade(self) -> None:
        data = [
            {"pd_range": "0.50-1.00%", "n_obligors": 75, "ead": 3000.0,
             "avg_pd": 0.007, "avg_lgd": 0.45, "avg_rw": 0.65, "rwa": 1950.0,
             "el": 9.45},
        ]
        result = generate_cr6_irb_overview(data)
        assert result["totals"]["weighted_avg_pd"] == pytest.approx(0.007)
        assert result["totals"]["weighted_avg_lgd"] == pytest.approx(0.45)
        assert result["totals"]["avg_risk_weight"] == pytest.approx(1950.0 / 3000.0)
