"""Tests for macro stress testing framework."""

import numpy as np
import pytest

from creditriskengine.portfolio.stress_testing import (
    CCARScenario,
    EBAStressTest,
    MacroScenario,
    RBIStressTest,
    apply_lgd_stress,
    apply_pd_stress,
    multi_period_projection,
    scenario_library,
    stress_test_rwa_impact,
)


class TestApplyPDStress:
    """PD stress multiplier with cap."""

    def test_basic_multiplier(self) -> None:
        pds = np.array([0.02, 0.05, 0.10])
        stressed = apply_pd_stress(pds, 2.0)
        np.testing.assert_allclose(stressed, [0.04, 0.10, 0.20])

    def test_caps_at_1(self) -> None:
        pds = np.array([0.60, 0.80])
        stressed = apply_pd_stress(pds, 2.0)
        np.testing.assert_allclose(stressed, [1.0, 1.0])

    def test_custom_cap(self) -> None:
        pds = np.array([0.40])
        stressed = apply_pd_stress(pds, 2.0, pd_cap=0.50)
        assert stressed[0] == pytest.approx(0.50)

    def test_multiplier_of_one(self) -> None:
        pds = np.array([0.05])
        stressed = apply_pd_stress(pds, 1.0)
        assert stressed[0] == pytest.approx(0.05)


class TestApplyLGDStress:
    """LGD additive stress with clip to [0, 1]."""

    def test_basic_add_on(self) -> None:
        lgds = np.array([0.30, 0.40])
        stressed = apply_lgd_stress(lgds, 0.10)
        np.testing.assert_allclose(stressed, [0.40, 0.50])

    def test_clip_at_1(self) -> None:
        lgds = np.array([0.90])
        stressed = apply_lgd_stress(lgds, 0.20)
        assert stressed[0] == pytest.approx(1.0)

    def test_clip_at_0(self) -> None:
        lgds = np.array([0.10])
        stressed = apply_lgd_stress(lgds, -0.20)
        assert stressed[0] == pytest.approx(0.0)

    def test_negative_add_on(self) -> None:
        lgds = np.array([0.50])
        stressed = apply_lgd_stress(lgds, -0.10)
        assert stressed[0] == pytest.approx(0.40)


class TestStressTestRWAImpact:
    """RWA impact calculation."""

    def test_positive_delta(self) -> None:
        result = stress_test_rwa_impact(1000.0, 1200.0)
        assert result["delta_rwa"] == pytest.approx(200.0)
        assert result["pct_change"] == pytest.approx(0.20)

    def test_zero_base_rwa(self) -> None:
        result = stress_test_rwa_impact(0.0, 100.0)
        assert result["pct_change"] == 0.0

    def test_negative_delta(self) -> None:
        result = stress_test_rwa_impact(1000.0, 800.0)
        assert result["delta_rwa"] == pytest.approx(-200.0)


class TestScenarioLibrary:
    """Predefined scenario library."""

    def test_returns_dict(self) -> None:
        lib = scenario_library()
        assert isinstance(lib, dict)

    def test_contains_standard_scenarios(self) -> None:
        lib = scenario_library()
        for name in ["baseline", "mild_downturn", "moderate_recession", "severe_recession"]:
            assert name in lib

    def test_scenarios_have_variables(self) -> None:
        lib = scenario_library()
        for scenario in lib.values():
            assert isinstance(scenario, MacroScenario)
            assert len(scenario.variables) > 0


class TestMultiPeriodProjection:
    """Multi-period credit risk projection."""

    def test_output_shapes(self) -> None:
        pds = np.array([0.02, 0.05])
        lgds = np.array([0.40, 0.30])
        eads = np.array([1e6, 2e6])
        pd_mult = np.array([1.5, 2.0, 1.8])
        lgd_add = np.array([0.05, 0.10, 0.08])

        result = multi_period_projection(pds, lgds, eads, pd_mult, lgd_add)
        assert result["stressed_pds"].shape == (3, 2)
        assert result["stressed_lgds"].shape == (3, 2)
        assert result["expected_losses"].shape == (3, 2)
        assert len(result["period_el"]) == 3

    def test_cumulative_el_positive(self) -> None:
        pds = np.array([0.02])
        lgds = np.array([0.40])
        eads = np.array([1e6])
        pd_mult = np.array([1.5, 2.0])
        lgd_add = np.array([0.05, 0.10])

        result = multi_period_projection(pds, lgds, eads, pd_mult, lgd_add)
        assert result["cumulative_el"] > 0

    def test_mismatched_lengths_raises(self) -> None:
        pds = np.array([0.02])
        lgds = np.array([0.40])
        eads = np.array([1e6])
        pd_mult = np.array([1.5, 2.0])
        lgd_add = np.array([0.05])
        with pytest.raises(ValueError, match="lgd_add_ons length"):
            multi_period_projection(pds, lgds, eads, pd_mult, lgd_add)

    def test_static_balance_sheet_eads_decrease(self) -> None:
        """Under defaults, EADs should decrease over periods."""
        pds = np.array([0.10])
        lgds = np.array([0.40])
        eads = np.array([1e6])
        pd_mult = np.array([1.0, 1.0, 1.0])
        lgd_add = np.array([0.0, 0.0, 0.0])

        result = multi_period_projection(pds, lgds, eads, pd_mult, lgd_add)
        period_eads = result["period_eads"]
        assert period_eads[0] > period_eads[2]


class TestEBAStressTest:
    """EBA stress test framework."""

    def test_run_returns_expected_keys(self) -> None:
        scenario = MacroScenario(
            name="Test Adverse",
            horizon_years=3,
            variables={
                "gdp_growth": np.array([-0.03, -0.01, 0.01]),
                "house_price_index": np.array([-0.10, -0.05, 0.0]),
            },
            severity="adverse",
        )
        eba = EBAStressTest(scenario)
        result = eba.run(
            np.array([0.02, 0.05]),
            np.array([0.40, 0.30]),
            np.array([1e6, 2e6]),
        )
        assert "cumulative_el" in result
        assert "baseline_el" in result
        assert "delta_el" in result
        assert "scenario" in result
        assert result["scenario"] == "Test Adverse"

    def test_minimum_3_year_horizon(self) -> None:
        scenario = MacroScenario(name="Short", horizon_years=2)
        with pytest.raises(ValueError, match="minimum 3-year"):
            EBAStressTest(scenario, horizon_years=2)

    def test_stressed_el_exceeds_baseline(self) -> None:
        """Adverse macro should produce higher stressed EL than baseline."""
        scenario = MacroScenario(
            name="Adverse",
            horizon_years=3,
            variables={
                "gdp_growth": np.array([-0.04, -0.02, 0.01]),
                "house_price_index": np.array([-0.15, -0.10, -0.03]),
            },
        )
        eba = EBAStressTest(scenario)
        pds = np.array([0.02, 0.03])
        lgds = np.array([0.40, 0.35])
        eads = np.array([1e6, 1e6])
        result = eba.run(pds, lgds, eads)
        # Cumulative stressed EL should exceed 3x baseline (since multipliers > 1)
        assert result["cumulative_el"] > result["baseline_el"]


class TestCCARScenario:
    """CCAR/DFAST stress testing."""

    def test_project_quarterly_losses_9_quarters(self) -> None:
        scenario = MacroScenario(name="CCAR Adverse", horizon_years=3)
        ccar = CCARScenario(scenario, horizon_quarters=9)
        result = ccar.project_quarterly_losses(
            np.array([0.02, 0.05]),
            np.array([0.40, 0.30]),
            np.array([1e6, 2e6]),
        )
        assert len(result["quarterly_totals"]) == 9
        assert len(result["cumulative_loss"]) == 9
        assert result["total_loss"] > 0

    def test_run_capital_trajectory(self) -> None:
        scenario = MacroScenario(name="CCAR Test", horizon_years=3)
        ccar = CCARScenario(
            scenario,
            horizon_quarters=9,
            ppnr_quarterly=np.full(9, 50_000.0),
        )
        result = ccar.run(
            np.array([0.02]),
            np.array([0.40]),
            np.array([1e6]),
            initial_capital=1_000_000,
        )
        assert len(result["capital_trajectory"]) == 9
        assert "min_capital" in result
        assert "min_capital_quarter" in result

    def test_ppnr_length_mismatch_raises(self) -> None:
        scenario = MacroScenario(name="Test", horizon_years=3)
        with pytest.raises(ValueError, match="ppnr_quarterly must have exactly"):
            CCARScenario(scenario, horizon_quarters=9, ppnr_quarterly=np.ones(5))


class TestRBIStressTest:
    """RBI stress testing with sensitivity analysis."""

    def test_credit_quality_stress_base_vs_stressed(self) -> None:
        rbi = RBIStressTest(severity="moderate")
        result = rbi.credit_quality_stress(
            np.array([0.02, 0.05]),
            np.array([0.30, 0.40]),
            np.array([1e6, 2e6]),
        )
        assert result["stressed_el"] > result["base_el"]
        assert result["severity"] == "moderate"

    def test_severe_stress_higher_than_mild(self) -> None:
        pds = np.array([0.03])
        lgds = np.array([0.35])
        eads = np.array([1e6])

        mild = RBIStressTest(severity="mild").credit_quality_stress(pds, lgds, eads)
        severe = RBIStressTest(severity="severe").credit_quality_stress(pds, lgds, eads)
        assert severe["stressed_el"] > mild["stressed_el"]

    def test_incremental_provisions_positive(self) -> None:
        rbi = RBIStressTest(severity="moderate")
        result = rbi.credit_quality_stress(
            np.array([0.02]),
            np.array([0.30]),
            np.array([1e6]),
        )
        assert result["incremental_provisions"] > 0
