"""Tests for stress testing, economic capital, VaR, backtesting, and reporting."""

import numpy as np
import pytest

from creditriskengine.portfolio.economic_capital import ec_single_factor
from creditriskengine.portfolio.stress_testing import (
    MacroScenario,
    apply_lgd_stress,
    apply_pd_stress,
    stress_test_rwa_impact,
)
from creditriskengine.portfolio.var import marginal_var, parametric_credit_var
from creditriskengine.validation.backtesting import pd_backtest_summary
from creditriskengine.validation.reporting import generate_validation_summary


class TestApplyPDStress:
    def test_basic(self) -> None:
        pds = np.array([0.01, 0.02, 0.03])
        stressed = apply_pd_stress(pds, stress_multiplier=2.0)
        np.testing.assert_allclose(stressed, [0.02, 0.04, 0.06])

    def test_cap(self) -> None:
        pds = np.array([0.80])
        stressed = apply_pd_stress(pds, stress_multiplier=2.0, pd_cap=1.0)
        assert stressed[0] == pytest.approx(1.0)


class TestApplyLGDStress:
    def test_basic(self) -> None:
        lgds = np.array([0.40, 0.45])
        stressed = apply_lgd_stress(lgds, stress_add_on=0.10)
        np.testing.assert_allclose(stressed, [0.50, 0.55])

    def test_cap_at_one(self) -> None:
        lgds = np.array([0.95])
        stressed = apply_lgd_stress(lgds, stress_add_on=0.20, lgd_cap=1.0)
        assert stressed[0] == pytest.approx(1.0)

    def test_floor_at_zero(self) -> None:
        lgds = np.array([0.10])
        stressed = apply_lgd_stress(lgds, stress_add_on=-0.50)
        assert stressed[0] == pytest.approx(0.0)


class TestStressTestRWAImpact:
    def test_basic(self) -> None:
        result = stress_test_rwa_impact(base_rwa=1000.0, stressed_rwa=1200.0)
        assert result["delta_rwa"] == pytest.approx(200.0)
        assert result["pct_change"] == pytest.approx(0.20)

    def test_zero_base(self) -> None:
        result = stress_test_rwa_impact(base_rwa=0.0, stressed_rwa=100.0)
        assert result["pct_change"] == pytest.approx(0.0)


class TestMacroScenario:
    def test_defaults(self) -> None:
        s = MacroScenario("adverse")
        assert s.name == "adverse"
        assert s.horizon_years == 3
        assert s.severity == "adverse"


class TestEconomicCapital:
    def test_basic(self) -> None:
        pds = np.array([0.02, 0.03])
        lgds = np.array([0.45, 0.40])
        eads = np.array([100.0, 200.0])
        result = ec_single_factor(pds, lgds, eads, rho=0.15, n_simulations=5000, seed=42)
        assert result["expected_loss"] == pytest.approx(np.sum(pds * lgds * eads))
        assert result["var"] >= result["expected_loss"]
        assert result["economic_capital"] >= 0.0
        assert result["total_ead"] == pytest.approx(300.0)


class TestParametricCreditVaR:
    def test_basic(self) -> None:
        var = parametric_credit_var(el=10.0, ul_std=5.0, confidence=0.999)
        # VaR = 10 + z_0.999 * 5 ≈ 10 + 3.09 * 5 ≈ 25.45
        assert var > 10.0
        assert var == pytest.approx(10.0 + 3.0902 * 5.0, abs=0.1)


class TestMarginalVaR:
    def test_basic(self) -> None:
        mvar = marginal_var(
            portfolio_var=100.0, portfolio_std=20.0,
            exposure_contribution_to_std=5.0,
        )
        assert mvar == pytest.approx(100.0 * 5.0 / 20.0)

    def test_zero_std(self) -> None:
        assert marginal_var(100.0, 0.0, 5.0) == 0.0


class TestPDBacktestSummary:
    def test_basic(self) -> None:
        pds = np.array([0.01, 0.02, 0.03, 0.01, 0.02])
        defaults = np.array([0, 0, 1, 0, 0])
        result = pd_backtest_summary(pds, defaults)
        assert result["n_observations"] == 5
        assert result["n_defaults"] == 1
        assert result["observed_default_rate"] == pytest.approx(0.2)
        assert result["average_predicted_pd"] == pytest.approx(0.018)

    def test_empty(self) -> None:
        result = pd_backtest_summary(np.array([]), np.array([]))
        assert result["n_observations"] == 0


class TestValidationReporting:
    def test_green(self) -> None:
        result = generate_validation_summary(
            "test_model",
            {"gini": 0.50},
            {"binomial": {"reject_h0": False}},
            {"psi": 0.05},
        )
        assert result["overall_assessment"] == "green"
        assert result["model_name"] == "test_model"

    def test_yellow_low_gini(self) -> None:
        result = generate_validation_summary(
            "model", {"gini": 0.25}, {}, {"psi": 0.05}
        )
        assert result["overall_assessment"] == "yellow"

    def test_red_high_psi(self) -> None:
        result = generate_validation_summary(
            "model", {"gini": 0.50}, {}, {"psi": 0.30}
        )
        assert result["overall_assessment"] == "red"

    def test_red_very_low_gini(self) -> None:
        result = generate_validation_summary(
            "model", {"gini": 0.10}, {}, {"psi": 0.01}
        )
        assert result["overall_assessment"] == "red"
