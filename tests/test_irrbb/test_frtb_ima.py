"""Tests for FRTB Internal Models Approach."""

from __future__ import annotations

import numpy as np
import pytest

from creditriskengine.rwa.frtb_ima import (
    ES_CONFIDENCE_LEVEL,
    LIQUIDITY_HORIZONS,
    NMRFCharge,
    PLATZone,
    default_risk_charge_ima,
    expected_shortfall,
    internal_model_capital_charge,
    liquidity_scaled_es,
    nmrf_stress_charge,
    plat_test,
)


class TestExpectedShortfall:
    def test_es_positive_for_losses(self) -> None:
        rng = np.random.default_rng(42)
        pnl = rng.normal(0, 1, 100000)
        es = expected_shortfall(pnl, 0.975)
        assert es > 0

    def test_es_exceeds_var(self) -> None:
        rng = np.random.default_rng(42)
        pnl = rng.normal(0, 1, 100000)
        es = expected_shortfall(pnl, 0.975)
        var = -np.quantile(pnl, 0.025)
        # ES (mean beyond VaR) >= VaR
        assert es >= var

    def test_confidence_level_constant(self) -> None:
        assert ES_CONFIDENCE_LEVEL == 0.975

    def test_invalid_confidence(self) -> None:
        with pytest.raises(ValueError, match="confidence_level"):
            expected_shortfall(np.array([1.0, -1.0]), 1.5)

    def test_normal_es_approx(self) -> None:
        # For N(0,1), ES at 97.5% ≈ phi(z)/(1-cl) ≈ 2.34
        rng = np.random.default_rng(7)
        pnl = rng.normal(0, 1, 500000)
        es = expected_shortfall(pnl, 0.975)
        assert es == pytest.approx(2.34, abs=0.1)


class TestLiquidityScaling:
    def test_base_horizon_unchanged(self) -> None:
        assert liquidity_scaled_es(100.0, 10, 10) == pytest.approx(100.0)

    def test_longer_horizon_larger_es(self) -> None:
        es40 = liquidity_scaled_es(100.0, 40, 10)
        # sqrt(40/10) = 2 → 200
        assert es40 == pytest.approx(200.0)

    def test_invalid_horizon(self) -> None:
        with pytest.raises(ValueError, match="horizons must be positive"):
            liquidity_scaled_es(100.0, 0, 10)

    def test_liquidity_horizon_table(self) -> None:
        assert LIQUIDITY_HORIZONS["interest_rate"] == 10
        assert LIQUIDITY_HORIZONS["credit_spread_other"] == 120
        assert LIQUIDITY_HORIZONS["credit_spread_ig"] == 20


class TestIMCC:
    def test_stressed_binds(self) -> None:
        charge = internal_model_capital_charge(es_current=80, es_stressed=120, multiplier=1.5)
        assert charge == pytest.approx(1.5 * 120)

    def test_current_binds_when_higher(self) -> None:
        charge = internal_model_capital_charge(es_current=150, es_stressed=120, multiplier=1.5)
        assert charge == pytest.approx(1.5 * 150)

    def test_multiplier_floor(self) -> None:
        with pytest.raises(ValueError, match="multiplier must be"):
            internal_model_capital_charge(80, 120, multiplier=1.0)


class TestPLAT:
    def test_green_when_aligned(self) -> None:
        rng = np.random.default_rng(42)
        hypo = rng.normal(0, 1, 250)
        # risk-theoretical nearly identical
        rt = hypo + rng.normal(0, 0.01, 250)
        zone, spearman, ks = plat_test(hypo, rt)
        assert zone == PLATZone.GREEN
        assert spearman >= 0.80

    def test_red_when_misaligned(self) -> None:
        rng = np.random.default_rng(42)
        hypo = rng.normal(0, 1, 250)
        rt = rng.normal(5, 3, 250)  # Unrelated
        zone, spearman, ks = plat_test(hypo, rt)
        assert zone == PLATZone.RED

    def test_returns_metrics(self) -> None:
        rng = np.random.default_rng(1)
        hypo = rng.normal(0, 1, 100)
        rt = hypo.copy()
        zone, spearman, ks = plat_test(hypo, rt)
        assert -1.0 <= spearman <= 1.0
        assert 0.0 <= ks <= 1.0


class TestDRC:
    def test_long_only(self) -> None:
        drc = default_risk_charge_ima(
            jtd_long=np.array([100.0, 50.0]),
            jtd_short=np.array([0.0]),
        )
        assert drc == pytest.approx(150.0)

    def test_hedge_benefit(self) -> None:
        # Long 100, short 100 → WtS = 0.5 → 100 - 0.5*100 = 50
        drc = default_risk_charge_ima(
            jtd_long=np.array([100.0]),
            jtd_short=np.array([100.0]),
        )
        assert drc == pytest.approx(50.0)

    def test_empty(self) -> None:
        drc = default_risk_charge_ima(np.array([0.0]), np.array([0.0]))
        assert drc == 0.0


class TestNMRF:
    def test_returns_charge(self) -> None:
        result = nmrf_stress_charge(
            idiosyncratic_stress_losses=np.array([10.0, 20.0]),
            non_idiosyncratic_stress_losses=np.array([30.0, 40.0]),
        )
        assert isinstance(result, NMRFCharge)
        assert result.total > 0

    def test_idiosyncratic_sum_of_squares(self) -> None:
        result = nmrf_stress_charge(
            idiosyncratic_stress_losses=np.array([3.0, 4.0]),
            non_idiosyncratic_stress_losses=np.array([0.0]),
        )
        # sqrt(9 + 16) = 5
        assert result.idiosyncratic_charge == pytest.approx(5.0)

    def test_total_combines_components(self) -> None:
        result = nmrf_stress_charge(
            idiosyncratic_stress_losses=np.array([3.0, 4.0]),
            non_idiosyncratic_stress_losses=np.array([10.0]),
            rho=0.6,
        )
        expected_total = np.sqrt(
            result.idiosyncratic_charge**2 + result.non_idiosyncratic_charge**2
        )
        assert result.total == pytest.approx(expected_total)
