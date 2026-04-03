"""Tests for TTC-to-PIT PD conversion and forward-looking adjustments."""

import numpy as np
import pytest

from creditriskengine.ecl.ifrs9.forward_looking import (
    SatelliteModelConfig,
    apply_fli_with_reversion,
    apply_macro_overlay,
    fli_impact_summary,
    lgd_macro_overlay,
    macro_adjustment_factor,
    mean_reversion_weights,
    satellite_model_predict,
)
from creditriskengine.ecl.ifrs9.ttc_to_pit import (
    estimate_z_factor,
    ttc_to_pit_pd,
    ttc_to_pit_pd_curve,
)


class TestTTCtoPIT:
    def test_neutral_z_reasonable(self) -> None:
        pit = ttc_to_pit_pd(0.02, z_factor=0.0, rho=0.15)
        assert 0.0 < pit < 0.10

    def test_adverse_z_increases_pd(self) -> None:
        pit = ttc_to_pit_pd(0.02, z_factor=-2.0, rho=0.15)
        assert pit > 0.02

    def test_favorable_z_decreases_pd(self) -> None:
        pit = ttc_to_pit_pd(0.02, z_factor=2.0, rho=0.15)
        assert pit < 0.02

    def test_boundary_pd_zero(self) -> None:
        assert ttc_to_pit_pd(0.0, z_factor=-2.0, rho=0.15) == 0.0

    def test_boundary_pd_one(self) -> None:
        assert ttc_to_pit_pd(1.0, z_factor=2.0, rho=0.15) == 1.0

    def test_invalid_rho_raises(self) -> None:
        with pytest.raises(ValueError, match="rho must be in"):
            ttc_to_pit_pd(0.02, 0.0, rho=0.0)
        with pytest.raises(ValueError, match="rho must be in"):
            ttc_to_pit_pd(0.02, 0.0, rho=1.0)

    def test_pit_in_unit_interval(self) -> None:
        pit = ttc_to_pit_pd(0.05, z_factor=-3.0, rho=0.30)
        assert 0.0 <= pit <= 1.0


class TestTTCtoPITCurve:
    def test_shape(self) -> None:
        curve = ttc_to_pit_pd_curve(0.02, [0.0, -1.0, -2.0], rho=0.15)
        assert len(curve) == 3

    def test_adverse_trend(self) -> None:
        curve = ttc_to_pit_pd_curve(0.02, [0.0, -1.0, -2.0], rho=0.15)
        assert curve[1] > curve[0]
        assert curve[2] > curve[1]


class TestEstimateZFactor:
    def test_roundtrip(self) -> None:
        rho = 0.15
        pd_ttc = 0.02
        z_true = -1.5
        pit = ttc_to_pit_pd(pd_ttc, z_true, rho)
        z_est = estimate_z_factor(pit, pd_ttc, rho)
        assert z_est == pytest.approx(z_true, abs=0.05)

    def test_invalid_rho_raises(self) -> None:
        with pytest.raises(ValueError, match="rho must be in"):
            estimate_z_factor(0.03, 0.02, rho=1.0)


class TestMacroAdjustmentFactor:
    def test_no_change(self) -> None:
        forecast = np.array([5.0, 5.0, 5.0])
        factors = macro_adjustment_factor(forecast, 5.0, sensitivity=1.0)
        np.testing.assert_allclose(factors, [1.0, 1.0, 1.0])

    def test_adverse_increase(self) -> None:
        forecast = np.array([6.0])
        factors = macro_adjustment_factor(forecast, 5.0, sensitivity=1.0)
        assert factors[0] == pytest.approx(1.2)

    def test_zero_baseline(self) -> None:
        forecast = np.array([1.0, 2.0])
        factors = macro_adjustment_factor(forecast, 0.0, sensitivity=1.0)
        np.testing.assert_allclose(factors, [1.0, 1.0])

    def test_floor_at_zero(self) -> None:
        forecast = np.array([0.0])
        factors = macro_adjustment_factor(forecast, 5.0, sensitivity=2.0)
        assert factors[0] == pytest.approx(0.0)


class TestApplyMacroOverlay:
    def test_basic(self) -> None:
        base_pds = np.array([0.01, 0.02, 0.03])
        factors = np.array([1.5, 1.5, 1.5])
        adjusted = apply_macro_overlay(base_pds, factors)
        np.testing.assert_allclose(adjusted, [0.015, 0.03, 0.045])

    def test_floor_applied(self) -> None:
        base_pds = np.array([0.0001])
        factors = np.array([0.01])
        adjusted = apply_macro_overlay(base_pds, factors, floor=0.0001)
        assert adjusted[0] == pytest.approx(0.0001)

    def test_cap_applied(self) -> None:
        base_pds = np.array([0.90])
        factors = np.array([2.0])
        adjusted = apply_macro_overlay(base_pds, factors, cap=1.0)
        assert adjusted[0] == pytest.approx(1.0)


# ---- New: Satellite model tests ----


class TestSatelliteModelPredict:
    def test_linear_link(self) -> None:
        config = SatelliteModelConfig(
            variable_names=["gdp_growth"],
            coefficients=[-2.0],
            intercept=1.0,
            link="linear",
        )
        forecasts = {"gdp_growth": np.array([0.02, -0.01, -0.03])}
        factors = satellite_model_predict(config, forecasts)
        # z(t) = 1.0 + (-2.0) * gdp
        # t=0: 1.0 - 0.04 = 0.96
        # t=1: 1.0 + 0.02 = 1.02
        # t=2: 1.0 + 0.06 = 1.06
        np.testing.assert_allclose(factors, [0.96, 1.02, 1.06])

    def test_logistic_link_centred(self) -> None:
        config = SatelliteModelConfig(
            variable_names=["unemp"],
            coefficients=[0.0],
            intercept=0.0,
            link="logistic",
        )
        forecasts = {"unemp": np.array([0.05])}
        factors = satellite_model_predict(config, forecasts)
        # z=0 → factor = 2/(1+exp(0)) = 2/2 = 1.0
        assert factors[0] == pytest.approx(1.0)

    def test_logistic_adverse(self) -> None:
        config = SatelliteModelConfig(
            variable_names=["unemp"],
            coefficients=[10.0],
            intercept=0.0,
            link="logistic",
        )
        forecasts = {"unemp": np.array([0.10])}
        factors = satellite_model_predict(config, forecasts)
        # z = 10 * 0.10 = 1.0 → factor = 2/(1+exp(-1)) ≈ 1.46
        assert factors[0] > 1.0
        assert factors[0] < 2.0

    def test_log_link(self) -> None:
        config = SatelliteModelConfig(
            variable_names=["gdp"],
            coefficients=[-1.0],
            intercept=0.0,
            link="log",
        )
        forecasts = {"gdp": np.array([0.0])}
        factors = satellite_model_predict(config, forecasts)
        assert factors[0] == pytest.approx(1.0)  # exp(0) = 1

    def test_multi_variable(self) -> None:
        config = SatelliteModelConfig(
            variable_names=["gdp", "unemp"],
            coefficients=[-2.0, 3.0],
            intercept=1.0,
            link="linear",
        )
        forecasts = {
            "gdp": np.array([0.01, -0.02]),
            "unemp": np.array([0.05, 0.08]),
        }
        factors = satellite_model_predict(config, forecasts)
        # t=0: 1.0 + (-2)(0.01) + 3(0.05) = 1.0 - 0.02 + 0.15 = 1.13
        # t=1: 1.0 + (-2)(-0.02) + 3(0.08) = 1.0 + 0.04 + 0.24 = 1.28
        np.testing.assert_allclose(factors, [1.13, 1.28])

    def test_missing_variable_raises(self) -> None:
        config = SatelliteModelConfig(
            variable_names=["gdp", "unemp"],
            coefficients=[-2.0, 3.0],
        )
        with pytest.raises(ValueError, match="Missing forecast"):
            satellite_model_predict(config, {"gdp": np.array([0.01])})

    def test_mismatched_config_raises(self) -> None:
        config = SatelliteModelConfig(
            variable_names=["gdp"],
            coefficients=[-2.0, 3.0],
        )
        with pytest.raises(ValueError, match="same length"):
            satellite_model_predict(config, {"gdp": np.array([0.01])})

    def test_invalid_link_raises(self) -> None:
        config = SatelliteModelConfig(
            variable_names=["gdp"],
            coefficients=[-2.0],
            link="logisitic",  # typo
        )
        with pytest.raises(ValueError, match="Unknown link function"):
            satellite_model_predict(config, {"gdp": np.array([0.01])})


class TestMeanReversionWeights:
    def test_within_forecast(self) -> None:
        w = mean_reversion_weights(total_periods=10, forecast_horizon=5, reversion_periods=3)
        np.testing.assert_allclose(w[:5], [1.0, 1.0, 1.0, 1.0, 1.0])

    def test_reversion_ramp(self) -> None:
        w = mean_reversion_weights(total_periods=10, forecast_horizon=3, reversion_periods=3)
        assert w[3] == pytest.approx(1.0 - 0 / 3)  # = 1.0 (start of ramp)
        assert w[4] == pytest.approx(1.0 - 1 / 3)  # ≈ 0.667
        assert w[5] == pytest.approx(1.0 - 2 / 3)  # ≈ 0.333

    def test_beyond_reversion(self) -> None:
        w = mean_reversion_weights(total_periods=10, forecast_horizon=3, reversion_periods=2)
        assert w[5] == pytest.approx(0.0)
        assert w[9] == pytest.approx(0.0)

    def test_no_reversion(self) -> None:
        w = mean_reversion_weights(total_periods=5, forecast_horizon=3, reversion_periods=0)
        np.testing.assert_allclose(w, [1.0, 1.0, 1.0, 0.0, 0.0])


class TestApplyFLIWithReversion:
    def test_blends_fli_and_longrun(self) -> None:
        base_pds = np.array([0.02, 0.02, 0.02, 0.02, 0.02])
        fli_factors = np.array([1.5, 1.5, 1.5, 1.5, 1.5])
        long_run = 0.02

        adjusted = apply_fli_with_reversion(
            base_pds, fli_factors, long_run,
            forecast_horizon=2, reversion_periods=2,
        )
        # Period 0-1: full FLI → 0.03
        assert adjusted[0] == pytest.approx(0.03)
        assert adjusted[1] == pytest.approx(0.03)
        # Period 4: fully reverted → 0.02
        assert adjusted[4] == pytest.approx(0.02)

    def test_floor_applied(self) -> None:
        base_pds = np.array([0.0001])
        fli_factors = np.array([0.001])
        adjusted = apply_fli_with_reversion(
            base_pds, fli_factors, 0.0001,
            forecast_horizon=1, reversion_periods=0, floor=0.0001,
        )
        assert adjusted[0] == pytest.approx(0.0001)


class TestLGDMacroOverlay:
    def test_hpi_decline_increases_lgd(self) -> None:
        base_lgds = np.array([0.40, 0.40])
        # HPI drops from 100 to 80 (20% decline)
        hpi_forecast = np.array([80.0, 85.0])
        adjusted = lgd_macro_overlay(base_lgds, hpi_forecast, 100.0, sensitivity=0.5)
        # decline_pct = [0.20, 0.15]
        # adj = [0.40 + 0.5*0.20, 0.40 + 0.5*0.15] = [0.50, 0.475]
        np.testing.assert_allclose(adjusted, [0.50, 0.475])

    def test_hpi_increase_no_effect(self) -> None:
        base_lgds = np.array([0.40])
        hpi_forecast = np.array([110.0])
        adjusted = lgd_macro_overlay(base_lgds, hpi_forecast, 100.0, sensitivity=0.5)
        assert adjusted[0] == pytest.approx(0.40)

    def test_cap_applied(self) -> None:
        base_lgds = np.array([0.80])
        hpi_forecast = np.array([20.0])
        adjusted = lgd_macro_overlay(base_lgds, hpi_forecast, 100.0, sensitivity=1.0, cap=1.0)
        assert adjusted[0] == pytest.approx(1.0)

    def test_zero_baseline(self) -> None:
        base_lgds = np.array([0.40])
        hpi_forecast = np.array([80.0])
        adjusted = lgd_macro_overlay(base_lgds, hpi_forecast, 0.0, sensitivity=0.5)
        assert adjusted[0] == pytest.approx(0.40)


class TestFLIImpactSummary:
    def test_summary_fields(self) -> None:
        base = np.array([0.02, 0.02, 0.02])
        adjusted = np.array([0.03, 0.03, 0.03])
        summary = fli_impact_summary(base, adjusted, model_type="satellite_logistic",
                                     variables_used=["gdp", "unemp"])
        assert summary["model_type"] == "satellite_logistic"
        assert summary["variables_used"] == ["gdp", "unemp"]
        assert summary["n_periods"] == 3
        assert summary["pct_change"] == pytest.approx(50.0)
        assert summary["max_adjustment_factor"] == pytest.approx(1.5)
