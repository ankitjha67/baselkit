"""Tests for TTC-to-PIT PD conversion and forward-looking adjustments."""

import numpy as np
import pytest

from creditriskengine.ecl.ifrs9.ttc_to_pit import (
    estimate_z_factor,
    ttc_to_pit_pd,
    ttc_to_pit_pd_curve,
)
from creditriskengine.ecl.ifrs9.forward_looking import (
    apply_macro_overlay,
    macro_adjustment_factor,
)


class TestTTCtoPIT:
    def test_neutral_z_reasonable(self):
        # Z=0 → PIT should be close to but not exactly TTC (model-dependent)
        pit = ttc_to_pit_pd(0.02, z_factor=0.0, rho=0.15)
        assert 0.0 < pit < 0.10  # reasonable range

    def test_adverse_z_increases_pd(self):
        pit = ttc_to_pit_pd(0.02, z_factor=-2.0, rho=0.15)
        assert pit > 0.02

    def test_favorable_z_decreases_pd(self):
        pit = ttc_to_pit_pd(0.02, z_factor=2.0, rho=0.15)
        assert pit < 0.02

    def test_boundary_pd_zero(self):
        assert ttc_to_pit_pd(0.0, z_factor=-2.0, rho=0.15) == 0.0

    def test_boundary_pd_one(self):
        assert ttc_to_pit_pd(1.0, z_factor=2.0, rho=0.15) == 1.0

    def test_invalid_rho_raises(self):
        with pytest.raises(ValueError, match="rho must be in"):
            ttc_to_pit_pd(0.02, 0.0, rho=0.0)
        with pytest.raises(ValueError, match="rho must be in"):
            ttc_to_pit_pd(0.02, 0.0, rho=1.0)

    def test_pit_in_unit_interval(self):
        pit = ttc_to_pit_pd(0.05, z_factor=-3.0, rho=0.30)
        assert 0.0 <= pit <= 1.0


class TestTTCtoPITCurve:
    def test_shape(self):
        curve = ttc_to_pit_pd_curve(0.02, [0.0, -1.0, -2.0], rho=0.15)
        assert len(curve) == 3

    def test_adverse_trend(self):
        curve = ttc_to_pit_pd_curve(0.02, [0.0, -1.0, -2.0], rho=0.15)
        # More adverse Z → higher PD
        assert curve[1] > curve[0]
        assert curve[2] > curve[1]


class TestEstimateZFactor:
    def test_roundtrip(self):
        # Convert TTC → PIT with known Z, then back-solve
        rho = 0.15
        pd_ttc = 0.02
        z_true = -1.5
        pit = ttc_to_pit_pd(pd_ttc, z_true, rho)
        z_est = estimate_z_factor(pit, pd_ttc, rho)
        assert z_est == pytest.approx(z_true, abs=0.05)

    def test_invalid_rho_raises(self):
        with pytest.raises(ValueError, match="rho must be in"):
            estimate_z_factor(0.03, 0.02, rho=1.0)


class TestMacroAdjustmentFactor:
    def test_no_change(self):
        forecast = np.array([5.0, 5.0, 5.0])
        factors = macro_adjustment_factor(forecast, 5.0, sensitivity=1.0)
        np.testing.assert_allclose(factors, [1.0, 1.0, 1.0])

    def test_adverse_increase(self):
        forecast = np.array([6.0])  # 20% above baseline
        factors = macro_adjustment_factor(forecast, 5.0, sensitivity=1.0)
        assert factors[0] == pytest.approx(1.2)

    def test_zero_baseline(self):
        forecast = np.array([1.0, 2.0])
        factors = macro_adjustment_factor(forecast, 0.0, sensitivity=1.0)
        np.testing.assert_allclose(factors, [1.0, 1.0])

    def test_floor_at_zero(self):
        forecast = np.array([0.0])  # -100% change
        factors = macro_adjustment_factor(forecast, 5.0, sensitivity=2.0)
        assert factors[0] == pytest.approx(0.0)


class TestApplyMacroOverlay:
    def test_basic(self):
        base_pds = np.array([0.01, 0.02, 0.03])
        factors = np.array([1.5, 1.5, 1.5])
        adjusted = apply_macro_overlay(base_pds, factors)
        np.testing.assert_allclose(adjusted, [0.015, 0.03, 0.045])

    def test_floor_applied(self):
        base_pds = np.array([0.0001])
        factors = np.array([0.01])
        adjusted = apply_macro_overlay(base_pds, factors, floor=0.0001)
        assert adjusted[0] == pytest.approx(0.0001)

    def test_cap_applied(self):
        base_pds = np.array([0.90])
        factors = np.array([2.0])
        adjusted = apply_macro_overlay(base_pds, factors, cap=1.0)
        assert adjusted[0] == pytest.approx(1.0)
