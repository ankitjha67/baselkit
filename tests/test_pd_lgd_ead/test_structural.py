"""Tests for the Merton structural model."""

import math

import pytest
from scipy.stats import norm

from creditriskengine.models.pd.structural import (
    distance_to_default,
    implied_asset_value,
    merton_default_probability,
)


class TestDistanceToDefault:
    """Distance to default calculation."""

    def test_basic_dd(self) -> None:
        dd = distance_to_default(
            asset_value=100.0,
            debt_face_value=80.0,
            asset_volatility=0.20,
            risk_free_rate=0.05,
        )
        expected = (
            math.log(100.0 / 80.0) + (0.05 - 0.5 * 0.04) * 1.0
        ) / (0.20 * 1.0)
        assert dd == pytest.approx(expected)

    def test_high_leverage_low_dd(self) -> None:
        """Highly leveraged firm should have low DD."""
        dd = distance_to_default(100.0, 95.0, 0.30, 0.02)
        assert dd < 1.0

    def test_low_leverage_high_dd(self) -> None:
        """Low leverage firm should have high DD."""
        dd = distance_to_default(200.0, 50.0, 0.15, 0.05)
        assert dd > 5.0

    def test_longer_horizon_changes_dd(self) -> None:
        dd_1y = distance_to_default(100.0, 80.0, 0.20, 0.05, time_horizon=1.0)
        dd_5y = distance_to_default(100.0, 80.0, 0.20, 0.05, time_horizon=5.0)
        # With positive drift, longer horizon can increase DD
        assert dd_1y != pytest.approx(dd_5y)

    def test_zero_asset_value_raises(self) -> None:
        with pytest.raises(ValueError, match="asset_value must be positive"):
            distance_to_default(0.0, 80.0, 0.20, 0.05)

    def test_zero_debt_raises(self) -> None:
        with pytest.raises(ValueError, match="debt_face_value must be positive"):
            distance_to_default(100.0, 0.0, 0.20, 0.05)

    def test_zero_volatility_raises(self) -> None:
        with pytest.raises(ValueError, match="asset_volatility must be positive"):
            distance_to_default(100.0, 80.0, 0.0, 0.05)

    def test_zero_horizon_raises(self) -> None:
        with pytest.raises(ValueError, match="time_horizon must be positive"):
            distance_to_default(100.0, 80.0, 0.20, 0.05, time_horizon=0.0)


class TestMertonDefaultProbability:
    """Merton PD = Phi(-DD)."""

    def test_pd_between_0_and_1(self) -> None:
        pd = merton_default_probability(100.0, 80.0, 0.20, 0.05)
        assert 0.0 < pd < 1.0

    def test_highly_leveraged_high_pd(self) -> None:
        """Near-default firm should have high PD."""
        pd = merton_default_probability(100.0, 99.0, 0.40, 0.02)
        assert pd > 0.10

    def test_safe_firm_low_pd(self) -> None:
        """Safe firm should have very low PD."""
        pd = merton_default_probability(200.0, 50.0, 0.10, 0.05)
        assert pd < 0.001

    def test_pd_consistent_with_dd(self) -> None:
        dd = distance_to_default(100.0, 80.0, 0.20, 0.05)
        pd = merton_default_probability(100.0, 80.0, 0.20, 0.05)
        assert pd == pytest.approx(float(norm.cdf(-dd)))


class TestImpliedAssetValue:
    """Newton's method solver for implied asset value."""

    def test_round_trip(self) -> None:
        """V -> Equity -> implied V should round-trip."""
        v_true = 150.0
        d = 80.0
        sigma = 0.25
        r = 0.05
        t = 1.0

        sqrt_t = math.sqrt(t)
        d1 = (
            math.log(v_true / d) + (r + 0.5 * sigma**2) * t
        ) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t
        equity = v_true * norm.cdf(d1) - d * math.exp(-r * t) * norm.cdf(d2)

        v_implied = implied_asset_value(equity, d, sigma, r, t)
        assert v_implied == pytest.approx(v_true, rel=1e-6)

    def test_zero_equity_raises(self) -> None:
        with pytest.raises(ValueError, match="equity_value must be positive"):
            implied_asset_value(0.0, 80.0, 0.20, 0.05)

    def test_zero_debt_raises(self) -> None:
        with pytest.raises(ValueError, match="debt_face_value must be positive"):
            implied_asset_value(50.0, 0.0, 0.20, 0.05)

    def test_implied_exceeds_equity(self) -> None:
        """Asset value should always exceed equity value."""
        v = implied_asset_value(50.0, 80.0, 0.25, 0.05)
        assert v > 50.0

    def test_implied_value_positive(self) -> None:
        v = implied_asset_value(30.0, 100.0, 0.30, 0.03)
        assert v > 0.0

    def test_zero_volatility_raises(self) -> None:
        with pytest.raises(ValueError, match="asset_volatility must be positive"):
            implied_asset_value(50.0, 80.0, 0.0, 0.05)

    def test_zero_horizon_raises(self) -> None:
        with pytest.raises(ValueError, match="time_horizon must be positive"):
            implied_asset_value(50.0, 80.0, 0.20, 0.05, time_horizon=0.0)

    def test_non_convergence_raises(self) -> None:
        """Extreme inputs should cause non-convergence with very few iterations."""
        with pytest.raises(RuntimeError, match="did not converge"):
            implied_asset_value(1e-15, 1e15, 0.01, 0.0, max_iterations=1)

    def test_near_zero_derivative_raises(self) -> None:
        """When d1 is extremely negative, N(d1) ≈ 0 → derivative near zero."""
        # Tiny equity relative to massive debt forces V ≈ debt,
        # making d1 extremely negative and N(d1) ≈ 0
        with pytest.raises(RuntimeError, match="derivative near zero"):
            implied_asset_value(
                equity_value=1e-100,
                debt_face_value=1e10,
                asset_volatility=0.001,
                risk_free_rate=0.0,
            )
