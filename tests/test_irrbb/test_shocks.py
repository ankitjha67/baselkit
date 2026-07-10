"""Tests for the IRRBB d578 shock recalibration framework."""

import numpy as np
import pytest

from creditriskengine.irrbb.eve import eve_sensitivity
from creditriskengine.irrbb.shocks import (
    D368_BASELINE,
    CurrencyShocks,
    apply_post_shock_floor,
    get_currency_shocks,
    is_valid_shock_rounding,
    post_shock_floor,
    register_currency_shocks,
)


class TestCurrencyShocks:
    def test_eur_d578_values(self) -> None:
        s = get_currency_shocks("EUR")
        assert s.parallel_bps == pytest.approx(225.0)
        assert s.short_bps == pytest.approx(350.0)
        assert s.long_bps == pytest.approx(200.0)

    def test_usd_d578_values(self) -> None:
        s = get_currency_shocks("usd")  # case-insensitive
        assert s.parallel_bps == pytest.approx(200.0)
        assert s.short_bps == pytest.approx(300.0)
        assert s.long_bps == pytest.approx(225.0)

    def test_unregistered_falls_back_to_baseline(self) -> None:
        s = get_currency_shocks("ZAR")
        assert s == D368_BASELINE

    def test_unregistered_strict_raises(self) -> None:
        with pytest.raises(KeyError, match="ZWL"):
            get_currency_shocks("ZWL", fallback_to_baseline=False)

    def test_register_override(self) -> None:
        register_currency_shocks("GBP", CurrencyShocks(250.0, 375.0, 175.0))
        s = get_currency_shocks("GBP", fallback_to_baseline=False)
        assert s.parallel_bps == pytest.approx(250.0)

    def test_caps_enforced(self) -> None:
        with pytest.raises(ValueError, match="parallel"):
            CurrencyShocks(parallel_bps=425.0, short_bps=100.0, long_bps=100.0)
        with pytest.raises(ValueError, match="short"):
            CurrencyShocks(parallel_bps=100.0, short_bps=525.0, long_bps=100.0)
        with pytest.raises(ValueError, match="long"):
            CurrencyShocks(parallel_bps=100.0, short_bps=100.0, long_bps=325.0)

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            CurrencyShocks(parallel_bps=-25.0, short_bps=100.0, long_bps=100.0)


class TestRounding:
    def test_valid_multiples_of_25(self) -> None:
        assert is_valid_shock_rounding(225.0) is True
        assert is_valid_shock_rounding(350.0) is True

    def test_invalid_rounding(self) -> None:
        assert is_valid_shock_rounding(230.0) is False


class TestPostShockFloor:
    def test_overnight_minus_100bps(self) -> None:
        assert post_shock_floor(0.0) == pytest.approx(-0.01)

    def test_ten_years_minus_50bps(self) -> None:
        # -100 + 5*10 = -50 bps.
        assert post_shock_floor(10.0) == pytest.approx(-0.005)

    def test_twenty_years_zero(self) -> None:
        assert post_shock_floor(20.0) == pytest.approx(0.0)

    def test_beyond_twenty_years_capped_at_zero(self) -> None:
        assert post_shock_floor(30.0) == pytest.approx(0.0)

    def test_negative_tenor_raises(self) -> None:
        with pytest.raises(ValueError, match="tenor_years"):
            post_shock_floor(-1.0)

    def test_apply_floor_to_curve(self) -> None:
        # Deeply negative shocked rates are floored per tenor; at >= 20y
        # the floor is 0%, so even a mildly negative rate floors to 0.
        shocked = np.array([-0.03, -0.02, -0.001])
        tenors = np.array([0.0, 10.0, 20.0])
        floored = apply_post_shock_floor(shocked, tenors)
        np.testing.assert_allclose(floored, [-0.01, -0.005, 0.0])

    def test_floor_does_not_lift_positive_rates(self) -> None:
        shocked = np.array([0.02, 0.03])
        tenors = np.array([1.0, 30.0])
        np.testing.assert_allclose(apply_post_shock_floor(shocked, tenors), shocked)

    def test_apply_floor_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            apply_post_shock_floor(np.array([0.01]), np.array([1.0, 2.0]))


class TestEVECurrencyIntegration:
    def _book(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cashflows = np.array([100.0, -50.0, 200.0])
        tenors = np.array([0.5, 5.0, 10.0])
        rates = np.array([0.03, 0.035, 0.04])
        return cashflows, tenors, rates

    def test_currency_overrides_generic_shocks(self) -> None:
        cf, t, r = self._book()
        generic = eve_sensitivity(cf, t, r)  # d368 200/250/100
        eur = eve_sensitivity(cf, t, r, currency="EUR")  # d578 225/350/200
        # EUR's larger parallel shock produces a different Delta-EVE.
        assert generic != eur

    def test_floor_binds_in_down_shock(self) -> None:
        cf, t, _ = self._book()
        low_rates = np.array([0.001, 0.002, 0.003])
        unfloored = eve_sensitivity(cf, t, low_rates)
        floored = eve_sensitivity(cf, t, low_rates, apply_floor=True)
        from creditriskengine.irrbb.eve import InterestRateShock

        # With near-zero base rates, the -200bps parallel-down shocked
        # curve breaches the floor, so the floored Delta-EVE differs.
        key = InterestRateShock.PARALLEL_DOWN
        assert unfloored[key] != floored[key]
