"""Tests for IRRBB (EVE, NII, Supervisory Outlier Test)."""

from __future__ import annotations

import numpy as np
import pytest

from creditriskengine.irrbb import (
    SHOCK_SCENARIOS,
    InterestRateShock,
    OutlierTestResult,
    eve_sensitivity,
    nii_sensitivity,
    repricing_gap,
    supervisory_outlier_test,
)


class TestEVE:
    def _book(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Asset-heavy long-duration book
        cashflows = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        tenors = np.array([1.0, 2.0, 3.0, 5.0, 10.0])
        rates = np.array([0.03, 0.032, 0.034, 0.036, 0.04])
        return cashflows, tenors, rates

    def test_six_scenarios(self) -> None:
        cf, t, r = self._book()
        result = eve_sensitivity(cf, t, r)
        assert len(result) == 6
        assert set(result.keys()) == set(SHOCK_SCENARIOS)

    def test_parallel_up_reduces_pv(self) -> None:
        cf, t, r = self._book()
        result = eve_sensitivity(cf, t, r)
        # Positive cash flows: rates up → PV down → negative Delta-EVE
        assert result[InterestRateShock.PARALLEL_UP] < 0

    def test_parallel_down_increases_pv(self) -> None:
        cf, t, r = self._book()
        result = eve_sensitivity(cf, t, r)
        assert result[InterestRateShock.PARALLEL_DOWN] > 0

    def test_repricing_gap_pv(self) -> None:
        cf, t, r = self._book()
        pv = repricing_gap(cf, t, r)
        # Manual PV
        expected = float(np.sum(cf / (1 + r) ** t))
        assert pv == pytest.approx(expected)

    def test_up_and_down_roughly_symmetric(self) -> None:
        cf, t, r = self._book()
        result = eve_sensitivity(cf, t, r)
        up = result[InterestRateShock.PARALLEL_UP]
        down = result[InterestRateShock.PARALLEL_DOWN]
        # Opposite signs
        assert up < 0 < down


class TestNII:
    def test_asset_sensitive_positive_under_up_shock(self) -> None:
        # RSA > RSL → asset sensitive → up shock raises NII
        delta = nii_sensitivity(
            rate_sensitive_assets=1_000_000,
            rate_sensitive_liabilities=600_000,
            shock_bps=200,
        )
        assert delta > 0

    def test_liability_sensitive_negative_under_up_shock(self) -> None:
        delta = nii_sensitivity(
            rate_sensitive_assets=400_000,
            rate_sensitive_liabilities=900_000,
            shock_bps=200,
        )
        assert delta < 0

    def test_down_shock_flips_sign(self) -> None:
        up = nii_sensitivity(1_000_000, 600_000, 200)
        down = nii_sensitivity(1_000_000, 600_000, -200)
        assert up == pytest.approx(-down)

    def test_magnitude(self) -> None:
        # 200bps on a 400k gap over 1y = 0.02 * 400000 = 8000
        delta = nii_sensitivity(1_000_000, 600_000, 200)
        assert delta == pytest.approx(8000.0)


class TestSupervisoryOutlierTest:
    def test_eve_breach(self) -> None:
        # Worst EVE loss 20% of Tier 1 → breach (> 15%)
        delta_eve = {
            InterestRateShock.PARALLEL_UP: -200.0,
            InterestRateShock.PARALLEL_DOWN: 50.0,
            InterestRateShock.STEEPENER: -30.0,
            InterestRateShock.FLATTENER: -10.0,
            InterestRateShock.SHORT_UP: -40.0,
            InterestRateShock.SHORT_DOWN: 20.0,
        }
        result = supervisory_outlier_test(delta_eve, tier1_capital=1000.0)
        assert result.eve_breach is True
        assert result.worst_eve_shock == InterestRateShock.PARALLEL_UP
        assert result.eve_ratio == pytest.approx(0.20)

    def test_eve_no_breach(self) -> None:
        delta_eve = {
            InterestRateShock.PARALLEL_UP: -100.0,
            InterestRateShock.PARALLEL_DOWN: 50.0,
            InterestRateShock.STEEPENER: -30.0,
            InterestRateShock.FLATTENER: -10.0,
            InterestRateShock.SHORT_UP: -40.0,
            InterestRateShock.SHORT_DOWN: 20.0,
        }
        result = supervisory_outlier_test(delta_eve, tier1_capital=1000.0)
        # 100/1000 = 10% < 15%
        assert result.eve_breach is False

    def test_nii_breach(self) -> None:
        delta_eve = {s: -10.0 for s in SHOCK_SCENARIOS}
        result = supervisory_outlier_test(
            delta_eve, tier1_capital=1000.0, delta_nii=-30.0
        )
        # 30/1000 = 3% > 2.5%
        assert result.nii_breach is True

    def test_nii_no_breach(self) -> None:
        delta_eve = {s: -10.0 for s in SHOCK_SCENARIOS}
        result = supervisory_outlier_test(
            delta_eve, tier1_capital=1000.0, delta_nii=-20.0
        )
        # 20/1000 = 2% < 2.5%
        assert result.nii_breach is False

    def test_zero_capital_raises(self) -> None:
        with pytest.raises(ValueError, match="tier1_capital must be positive"):
            supervisory_outlier_test({}, 0.0)

    def test_result_type(self) -> None:
        delta_eve = {s: -10.0 for s in SHOCK_SCENARIOS}
        result = supervisory_outlier_test(delta_eve, 1000.0)
        assert isinstance(result, OutlierTestResult)
