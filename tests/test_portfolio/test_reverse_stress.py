"""Tests for reverse stress testing functions."""

import numpy as np
import pytest

from creditriskengine.portfolio.stress_testing import (
    reverse_stress_capital_breach,
    reverse_stress_test,
)


class TestReverseStressTest:
    """Find PD multiplier that produces a target EL."""

    def test_basic_bisection(self) -> None:
        pds = np.array([0.02, 0.03])
        lgds = np.array([0.40, 0.35])
        eads = np.array([1e6, 2e6])
        # Baseline EL = 0.02*0.40*1e6 + 0.03*0.35*2e6 = 8000 + 21000 = 29000
        target = 58000.0  # ~2x baseline
        result = reverse_stress_test(pds, lgds, eads, target)
        assert result["stressed_el"] == pytest.approx(target, abs=1.0)
        assert result["multiplier"] > 1.0
        assert result["iterations"] > 0

    def test_multiplier_near_1_for_baseline_target(self) -> None:
        pds = np.array([0.05])
        lgds = np.array([0.40])
        eads = np.array([1e6])
        baseline_el = 0.05 * 0.40 * 1e6  # 20000
        result = reverse_stress_test(pds, lgds, eads, baseline_el + 1.0)
        assert result["multiplier"] == pytest.approx(1.0, abs=0.05)

    def test_target_outside_range_raises(self) -> None:
        pds = np.array([0.05])
        lgds = np.array([0.40])
        eads = np.array([1e6])
        with pytest.raises(ValueError, match="outside achievable range"):
            reverse_stress_test(pds, lgds, eads, target_el=1e12)

    def test_target_below_low_raises(self) -> None:
        pds = np.array([0.05])
        lgds = np.array([0.40])
        eads = np.array([1e6])
        with pytest.raises(ValueError, match="outside achievable range"):
            reverse_stress_test(pds, lgds, eads, target_el=0.0)

    def test_stressed_pds_returned(self) -> None:
        pds = np.array([0.02])
        lgds = np.array([0.40])
        eads = np.array([1e6])
        target = 0.02 * 0.40 * 1e6 * 3.0  # 3x baseline
        result = reverse_stress_test(pds, lgds, eads, target)
        assert len(result["stressed_pds"]) == 1
        assert result["stressed_pds"][0] > pds[0]

    def test_custom_range(self) -> None:
        pds = np.array([0.10])
        lgds = np.array([0.50])
        eads = np.array([1e6])
        target = 0.10 * 0.50 * 1e6 * 1.5  # 1.5x
        result = reverse_stress_test(
            pds, lgds, eads, target, pd_multiplier_range=(1.0, 5.0),
        )
        assert result["stressed_el"] == pytest.approx(target, abs=1.0)


class TestReverseStressCapitalBreach:
    """Find PD multiplier that breaches CET1 floor."""

    def test_basic_breach(self) -> None:
        pds = np.array([0.02, 0.03])
        lgds = np.array([0.40, 0.35])
        eads = np.array([1e6, 2e6])
        # RWA = sum(eads) = 3e6
        # Need CET1 such that breach is feasible
        cet1_capital = 200_000.0  # ~6.67% of RWA
        result = reverse_stress_capital_breach(pds, lgds, eads, cet1_capital)
        assert result["breach_multiplier"] > 1.0
        assert result["cet1_at_breach"] == pytest.approx(0.045, abs=0.001)
        assert result["iterations"] > 0

    def test_already_breached_raises(self) -> None:
        pds = np.array([0.50])
        lgds = np.array([0.80])
        eads = np.array([1e6])
        # EL at mult=1 = 400,000; RWA = 1e6; CET1 capital = 50,000
        # CET1 ratio = (50000 - 400000) / 1e6 = -0.35 < 0.045
        with pytest.raises(ValueError, match="already below floor"):
            reverse_stress_capital_breach(pds, lgds, eads, 50_000.0)

    def test_no_breach_possible_raises(self) -> None:
        pds = np.array([0.001])
        lgds = np.array([0.10])
        eads = np.array([1e6])
        # Even at 10x: EL = 0.01*0.10*1e6 = 1000; CET1 = (1e6-1000)/1e6 = 0.999
        with pytest.raises(ValueError, match="does not breach"):
            reverse_stress_capital_breach(pds, lgds, eads, 1_000_000.0)

    def test_custom_rwa_func(self) -> None:
        pds = np.array([0.02])
        lgds = np.array([0.40])
        eads = np.array([1e6])
        cet1_capital = 80_000.0

        def custom_rwa(stressed_pds, lgds, eads):
            return float(np.sum(eads) * 1.5)  # Higher RWA

        result = reverse_stress_capital_breach(
            pds, lgds, eads, cet1_capital, rwa_func=custom_rwa,
        )
        assert result["breach_multiplier"] > 1.0

    def test_custom_floor(self) -> None:
        pds = np.array([0.02])
        lgds = np.array([0.40])
        eads = np.array([1e6])
        cet1_capital = 100_000.0
        result_low = reverse_stress_capital_breach(
            pds, lgds, eads, cet1_capital, cet1_floor_pct=0.03,
        )
        result_high = reverse_stress_capital_breach(
            pds, lgds, eads, cet1_capital, cet1_floor_pct=0.06,
        )
        # Higher floor should be breached at a lower multiplier
        assert result_high["breach_multiplier"] < result_low["breach_multiplier"]

    def test_zero_rwa_func(self) -> None:
        """When rwa_func returns 0, CET1 ratio is 0 → immediate breach."""
        pds = np.array([0.01])
        lgds = np.array([0.40])
        eads = np.array([1e6])
        with pytest.raises(ValueError):
            reverse_stress_capital_breach(
                pds, lgds, eads,
                cet1_capital=500_000,
                cet1_floor_pct=0.045,
                rwa_func=lambda sp, lg, ea: 0.0,
            )
