"""Tests for cure rate modeling — EBA GL/2017/16 Section 6.3.2."""

import numpy as np
import pytest

from creditriskengine.models.lgd.cure_rate import (
    CureRateResult,
    cure_rate_by_segment,
    cure_rate_term_structure,
    estimate_cure_rate,
    lgd_with_cure_adjustment,
    macro_adjusted_cure_rate,
)


class TestEstimateCureRate:
    """Test basic cure rate estimation."""

    def test_basic_cure_rate(self) -> None:
        outcomes = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])  # 30% cured
        result = estimate_cure_rate(outcomes)
        assert isinstance(result, CureRateResult)
        assert result.overall_cure_rate == pytest.approx(0.3)
        assert result.n_defaults == 10
        assert result.n_cured == 3

    def test_all_cured(self) -> None:
        outcomes = np.ones(50, dtype=np.int64)
        result = estimate_cure_rate(outcomes)
        assert result.overall_cure_rate == pytest.approx(1.0)

    def test_none_cured(self) -> None:
        outcomes = np.zeros(50, dtype=np.int64)
        result = estimate_cure_rate(outcomes)
        assert result.overall_cure_rate == pytest.approx(0.0)

    def test_with_time_in_default(self) -> None:
        outcomes = np.array([1, 1, 0, 0, 0])
        times = np.array([3.0, 6.0, 12.0, 18.0, 24.0])
        result = estimate_cure_rate(outcomes, times)
        assert result.time_to_cure_months_mean > 0


class TestCureRateBySegment:
    """Test segment-level cure rates."""

    def test_two_segments(self) -> None:
        outcomes = np.array([1, 1, 0, 0, 1, 0, 0, 0])
        segments = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        result = cure_rate_by_segment(outcomes, segments, ["retail", "corporate"])
        assert "retail" in result
        assert "corporate" in result
        assert result["retail"] == pytest.approx(0.5)
        assert result["corporate"] == pytest.approx(0.25)


class TestMacroAdjustedCureRate:
    """Test macro-sensitive cure rate adjustments."""

    def test_adverse_reduces_cure_rate(self) -> None:
        base = 0.30
        adjusted = macro_adjusted_cure_rate(base, gdp_growth_current=-0.02,
                                             gdp_growth_baseline=0.02, sensitivity=0.5)
        assert adjusted < base

    def test_favorable_increases_cure_rate(self) -> None:
        base = 0.30
        adjusted = macro_adjusted_cure_rate(base, gdp_growth_current=0.04,
                                             gdp_growth_baseline=0.02, sensitivity=0.5)
        assert adjusted > base

    def test_bounded_0_1(self) -> None:
        # Extreme scenario shouldn't go below 0
        adjusted = macro_adjusted_cure_rate(0.10, gdp_growth_current=-0.10,
                                             gdp_growth_baseline=0.02, sensitivity=2.0)
        assert 0.0 <= adjusted <= 1.0


class TestLGDWithCureAdjustment:
    """Test cure-adjusted LGD per EBA GL/2017/16."""

    def test_basic_adjustment(self) -> None:
        # LGD_adj = (1 - 0.3) * 0.45 + 0.3 * 0.0 = 0.315
        lgd_adj = lgd_with_cure_adjustment(workout_lgd=0.45, cure_rate=0.30)
        assert lgd_adj == pytest.approx(0.315)

    def test_no_cure(self) -> None:
        lgd_adj = lgd_with_cure_adjustment(workout_lgd=0.45, cure_rate=0.0)
        assert lgd_adj == pytest.approx(0.45)

    def test_full_cure(self) -> None:
        lgd_adj = lgd_with_cure_adjustment(workout_lgd=0.45, cure_rate=1.0)
        assert lgd_adj == pytest.approx(0.0)

    def test_nonzero_cured_lgd(self) -> None:
        lgd_adj = lgd_with_cure_adjustment(workout_lgd=0.45, cure_rate=0.30,
                                            lgd_if_cured=0.05)
        expected = 0.70 * 0.45 + 0.30 * 0.05
        assert lgd_adj == pytest.approx(expected)


class TestCureRateTermStructure:
    """Test cumulative cure probability curve."""

    def test_monotonic_increasing(self) -> None:
        monthly_hazard = np.array([0.05, 0.04, 0.03, 0.02, 0.01, 0.01])
        cumulative = cure_rate_term_structure(monthly_hazard)
        assert np.all(np.diff(cumulative) >= 0)

    def test_bounded(self) -> None:
        monthly_hazard = np.full(24, 0.05)
        cumulative = cure_rate_term_structure(monthly_hazard)
        assert np.all(cumulative >= 0)
        assert np.all(cumulative <= 1)
