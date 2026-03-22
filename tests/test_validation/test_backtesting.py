"""Tests for PD backtesting framework — BCBS WP14, EBA GL/2017/16."""

import numpy as np
import pytest

from creditriskengine.validation.backtesting import (
    FullBacktestResult,
    GradeBacktestResult,
    MultiPeriodBacktestResult,
    VintageResult,
    multi_period_backtest,
    pd_backtest_full,
    pd_backtest_summary,
)


class TestPDBacktestSummary:
    """Test summary statistics computation."""

    def test_basic(self) -> None:
        pds = np.array([0.05, 0.05, 0.05, 0.05])
        defaults = np.array([0, 0, 1, 0])
        result = pd_backtest_summary(pds, defaults)
        assert result["n_observations"] == 4
        assert result["n_defaults"] == 1
        assert result["average_predicted_pd"] == pytest.approx(0.05)
        assert result["observed_default_rate"] == pytest.approx(0.25)

    def test_no_defaults(self) -> None:
        pds = np.array([0.01, 0.02, 0.03])
        defaults = np.array([0, 0, 0])
        result = pd_backtest_summary(pds, defaults)
        assert result["n_defaults"] == 0
        assert result["observed_default_rate"] == pytest.approx(0.0)

    def test_all_defaults(self) -> None:
        pds = np.array([0.50, 0.50])
        defaults = np.array([1, 1])
        result = pd_backtest_summary(pds, defaults)
        assert result["observed_default_rate"] == pytest.approx(1.0)

    def test_ratio_observed_to_predicted(self) -> None:
        pds = np.array([0.10, 0.10])
        defaults = np.array([0, 1])
        result = pd_backtest_summary(pds, defaults)
        # observed_dr = 0.5, avg_pd = 0.1, ratio = 5.0
        assert result["ratio_observed_to_predicted"] == pytest.approx(5.0)

    def test_zero_avg_pd(self) -> None:
        pds = np.array([0.0, 0.0])
        defaults = np.array([0, 0])
        result = pd_backtest_summary(pds, defaults)
        assert result["ratio_observed_to_predicted"] == pytest.approx(0.0)

    def test_with_rating_grades(self) -> None:
        pds = np.array([0.01, 0.05])
        defaults = np.array([0, 1])
        grades = np.array(["A", "B"])
        result = pd_backtest_summary(pds, defaults, grades)
        assert result["n_observations"] == 2


class TestPDBacktestFull:
    """Test full backtest with per-grade detail."""

    def _make_data(self) -> tuple:
        rng = np.random.default_rng(42)
        n = 500
        grades = np.array(["A"] * 200 + ["B"] * 200 + ["C"] * 100)
        # Grade A: low PD, Grade B: mid, Grade C: high
        pds = np.concatenate([
            np.full(200, 0.01),
            np.full(200, 0.05),
            np.full(100, 0.15),
        ])
        defaults = np.zeros(n, dtype=int)
        # Sprinkle some defaults consistent with PDs
        defaults[rng.choice(200, 2, replace=False)] = 1  # 2 in A
        defaults[200 + rng.choice(200, 10, replace=False)] = 1  # 10 in B
        defaults[400 + rng.choice(100, 15, replace=False)] = 1  # 15 in C
        return pds, defaults, grades

    def test_returns_full_result(self) -> None:
        pds, defaults, grades = self._make_data()
        result = pd_backtest_full(pds, defaults, grades)
        assert isinstance(result, FullBacktestResult)
        assert len(result.grade_results) == 3

    def test_grade_result_types(self) -> None:
        pds, defaults, grades = self._make_data()
        result = pd_backtest_full(pds, defaults, grades)
        for gr in result.grade_results:
            assert isinstance(gr, GradeBacktestResult)
            assert gr.traffic_light in ("green", "yellow", "red")

    def test_overall_traffic_light(self) -> None:
        pds, defaults, grades = self._make_data()
        result = pd_backtest_full(pds, defaults, grades)
        assert result.overall_traffic_light in ("green", "yellow", "red")

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            pd_backtest_full(
                np.array([0.01, 0.02]),
                np.array([0, 1, 0]),
                np.array(["A", "B"]),
            )

    def test_overall_assessment_all_green(self) -> None:
        # Very few defaults relative to PD -> green
        pds = np.full(1000, 0.50)
        defaults = np.zeros(1000, dtype=int)
        defaults[:100] = 1  # 10% default rate vs 50% predicted -> well below
        grades = np.full(1000, "A")
        result = pd_backtest_full(pds, defaults, grades)
        assert "PASS" in result.overall_assessment

    def test_overall_assessment_red(self) -> None:
        # Massive under-estimation
        pds = np.full(1000, 0.01)
        defaults = np.zeros(1000, dtype=int)
        defaults[:200] = 1  # 20% observed vs 1% predicted
        grades = np.full(1000, "A")
        result = pd_backtest_full(pds, defaults, grades)
        assert result.overall_traffic_light == "red"
        assert "FAIL" in result.overall_assessment

    def test_custom_confidence(self) -> None:
        pds, defaults, grades = self._make_data()
        result = pd_backtest_full(pds, defaults, grades, confidence=0.95)
        assert isinstance(result, FullBacktestResult)


class TestMultiPeriodBacktest:
    """Test time-series backtesting across vintages."""

    def test_basic(self) -> None:
        pds = np.array([0.05] * 100 + [0.05] * 100)
        defaults = np.zeros(200, dtype=int)
        defaults[:5] = 1
        defaults[100:105] = 1
        periods = np.array(["2024-Q1"] * 100 + ["2024-Q2"] * 100)
        result = multi_period_backtest(pds, defaults, periods)
        assert isinstance(result, MultiPeriodBacktestResult)
        assert result.n_periods == 2

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            multi_period_backtest(
                np.array([0.01]),
                np.array([0, 1]),
                np.array(["Q1"]),
            )

    def test_all_green(self) -> None:
        pds = np.full(300, 0.50)
        defaults = np.zeros(300, dtype=int)
        defaults[:30] = 1  # 10% vs 50%
        periods = np.array(["Q1"] * 100 + ["Q2"] * 100 + ["Q3"] * 100)
        result = multi_period_backtest(pds, defaults, periods)
        assert "PASS" in result.overall_assessment
        assert result.n_green == 3

    def test_systematic_red(self) -> None:
        pds = np.full(200, 0.01)
        defaults = np.zeros(200, dtype=int)
        # 50% default rate in each period vs 1% PD -> red
        defaults[:50] = 1
        defaults[100:150] = 1
        periods = np.array(["Q1"] * 100 + ["Q2"] * 100)
        result = multi_period_backtest(pds, defaults, periods)
        assert result.n_red >= 2
        assert "FAIL" in result.overall_assessment

    def test_single_red(self) -> None:
        # One period red, one green
        pds = np.full(200, 0.01)
        defaults = np.zeros(200, dtype=int)
        defaults[:50] = 1  # Q1: 50% vs 1% -> red
        # Q2: 0% vs 1% -> green
        periods = np.array(["Q1"] * 100 + ["Q2"] * 100)
        result = multi_period_backtest(pds, defaults, periods)
        assert result.n_red >= 1
        assert "WARNING" in result.overall_assessment or "FAIL" in result.overall_assessment

    def test_vintage_results_populated(self) -> None:
        pds = np.full(100, 0.05)
        defaults = np.zeros(100, dtype=int)
        defaults[:5] = 1
        periods = np.full(100, "Q1")
        result = multi_period_backtest(pds, defaults, periods)
        assert len(result.vintage_results) == 1
        v = result.vintage_results[0]
        assert isinstance(v, VintageResult)
        assert v.period == "Q1"
        assert v.n_observations == 100

    def test_custom_confidence(self) -> None:
        pds = np.full(100, 0.05)
        defaults = np.zeros(100, dtype=int)
        periods = np.full(100, "Q1")
        result = multi_period_backtest(pds, defaults, periods, confidence=0.95)
        assert isinstance(result, MultiPeriodBacktestResult)

    def test_many_yellow_periods(self) -> None:
        # Create data where many periods are borderline yellow
        n_per = 100
        pds_list = []
        def_list = []
        per_list = []
        for i in range(4):
            pds_list.append(np.full(n_per, 0.03))
            d = np.zeros(n_per, dtype=int)
            d[:6] = 1  # 6% vs 3% - borderline
            def_list.append(d)
            per_list.append(np.full(n_per, f"Q{i + 1}"))
        pds = np.concatenate(pds_list)
        defaults = np.concatenate(def_list)
        periods = np.concatenate(per_list)
        result = multi_period_backtest(pds, defaults, periods)
        assert isinstance(result, MultiPeriodBacktestResult)
