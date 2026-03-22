"""Tests for model benchmarking framework."""

import pytest

from creditriskengine.validation.benchmarking import benchmark_comparison


class TestBenchmarkComparison:
    def test_exact_match(self):
        result = benchmark_comparison(0.80, 0.80, metric_name="auroc")
        assert result["deviation_pct"] == pytest.approx(0.0)
        assert result["within_tolerance"] is True

    def test_within_tolerance(self):
        result = benchmark_comparison(0.82, 0.80, tolerance_pct=0.10)
        assert result["deviation_pct"] == pytest.approx(0.025)
        assert result["within_tolerance"] is True

    def test_outside_tolerance(self):
        result = benchmark_comparison(0.90, 0.80, tolerance_pct=0.10)
        assert result["deviation_pct"] == pytest.approx(0.125)
        assert result["within_tolerance"] is False

    def test_zero_benchmark(self):
        result = benchmark_comparison(0.05, 0.0)
        assert result["deviation_pct"] == pytest.approx(0.0)

    def test_negative_deviation(self):
        result = benchmark_comparison(0.70, 0.80)
        assert result["deviation_pct"] == pytest.approx(-0.125)

    def test_result_keys(self):
        result = benchmark_comparison(0.80, 0.75, "gini", 0.15)
        assert result["metric_name"] == "gini"
        assert result["model_value"] == pytest.approx(0.80)
        assert result["benchmark_value"] == pytest.approx(0.75)
