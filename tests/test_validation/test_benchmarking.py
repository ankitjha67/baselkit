"""Tests for model benchmarking framework."""

import pytest

from creditriskengine.validation.benchmarking import (
    BenchmarkResult,
    benchmark_comparison,
    multi_model_benchmark,
    regulatory_benchmark_check,
)


class TestBenchmarkComparison:
    def test_exact_match(self) -> None:
        result = benchmark_comparison(0.80, 0.80, metric_name="auroc")
        assert result["deviation_pct"] == pytest.approx(0.0)
        assert result["within_tolerance"] is True

    def test_within_tolerance(self) -> None:
        result = benchmark_comparison(0.82, 0.80, tolerance_pct=0.10)
        assert result["deviation_pct"] == pytest.approx(0.025)
        assert result["within_tolerance"] is True

    def test_outside_tolerance(self) -> None:
        result = benchmark_comparison(0.90, 0.80, tolerance_pct=0.10)
        assert result["deviation_pct"] == pytest.approx(0.125)
        assert result["within_tolerance"] is False

    def test_zero_benchmark(self) -> None:
        result = benchmark_comparison(0.05, 0.0)
        assert result["deviation_pct"] == pytest.approx(0.0)

    def test_negative_deviation(self) -> None:
        result = benchmark_comparison(0.70, 0.80)
        assert result["deviation_pct"] == pytest.approx(-0.125)

    def test_result_keys(self) -> None:
        result = benchmark_comparison(0.80, 0.75, "gini", 0.15)
        assert result["metric_name"] == "gini"
        assert result["model_value"] == pytest.approx(0.80)
        assert result["benchmark_value"] == pytest.approx(0.75)

    def test_negative_benchmark(self) -> None:
        result = benchmark_comparison(0.10, -0.50)
        assert result["deviation_pct"] == pytest.approx((0.10 - (-0.50)) / 0.50)


class TestMultiModelBenchmark:
    """Test multi-model comparison."""

    def test_requires_at_least_two_models(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            multi_model_benchmark({"ModelA": {"auroc": 0.80}})

    def test_basic_two_models(self) -> None:
        models = {
            "ModelA": {"auroc": 0.82, "psi": 0.05},
            "ModelB": {"auroc": 0.78, "psi": 0.12},
        }
        result = multi_model_benchmark(models)
        assert "comparison_table" in result
        assert "pairwise" in result
        assert "best_model_per_metric" in result
        assert "ranking" in result

    def test_comparison_table_has_rows(self) -> None:
        models = {
            "ModelA": {"auroc": 0.82, "psi": 0.05},
            "ModelB": {"auroc": 0.78, "psi": 0.12},
        }
        result = multi_model_benchmark(models)
        assert len(result["comparison_table"]) == 2

    def test_best_model_per_metric(self) -> None:
        models = {
            "ModelA": {"auroc": 0.82, "psi": 0.05},
            "ModelB": {"auroc": 0.78, "psi": 0.12},
        }
        result = multi_model_benchmark(models)
        assert result["best_model_per_metric"]["auroc"] == "ModelA"
        assert result["best_model_per_metric"]["psi"] == "ModelA"

    def test_pairwise_comparisons(self) -> None:
        models = {
            "ModelA": {"auroc": 0.82},
            "ModelB": {"auroc": 0.78},
        }
        result = multi_model_benchmark(models)
        assert len(result["pairwise"]) == 1
        pw = result["pairwise"][0]
        assert pw["primary_model"] == "ModelA"
        assert pw["challenger_model"] == "ModelB"

    def test_explicit_metrics_list(self) -> None:
        models = {
            "ModelA": {"auroc": 0.82, "psi": 0.05, "extra": 1.0},
            "ModelB": {"auroc": 0.78, "psi": 0.12, "extra": 2.0},
        }
        result = multi_model_benchmark(models, metrics=["auroc"])
        assert len(result["comparison_table"]) == 1

    def test_three_models(self) -> None:
        models = {
            "ModelA": {"auroc": 0.82},
            "ModelB": {"auroc": 0.78},
            "ModelC": {"auroc": 0.85},
        }
        result = multi_model_benchmark(models)
        assert result["best_model_per_metric"]["auroc"] == "ModelC"
        assert len(result["pairwise"]) == 2

    def test_ranking_order(self) -> None:
        models = {
            "ModelA": {"auroc": 0.80, "gini": 0.50},
            "ModelB": {"auroc": 0.85, "gini": 0.60},
        }
        result = multi_model_benchmark(models)
        assert result["ranking"][0][0] == "ModelB"

    def test_missing_metric_for_model(self) -> None:
        models = {
            "ModelA": {"auroc": 0.82},
            "ModelB": {"psi": 0.05},
        }
        result = multi_model_benchmark(models)
        assert len(result["comparison_table"]) == 2

    def test_custom_tolerance(self) -> None:
        models = {
            "ModelA": {"auroc": 0.80},
            "ModelB": {"auroc": 0.79},
        }
        result = multi_model_benchmark(models, tolerance_pct=0.05)
        pw = result["pairwise"][0]
        assert pw["within_tolerance"] is True


class TestRegulatoryBenchmarkCheck:
    """Test regulatory threshold checking."""

    def test_all_pass(self) -> None:
        metrics = {"auroc": 0.85, "gini": 0.60, "psi": 0.05, "ks": 0.45, "brier_score": 0.10}
        result = regulatory_benchmark_check(metrics)
        assert isinstance(result, BenchmarkResult)
        assert result.overall_pass is True
        assert "All metrics within" in result.commentary

    def test_auroc_fail(self) -> None:
        metrics = {"auroc": 0.60}
        result = regulatory_benchmark_check(metrics)
        assert result.overall_pass is False

    def test_gini_fail(self) -> None:
        metrics = {"gini": 0.30}
        result = regulatory_benchmark_check(metrics)
        assert result.overall_pass is False

    def test_psi_fail(self) -> None:
        metrics = {"psi": 0.30}
        result = regulatory_benchmark_check(metrics)
        assert result.overall_pass is False

    def test_brier_score_fail(self) -> None:
        metrics = {"brier_score": 0.30}
        result = regulatory_benchmark_check(metrics)
        assert result.overall_pass is False

    def test_iv_fail(self) -> None:
        metrics = {"iv": 0.05}
        result = regulatory_benchmark_check(metrics)
        assert result.overall_pass is False

    def test_unknown_metric_passes(self) -> None:
        metrics = {"custom_metric": 999.0}
        result = regulatory_benchmark_check(metrics)
        assert result.overall_pass is True
        assert result.metric_results[0]["note"] == "No standard threshold defined."

    def test_custom_thresholds_override(self) -> None:
        metrics = {"auroc": 0.65}
        result = regulatory_benchmark_check(
            metrics, custom_thresholds={"auroc": {"min": 0.60}}
        )
        assert result.overall_pass is True

    def test_custom_thresholds_new_metric(self) -> None:
        metrics = {"custom": 0.50}
        result = regulatory_benchmark_check(
            metrics, custom_thresholds={"custom": {"min": 0.40}}
        )
        assert result.overall_pass is True

    def test_commentary_on_failure(self) -> None:
        metrics = {"auroc": 0.50, "psi": 0.30}
        result = regulatory_benchmark_check(metrics)
        assert "failed regulatory thresholds" in result.commentary

    def test_result_fields(self) -> None:
        result = regulatory_benchmark_check({"auroc": 0.85})
        assert result.model_name == "model_under_review"
        assert result.benchmark_name == "regulatory_standard"
        assert "SR 11-7" in result.regulatory_context

    def test_metric_result_has_description(self) -> None:
        result = regulatory_benchmark_check({"auroc": 0.85})
        mr = result.metric_results[0]
        assert "description" in mr
        assert "reference" in mr

    def test_metric_with_both_min_and_max(self) -> None:
        """Line 274: threshold_desc += ', ' when a metric has both min and max."""
        metrics = {"custom_bounded": 0.50}
        custom = {
            "custom_bounded": {
                "min": 0.20, "max": 0.80,
                "description": "test", "reference": "test",
            },
        }
        result = regulatory_benchmark_check(metrics, custom_thresholds=custom)
        assert result.overall_pass is True
        mr = result.metric_results[0]
        assert "min=0.2" in mr["threshold"]
        assert "max=0.8" in mr["threshold"]
        assert ", " in mr["threshold"]

    def test_metric_with_both_min_and_max_fail(self) -> None:
        """Both min and max present, value violates max."""
        metrics = {"custom_bounded": 0.90}
        custom = {"custom_bounded": {"min": 0.20, "max": 0.80}}
        result = regulatory_benchmark_check(metrics, custom_thresholds=custom)
        assert result.overall_pass is False
