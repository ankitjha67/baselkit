"""Tests for automated validation report generator."""

import json
from unittest import mock

import numpy as np
import pytest

import creditriskengine.validation.reporting as reporting_module
from creditriskengine.validation.reporting import (
    MetricSnapshot,
    _format_dict_section,
    export_validation_json,
    generate_validation_report_text,
    generate_validation_summary,
    time_series_tracking,
)


class TestGenerateValidationSummary:
    """Test traffic-light assessment logic."""

    def test_all_green(self) -> None:
        summary = generate_validation_summary(
            "TestModel",
            {"gini": 0.50},
            {"hl_stat": 5.0},
            {"psi": 0.05},
        )
        assert summary["overall_assessment"] == "green"
        assert summary["model_name"] == "TestModel"

    def test_low_gini_yellow(self) -> None:
        summary = generate_validation_summary(
            "TestModel",
            {"gini": 0.20},
            {},
            {"psi": 0.05},
        )
        assert summary["overall_assessment"] == "yellow"

    def test_very_low_gini_red(self) -> None:
        summary = generate_validation_summary(
            "TestModel",
            {"gini": 0.10},
            {},
            {"psi": 0.05},
        )
        assert summary["overall_assessment"] == "red"

    def test_high_psi_red(self) -> None:
        summary = generate_validation_summary(
            "TestModel",
            {"gini": 0.50},
            {},
            {"psi": 0.30},
        )
        assert summary["overall_assessment"] == "red"

    def test_moderate_psi_yellow(self) -> None:
        summary = generate_validation_summary(
            "TestModel",
            {"gini": 0.50},
            {},
            {"psi": 0.15},
        )
        assert summary["overall_assessment"] == "yellow"

    def test_psi_does_not_downgrade_red_gini(self) -> None:
        # Gini already makes it red; moderate PSI shouldn't change that
        summary = generate_validation_summary(
            "TestModel",
            {"gini": 0.10},
            {},
            {"psi": 0.15},
        )
        assert summary["overall_assessment"] == "red"

    def test_missing_gini_key(self) -> None:
        summary = generate_validation_summary(
            "TestModel",
            {},
            {},
            {"psi": 0.05},
        )
        # gini defaults to 0 -> red
        assert summary["overall_assessment"] == "red"

    def test_missing_psi_key(self) -> None:
        summary = generate_validation_summary(
            "TestModel",
            {"gini": 0.50},
            {},
            {},
        )
        assert summary["overall_assessment"] == "green"


class TestFormatDictSection:
    """Test the plain text section formatter."""

    def test_basic(self) -> None:
        text = _format_dict_section("SECTION", {"key1": "val1", "key2": 0.5})
        assert "SECTION" in text
        assert "key1" in text
        assert "val1" in text

    def test_empty_dict(self) -> None:
        text = _format_dict_section("EMPTY", {})
        assert "EMPTY" in text


class TestGenerateValidationReportText:
    """Test text report generation."""

    def test_returns_string(self) -> None:
        report = generate_validation_report_text(
            "TestModel",
            {"gini": 0.50, "auroc": 0.75},
            {"hl_stat": 5.0},
            {"psi": 0.05},
        )
        assert isinstance(report, str)
        assert "TestModel" in report

    def test_contains_discrimination_section(self) -> None:
        report = generate_validation_report_text(
            "M", {"gini": 0.50}, {"hl": 1.0}, {"psi": 0.05}
        )
        assert "DISCRIMINATION" in report

    def test_contains_overall_status(self) -> None:
        report = generate_validation_report_text(
            "M", {"gini": 0.50}, {}, {"psi": 0.05}
        )
        assert "GREEN" in report

    def test_with_backtesting(self) -> None:
        report = generate_validation_report_text(
            "M",
            {"gini": 0.50},
            {},
            {"psi": 0.05},
            backtesting_results={"traffic_light": "green"},
        )
        assert "BACKTESTING" in report

    def test_with_benchmarking(self) -> None:
        report = generate_validation_report_text(
            "M",
            {"gini": 0.50},
            {},
            {"psi": 0.05},
            benchmarking_results={"auroc_diff": 0.02},
        )
        assert "BENCHMARKING" in report

    def test_with_recommendations(self) -> None:
        report = generate_validation_report_text(
            "M",
            {"gini": 0.50},
            {},
            {"psi": 0.05},
            recommendations=["Recalibrate model"],
        )
        assert "Recalibrate model" in report

    def test_custom_report_date(self) -> None:
        report = generate_validation_report_text(
            "M",
            {"gini": 0.50},
            {},
            {"psi": 0.05},
            report_date="2025-01-15",
        )
        assert "2025-01-15" in report

    def test_custom_template_string(self) -> None:
        template = "Model: {{ model_name }} Status: {{ overall_assessment }}"
        report = generate_validation_report_text(
            "MyModel",
            {"gini": 0.50},
            {},
            {"psi": 0.05},
            template_string=template,
        )
        assert "MyModel" in report
        assert "green" in report


class TestExportValidationJson:
    """Test JSON export."""

    def test_valid_json(self) -> None:
        result = export_validation_json(
            "TestModel",
            {"gini": 0.50},
            {"hl": 5.0},
            {"psi": 0.05},
        )
        data = json.loads(result)
        assert data["model_name"] == "TestModel"

    def test_contains_all_sections(self) -> None:
        result = export_validation_json(
            "M",
            {"gini": 0.50},
            {"hl": 1.0},
            {"psi": 0.05},
            backtesting_results={"tl": "green"},
            benchmarking_results={"diff": 0.01},
            recommendations=["Fix it"],
        )
        data = json.loads(result)
        assert data["backtesting"] is not None
        assert data["benchmarking"] is not None
        assert "Fix it" in data["recommendations"]

    def test_custom_indent(self) -> None:
        result = export_validation_json(
            "M", {"gini": 0.50}, {}, {"psi": 0.05}, indent=4
        )
        # 4-space indent should be present
        assert "    " in result

    def test_custom_report_date(self) -> None:
        result = export_validation_json(
            "M", {"gini": 0.50}, {}, {"psi": 0.05}, report_date="2025-06-01"
        )
        data = json.loads(result)
        assert data["report_date"] == "2025-06-01"

    def test_numpy_serialization(self) -> None:
        result = export_validation_json(
            "M",
            {"gini": np.float64(0.55)},
            {"count": np.int64(100)},
            {"psi": np.float64(0.03), "arr": np.array([1.0, 2.0])},
        )
        data = json.loads(result)
        assert data["discrimination"]["gini"] == pytest.approx(0.55)

    def test_non_serializable_raises(self) -> None:
        with pytest.raises(TypeError):
            export_validation_json(
                "M",
                {"gini": 0.50},
                {"bad": object()},
                {"psi": 0.05},
            )


class TestMetricSnapshot:
    """Test MetricSnapshot data class."""

    def test_creation(self) -> None:
        snap = MetricSnapshot("2025-01-01", "gini", 0.55)
        assert snap.date == "2025-01-01"
        assert snap.metric_name == "gini"
        assert snap.value == 0.55
        assert snap.threshold_breach is False

    def test_with_breach(self) -> None:
        snap = MetricSnapshot("2025-01-01", "psi", 0.30, threshold_breach=True)
        assert snap.threshold_breach is True

    def test_to_dict(self) -> None:
        snap = MetricSnapshot("2025-01-01", "gini", 0.55, False)
        d = snap.to_dict()
        assert d["date"] == "2025-01-01"
        assert d["metric_name"] == "gini"
        assert d["value"] == 0.55
        assert d["threshold_breach"] is False


class TestTimeSeriesTracking:
    """Test time-series metric tracking."""

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            time_series_tracking([])

    def test_single_metric_stable(self) -> None:
        history = [
            {"date": "2025-01-01", "metric_name": "gini", "value": 0.50},
            {"date": "2025-02-01", "metric_name": "gini", "value": 0.51},
            {"date": "2025-03-01", "metric_name": "gini", "value": 0.49},
        ]
        result = time_series_tracking(history)
        assert result["overall_stable"] is True
        assert "gini" in result["summary_by_metric"]
        summary = result["summary_by_metric"]["gini"]
        assert summary["n_observations"] == 3
        assert summary["latest"] == pytest.approx(0.49)

    def test_breach_detection(self) -> None:
        history = [
            {"date": "2025-01-01", "metric_name": "psi", "value": 0.05},
            {"date": "2025-02-01", "metric_name": "psi", "value": 0.15},
            {"date": "2025-03-01", "metric_name": "psi", "value": 0.30},
        ]
        thresholds = {"psi": {"max": 0.25}}
        result = time_series_tracking(history, thresholds)
        summary = result["summary_by_metric"]["psi"]
        assert summary["n_breaches"] == 1  # only 0.30 breaches

    def test_min_threshold_breach(self) -> None:
        history = [
            {"date": "2025-01-01", "metric_name": "gini", "value": 0.50},
            {"date": "2025-02-01", "metric_name": "gini", "value": 0.30},
            {"date": "2025-03-01", "metric_name": "gini", "value": 0.20},
        ]
        thresholds = {"gini": {"min": 0.40}}
        result = time_series_tracking(history, thresholds)
        summary = result["summary_by_metric"]["gini"]
        assert summary["n_breaches"] == 2

    def test_deteriorating_trend(self) -> None:
        # Clearly declining gini with higher_is_better
        history = [
            {"date": f"2025-{i:02d}-01", "metric_name": "gini", "value": 0.60 - i * 0.08}
            for i in range(1, 6)
        ]
        thresholds = {"gini": {"direction": "higher_is_better"}}
        result = time_series_tracking(history, thresholds)
        summary = result["summary_by_metric"]["gini"]
        assert summary["trend_direction"] == "deteriorating"
        assert result["overall_stable"] is False

    def test_improving_trend(self) -> None:
        # Clearly increasing gini
        history = [
            {"date": f"2025-{i:02d}-01", "metric_name": "gini", "value": 0.30 + i * 0.08}
            for i in range(1, 6)
        ]
        thresholds = {"gini": {"direction": "higher_is_better"}}
        result = time_series_tracking(history, thresholds)
        summary = result["summary_by_metric"]["gini"]
        assert summary["trend_direction"] == "improving"

    def test_lower_is_better_deteriorating(self) -> None:
        # PSI increasing (lower_is_better) -> deteriorating
        history = [
            {"date": f"2025-{i:02d}-01", "metric_name": "psi", "value": 0.02 + i * 0.05}
            for i in range(1, 6)
        ]
        thresholds = {"psi": {"direction": "lower_is_better"}}
        result = time_series_tracking(history, thresholds)
        summary = result["summary_by_metric"]["psi"]
        assert summary["trend_direction"] == "deteriorating"

    def test_multiple_metrics(self) -> None:
        history = [
            {"date": "2025-01-01", "metric_name": "gini", "value": 0.50},
            {"date": "2025-01-01", "metric_name": "psi", "value": 0.05},
            {"date": "2025-02-01", "metric_name": "gini", "value": 0.51},
            {"date": "2025-02-01", "metric_name": "psi", "value": 0.06},
        ]
        result = time_series_tracking(history)
        assert len(result["summary_by_metric"]) == 2
        assert len(result["snapshots"]) == 4

    def test_few_points_stable(self) -> None:
        # Only 2 points -> not enough for trend -> stays stable
        history = [
            {"date": "2025-01-01", "metric_name": "gini", "value": 0.50},
            {"date": "2025-02-01", "metric_name": "gini", "value": 0.30},
        ]
        result = time_series_tracking(history)
        summary = result["summary_by_metric"]["gini"]
        assert summary["trend_direction"] == "stable"
        assert summary["trend_slope"] == pytest.approx(0.0)

    def test_no_thresholds(self) -> None:
        history = [
            {"date": "2025-01-01", "metric_name": "gini", "value": 0.50},
        ]
        result = time_series_tracking(history)
        assert result["summary_by_metric"]["gini"]["n_breaches"] == 0

    def test_snapshots_contain_breach_flag(self) -> None:
        history = [
            {"date": "2025-01-01", "metric_name": "psi", "value": 0.30},
        ]
        thresholds = {"psi": {"max": 0.25}}
        result = time_series_tracking(history, thresholds)
        assert result["snapshots"][0]["threshold_breach"] is True

    def test_summary_statistics(self) -> None:
        history = [
            {"date": "2025-01-01", "metric_name": "gini", "value": 0.40},
            {"date": "2025-02-01", "metric_name": "gini", "value": 0.60},
        ]
        result = time_series_tracking(history)
        summary = result["summary_by_metric"]["gini"]
        assert summary["mean"] == pytest.approx(0.50)
        assert summary["min"] == pytest.approx(0.40)
        assert summary["max"] == pytest.approx(0.60)
        assert summary["std"] == pytest.approx(0.10)


class TestFallbackTextReport:
    """Test text report generation when Jinja2 is not available (lines 177-203)."""

    def test_fallback_basic(self) -> None:
        """Exercise the manual formatting fallback path."""
        with mock.patch.object(reporting_module, "_HAS_JINJA2", False):
            report = generate_validation_report_text(
                "FallbackModel",
                {"gini": 0.50, "auroc": 0.75},
                {"hl_stat": 5.0},
                {"psi": 0.05},
                report_date="2025-06-01",
            )
        assert "FallbackModel" in report
        assert "2025-06-01" in report
        assert "GREEN" in report
        assert "DISCRIMINATION" in report
        assert "CALIBRATION" in report
        assert "STABILITY" in report

    def test_fallback_with_backtesting(self) -> None:
        with mock.patch.object(reporting_module, "_HAS_JINJA2", False):
            report = generate_validation_report_text(
                "M",
                {"gini": 0.50},
                {},
                {"psi": 0.05},
                backtesting_results={"traffic_light": "green"},
                report_date="2025-01-01",
            )
        assert "BACKTESTING" in report

    def test_fallback_with_benchmarking(self) -> None:
        with mock.patch.object(reporting_module, "_HAS_JINJA2", False):
            report = generate_validation_report_text(
                "M",
                {"gini": 0.50},
                {},
                {"psi": 0.05},
                benchmarking_results={"auroc_diff": 0.02},
                report_date="2025-01-01",
            )
        assert "BENCHMARKING" in report

    def test_fallback_with_recommendations(self) -> None:
        with mock.patch.object(reporting_module, "_HAS_JINJA2", False):
            report = generate_validation_report_text(
                "M",
                {"gini": 0.50},
                {},
                {"psi": 0.05},
                recommendations=["Recalibrate model", "Increase sample size"],
                report_date="2025-01-01",
            )
        assert "RECOMMENDATIONS" in report
        assert "Recalibrate model" in report
        assert "Increase sample size" in report

    def test_fallback_regulatory_references(self) -> None:
        with mock.patch.object(reporting_module, "_HAS_JINJA2", False):
            report = generate_validation_report_text(
                "M",
                {"gini": 0.50},
                {},
                {"psi": 0.05},
                report_date="2025-01-01",
            )
        assert "Regulatory references" in report


class TestExportJsonNumpyFloating:
    """Test the np.floating branch in _default (line 259)."""

    def test_numpy_floating_serialization(self) -> None:
        """np.float32 is np.floating but not np.integer or ndarray."""
        result = export_validation_json(
            "M",
            {"gini": np.float32(0.55)},
            {},
            {"psi": 0.05},
        )
        data = json.loads(result)
        assert data["discrimination"]["gini"] == pytest.approx(0.55, abs=1e-5)


class TestExportJsonNumpyImportError:
    """Test the ImportError fallback in _default (lines 262-263)."""

    def test_numpy_import_error_fallback(self) -> None:
        """When numpy import fails inside _default, non-serializable objects raise TypeError."""
        import builtins

        real_import = builtins.__import__

        def fake_import(
            name: str, *args: object, **kwargs: object,
        ) -> object:
            if name == "numpy":
                raise ImportError("mocked numpy unavailable")
            return real_import(name, *args, **kwargs)

        # We need an object that would normally be caught by numpy checks
        # but since numpy import fails, it falls through to TypeError
        with (
            mock.patch("builtins.__import__", side_effect=fake_import),
            pytest.raises(TypeError, match="not JSON serializable"),
        ):
                export_validation_json(
                    "M",
                    {"gini": 0.50},
                    {"bad": object()},
                    {"psi": 0.05},
                )


class TestTimeSeriesTrackingTrendException:
    """Test the except block in trend computation (lines 390-391)."""

    def test_polyfit_exception_falls_back_to_stable(self) -> None:
        """When np.polyfit raises, trend should remain 'stable'."""
        history = [
            {"date": f"2025-{i:02d}-01", "metric_name": "gini", "value": 0.60 - i * 0.08}
            for i in range(1, 6)
        ]
        with mock.patch("numpy.polyfit", side_effect=RuntimeError("mocked polyfit failure")):
            result = time_series_tracking(history)
        summary = result["summary_by_metric"]["gini"]
        assert summary["trend_direction"] == "stable"
        assert summary["trend_slope"] == pytest.approx(0.0)
