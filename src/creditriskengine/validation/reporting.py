"""
Automated validation report generator.

Generates standardized model validation reports combining
discrimination, calibration, and stability results.
"""

import datetime
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Attempt to import Jinja2 for template-based reporting; fall back gracefully.
try:
    import jinja2

    _HAS_JINJA2 = True
except ImportError:  # pragma: no cover
    _HAS_JINJA2 = False


def generate_validation_summary(
    model_name: str,
    discrimination_results: dict[str, float],
    calibration_results: dict[str, Any],
    stability_results: dict[str, float],
) -> dict[str, Any]:
    """Generate a validation summary report.

    Args:
        model_name: Name of the model being validated.
        discrimination_results: Results from discrimination tests.
        calibration_results: Results from calibration tests.
        stability_results: Results from stability tests.

    Returns:
        Structured validation summary.
    """
    # Traffic light assessment
    overall = "green"
    if discrimination_results.get("gini", 0) < 0.3:
        overall = "yellow"
    if discrimination_results.get("gini", 0) < 0.15:
        overall = "red"

    psi = stability_results.get("psi", 0)
    if psi >= 0.25:
        overall = "red"
    elif psi >= 0.10 and overall != "red":
        overall = "yellow"

    return {
        "model_name": model_name,
        "overall_assessment": overall,
        "discrimination": discrimination_results,
        "calibration": calibration_results,
        "stability": stability_results,
    }


# ---------------------------------------------------------------------------
# Default Jinja2 template for text report generation
# ---------------------------------------------------------------------------
_DEFAULT_TEMPLATE = """\
================================================================================
MODEL VALIDATION REPORT
================================================================================
Model Name      : {{ model_name }}
Report Date     : {{ report_date }}
Overall Status  : {{ overall_assessment | upper }}
================================================================================

1. DISCRIMINATION
{% for key, val in discrimination.items() -%}
   {{ "%-25s" | format(key) }}: {{ val }}
{% endfor %}
2. CALIBRATION
{% for key, val in calibration.items() -%}
   {{ "%-25s" | format(key) }}: {{ val }}
{% endfor %}
3. STABILITY
{% for key, val in stability.items() -%}
   {{ "%-25s" | format(key) }}: {{ val }}
{% endfor %}
{% if backtesting -%}
4. BACKTESTING
{% for key, val in backtesting.items() -%}
   {{ "%-25s" | format(key) }}: {{ val }}
{% endfor %}
{% endif -%}
{% if benchmarking -%}
5. BENCHMARKING
{% for key, val in benchmarking.items() -%}
   {{ "%-25s" | format(key) }}: {{ val }}
{% endfor %}
{% endif -%}
{% if recommendations %}
RECOMMENDATIONS
{% for rec in recommendations -%}
 - {{ rec }}
{% endfor %}
{% endif -%}
================================================================================
Regulatory references: BCBS WP14, SR 11-7, ECB Guide, EBA GL/2017/16.
================================================================================
"""

# Plain-text fallback when Jinja2 is not available
_SECTION_SEP = "=" * 80


def _format_dict_section(title: str, data: dict[str, Any]) -> str:
    """Format a dict as a titled text section."""
    lines = [f"\n{title}"]
    for key, val in data.items():
        lines.append(f"   {key:<25s}: {val}")
    return "\n".join(lines)


def generate_validation_report_text(
    model_name: str,
    discrimination_results: dict[str, float],
    calibration_results: dict[str, Any],
    stability_results: dict[str, float],
    backtesting_results: dict[str, Any] | None = None,
    benchmarking_results: dict[str, Any] | None = None,
    recommendations: list[str] | None = None,
    report_date: str | None = None,
    template_string: str | None = None,
) -> str:
    """Produce a formatted text validation report.

    If Jinja2 is available, uses a Jinja2 template (either the provided
    *template_string* or the built-in default). Otherwise falls back to
    plain string formatting.

    Args:
        model_name: Name of the model.
        discrimination_results: Discrimination test outcomes.
        calibration_results: Calibration test outcomes.
        stability_results: Stability test outcomes.
        backtesting_results: Optional backtesting outcomes.
        benchmarking_results: Optional benchmarking outcomes.
        recommendations: Optional list of recommendation strings.
        report_date: Optional report date string (defaults to today).
        template_string: Optional custom Jinja2 template.

    Returns:
        Formatted report as a string.
    """
    if report_date is None:
        report_date = datetime.date.today().isoformat()

    summary = generate_validation_summary(
        model_name, discrimination_results, calibration_results, stability_results
    )

    context: dict[str, Any] = {
        **summary,
        "report_date": report_date,
        "backtesting": backtesting_results,
        "benchmarking": benchmarking_results,
        "recommendations": recommendations or [],
    }

    if _HAS_JINJA2:
        tmpl_src = template_string or _DEFAULT_TEMPLATE
        env = jinja2.Environment(undefined=jinja2.StrictUndefined)
        tmpl = env.from_string(tmpl_src)
        report = tmpl.render(**context)
        logger.debug("Validation report generated via Jinja2 template.")
        return report

    # Fallback: manual formatting
    lines: list[str] = [
        _SECTION_SEP,
        "MODEL VALIDATION REPORT",
        _SECTION_SEP,
        f"Model Name      : {model_name}",
        f"Report Date     : {report_date}",
        f"Overall Status  : {summary['overall_assessment'].upper()}",
        _SECTION_SEP,
    ]
    lines.append(_format_dict_section("1. DISCRIMINATION", discrimination_results))
    lines.append(_format_dict_section("2. CALIBRATION", calibration_results))
    lines.append(_format_dict_section("3. STABILITY", stability_results))
    if backtesting_results:
        lines.append(_format_dict_section("4. BACKTESTING", backtesting_results))
    if benchmarking_results:
        lines.append(_format_dict_section("5. BENCHMARKING", benchmarking_results))
    if recommendations:
        lines.append("\nRECOMMENDATIONS")
        for rec in recommendations:
            lines.append(f" - {rec}")
    lines.append(_SECTION_SEP)
    lines.append("Regulatory references: BCBS WP14, SR 11-7, ECB Guide, EBA GL/2017/16.")
    lines.append(_SECTION_SEP)

    report = "\n".join(lines)
    logger.debug("Validation report generated via fallback formatter.")
    return report


def export_validation_json(
    model_name: str,
    discrimination_results: dict[str, float],
    calibration_results: dict[str, Any],
    stability_results: dict[str, float],
    backtesting_results: dict[str, Any] | None = None,
    benchmarking_results: dict[str, Any] | None = None,
    recommendations: list[str] | None = None,
    report_date: str | None = None,
    indent: int = 2,
) -> str:
    """Export validation results as a JSON string.

    Combines all validation dimensions into a single JSON document suitable
    for storage, transmission, or ingestion by downstream systems.

    Args:
        model_name: Name of the model.
        discrimination_results: Discrimination test outcomes.
        calibration_results: Calibration test outcomes.
        stability_results: Stability test outcomes.
        backtesting_results: Optional backtesting outcomes.
        benchmarking_results: Optional benchmarking outcomes.
        recommendations: Optional list of recommendation strings.
        report_date: Optional report date (defaults to today).
        indent: JSON indentation level.

    Returns:
        JSON string representation of the validation report.
    """
    if report_date is None:
        report_date = datetime.date.today().isoformat()

    summary = generate_validation_summary(
        model_name, discrimination_results, calibration_results, stability_results
    )

    payload: dict[str, Any] = {
        **summary,
        "report_date": report_date,
        "backtesting": backtesting_results,
        "benchmarking": benchmarking_results,
        "recommendations": recommendations or [],
    }

    # Sanitise numpy types that are not JSON-serialisable
    def _default(obj: Any) -> Any:  # noqa: ANN401
        try:
            import numpy as np

            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    result = json.dumps(payload, indent=indent, default=_default)
    logger.debug("Validation results exported to JSON (%d bytes).", len(result))
    return result


class MetricSnapshot:
    """A single point-in-time metric observation for time-series tracking.

    Attributes:
        date: Observation date string (ISO 8601).
        metric_name: Name of the metric.
        value: Metric value.
        threshold_breach: Whether the value breaches a threshold.
    """

    __slots__ = ("date", "metric_name", "value", "threshold_breach")

    def __init__(
        self,
        date: str,
        metric_name: str,
        value: float,
        threshold_breach: bool = False,
    ) -> None:
        self.date = date
        self.metric_name = metric_name
        self.value = value
        self.threshold_breach = threshold_breach

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "metric_name": self.metric_name,
            "value": self.value,
            "threshold_breach": self.threshold_breach,
        }


def time_series_tracking(
    metric_history: list[dict[str, Any]],
    thresholds: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Track validation metrics over time and detect trends or breaches.

    Designed for ongoing model monitoring as required by SR 11-7 Section V
    and ECB Guide Chapter 7 — ensuring that model performance is tracked
    continuously, not only at periodic validation.

    Args:
        metric_history: List of observation dicts, each with keys:
            - 'date': ISO date string
            - 'metric_name': Name of the metric
            - 'value': Float metric value
        thresholds: Optional dict mapping metric_name -> {'min': x} or {'max': y}
            for breach detection. If None, no breach detection is performed.

    Returns:
        Dict with:
            - 'snapshots': list of MetricSnapshot dicts
            - 'summary_by_metric': per-metric summary (mean, std, min, max, trend, n_breaches)
            - 'overall_stable': bool — True if no metric shows a deteriorating trend

    Raises:
        ValueError: If metric_history is empty.
    """
    if not metric_history:
        raise ValueError("metric_history must be a non-empty list.")

    thresholds = thresholds or {}

    # Group by metric name
    by_metric: dict[str, list[dict[str, Any]]] = {}
    for obs in metric_history:
        name = obs["metric_name"]
        by_metric.setdefault(name, []).append(obs)

    snapshots: list[dict[str, Any]] = []
    summary_by_metric: dict[str, dict[str, Any]] = {}
    all_stable = True

    for metric_name, observations in by_metric.items():
        # Sort by date
        observations = sorted(observations, key=lambda x: x["date"])
        values = [o["value"] for o in observations]
        dates = [o["date"] for o in observations]

        n_breaches = 0
        spec = thresholds.get(metric_name, {})

        for obs in observations:
            breach = False
            if "min" in spec and obs["value"] < spec["min"]:
                breach = True
            if "max" in spec and obs["value"] > spec["max"]:
                breach = True
            if breach:
                n_breaches += 1
            snap = MetricSnapshot(
                date=obs["date"],
                metric_name=metric_name,
                value=obs["value"],
                threshold_breach=breach,
            )
            snapshots.append(snap.to_dict())

        # Compute simple linear trend (slope) if enough points
        trend_direction = "stable"
        slope = 0.0
        if len(values) >= 3:
            try:
                import numpy as np

                x = np.arange(len(values), dtype=np.float64)
                y = np.array(values, dtype=np.float64)
                slope = float(np.polyfit(x, y, 1)[0])
                # Determine if trend is material (> 5% relative change per period)
                mean_val = float(np.mean(y))
                if mean_val != 0 and abs(slope / mean_val) > 0.05:
                    higher_better = spec.get("direction", "higher_is_better") == "higher_is_better"
                    if (higher_better and slope < 0) or (not higher_better and slope > 0):
                        trend_direction = "deteriorating"
                        all_stable = False
                    else:
                        trend_direction = "improving"
            except Exception:  # noqa: BLE001
                logger.warning("Could not compute trend for metric '%s'.", metric_name)

        summary_by_metric[metric_name] = {
            "n_observations": len(values),
            "mean": sum(values) / len(values),
            "std": (
                (sum((v - sum(values) / len(values)) ** 2 for v in values) / len(values))
                ** 0.5
            ),
            "min": min(values),
            "max": max(values),
            "latest": values[-1],
            "trend_slope": slope,
            "trend_direction": trend_direction,
            "n_breaches": n_breaches,
            "dates": dates,
        }

        if n_breaches > 0:
            logger.warning(
                "Metric '%s': %d threshold breach(es) detected over %d observations.",
                metric_name,
                n_breaches,
                len(values),
            )

    logger.info(
        "Time-series tracking: %d metric(s), overall_stable=%s",
        len(by_metric),
        all_stable,
    )

    return {
        "snapshots": snapshots,
        "summary_by_metric": summary_by_metric,
        "overall_stable": all_stable,
    }
