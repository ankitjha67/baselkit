"""
Model benchmarking framework.

Reference: SR 11-7, ECB Guide to Internal Models.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Structured result of a benchmark comparison.

    Attributes:
        model_name: Identifier for the model under review.
        benchmark_name: Identifier for the benchmark model or standard.
        metric_results: Per-metric comparison results.
        overall_pass: Whether the model meets all benchmark thresholds.
        regulatory_context: Applicable regulatory reference.
        commentary: Human-readable assessment summary.
    """

    model_name: str
    benchmark_name: str
    metric_results: list[dict[str, Any]]
    overall_pass: bool
    regulatory_context: str = ""
    commentary: str = ""


def benchmark_comparison(
    model_metric: float,
    benchmark_metric: float,
    metric_name: str = "metric",
    tolerance_pct: float = 0.10,
) -> dict[str, Any]:
    """Compare model metric against a benchmark.

    Args:
        model_metric: Metric value from the model under review.
        benchmark_metric: Metric value from the benchmark model.
        metric_name: Name of the metric.
        tolerance_pct: Acceptable deviation tolerance.

    Returns:
        Comparison result dict.
    """
    if benchmark_metric == 0:
        deviation = 0.0
    else:
        deviation = (model_metric - benchmark_metric) / abs(benchmark_metric)

    return {
        "metric_name": metric_name,
        "model_value": model_metric,
        "benchmark_value": benchmark_metric,
        "deviation_pct": deviation,
        "within_tolerance": abs(deviation) <= tolerance_pct,
    }


# ---------------------------------------------------------------------------
# Standard regulatory thresholds for model performance metrics.
#
# These thresholds are derived from:
# - SR 11-7 (Fed): Guidance on Model Risk Management
# - ECB Guide to Internal Models (2019)
# - EBA GL/2017/16: Guidelines on PD estimation, LGD estimation, and
#   treatment of defaulted exposures
# ---------------------------------------------------------------------------
_REGULATORY_THRESHOLDS: dict[str, dict[str, Any]] = {
    "auroc": {
        "min": 0.70,
        "direction": "higher_is_better",
        "description": "Area Under ROC Curve — discriminatory power",
        "reference": "ECB Guide to Internal Models, SR 11-7",
    },
    "gini": {
        "min": 0.40,
        "direction": "higher_is_better",
        "description": "Gini coefficient (= 2*AUROC - 1)",
        "reference": "ECB Guide to Internal Models",
    },
    "psi": {
        "max": 0.25,
        "direction": "lower_is_better",
        "description": "Population Stability Index — distribution stability",
        "reference": "SR 11-7, EBA GL/2017/16",
    },
    "ks": {
        "min": 0.30,
        "direction": "higher_is_better",
        "description": "Kolmogorov-Smirnov statistic — separation power",
        "reference": "SR 11-7",
    },
    "brier_score": {
        "max": 0.25,
        "direction": "lower_is_better",
        "description": "Brier score — probability calibration quality",
        "reference": "BCBS WP14",
    },
    "iv": {
        "min": 0.10,
        "direction": "higher_is_better",
        "description": "Information Value — predictive power of features",
        "reference": "SR 11-7",
    },
}


def multi_model_benchmark(
    models: dict[str, dict[str, float]],
    metrics: list[str] | None = None,
    tolerance_pct: float = 0.10,
) -> dict[str, Any]:
    """Compare multiple models across multiple metrics.

    Produces a comparative table where the first model supplied is treated as
    the primary model and subsequent models serve as benchmarks.

    Per SR 11-7 Section IV.B, outcomes analysis should include comparison
    against challenger models and industry benchmarks.

    Args:
        models: Mapping of model name -> dict of metric name -> metric value.
            Example: {"ModelA": {"auroc": 0.82, "psi": 0.05},
                      "ModelB": {"auroc": 0.78, "psi": 0.12}}
        metrics: Optional list of metrics to compare. If None, uses the union
            of all metrics present across models.
        tolerance_pct: Acceptable deviation tolerance for pairwise comparisons.

    Returns:
        Dict with:
            - 'comparison_table': list of per-metric dicts with each model's value
            - 'pairwise': list of pairwise benchmark_comparison results
            - 'best_model_per_metric': dict mapping metric -> best model name
            - 'ranking': list of (model_name, n_wins) sorted by wins descending

    Raises:
        ValueError: If fewer than two models are supplied.
    """
    if len(models) < 2:
        raise ValueError("multi_model_benchmark requires at least two models.")

    model_names = list(models.keys())

    if metrics is None:
        metrics_set: set[str] = set()
        for m in models.values():
            metrics_set.update(m.keys())
        metrics = sorted(metrics_set)

    comparison_table: list[dict[str, Any]] = []
    best_per_metric: dict[str, str] = {}
    wins: dict[str, int] = {name: 0 for name in model_names}

    for metric in metrics:
        row: dict[str, Any] = {"metric": metric}
        best_val: float | None = None
        best_name: str = ""
        higher_better = _REGULATORY_THRESHOLDS.get(metric, {}).get(
            "direction", "higher_is_better"
        ) == "higher_is_better"

        for name in model_names:
            val = models[name].get(metric)
            row[name] = val
            if val is not None:
                if best_val is None:
                    best_val = val
                    best_name = name
                elif (higher_better and val > best_val) or (
                    not higher_better and val < best_val
                ):
                    best_val = val
                    best_name = name

        best_per_metric[metric] = best_name
        wins[best_name] = wins.get(best_name, 0) + 1
        comparison_table.append(row)

    # Pairwise comparisons of every non-primary model against the primary
    primary = model_names[0]
    pairwise: list[dict[str, Any]] = []
    for challenger in model_names[1:]:
        for metric in metrics:
            prim_val = models[primary].get(metric)
            chal_val = models[challenger].get(metric)
            if prim_val is not None and chal_val is not None:
                result = benchmark_comparison(
                    prim_val, chal_val, metric_name=metric, tolerance_pct=tolerance_pct
                )
                result["primary_model"] = primary
                result["challenger_model"] = challenger
                pairwise.append(result)

    ranking = sorted(wins.items(), key=lambda x: x[1], reverse=True)

    logger.info(
        "Multi-model benchmark: %d models, %d metrics. Ranking: %s",
        len(model_names),
        len(metrics),
        ranking,
    )

    return {
        "comparison_table": comparison_table,
        "pairwise": pairwise,
        "best_model_per_metric": best_per_metric,
        "ranking": ranking,
    }


def regulatory_benchmark_check(
    metrics: dict[str, float],
    custom_thresholds: dict[str, dict[str, float]] | None = None,
) -> BenchmarkResult:
    """Check model metrics against standard regulatory thresholds.

    Default thresholds (overridable via *custom_thresholds*):
        - AUROC  > 0.70   (ECB Guide, SR 11-7)
        - Gini   > 0.40   (ECB Guide)
        - PSI    < 0.25   (SR 11-7, EBA GL/2017/16)
        - KS     > 0.30   (SR 11-7)
        - Brier  < 0.25   (BCBS WP14)
        - IV     > 0.10   (SR 11-7)

    Args:
        metrics: Dict of metric name -> value (e.g. {"auroc": 0.82, "psi": 0.05}).
        custom_thresholds: Optional overrides. Keys are metric names; values are
            dicts with 'min' and/or 'max' keys.

    Returns:
        BenchmarkResult with per-metric pass/fail and overall assessment.
    """
    thresholds = dict(_REGULATORY_THRESHOLDS)
    if custom_thresholds:
        for k, v in custom_thresholds.items():
            if k in thresholds:
                thresholds[k].update(v)
            else:
                thresholds[k] = v

    results: list[dict[str, Any]] = []
    all_pass = True

    for metric_name, value in metrics.items():
        spec = thresholds.get(metric_name)
        if spec is None:
            logger.debug("No regulatory threshold defined for '%s'; skipping.", metric_name)
            results.append({
                "metric": metric_name,
                "value": value,
                "threshold": None,
                "pass": True,
                "note": "No standard threshold defined.",
            })
            continue

        passed = True
        threshold_desc = ""

        if "min" in spec:
            if value < spec["min"]:
                passed = False
            threshold_desc += f"min={spec['min']}"

        if "max" in spec:
            if value > spec["max"]:
                passed = False
            if threshold_desc:
                threshold_desc += ", "
            threshold_desc += f"max={spec['max']}"

        if not passed:
            all_pass = False

        results.append({
            "metric": metric_name,
            "value": value,
            "threshold": threshold_desc,
            "pass": passed,
            "description": spec.get("description", ""),
            "reference": spec.get("reference", ""),
        })

        if not passed:
            logger.warning(
                "Regulatory check FAIL: %s=%.4f (threshold: %s). Ref: %s",
                metric_name,
                value,
                threshold_desc,
                spec.get("reference", "N/A"),
            )

    # Build commentary
    failed = [r for r in results if not r["pass"]]
    if failed:
        commentary = (
            f"{len(failed)} metric(s) failed regulatory thresholds: "
            + ", ".join(f"{r['metric']}={r['value']}" for r in failed)
            + ". Review required per SR 11-7 Section IV and ECB Guide Chapter 6."
        )
    else:
        commentary = "All metrics within regulatory thresholds."

    return BenchmarkResult(
        model_name="model_under_review",
        benchmark_name="regulatory_standard",
        metric_results=results,
        overall_pass=all_pass,
        regulatory_context="SR 11-7 (Fed), ECB Guide to Internal Models, EBA GL/2017/16",
        commentary=commentary,
    )
