"""
Model benchmarking framework.

Reference: SR 11-7, ECB Guide to Internal Models.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


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
