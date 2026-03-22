"""
Multi-scenario probability-weighted ECL.

Reference: IFRS 9.5.5.17-5.5.20.

ECL_weighted = Sum(s=1..S) [w_s * ECL_s]

Where w_s = probability weight for scenario s (must sum to 1.0).
Typical scenarios: base (40-60%), upside (15-25%),
downside (15-25%), severe downside (5-10%).
"""

import logging
from typing import NamedTuple

logger = logging.getLogger(__name__)


class Scenario(NamedTuple):
    """A macroeconomic scenario with its probability weight and ECL."""
    name: str
    weight: float
    ecl: float


def weighted_ecl(scenarios: list[Scenario]) -> float:
    """Calculate probability-weighted ECL across scenarios.

    Formula (IFRS 9.5.5.17):
        ECL_weighted = Sum(w_s * ECL_s)

    Args:
        scenarios: List of (name, weight, ecl) scenarios.

    Returns:
        Probability-weighted ECL.

    Raises:
        ValueError: If weights don't sum to approximately 1.0.
    """
    total_weight = sum(s.weight for s in scenarios)
    if abs(total_weight - 1.0) > 1e-6:
        raise ValueError(
            f"Scenario weights must sum to 1.0, got {total_weight:.6f}"
        )

    result = sum(s.weight * s.ecl for s in scenarios)
    logger.debug(
        "Weighted ECL: %s -> %.2f",
        [(s.name, f"w={s.weight:.2f}", f"ecl={s.ecl:.2f}") for s in scenarios],
        result,
    )
    return result


def standard_scenario_weights() -> dict[str, float]:
    """Return commonly used scenario probability weights.

    These are typical weights used in practice. Banks must
    determine their own weights based on their assessment.

    Returns:
        Dict of scenario name to weight.
    """
    return {
        "base": 0.50,
        "upside": 0.20,
        "downside": 0.20,
        "severe_downside": 0.10,
    }
