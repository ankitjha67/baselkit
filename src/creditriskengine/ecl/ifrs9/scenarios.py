"""
Multi-scenario probability-weighted ECL.

Reference: IFRS 9.5.5.17-5.5.20, IFRS 9.B5.5.41-B5.5.43.

ECL_weighted = Sum(s=1..S) [w_s * ECL_s]

Where w_s = probability weight for scenario s (must sum to 1.0).
Typical scenarios: base (40-60%), upside (15-25%),
downside (15-25%), severe downside (5-10%).

Scenario governance (weight approval, sensitivity analysis) supports
the "unbiased and probability-weighted" requirement in IFRS 9.5.5.17
and the ECB Guide to Internal Models (Feb 2017) expectations on
scenario design and weight calibration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
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


# ---------------------------------------------------------------------------
# Scenario governance — approval metadata and sensitivity analysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioSetMetadata:
    """Governance metadata for a set of probability-weighted scenarios.

    Captures the approval chain, review cadence, and methodology
    documentation required by auditors and validators.

    Reference:
        IFRS 9.5.5.17 — unbiased and probability-weighted.
        IFRS 9.B5.5.41-B5.5.43 — range of possible outcomes.
        ECB Guide to Internal Models (Feb 2017) — scenario design.
        EBA/GL/2017/06 para 74 — scenario governance expectations.

    Attributes:
        scenarios: The scenario set being governed.
        approved_by: Approving authority (e.g., "Model Risk Committee").
        approval_date: Date the weights were approved.
        next_review_date: Date by which weights must be re-assessed.
        methodology: Brief description of how weights were determined
            (e.g., "Expert panel + GDP forecast consensus").
        data_sources: External data sources used to calibrate weights
            (e.g., "IMF WEO Oct 2025, Bloomberg consensus").
        notes: Free-text notes for auditors.
    """

    scenarios: list[Scenario]
    approved_by: str = ""
    approval_date: datetime | None = None
    next_review_date: datetime | None = None
    methodology: str = ""
    data_sources: str = ""
    notes: str = ""


def validate_scenario_governance(
    meta: ScenarioSetMetadata,
) -> list[str]:
    """Check governance completeness of a scenario set.

    Returns a list of warnings that an auditor would raise.
    An empty list means the scenario set passes all governance checks.

    Reference:
        EBA/GL/2017/06 para 74 — scenario governance expectations.
        PRA SS1/23 — model risk management expectations.

    Args:
        meta: Scenario governance metadata to validate.

    Returns:
        List of warning strings (empty if compliant).
    """
    warnings: list[str] = []

    if not meta.scenarios:
        warnings.append("No scenarios defined")
        return warnings

    if len(meta.scenarios) < 3:
        warnings.append(
            "Fewer than 3 scenarios — IFRS 9.B5.5.42 expects a range "
            "of possible outcomes"
        )

    total_w = sum(s.weight for s in meta.scenarios)
    if abs(total_w - 1.0) > 1e-6:
        warnings.append(
            f"Scenario weights sum to {total_w:.6f}, expected 1.0"
        )

    if not meta.approved_by:
        warnings.append("Missing approval authority")
    if meta.approval_date is None:
        warnings.append("Missing approval date")
    if meta.next_review_date is None:
        warnings.append(
            "No review date — weights should be reassessed at least quarterly"
        )
    if (
        meta.next_review_date is not None
        and meta.approval_date is not None
        and meta.next_review_date <= meta.approval_date
    ):
        warnings.append("Review date must be after approval date")
    if not meta.methodology:
        warnings.append("Missing weight calibration methodology description")

    return warnings


@dataclass
class SensitivityResult:
    """Output of :func:`scenario_sensitivity_analysis`.

    Attributes:
        base_ecl: ECL under the original scenario weights.
        shifted_results: Mapping of scenario name to the ECL that
            results when that scenario's weight is increased by
            *shift_size* (with other weights normalised).
        max_sensitivity_scenario: Name of the scenario producing the
            largest ECL change when its weight is increased.
        max_sensitivity_pct: Percentage change in ECL for the most
            sensitive scenario.
    """

    base_ecl: float
    shifted_results: dict[str, float] = field(default_factory=dict)
    max_sensitivity_scenario: str = ""
    max_sensitivity_pct: float = 0.0


def scenario_sensitivity_analysis(
    scenarios: list[Scenario],
    shift_size: float = 0.10,
) -> SensitivityResult:
    """Analyse ECL sensitivity to scenario weight perturbations.

    For each scenario *i*, the weight is increased by *shift_size*
    and remaining weights are proportionally renormalised.  This
    quantifies how dependent the ECL estimate is on a single
    scenario's probability.

    Reference:
        IFRS 9.B5.5.43 — evaluation of range of scenarios.
        EBA/GL/2017/06 para 75 — sensitivity testing of scenario weights.

    Args:
        scenarios: Original probability-weighted scenarios.
        shift_size: Amount to increase each scenario's weight
            (default 10 pp).  Must be in (0, 1).

    Returns:
        :class:`SensitivityResult` with per-scenario sensitivity.

    Raises:
        ValueError: If *shift_size* is out of range or scenarios are
            empty.
    """
    if not scenarios:
        raise ValueError("At least one scenario required")
    if not 0.0 < shift_size < 1.0:
        raise ValueError(f"shift_size must be in (0, 1), got {shift_size}")

    base_ecl = weighted_ecl(scenarios)
    shifted: dict[str, float] = {}
    max_name = ""
    max_pct = 0.0

    for i, target in enumerate(scenarios):
        new_weight = min(target.weight + shift_size, 1.0)
        remaining = 1.0 - new_weight
        others_total = sum(s.weight for j, s in enumerate(scenarios) if j != i)

        new_scenarios: list[Scenario] = []
        for j, s in enumerate(scenarios):
            if j == i:
                new_scenarios.append(Scenario(s.name, new_weight, s.ecl))
            else:
                scaled_w = (
                    s.weight / others_total * remaining
                    if others_total > 0
                    else 0.0
                )
                new_scenarios.append(Scenario(s.name, scaled_w, s.ecl))

        shifted_ecl = weighted_ecl(new_scenarios)
        shifted[target.name] = shifted_ecl

        pct_change = (
            abs(shifted_ecl - base_ecl) / base_ecl * 100.0
            if base_ecl != 0
            else 0.0
        )
        if pct_change > max_pct:
            max_pct = pct_change
            max_name = target.name

    return SensitivityResult(
        base_ecl=base_ecl,
        shifted_results=shifted,
        max_sensitivity_scenario=max_name,
        max_sensitivity_pct=round(max_pct, 2),
    )
