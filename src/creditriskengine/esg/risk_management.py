"""ESG risk management toolkit per EBA Guidelines EBA/GL/2025/01.

Reference:
    - EBA Guidelines on the management of ESG risks (EBA/GL/2025/01),
      published 8 January 2025, applicable from 11 January 2026 (large
      institutions) and 11 January 2027 (small and non-complex
      institutions).
    - CRD Article 76(2) and Article 87a — institution transition plans
      with quantifiable intermediate targets.
    - EBA Report on ESG Risks Management (October 2023).

Implements two quantitative centrepieces of EBA/GL/2025/01 that sit
above the existing climate (NGFS, PCAF, GAR) and ESG-ratings modules:

1. **Materiality assessment** — a likelihood x impact, exposure-weighted
   score per ESG risk driver, with the EBA's recommended mapping of
   assessment method to time horizon (Title 4):
       short term  -> exposure-based
       medium term -> sector- / portfolio-alignment-based
       long term   -> scenario-based
2. **Transition-plan monitoring** — tracks progress of a portfolio metric
   (e.g. financed-emissions intensity) against an intermediate target on
   the net-zero / regulatory-objective pathway required by CRD Art. 76(2).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

logger = logging.getLogger(__name__)


class TimeHorizon(StrEnum):
    """ESG risk assessment time horizons (EBA/GL/2025/01, Title 4)."""

    SHORT = "short"  # <= ~3 years
    MEDIUM = "medium"  # ~3-5 up to ~10 years
    LONG = "long"  # >= ~10 years, out to 2050


class MaterialityMethod(StrEnum):
    """ESG materiality assessment methodologies (EBA/GL/2025/01)."""

    EXPOSURE_BASED = "exposure_based"
    SECTOR_BASED = "sector_based"
    PORTFOLIO_ALIGNMENT = "portfolio_alignment"
    SCENARIO_BASED = "scenario_based"


class ESGFactor(StrEnum):
    """ESG risk factor categories (EBA/GL/2025/01)."""

    ENV_PHYSICAL = "environmental_physical"
    ENV_TRANSITION = "environmental_transition"
    SOCIAL = "social"
    GOVERNANCE = "governance"


class MaterialityLevel(StrEnum):
    """Outcome of an ESG materiality assessment."""

    NOT_MATERIAL = "not_material"
    MATERIAL = "material"


# EBA-recommended assessment method by time horizon (Title 4). Short
# horizons rely on current exposures; medium horizons on sector and
# portfolio-alignment views; long horizons on forward-looking scenarios.
_HORIZON_METHOD: dict[TimeHorizon, MaterialityMethod] = {
    TimeHorizon.SHORT: MaterialityMethod.EXPOSURE_BASED,
    TimeHorizon.MEDIUM: MaterialityMethod.SECTOR_BASED,
    TimeHorizon.LONG: MaterialityMethod.SCENARIO_BASED,
}


def recommended_method(horizon: TimeHorizon) -> MaterialityMethod:
    """Return the EBA-recommended materiality method for a time horizon.

    Per EBA/GL/2025/01 Title 4, institutions should combine
    methodologies across horizons: exposure-based for the short term,
    sector- and portfolio-alignment-based for the medium term, and
    scenario-based for the long term.

    Args:
        horizon: Assessment time horizon.

    Returns:
        The recommended :class:`MaterialityMethod` for that horizon.
    """
    return _HORIZON_METHOD[horizon]


@dataclass(frozen=True)
class ESGRiskDriver:
    """A single ESG risk driver acting on part of the portfolio.

    Attributes:
        factor: The ESG risk factor category.
        exposure_amount: Exposure (e.g. EAD) subject to this driver.
        likelihood: Probability the risk materialises over the assessment
            horizon, in [0, 1].
        impact: Severity of the loss given the risk materialises, in
            [0, 1] (1.0 = total impairment of the affected exposure).
        label: Optional free-text label (e.g. "flood risk, Region X").
    """

    factor: ESGFactor
    exposure_amount: float
    likelihood: float
    impact: float
    label: str = ""

    def __post_init__(self) -> None:
        if self.exposure_amount < 0.0:
            raise ValueError("exposure_amount must be non-negative")
        if not 0.0 <= self.likelihood <= 1.0:
            raise ValueError("likelihood must be in [0, 1]")
        if not 0.0 <= self.impact <= 1.0:
            raise ValueError("impact must be in [0, 1]")

    @property
    def severity(self) -> float:
        """Likelihood x impact severity score in [0, 1]."""
        return self.likelihood * self.impact


@dataclass(frozen=True)
class MaterialityResult:
    """Result of an ESG materiality assessment.

    Attributes:
        method: Methodology used (driven by the time horizon).
        horizon: Time horizon assessed.
        score: Exposure-weighted materiality score in [0, 1].
        level: Categorical outcome relative to the materiality threshold.
        threshold: Threshold applied to ``score`` for the verdict.
        score_by_factor: Materiality contribution split by ESG factor.
        exposure_at_risk: Exposure exposed to at least one ESG driver.
    """

    method: MaterialityMethod
    horizon: TimeHorizon
    score: float
    level: MaterialityLevel
    threshold: float
    score_by_factor: dict[ESGFactor, float]
    exposure_at_risk: float


def assess_esg_materiality(
    drivers: Sequence[ESGRiskDriver],
    total_exposure: float,
    horizon: TimeHorizon = TimeHorizon.SHORT,
    materiality_threshold: float = 0.10,
) -> MaterialityResult:
    """Assess the materiality of ESG risks over a time horizon.

    Implements the EBA/GL/2025/01 reference approach: each risk driver is
    scored on a likelihood x impact basis, weighted by the share of total
    portfolio exposure it affects. The exposure-weighted sum is the
    portfolio materiality score; it is compared with a threshold to give a
    material / not-material verdict. The assessment method is selected
    from the time horizon per :func:`recommended_method`.

        score = sum_i (exposure_i / total_exposure) * likelihood_i * impact_i

    Args:
        drivers: ESG risk drivers acting on the portfolio.
        total_exposure: Total portfolio exposure (denominator). Must be
            strictly positive.
        horizon: Assessment time horizon; selects the EBA method.
        materiality_threshold: Score at or above which ESG risk is deemed
            material (default 0.10). Must be in [0, 1].

    Returns:
        A :class:`MaterialityResult` with the score, per-factor breakdown,
        and verdict.

    Raises:
        ValueError: If ``total_exposure`` is not positive or the threshold
            is outside [0, 1].
    """
    if total_exposure <= 0.0:
        raise ValueError("total_exposure must be positive")
    if not 0.0 <= materiality_threshold <= 1.0:
        raise ValueError("materiality_threshold must be in [0, 1]")

    score_by_factor: dict[ESGFactor, float] = dict.fromkeys(ESGFactor, 0.0)
    exposure_at_risk = 0.0

    for driver in drivers:
        weight = driver.exposure_amount / total_exposure
        contribution = weight * driver.severity
        score_by_factor[driver.factor] += contribution
        if driver.severity > 0.0:
            exposure_at_risk += driver.exposure_amount

    # Total score is bounded at 1.0: pathological inputs (overlapping
    # drivers summing above total_exposure) cannot push materiality past
    # a full write-down of the book.
    score = min(sum(score_by_factor.values()), 1.0)
    level = (
        MaterialityLevel.MATERIAL
        if score >= materiality_threshold
        else MaterialityLevel.NOT_MATERIAL
    )

    return MaterialityResult(
        method=recommended_method(horizon),
        horizon=horizon,
        score=round(score, 6),
        level=level,
        threshold=materiality_threshold,
        score_by_factor={k: round(v, 6) for k, v in score_by_factor.items()},
        exposure_at_risk=round(min(exposure_at_risk, total_exposure), 6),
    )


@dataclass(frozen=True)
class TransitionPlanStatus:
    """Progress of a portfolio metric against its intermediate target.

    Attributes:
        expected_value: Value implied by the linear pathway at the current
            year (the intermediate target for "now").
        actual_value: Observed current value of the metric.
        gap: actual - expected. For a reduction target, a positive gap
            means the portfolio is behind plan.
        on_track: True when the portfolio is at or ahead of the pathway.
        alignment_pct: Share of the required-to-date change achieved, in
            percent (can exceed 100 when ahead of plan, or be negative
            when moving the wrong way).
        required_annual_change: Constant annual change needed from the base
            year to hit the target on a straight-line path.
    """

    expected_value: float
    actual_value: float
    gap: float
    on_track: bool
    alignment_pct: float
    required_annual_change: float


def transition_plan_alignment(
    current_value: float,
    base_year_value: float,
    target_value: float,
    base_year: int,
    target_year: int,
    current_year: int,
) -> TransitionPlanStatus:
    """Monitor a transition plan against its intermediate target.

    Required by CRD Art. 76(2) and Art. 87a and operationalised by
    EBA/GL/2025/01: institutions set quantifiable intermediate targets on
    a pathway towards their long-term ESG objective (e.g. a 2050 net-zero
    financed-emissions intensity) and monitor progress against them.

    A straight-line pathway is assumed between ``base_year_value`` and
    ``target_value``. The metric direction is inferred from the target:
    when ``target_value < base_year_value`` the plan is a reduction path
    (lower is better), otherwise it is an increase path (higher is
    better, e.g. a Green Asset Ratio uplift target).

    Args:
        current_value: Observed current value of the metric.
        base_year_value: Metric value in the base (starting) year.
        target_value: Target value to reach by ``target_year``.
        base_year: Pathway start year.
        target_year: Year the target must be met. Must exceed base_year.
        current_year: Year being assessed; expected within the pathway.

    Returns:
        A :class:`TransitionPlanStatus` describing progress.

    Raises:
        ValueError: If ``target_year`` does not exceed ``base_year`` or
            ``current_year`` falls outside the pathway window.
    """
    if target_year <= base_year:
        raise ValueError("target_year must be after base_year")
    if not base_year <= current_year <= target_year:
        raise ValueError("current_year must be within [base_year, target_year]")

    span = target_year - base_year
    elapsed_fraction = (current_year - base_year) / span
    total_change = target_value - base_year_value
    expected_value = base_year_value + elapsed_fraction * total_change

    # Reduction path when the target is below the base; otherwise growth.
    is_reduction = target_value < base_year_value
    gap = current_value - expected_value
    on_track = gap <= 0.0 if is_reduction else gap >= 0.0

    required_to_date = elapsed_fraction * total_change
    achieved_to_date = current_value - base_year_value
    if required_to_date == 0.0:
        # No movement required yet (current_year == base_year): treat any
        # alignment as fully on plan.
        alignment_pct = 100.0
    else:
        alignment_pct = (achieved_to_date / required_to_date) * 100.0

    required_annual_change = total_change / span

    return TransitionPlanStatus(
        expected_value=round(expected_value, 6),
        actual_value=current_value,
        gap=round(gap, 6),
        on_track=on_track,
        alignment_pct=round(alignment_pct, 4),
        required_annual_change=round(required_annual_change, 6),
    )
