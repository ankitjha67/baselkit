"""Climate scenario analysis for credit portfolios (EBA/GL/2025/02, NGFS).

Reference:
    - EBA Guidelines on ESG scenario analysis (EBA/GL/2025/02).
    - NGFS Phase V Scenarios (Vintage 5.0, November 2024).
    - ECB/SSM climate risk stress test methodology (2022).

Orchestrates the existing climate building blocks — NGFS scenarios,
transition-risk PD multipliers, and physical-risk PD/LGD adjustments —
into a portfolio-level projection of climate-adjusted expected credit
loss. For a given scenario and horizon year it computes, per exposure:

    stressed_PD  = min(PD * transition_mult * physical_mult, 1)
    stressed_LGD = min(LGD + physical_LGD_haircut, 1)
    stressed_ECL = EAD * stressed_PD * stressed_LGD

and aggregates the baseline-vs-stressed ECL uplift across the book,
decomposed into transition and physical contributions.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from creditriskengine.climate.ngfs_scenarios import NGFSScenario
from creditriskengine.climate.physical_risk import (
    PhysicalHazard,
    physical_risk_lgd_haircut,
    physical_risk_pd_multiplier,
)
from creditriskengine.climate.transition_risk import transition_risk_pd_multiplier

logger = logging.getLogger(__name__)

# Carbon-price anchor years in the NGFS scenario definitions.
_CP_BASE_YEAR = 2030
_CP_END_YEAR = 2050


@dataclass(frozen=True)
class ClimateExposure:
    """A credit exposure with the attributes needed for climate stressing.

    Attributes:
        exposure_id: Identifier.
        ead: Exposure at default.
        pd: Baseline (point-in-time) probability of default in [0, 1].
        lgd: Baseline loss given default in [0, 1].
        sector: Sector key for transition-risk elasticity (e.g. "energy").
        scope1_emissions_tco2e: Counterparty Scope 1 emissions (tCO2e).
        ebitda: Counterparty EBITDA (for the carbon-cost-to-EBITDA ratio).
            Transition risk is only applied when EBITDA is positive.
        physical_hazard: Dominant physical hazard for the exposure, or
            None if not exposed to a modelled hazard.
    """

    exposure_id: str
    ead: float
    pd: float
    lgd: float
    sector: str = "other"
    scope1_emissions_tco2e: float = 0.0
    ebitda: float = 0.0
    physical_hazard: PhysicalHazard | None = None

    def __post_init__(self) -> None:
        if self.ead < 0.0:
            raise ValueError("ead must be non-negative")
        if not 0.0 <= self.pd <= 1.0:
            raise ValueError("pd must be in [0, 1]")
        if not 0.0 <= self.lgd <= 1.0:
            raise ValueError("lgd must be in [0, 1]")


@dataclass(frozen=True)
class ClimateExposureResult:
    """Per-exposure climate stress result."""

    exposure_id: str
    baseline_ecl: float
    stressed_ecl: float
    stressed_pd: float
    stressed_lgd: float
    transition_pd_multiplier: float
    physical_pd_multiplier: float


@dataclass(frozen=True)
class ClimateScenarioResult:
    """Portfolio-level climate scenario analysis result.

    Attributes:
        scenario: Scenario name.
        horizon_year: Projection horizon.
        baseline_ecl: Sum of baseline ECL across the book.
        stressed_ecl: Sum of climate-stressed ECL.
        ecl_uplift: stressed_ecl - baseline_ecl.
        ecl_uplift_pct: Uplift as a percentage of baseline ECL.
        transition_ecl_uplift: Uplift attributable to transition risk only.
        physical_ecl_uplift: Uplift attributable to physical risk only.
        n_exposures: Number of exposures.
        by_exposure: Per-exposure results.
    """

    scenario: str
    horizon_year: int
    baseline_ecl: float
    stressed_ecl: float
    ecl_uplift: float
    ecl_uplift_pct: float
    transition_ecl_uplift: float
    physical_ecl_uplift: float
    n_exposures: int
    by_exposure: tuple[ClimateExposureResult, ...]


def scenario_carbon_price(scenario: NGFSScenario, horizon_year: int) -> float:
    """Carbon price ($/tCO2e) for a scenario at a horizon year.

    Linearly interpolates between the scenario's 2030 and 2050 carbon
    prices, holding flat outside the [2030, 2050] window.

    Args:
        scenario: NGFS scenario.
        horizon_year: Projection year.

    Returns:
        Carbon price in USD per tonne of CO2-equivalent.
    """
    cp_2030 = scenario.carbon_price_2030_usd
    cp_2050 = scenario.carbon_price_2050_usd
    if horizon_year <= _CP_BASE_YEAR:
        return cp_2030
    if horizon_year >= _CP_END_YEAR:
        return cp_2050
    frac = (horizon_year - _CP_BASE_YEAR) / (_CP_END_YEAR - _CP_BASE_YEAR)
    return cp_2030 + frac * (cp_2050 - cp_2030)


def project_climate_ecl(
    exposures: Sequence[ClimateExposure],
    scenario: NGFSScenario,
    horizon_year: int = 2030,
) -> ClimateScenarioResult:
    """Project climate-adjusted ECL for a portfolio under an NGFS scenario.

    Args:
        exposures: Portfolio exposures.
        scenario: NGFS climate scenario.
        horizon_year: Projection horizon (default 2030).

    Returns:
        A :class:`ClimateScenarioResult` with the aggregate uplift and a
        transition/physical decomposition.

    Raises:
        ValueError: If ``exposures`` is empty.
    """
    if not exposures:
        raise ValueError("at least one exposure is required")

    carbon_price = scenario_carbon_price(scenario, horizon_year)
    severity = scenario.physical_risk_severity

    baseline_total = 0.0
    stressed_total = 0.0
    transition_only_total = 0.0
    physical_only_total = 0.0
    results: list[ClimateExposureResult] = []

    for exp in exposures:
        baseline_ecl = exp.ead * exp.pd * exp.lgd

        # Transition channel (PD only).
        if exp.ebitda > 0.0:
            trans_mult = transition_risk_pd_multiplier(
                exp.scope1_emissions_tco2e, carbon_price, exp.ebitda, exp.sector
            )
        else:
            trans_mult = 1.0

        # Physical channel (PD multiplier + additive LGD haircut).
        if exp.physical_hazard is not None:
            phys_mult = physical_risk_pd_multiplier(exp.physical_hazard, severity)
            lgd_haircut = physical_risk_lgd_haircut(exp.physical_hazard, severity)
        else:
            phys_mult = 1.0
            lgd_haircut = 0.0

        stressed_pd = min(exp.pd * trans_mult * phys_mult, 1.0)
        stressed_lgd = min(exp.lgd + lgd_haircut, 1.0)
        stressed_ecl = exp.ead * stressed_pd * stressed_lgd

        # Single-channel ECLs for the additive decomposition.
        trans_pd = min(exp.pd * trans_mult, 1.0)
        transition_only_ecl = exp.ead * trans_pd * exp.lgd
        phys_pd = min(exp.pd * phys_mult, 1.0)
        physical_only_ecl = exp.ead * phys_pd * min(exp.lgd + lgd_haircut, 1.0)

        baseline_total += baseline_ecl
        stressed_total += stressed_ecl
        transition_only_total += transition_only_ecl - baseline_ecl
        physical_only_total += physical_only_ecl - baseline_ecl

        results.append(
            ClimateExposureResult(
                exposure_id=exp.exposure_id,
                baseline_ecl=round(baseline_ecl, 6),
                stressed_ecl=round(stressed_ecl, 6),
                stressed_pd=round(stressed_pd, 8),
                stressed_lgd=round(stressed_lgd, 8),
                transition_pd_multiplier=round(trans_mult, 6),
                physical_pd_multiplier=round(phys_mult, 6),
            )
        )

    uplift = stressed_total - baseline_total
    uplift_pct = (uplift / baseline_total * 100.0) if baseline_total > 0.0 else 0.0

    return ClimateScenarioResult(
        scenario=scenario.name,
        horizon_year=horizon_year,
        baseline_ecl=round(baseline_total, 6),
        stressed_ecl=round(stressed_total, 6),
        ecl_uplift=round(uplift, 6),
        ecl_uplift_pct=round(uplift_pct, 4),
        transition_ecl_uplift=round(transition_only_total, 6),
        physical_ecl_uplift=round(physical_only_total, 6),
        n_exposures=len(exposures),
        by_exposure=tuple(results),
    )
