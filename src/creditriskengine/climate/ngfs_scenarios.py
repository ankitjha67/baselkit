"""NGFS Phase V climate scenario library.

Reference: Network for Greening the Financial System (NGFS),
Phase V Scenarios (Vintage 5.0), November 2024.

Provides the six canonical NGFS scenarios with macro variable paths
(carbon price, GDP impact, temperature) for use in climate stress
testing and IFRS 9 forward-looking overlays.

Categories:
    Orderly:    Net Zero 2050, Below 2°C
    Disorderly: Delayed Transition, Fragmented World
    Hot House:  Current Policies, NDCs (Nationally Determined Contributions)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NGFSScenario:
    """A single NGFS climate scenario with macro paths.

    Attributes:
        name: Scenario name.
        category: ``"orderly"``, ``"disorderly"``, or ``"hot_house"``.
        description: One-line scenario summary.
        carbon_price_2030_usd: Carbon price ($/tCO2e) at 2030.
        carbon_price_2050_usd: Carbon price ($/tCO2e) at 2050.
        gdp_impact_2050_pct: Cumulative GDP impact by 2050 (negative = loss).
        temperature_2100_c: Projected global temperature rise by 2100 (°C).
        physical_risk_severity: Qualitative severity (``"low"``, ``"medium"``,
            ``"high"``, ``"very_high"``).
        transition_risk_severity: Qualitative severity.
    """

    name: str
    category: str
    description: str
    carbon_price_2030_usd: float
    carbon_price_2050_usd: float
    gdp_impact_2050_pct: float
    temperature_2100_c: float
    physical_risk_severity: str
    transition_risk_severity: str


_SCENARIOS: dict[str, NGFSScenario] = {
    "net_zero_2050": NGFSScenario(
        name="Net Zero 2050",
        category="orderly",
        description="Immediate, stringent action to limit warming to 1.5°C",
        carbon_price_2030_usd=140.0,
        carbon_price_2050_usd=725.0,
        gdp_impact_2050_pct=-2.5,
        temperature_2100_c=1.5,
        physical_risk_severity="low",
        transition_risk_severity="high",
    ),
    "below_2c": NGFSScenario(
        name="Below 2°C",
        category="orderly",
        description="Gradual policy action limiting warming to below 2°C",
        carbon_price_2030_usd=70.0,
        carbon_price_2050_usd=400.0,
        gdp_impact_2050_pct=-1.5,
        temperature_2100_c=1.7,
        physical_risk_severity="low",
        transition_risk_severity="medium",
    ),
    "delayed_transition": NGFSScenario(
        name="Delayed Transition",
        category="disorderly",
        description="Late, sudden and disorderly policy action after 2030",
        carbon_price_2030_usd=25.0,
        carbon_price_2050_usd=850.0,
        gdp_impact_2050_pct=-4.0,
        temperature_2100_c=1.8,
        physical_risk_severity="medium",
        transition_risk_severity="very_high",
    ),
    "fragmented_world": NGFSScenario(
        name="Fragmented World",
        category="disorderly",
        description="Divergent climate policies across regions",
        carbon_price_2030_usd=15.0,
        carbon_price_2050_usd=250.0,
        gdp_impact_2050_pct=-5.0,
        temperature_2100_c=2.5,
        physical_risk_severity="high",
        transition_risk_severity="medium",
    ),
    "current_policies": NGFSScenario(
        name="Current Policies",
        category="hot_house",
        description="No new climate policies beyond those already implemented",
        carbon_price_2030_usd=10.0,
        carbon_price_2050_usd=15.0,
        gdp_impact_2050_pct=-8.0,
        temperature_2100_c=3.0,
        physical_risk_severity="very_high",
        transition_risk_severity="low",
    ),
    "ndcs": NGFSScenario(
        name="Nationally Determined Contributions",
        category="hot_house",
        description="Only currently pledged NDC targets are met",
        carbon_price_2030_usd=20.0,
        carbon_price_2050_usd=50.0,
        gdp_impact_2050_pct=-6.0,
        temperature_2100_c=2.5,
        physical_risk_severity="high",
        transition_risk_severity="low",
    ),
}


def get_ngfs_scenario(name: str) -> NGFSScenario:
    """Return a single NGFS Phase V scenario by key.

    Args:
        name: Scenario key (e.g., ``"net_zero_2050"``).

    Returns:
        :class:`NGFSScenario`.

    Raises:
        KeyError: If the scenario name is not found.
    """
    if name not in _SCENARIOS:
        raise KeyError(
            f"Unknown NGFS scenario '{name}'. "
            f"Available: {sorted(_SCENARIOS.keys())}"
        )
    return _SCENARIOS[name]


def list_ngfs_scenarios() -> list[NGFSScenario]:
    """Return all six NGFS Phase V scenarios."""
    return list(_SCENARIOS.values())
