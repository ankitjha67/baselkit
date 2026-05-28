"""Transition climate risk — carbon-price-driven PD adjustment.

Reference:
    - NGFS Phase V Scenarios (November 2024).
    - ECB Fit-for-55 Climate Scenario Analysis (November 2024).
    - BCBS d517 Climate-related risk drivers (2021).

Transition risks arise from policy, technology, and market changes
during the shift to a low-carbon economy. The primary channel for
credit risk is carbon-cost pass-through to corporate EBITDA, which
compresses debt-service coverage ratios and increases PD.

Methodology:
    PD_adj = PD_base × (1 + elasticity × carbon_cost_impact)

where:
    carbon_cost_impact = (Scope1_emissions × carbon_price) / EBITDA
    elasticity = sector-specific PD-to-carbon-cost sensitivity
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


SECTOR_ELASTICITY: dict[str, float] = {
    "oil_gas": 2.5,
    "coal_mining": 3.0,
    "steel": 2.0,
    "cement": 2.2,
    "aluminium": 1.8,
    "chemicals": 1.5,
    "power_generation_fossil": 2.5,
    "power_generation_renewable": 0.2,
    "aviation": 1.8,
    "shipping": 1.5,
    "road_transport": 1.2,
    "agriculture": 0.8,
    "real_estate": 0.5,
    "technology": 0.1,
    "financial_services": 0.2,
    "other": 0.5,
}
"""Sector-specific PD elasticity to carbon cost impact.

Calibrated to ECB Fit-for-55 findings (November 2024) showing
~50% average PD uplift for high energy-intensity sectors under
stressed paths, ~91% for the most exposed.
"""

CBAM_SECTORS: frozenset[str] = frozenset({
    "steel",
    "cement",
    "aluminium",
    "fertilisers",
    "electricity",
    "hydrogen",
    "ammonia",
})
"""EU Carbon Border Adjustment Mechanism (CBAM) sectors.

Reference: Regulation (EU) 2023/956. Definitive phase from
1 January 2026.
"""


def transition_risk_pd_multiplier(
    scope1_emissions_tco2e: float,
    carbon_price_usd: float,
    ebitda: float,
    sector: str = "other",
) -> float:
    """Calculate the PD multiplier from transition risk.

    Formula:
        carbon_cost_impact = (Scope1 × carbon_price) / EBITDA
        PD_multiplier = 1 + elasticity × carbon_cost_impact

    The multiplier is floored at 1.0 (no benefit from high carbon
    prices) and capped at 5.0 (to prevent runaway PD inflation in
    edge cases).

    Args:
        scope1_emissions_tco2e: Annual Scope 1 GHG emissions (tCO2e).
        carbon_price_usd: Carbon price per tonne CO2e (e.g., from
            NGFS scenario).
        ebitda: Annual EBITDA of the counterparty.
        sector: Sector key from ``SECTOR_ELASTICITY``.

    Returns:
        PD multiplier (1.0 = no impact; >1.0 = PD increase).

    Reference:
        BCBS d517, ECB Fit-for-55 (November 2024).
    """
    if ebitda <= 0 or scope1_emissions_tco2e <= 0 or carbon_price_usd <= 0:
        return 1.0

    carbon_cost = scope1_emissions_tco2e * carbon_price_usd
    carbon_cost_impact = carbon_cost / ebitda

    elasticity = SECTOR_ELASTICITY.get(
        sector.lower(), SECTOR_ELASTICITY["other"]
    )
    multiplier = 1.0 + elasticity * carbon_cost_impact
    return min(max(multiplier, 1.0), 5.0)


def is_cbam_sector(sector: str) -> bool:
    """Check if a sector is subject to the EU CBAM.

    Reference: Regulation (EU) 2023/956.

    Args:
        sector: Sector key.

    Returns:
        ``True`` if the sector is in the CBAM scope.
    """
    return sector.lower() in CBAM_SECTORS
