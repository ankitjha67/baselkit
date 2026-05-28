"""PCAF financed emissions calculation.

Reference: Partnership for Carbon Accounting Financials (PCAF),
Global GHG Accounting and Reporting Standard for the Financial
Industry, 2nd edition (2022).

Financed emissions = Sum_i [ attribution_factor_i × emissions_i ]

where:
    attribution_factor = outstanding_amount / EVIC (or total assets)
    emissions = Scope 1, Scope 2, or Scope 3 of the counterparty

Data quality is scored on a 1-5 PCAF scale:
    1 = Verified counterparty-reported emissions
    2 = Unverified counterparty-reported
    3 = Estimated from physical activity data
    4 = Estimated from production data
    5 = Estimated from sector averages (PCAF DB, EXIOBASE)
"""

from __future__ import annotations

import logging
from enum import IntEnum

logger = logging.getLogger(__name__)


class PCAFScore(IntEnum):
    """PCAF data quality score (1 = best, 5 = worst)."""

    VERIFIED = 1
    REPORTED = 2
    PHYSICAL_ACTIVITY = 3
    PRODUCTION = 4
    SECTOR_AVERAGE = 5


def financed_emissions(
    outstanding: float,
    evic_or_total_assets: float,
    counterparty_emissions_tco2e: float,
) -> float:
    """Calculate financed emissions for a single exposure.

    Formula (PCAF):
        FE = (outstanding / EVIC) × counterparty_emissions

    EVIC = Enterprise Value Including Cash. For sovereigns and
    project finance, total assets or total revenue may be used
    as a proxy per PCAF asset-class methodology.

    Args:
        outstanding: Outstanding amount to the counterparty.
        evic_or_total_assets: Enterprise Value Including Cash (or
            total assets for sovereigns/project finance).
        counterparty_emissions_tco2e: Annual GHG emissions of the
            counterparty (tCO2e — Scope 1, 2, or 3 as applicable).

    Returns:
        Financed emissions in tCO2e attributable to this exposure.

    Reference:
        PCAF Standard, 2nd edition (2022), Section 4.
    """
    if evic_or_total_assets <= 0 or outstanding <= 0:
        return 0.0
    attribution_factor = outstanding / evic_or_total_assets
    return attribution_factor * counterparty_emissions_tco2e


def weighted_data_quality_score(
    emissions_and_scores: list[tuple[float, PCAFScore]],
) -> float:
    """Calculate the weighted-average PCAF data quality score.

    Weights each score by the magnitude of financed emissions.

    Args:
        emissions_and_scores: List of ``(financed_emissions, pcaf_score)``
            tuples.

    Returns:
        Weighted average score (1.0 = best, 5.0 = worst). Returns
        0.0 if no emissions.

    Reference:
        PCAF Standard, 2nd edition (2022), Section 6.
    """
    total_emissions = sum(abs(fe) for fe, _ in emissions_and_scores)
    if total_emissions == 0:
        return 0.0
    weighted = sum(abs(fe) * int(score) for fe, score in emissions_and_scores)
    return weighted / total_emissions
