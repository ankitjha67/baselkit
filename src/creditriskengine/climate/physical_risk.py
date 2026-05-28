"""Physical climate risk adjustments to PD and LGD.

Reference:
    - BCBS Principles for Management of Climate-Related Financial
      Risks (June 2022), Principle 4.
    - ECB Guide on Climate-Related and Environmental Risks (Nov 2020).
    - PRA SS5/25 (December 2025) — climate into ECL / IFRS 9.

Physical risks (floods, wildfires, drought, sea-level rise, storms)
affect credit risk through:
    1. Collateral devaluation (LGD channel) — property damage, insurance
       gaps, uninsurability.
    2. Business interruption (PD channel) — revenue loss, supply chain
       disruption, uninsured losses.
"""

from __future__ import annotations

import logging
from enum import StrEnum

logger = logging.getLogger(__name__)


class PhysicalHazard(StrEnum):
    """Acute and chronic physical climate hazards."""

    FLOOD = "flood"
    WILDFIRE = "wildfire"
    DROUGHT = "drought"
    SEA_LEVEL_RISE = "sea_level_rise"
    STORM = "storm"
    EXTREME_HEAT = "extreme_heat"


_LGD_HAIRCUT_MATRIX: dict[PhysicalHazard, dict[str, float]] = {
    PhysicalHazard.FLOOD: {
        "low": 0.00, "medium": 0.05, "high": 0.15, "very_high": 0.30,
    },
    PhysicalHazard.WILDFIRE: {
        "low": 0.00, "medium": 0.05, "high": 0.20, "very_high": 0.40,
    },
    PhysicalHazard.DROUGHT: {
        "low": 0.00, "medium": 0.02, "high": 0.08, "very_high": 0.15,
    },
    PhysicalHazard.SEA_LEVEL_RISE: {
        "low": 0.00, "medium": 0.05, "high": 0.20, "very_high": 0.50,
    },
    PhysicalHazard.STORM: {
        "low": 0.00, "medium": 0.05, "high": 0.15, "very_high": 0.35,
    },
    PhysicalHazard.EXTREME_HEAT: {
        "low": 0.00, "medium": 0.01, "high": 0.05, "very_high": 0.10,
    },
}

_PD_MULTIPLIER_MATRIX: dict[PhysicalHazard, dict[str, float]] = {
    PhysicalHazard.FLOOD: {
        "low": 1.00, "medium": 1.05, "high": 1.15, "very_high": 1.30,
    },
    PhysicalHazard.WILDFIRE: {
        "low": 1.00, "medium": 1.05, "high": 1.15, "very_high": 1.35,
    },
    PhysicalHazard.DROUGHT: {
        "low": 1.00, "medium": 1.03, "high": 1.10, "very_high": 1.20,
    },
    PhysicalHazard.SEA_LEVEL_RISE: {
        "low": 1.00, "medium": 1.05, "high": 1.15, "very_high": 1.40,
    },
    PhysicalHazard.STORM: {
        "low": 1.00, "medium": 1.05, "high": 1.15, "very_high": 1.30,
    },
    PhysicalHazard.EXTREME_HEAT: {
        "low": 1.00, "medium": 1.02, "high": 1.08, "very_high": 1.15,
    },
}


def physical_risk_lgd_haircut(
    hazard: PhysicalHazard,
    severity: str,
) -> float:
    """Return the LGD haircut for a given physical hazard and severity.

    The haircut is an additive increase to the base LGD reflecting
    collateral devaluation from physical climate damage.

    Args:
        hazard: Type of physical hazard.
        severity: ``"low"``, ``"medium"``, ``"high"``, or ``"very_high"``.

    Returns:
        LGD add-on as a decimal (e.g., 0.15 = +15pp LGD increase).

    Raises:
        ValueError: If severity is not recognized.
    """
    haircuts = _LGD_HAIRCUT_MATRIX.get(hazard)
    if haircuts is None or severity not in haircuts:
        raise ValueError(
            f"Unknown hazard/severity: {hazard}/{severity}. "
            f"Severities: low, medium, high, very_high"
        )
    return haircuts[severity]


def physical_risk_pd_multiplier(
    hazard: PhysicalHazard,
    severity: str,
) -> float:
    """Return the PD multiplier for a given physical hazard and severity.

    The multiplier captures business-interruption and revenue-loss
    channels from physical climate events.

    Args:
        hazard: Type of physical hazard.
        severity: ``"low"``, ``"medium"``, ``"high"``, or ``"very_high"``.

    Returns:
        PD multiplier (1.0 = no impact; 1.30 = +30% PD increase).

    Raises:
        ValueError: If severity is not recognized.
    """
    multipliers = _PD_MULTIPLIER_MATRIX.get(hazard)
    if multipliers is None or severity not in multipliers:
        raise ValueError(
            f"Unknown hazard/severity: {hazard}/{severity}. "
            f"Severities: low, medium, high, very_high"
        )
    return multipliers[severity]
