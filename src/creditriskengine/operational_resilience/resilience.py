"""
Operational resilience — Important Business Services and impact tolerances.

Reference:
    - BCBS Principles for Operational Resilience (BCBS d516, March 2021).
    - PRA SS1/21 / PS6/21 Operational Resilience.
    - DORA (Reg 2022/2554) third-party concentration.

Banks must identify Important Business Services (IBS), set impact
tolerances (the maximum tolerable disruption), and manage third-party
concentration risk for critical ICT providers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImpactTolerance:
    """Impact tolerance for an Important Business Service.

    Attributes:
        service_name: Name of the important business service.
        max_tolerable_downtime_hours: Maximum tolerable disruption.
        max_tolerable_clients_affected: Maximum tolerable client impact.
    """

    service_name: str
    max_tolerable_downtime_hours: float
    max_tolerable_clients_affected: int


def within_impact_tolerance(
    tolerance: ImpactTolerance,
    actual_downtime_hours: float,
    actual_clients_affected: int,
) -> bool:
    """Check whether a disruption stays within the impact tolerance.

    Args:
        tolerance: The service's impact tolerance.
        actual_downtime_hours: Actual disruption duration.
        actual_clients_affected: Actual clients affected.

    Returns:
        ``True`` if BOTH downtime and client impact are within tolerance.

    Reference:
        BCBS d516, PRA SS1/21.
    """
    return (
        actual_downtime_hours <= tolerance.max_tolerable_downtime_hours
        and actual_clients_affected <= tolerance.max_tolerable_clients_affected
    )


def third_party_concentration(
    provider_exposures: dict[str, float],
) -> dict[str, float | str]:
    """Measure third-party (ICT provider) concentration via HHI.

    A high Herfindahl-Hirschman Index indicates over-reliance on a few
    critical ICT third-party providers — a key DORA/BCBS concern.

    Args:
        provider_exposures: Mapping of provider name → exposure measure
            (e.g., share of critical services, spend, or criticality
            weight).

    Returns:
        Dict with ``hhi`` (0-1), ``largest_provider``, ``largest_share``,
        and ``concentration_level`` (``"low"``/``"moderate"``/``"high"``).

    Reference:
        BCBS d516, DORA Art. 28-30 (concentration risk).
    """
    if not provider_exposures:
        return {
            "hhi": 0.0,
            "largest_provider": "",
            "largest_share": 0.0,
            "concentration_level": "low",
        }

    total = sum(provider_exposures.values())
    if total <= 0:
        return {
            "hhi": 0.0,
            "largest_provider": "",
            "largest_share": 0.0,
            "concentration_level": "low",
        }

    shares = {name: exp / total for name, exp in provider_exposures.items()}
    hhi = float(np.sum(np.array(list(shares.values())) ** 2))
    largest_provider = max(shares, key=lambda k: shares[k])
    largest_share = shares[largest_provider]

    if hhi >= 0.25:
        level = "high"
    elif hhi >= 0.15:
        level = "moderate"
    else:
        level = "low"

    return {
        "hhi": round(hhi, 4),
        "largest_provider": largest_provider,
        "largest_share": round(largest_share, 4),
        "concentration_level": level,
    }
