"""
EU DORA ICT incident classification.

Reference:
    - Regulation (EU) 2022/2554 (Digital Operational Resilience Act),
      in force January 17, 2025.
    - Commission Delegated Regulation (EU) 2024/1772 (RTS on
      classification of major ICT-related incidents).

DORA requires financial entities to classify ICT-related incidents and
report "major" incidents to competent authorities (initial notification
within tight deadlines). Major-incident classification is driven by
materiality thresholds across several criteria.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Materiality thresholds for major-incident classification
# (Commission Delegated Regulation (EU) 2024/1772).
CLIENTS_AFFECTED_THRESHOLD_PCT: float = 0.10
"""Major if > 10% of clients affected."""

CLIENTS_AFFECTED_ABSOLUTE: int = 100_000
"""Major if > 100,000 clients affected (absolute)."""

DOWNTIME_THRESHOLD_HOURS: float = 2.0
"""Major if critical-service downtime exceeds 2 hours."""

ECONOMIC_IMPACT_THRESHOLD_EUR: float = 100_000.0
"""Major if direct + indirect costs exceed EUR 100,000."""


@dataclass(frozen=True)
class DORAIncidentClassification:
    """Result of DORA ICT incident classification.

    Attributes:
        is_major: Whether the incident is classified as major.
        triggered_criteria: List of criteria that were breached.
        clients_affected_pct: Fraction of clients affected.
        downtime_hours: Critical-service downtime in hours.
        economic_impact_eur: Total economic impact.
    """

    is_major: bool
    triggered_criteria: list[str]
    clients_affected_pct: float
    downtime_hours: float
    economic_impact_eur: float


def classify_ict_incident(
    clients_affected: int,
    total_clients: int,
    downtime_hours: float,
    economic_impact_eur: float,
    data_losses: bool = False,
    critical_services_affected: bool = False,
) -> DORAIncidentClassification:
    """Classify an ICT incident per DORA RTS (EU) 2024/1772.

    An incident is "major" if it breaches any of the materiality
    criteria: clients affected (relative or absolute), service downtime,
    economic impact, data losses, or critical-service impact.

    Args:
        clients_affected: Number of clients affected.
        total_clients: Total client base.
        downtime_hours: Critical-service downtime (hours).
        economic_impact_eur: Direct + indirect cost (EUR).
        data_losses: Whether data integrity/confidentiality was lost.
        critical_services_affected: Whether critical/important functions
            were affected.

    Returns:
        :class:`DORAIncidentClassification`.

    Reference:
        Commission Delegated Regulation (EU) 2024/1772.
    """
    triggered: list[str] = []
    clients_pct = clients_affected / total_clients if total_clients > 0 else 0.0

    if clients_pct > CLIENTS_AFFECTED_THRESHOLD_PCT:
        triggered.append("clients_affected_pct")
    if clients_affected > CLIENTS_AFFECTED_ABSOLUTE:
        triggered.append("clients_affected_absolute")
    if downtime_hours > DOWNTIME_THRESHOLD_HOURS:
        triggered.append("downtime")
    if economic_impact_eur > ECONOMIC_IMPACT_THRESHOLD_EUR:
        triggered.append("economic_impact")
    if data_losses:
        triggered.append("data_losses")
    if critical_services_affected:
        triggered.append("critical_services")

    return DORAIncidentClassification(
        is_major=len(triggered) > 0,
        triggered_criteria=triggered,
        clients_affected_pct=clients_pct,
        downtime_hours=downtime_hours,
        economic_impact_eur=economic_impact_eur,
    )


def is_major_incident(classification: DORAIncidentClassification) -> bool:
    """Return whether a classification is a major incident.

    Args:
        classification: A :class:`DORAIncidentClassification`.

    Returns:
        ``True`` if the incident is major (reportable to authorities).
    """
    return classification.is_major
