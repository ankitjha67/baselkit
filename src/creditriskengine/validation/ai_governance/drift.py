"""Data and model drift detection.

References:
    - SR 11-7 (April 2011) — ongoing monitoring requirements.
    - PRA SS1/23 Principle 3 — model performance monitoring.
    - EBA Follow-up Report on ML for IRB (August 2023).

Standard thresholds for PSI (Population Stability Index):
    < 0.10  : Stable — no action needed.
    0.10-0.25: Minor drift — investigate.
    > 0.25  : Material drift — retrain / recalibrate.
"""

from __future__ import annotations

import logging
from enum import StrEnum

import numpy as np

logger = logging.getLogger(__name__)


class DriftSeverity(StrEnum):
    """PSI-based drift severity classification."""

    STABLE = "stable"
    """PSI < 0.10 — no material drift."""

    MINOR = "minor"
    """0.10 <= PSI < 0.25 — investigate."""

    MATERIAL = "material"
    """PSI >= 0.25 — retrain / recalibrate."""


PSI_MINOR_THRESHOLD: float = 0.10
PSI_MATERIAL_THRESHOLD: float = 0.25


def psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Calculate the Population Stability Index (PSI).

    PSI = Sum_i [ (actual_i - expected_i) × ln(actual_i / expected_i) ]

    where actual_i and expected_i are the proportions of observations
    in bin i.

    Args:
        expected: Reference (development) sample values.
        actual: Current (monitoring) sample values.
        n_bins: Number of equal-frequency bins.
        epsilon: Small constant to avoid log(0).

    Returns:
        PSI value (non-negative; higher = more drift).

    Reference:
        SR 11-7 ongoing monitoring, PRA SS1/23 Principle 3.
    """
    expected = np.asarray(expected, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)

    breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    expected_pct = expected_counts / len(expected) + epsilon
    actual_pct = actual_counts / len(actual) + epsilon

    psi_value = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
    return psi_value


def detect_psi_drift(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, DriftSeverity]:
    """Calculate PSI and classify drift severity.

    Args:
        expected: Reference sample values.
        actual: Current sample values.
        n_bins: Number of bins for PSI calculation.

    Returns:
        Tuple of (psi_value, severity).
    """
    psi_value = psi(expected, actual, n_bins=n_bins)

    if psi_value >= PSI_MATERIAL_THRESHOLD:
        severity = DriftSeverity.MATERIAL
    elif psi_value >= PSI_MINOR_THRESHOLD:
        severity = DriftSeverity.MINOR
    else:
        severity = DriftSeverity.STABLE

    return psi_value, severity
