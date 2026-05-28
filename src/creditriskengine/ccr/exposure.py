"""
Counterparty credit exposure profiles — EPE / EEPE / PFE.

Reference:
    - BCBS CRE53 (Internal Model Method for counterparty credit risk).
    - Basel III CCR framework, Annex 4.
    - Gregory (2015) — The xVA Challenge, Ch. 8-11.

Definitions (over a set of simulated exposure paths E(t)):
    Expected Exposure          EE(t)   = E[max(V(t), 0)]
    Expected Positive Exposure EPE     = time-average of EE(t)
    Effective EE               EEE(t)  = max(EEE(t-1), EE(t))  (non-decreasing)
    Effective EPE              EEPE    = time-average of EEE(t)
    Potential Future Exposure  PFE(t)  = high quantile of E(t) (e.g., 95th/99th)

Regulatory EAD under IMM = alpha * EEPE, with alpha = 1.4.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


IMM_ALPHA: float = 1.4
"""Regulatory alpha multiplier for IMM EAD = alpha * EEPE.

Reference: BCBS CRE53.1, CRE52.52.
"""


def expected_exposure(exposure_paths: np.ndarray) -> np.ndarray:
    """Expected Exposure profile EE(t) = mean of positive exposures.

    Args:
        exposure_paths: (n_paths, n_timesteps) simulated portfolio
            values at each future time step.

    Returns:
        EE(t) per time step (n_timesteps,).
    """
    exposure_paths = np.asarray(exposure_paths, dtype=np.float64)
    positive = np.maximum(exposure_paths, 0.0)
    return np.mean(positive, axis=0)


def expected_positive_exposure(
    exposure_paths: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """Expected Positive Exposure (EPE) — time-average of EE(t).

    Args:
        exposure_paths: (n_paths, n_timesteps) simulated exposures.
        weights: Optional per-timestep time weights (e.g., year
            fractions). If None, a simple average is used.

    Returns:
        EPE (scalar).
    """
    ee = expected_exposure(exposure_paths)
    if weights is None:
        return float(np.mean(ee))
    weights = np.asarray(weights, dtype=np.float64)
    return float(np.sum(ee * weights) / np.sum(weights))


def effective_expected_exposure(exposure_paths: np.ndarray) -> np.ndarray:
    """Effective Expected Exposure EEE(t) — non-decreasing running max of EE.

    EEE(t) = max(EEE(t-1), EE(t))

    Args:
        exposure_paths: (n_paths, n_timesteps) simulated exposures.

    Returns:
        EEE(t) per time step (non-decreasing).

    Reference:
        BCBS CRE53.
    """
    ee = expected_exposure(exposure_paths)
    return np.maximum.accumulate(ee)


def effective_epe(
    exposure_paths: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """Effective EPE (EEPE) — time-average of the Effective EE profile.

    Args:
        exposure_paths: (n_paths, n_timesteps) simulated exposures.
        weights: Optional per-timestep time weights.

    Returns:
        EEPE (scalar). Regulatory EAD = IMM_ALPHA * EEPE.

    Reference:
        BCBS CRE53.
    """
    eee = effective_expected_exposure(exposure_paths)
    if weights is None:
        return float(np.mean(eee))
    weights = np.asarray(weights, dtype=np.float64)
    return float(np.sum(eee * weights) / np.sum(weights))


def potential_future_exposure(
    exposure_paths: np.ndarray,
    confidence_level: float = 0.95,
) -> np.ndarray:
    """Potential Future Exposure PFE(t) — high quantile of exposure.

    Args:
        exposure_paths: (n_paths, n_timesteps) simulated exposures.
        confidence_level: Quantile (e.g., 0.95 = 95th percentile).

    Returns:
        PFE(t) per time step.

    Raises:
        ValueError: If confidence_level not in (0, 1).

    Reference:
        Basel III CCR framework, Gregory (2015).
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    exposure_paths = np.asarray(exposure_paths, dtype=np.float64)
    positive = np.maximum(exposure_paths, 0.0)
    return np.quantile(positive, confidence_level, axis=0)


@dataclass(frozen=True)
class ExposureProfile:
    """Summary of a counterparty exposure profile.

    Attributes:
        epe: Expected Positive Exposure.
        eepe: Effective EPE.
        peak_pfe: Peak Potential Future Exposure over the profile.
        regulatory_ead: IMM EAD = alpha * EEPE.
    """

    epe: float
    eepe: float
    peak_pfe: float
    regulatory_ead: float


def summarise_exposure(
    exposure_paths: np.ndarray,
    confidence_level: float = 0.95,
    weights: np.ndarray | None = None,
) -> ExposureProfile:
    """Compute the full set of CCR exposure metrics from paths.

    Args:
        exposure_paths: (n_paths, n_timesteps) simulated exposures.
        confidence_level: PFE quantile.
        weights: Optional per-timestep time weights.

    Returns:
        :class:`ExposureProfile`.
    """
    epe = expected_positive_exposure(exposure_paths, weights)
    eepe = effective_epe(exposure_paths, weights)
    pfe = potential_future_exposure(exposure_paths, confidence_level)
    return ExposureProfile(
        epe=epe,
        eepe=eepe,
        peak_pfe=float(np.max(pfe)),
        regulatory_ead=IMM_ALPHA * eepe,
    )


def netting_set_exposure(
    trade_values: np.ndarray,
    collateral: float = 0.0,
) -> np.ndarray:
    """Net the value of multiple trades within a netting set.

    Under a qualifying master netting agreement, the netting-set
    exposure is max(sum of trade values - collateral, 0).

    Args:
        trade_values: (n_paths, n_trades) or (n_trades,) trade values.
            If 2-D, netting is applied per path/timestep row.
        collateral: Collateral held against the netting set.

    Returns:
        Netted positive exposure (same leading dimension as input).

    Reference:
        BCBS CRE53 (netting sets), ISDA Master Agreement.
    """
    trade_values = np.asarray(trade_values, dtype=np.float64)
    netted = np.sum(trade_values, axis=-1) - collateral
    return np.maximum(netted, 0.0)
