"""
Macro stress testing framework.

Supports EBA, BoE ACS, US CCAR/DFAST, and RBI methodologies.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class MacroScenario:
    """A macroeconomic stress scenario."""

    def __init__(
        self,
        name: str,
        horizon_years: int = 3,
        variables: dict[str, np.ndarray] | None = None,
        severity: str = "adverse",
    ) -> None:
        self.name = name
        self.horizon_years = horizon_years
        self.variables = variables or {}
        self.severity = severity


def apply_pd_stress(
    base_pds: np.ndarray,
    stress_multiplier: float,
    pd_cap: float = 1.0,
) -> np.ndarray:
    """Apply stress multiplier to PDs.

    Args:
        base_pds: Baseline PD array.
        stress_multiplier: Multiplicative stress factor.
        pd_cap: Maximum PD cap.

    Returns:
        Stressed PD array.
    """
    stressed = base_pds * stress_multiplier
    return np.minimum(stressed, pd_cap)


def apply_lgd_stress(
    base_lgds: np.ndarray,
    stress_add_on: float,
    lgd_cap: float = 1.0,
) -> np.ndarray:
    """Apply additive stress to LGDs.

    Args:
        base_lgds: Baseline LGD array.
        stress_add_on: Additive stress increase.
        lgd_cap: Maximum LGD cap.

    Returns:
        Stressed LGD array.
    """
    stressed = base_lgds + stress_add_on
    return np.clip(stressed, 0.0, lgd_cap)


def stress_test_rwa_impact(
    base_rwa: float,
    stressed_rwa: float,
) -> dict[str, float]:
    """Calculate RWA impact from stress test.

    Args:
        base_rwa: Baseline RWA.
        stressed_rwa: Stressed RWA.

    Returns:
        Dict with absolute and relative impact.
    """
    delta = stressed_rwa - base_rwa
    pct = delta / base_rwa if base_rwa > 0 else 0.0

    return {
        "base_rwa": base_rwa,
        "stressed_rwa": stressed_rwa,
        "delta_rwa": delta,
        "pct_change": pct,
    }
