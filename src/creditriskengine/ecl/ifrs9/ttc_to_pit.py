"""
TTC-to-PIT PD conversion using the Z-factor (Merton/Vasicek) method.

Reference: Merton (1974), Vasicek (2002).
This follows the single-factor model framework underlying Basel III IRB.

Formula:
    PD_PIT(t) = Phi( (Phi^-1(PD_TTC) - rho^0.5 * Z(t)) / (1-rho)^0.5 )

Where:
    Phi = standard normal CDF
    Phi^-1 = standard normal inverse CDF (quantile)
    rho = systematic factor loading (asset correlation proxy)
    Z(t) = macroeconomic index value at time t
"""

import logging
import math

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)


def ttc_to_pit_pd(
    pd_ttc: float,
    z_factor: float,
    rho: float,
) -> float:
    """Convert TTC PD to PIT PD using the Z-factor method.

    Formula:
        PD_PIT = Phi( (Phi^-1(PD_TTC) - sqrt(rho) * Z) / sqrt(1 - rho) )

    Args:
        pd_ttc: Through-the-cycle PD.
        z_factor: Macroeconomic index value (standard normal).
        rho: Asset correlation / systematic factor loading.

    Returns:
        Point-in-time PD.
    """
    if rho <= 0.0 or rho >= 1.0:
        raise ValueError(f"rho must be in (0, 1), got {rho}")
    if pd_ttc <= 0.0:
        return 0.0
    if pd_ttc >= 1.0:
        return 1.0

    g_ttc = norm.ppf(pd_ttc)
    pit_input = (g_ttc - math.sqrt(rho) * z_factor) / math.sqrt(1.0 - rho)
    return float(norm.cdf(pit_input))


def ttc_to_pit_pd_curve(
    pd_ttc: float,
    z_factors: list[float],
    rho: float,
) -> np.ndarray:
    """Convert TTC PD to a PIT PD curve over multiple periods.

    Args:
        pd_ttc: Through-the-cycle PD.
        z_factors: List of Z-factor values for each period.
        rho: Asset correlation.

    Returns:
        Array of PIT PDs for each period.
    """
    return np.array([ttc_to_pit_pd(pd_ttc, z, rho) for z in z_factors])


def estimate_z_factor(
    observed_default_rate: float,
    long_run_pd: float,
    rho: float,
) -> float:
    """Back-solve for the Z-factor from observed default rate.

    Inverse of the TTC-to-PIT formula:
        Z = (Phi^-1(PD_TTC) - sqrt(1-rho) * Phi^-1(DR_observed)) / sqrt(rho)

    Args:
        observed_default_rate: Observed period default rate.
        long_run_pd: Long-run average (TTC) PD.
        rho: Asset correlation.

    Returns:
        Estimated Z-factor value.
    """
    if rho <= 0.0 or rho >= 1.0:
        raise ValueError(f"rho must be in (0, 1), got {rho}")

    g_ttc = norm.ppf(max(min(long_run_pd, 0.9999), 0.0001))
    g_obs = norm.ppf(max(min(observed_default_rate, 0.9999), 0.0001))

    return (g_ttc - math.sqrt(1.0 - rho) * g_obs) / math.sqrt(rho)
