"""
Recovery curve modeling for LGD.

Reference:
    - BBVA / HSBC / Barclays workout LGD methodology disclosures.
    - EBA/GL/2017/16 — LGD estimation, recovery cash flow discounting.
    - Araten, Jacobs & Varshney (2004) — measuring LGD on defaulted loans.

Fits parametric distributions (Weibull, lognormal, gamma) to observed
time-to-recovery data, enabling estimation of the cumulative recovery
profile and discounted workout LGD with stochastic recovery timing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class RecoveryCurveType(StrEnum):
    """Parametric family for the recovery-time distribution."""

    WEIBULL = "weibull"
    LOGNORMAL = "lognormal"
    GAMMA = "gamma"


@dataclass(frozen=True)
class RecoveryCurveFit:
    """Fitted recovery curve parameters.

    Attributes:
        curve_type: Distribution family used.
        shape: Shape parameter.
        scale: Scale parameter.
        mean_recovery_time: Mean time to recovery (years).
    """

    curve_type: RecoveryCurveType
    shape: float
    scale: float
    mean_recovery_time: float


def fit_recovery_curve(
    recovery_times: np.ndarray,
    curve_type: RecoveryCurveType = RecoveryCurveType.WEIBULL,
) -> RecoveryCurveFit:
    """Fit a parametric recovery-time distribution.

    Args:
        recovery_times: Observed times-to-recovery (years), all > 0.
        curve_type: Distribution family to fit.

    Returns:
        :class:`RecoveryCurveFit`.

    Raises:
        ValueError: If recovery_times is empty or non-positive.

    Reference:
        EBA/GL/2017/16, Araten et al. (2004).
    """
    recovery_times = np.asarray(recovery_times, dtype=np.float64)
    if len(recovery_times) == 0 or np.any(recovery_times <= 0):
        raise ValueError("recovery_times must be non-empty and positive")

    if curve_type == RecoveryCurveType.WEIBULL:
        shape, _loc, scale = stats.weibull_min.fit(recovery_times, floc=0)
        mean = scale * float(stats.gamma(1.0 + 1.0 / shape).mean()) \
            if shape > 0 else float(np.mean(recovery_times))
        # Use distribution mean directly for robustness
        mean = float(stats.weibull_min(shape, loc=0, scale=scale).mean())
    elif curve_type == RecoveryCurveType.LOGNORMAL:
        shape, _loc, scale = stats.lognorm.fit(recovery_times, floc=0)
        mean = float(stats.lognorm(shape, loc=0, scale=scale).mean())
    else:  # GAMMA
        shape, _loc, scale = stats.gamma.fit(recovery_times, floc=0)
        mean = float(stats.gamma(shape, loc=0, scale=scale).mean())

    return RecoveryCurveFit(
        curve_type=curve_type,
        shape=float(shape),
        scale=float(scale),
        mean_recovery_time=mean,
    )


def cumulative_recovery_fraction(
    fit: RecoveryCurveFit,
    t: float,
) -> float:
    """Fraction of recoveries realised by time t (the CDF).

    Args:
        fit: A fitted recovery curve.
        t: Time (years) at which to evaluate the cumulative fraction.

    Returns:
        Cumulative recovery fraction in [0, 1].
    """
    if t <= 0:
        return 0.0
    if fit.curve_type == RecoveryCurveType.WEIBULL:
        return float(stats.weibull_min.cdf(t, fit.shape, loc=0, scale=fit.scale))
    if fit.curve_type == RecoveryCurveType.LOGNORMAL:
        return float(stats.lognorm.cdf(t, fit.shape, loc=0, scale=fit.scale))
    return float(stats.gamma.cdf(t, fit.shape, loc=0, scale=fit.scale))


def discounted_workout_lgd(
    exposure_at_default: float,
    total_nominal_recovery: float,
    fit: RecoveryCurveFit,
    discount_rate: float,
    workout_costs: float = 0.0,
    n_steps: int = 120,
) -> float:
    """Discounted workout LGD using the recovery-time distribution.

    Discounts nominal recoveries by the expected timing implied by the
    fitted recovery curve, then computes:
        LGD = 1 - (PV(recoveries) - workout_costs) / EAD

    Args:
        exposure_at_default: EAD.
        total_nominal_recovery: Total undiscounted recovery expected.
        fit: Fitted recovery curve giving the timing profile.
        discount_rate: Annual discount rate (e.g., effective interest
            rate or contractual rate per EBA/GL/2017/16).
        workout_costs: Direct/indirect workout costs (undiscounted,
            applied at present value approximation).
        n_steps: Monthly steps over which to integrate the recovery
            timing (default 120 = 10 years).

    Returns:
        Discounted workout LGD, floored at 0 and capped at 1.

    Reference:
        EBA/GL/2017/16, Araten et al. (2004).
    """
    if exposure_at_default <= 0:
        return 0.0

    # Integrate marginal recovery fraction × discount factor over time
    dt = 1.0 / 12.0
    pv = 0.0
    prev_cdf = 0.0
    for step in range(1, n_steps + 1):
        t = step * dt
        cdf = cumulative_recovery_fraction(fit, t)
        marginal = cdf - prev_cdf
        prev_cdf = cdf
        df = 1.0 / (1.0 + discount_rate) ** t
        pv += marginal * total_nominal_recovery * df

    net_recovery = pv - workout_costs
    lgd = 1.0 - net_recovery / exposure_at_default
    return float(min(max(lgd, 0.0), 1.0))
