"""
Behavioural scoring for retail credit risk.

Reference:
    - Thomas, Edelman & Crook (2002) — Credit Scoring and Its Applications.
    - EBA/GL/2017/16 — behavioural PD estimation for retail.
    - IFRS 9.B5.5.40 — behavioural life for revolving facilities.

Behavioural scores are recomputed periodically (typically monthly)
from account-conduct attributes — utilisation, payment patterns,
delinquency, and balance dynamics — to produce a refreshed PD for
performing retail exposures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BehaviouralAttributes:
    """Account-conduct attributes for behavioural scoring.

    Attributes:
        utilisation: Current balance / credit limit (0-1+).
        payment_ratio: Payments made / minimum due over the window
            (>= 1.0 means at least minimum paid).
        months_on_book: Account age in months.
        max_dpd_12m: Maximum days past due in the last 12 months.
        balance_velocity: Month-on-month balance growth rate (decimal;
            positive = increasing balance).
        n_times_overlimit_12m: Count of months over limit in last 12.
    """

    utilisation: float
    payment_ratio: float
    months_on_book: int
    max_dpd_12m: int
    balance_velocity: float
    n_times_overlimit_12m: int


# Logistic behavioural scorecard weights (illustrative, sign-correct).
# Calibrated so that higher utilisation, lower payment ratio, higher
# DPD, faster balance growth, and more over-limit events raise PD.
_INTERCEPT: float = -3.0
_W_UTILISATION: float = 1.8
_W_PAYMENT_RATIO: float = -1.5
_W_MONTHS_ON_BOOK: float = -0.02
_W_MAX_DPD: float = 0.04
_W_BALANCE_VELOCITY: float = 0.8
_W_OVERLIMIT: float = 0.25


def behavioural_score(attrs: BehaviouralAttributes) -> float:
    """Compute the linear behavioural score (log-odds).

    score = intercept
            + w_util  * utilisation
            + w_pay   * payment_ratio
            + w_mob   * months_on_book
            + w_dpd   * max_dpd_12m
            + w_vel   * balance_velocity
            + w_ovl   * n_times_overlimit_12m

    Args:
        attrs: Behavioural attributes.

    Returns:
        Linear score (log-odds of default).
    """
    return (
        _INTERCEPT
        + _W_UTILISATION * attrs.utilisation
        + _W_PAYMENT_RATIO * attrs.payment_ratio
        + _W_MONTHS_ON_BOOK * attrs.months_on_book
        + _W_MAX_DPD * attrs.max_dpd_12m
        + _W_BALANCE_VELOCITY * attrs.balance_velocity
        + _W_OVERLIMIT * attrs.n_times_overlimit_12m
    )


def behavioural_pd(attrs: BehaviouralAttributes) -> float:
    """Compute the behavioural 12-month PD from attributes.

    PD = 1 / (1 + exp(-score))

    Args:
        attrs: Behavioural attributes.

    Returns:
        12-month PD in (0, 1).
    """
    score = behavioural_score(attrs)
    return float(1.0 / (1.0 + np.exp(-score)))


def early_warning_flag(
    attrs: BehaviouralAttributes,
    utilisation_threshold: float = 0.90,
    dpd_threshold: int = 30,
    payment_ratio_threshold: float = 1.0,
) -> bool:
    """Raise an early-warning flag for deteriorating retail accounts.

    Flags accounts showing any of: high utilisation, recent delinquency,
    or sub-minimum payments — leading indicators of SICR / default.

    Reference:
        EBA/GL/2017/16, RBI SMA framework (early warning).

    Args:
        attrs: Behavioural attributes.
        utilisation_threshold: Utilisation above which to flag.
        dpd_threshold: DPD above which to flag.
        payment_ratio_threshold: Payment ratio below which to flag.

    Returns:
        ``True`` if any early-warning trigger fires.
    """
    return (
        attrs.utilisation >= utilisation_threshold
        or attrs.max_dpd_12m >= dpd_threshold
        or attrs.payment_ratio < payment_ratio_threshold
    )
