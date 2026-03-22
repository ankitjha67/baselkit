"""Margin of Conservatism (MoC) -- ECB Guide to Internal Models, Chapter 7.

The MoC framework ensures conservatism in PD estimates by adding adjustments
for estimation error, model uncertainty, and data quality deficiencies.

References:
    - ECB Guide to Internal Models (July 2025), Chapter 7
    - EBA GL/2017/16, Section 5.3.5
    - BCBS d350, "Regulatory treatment of accounting provisions"

MoC categories:
    A: General estimation error (sampling, parameter uncertainty)
    B: Data quality/representativeness deficiencies
    C: Model uncertainty / methodology weaknesses
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)


class MoCCategory(str, Enum):  # noqa: UP042
    """MoC adjustment categories per ECB Guide."""

    ESTIMATION_ERROR = "A"
    DATA_DEFICIENCY = "B"
    MODEL_UNCERTAINTY = "C"


@dataclass
class MoCComponent:
    """Individual MoC adjustment component."""

    category: MoCCategory
    description: str
    adjustment_bps: float  # Additive adjustment in basis points
    confidence_level: float = 0.75  # ECB default: 75th percentile


@dataclass
class MoCResult:
    """Result of MoC calculation."""

    base_pd: float
    moc_additive: float  # Total additive MoC
    adjusted_pd: float  # PD after MoC
    components: list[MoCComponent] = field(default_factory=list)


# ── Category A: Estimation Error ──────────────────────────────────


def estimation_error_moc(
    pd_estimate: float,
    n_observations: int,
    n_defaults: int,
    confidence_level: float = 0.75,
) -> float:
    """Category A: MoC for estimation/sampling error.

    Uses the upper bound of a normal approximation to the binomial
    confidence interval for the default rate.  The MoC is the difference
    between the upper bound and the point estimate.

    MoC_A = z(confidence) * sqrt(pd * (1 - pd) / n) [normal approx.]

    For small samples or extreme default rates a Wilson interval is used
    for better coverage.

    Args:
        pd_estimate: Point estimate of PD.
        n_observations: Number of observations in estimation sample.
        n_defaults: Number of observed defaults.
        confidence_level: Confidence level (ECB default 75 %).

    Returns:
        Additive MoC in absolute PD terms (e.g. 0.005 = 50 bps).
    """
    if n_observations <= 0:
        logger.warning("n_observations <= 0; returning pd_estimate as MoC")
        return pd_estimate

    pd_hat = n_defaults / n_observations if n_observations > 0 else pd_estimate
    z = float(norm.ppf(confidence_level))

    # Wilson score interval upper bound (better coverage for small n / extreme p)
    denominator = 1.0 + z**2 / n_observations
    centre = (pd_hat + z**2 / (2.0 * n_observations)) / denominator
    margin = (
        z
        * math.sqrt(pd_hat * (1.0 - pd_hat) / n_observations + z**2 / (4.0 * n_observations**2))
        / denominator
    )
    upper_bound = min(centre + margin, 1.0)

    moc = max(upper_bound - pd_estimate, 0.0)
    logger.debug(
        "Estimation error MoC: pd=%.4f, n=%d, defaults=%d, moc=%.6f",
        pd_estimate,
        n_observations,
        n_defaults,
        moc,
    )
    return moc


# ── Category B: Data Representativeness ───────────────────────────


def data_representativeness_moc(
    development_default_rate: float,
    application_default_rate: float,
    psi: float = 0.0,
) -> float:
    """Category B: MoC for data representativeness.

    Combines two signals:
        1. Difference in default rates between development and application
           samples, which may indicate population drift.
        2. Population Stability Index (PSI) measuring distributional shift
           of model scores / features.

    Heuristic:
        MoC_B = |DR_dev - DR_app| + psi_penalty

    where psi_penalty is:
        PSI < 0.10  -> 0 bps  (insignificant change)
        0.10 <= PSI < 0.25 -> 25 bps (moderate shift)
        PSI >= 0.25 -> 50 bps (significant shift)

    Args:
        development_default_rate: Default rate in the development sample.
        application_default_rate: Default rate in the application sample.
        psi: Population Stability Index between the two samples.

    Returns:
        Additive MoC in absolute PD terms.
    """
    dr_diff = abs(development_default_rate - application_default_rate)

    if psi >= 0.25:
        psi_penalty = 0.0050  # 50 bps
    elif psi >= 0.10:
        psi_penalty = 0.0025  # 25 bps
    else:
        psi_penalty = 0.0

    moc = dr_diff + psi_penalty
    logger.debug(
        "Data representativeness MoC: dr_diff=%.4f, psi=%.4f, moc=%.6f",
        dr_diff,
        psi,
        moc,
    )
    return moc


# ── Category C: Model Uncertainty ─────────────────────────────────


def model_uncertainty_moc(
    base_pd: float,
    challenger_pd: float,
    n_challengers: int = 1,
) -> float:
    """Category C: MoC for model risk.

    Based on the difference between champion and challenger model PDs.
    When the challenger yields a higher PD, the MoC captures part of
    that gap to reflect model selection uncertainty.

    MoC_C = max(challenger_pd - base_pd, 0) / sqrt(n_challengers)

    Dividing by sqrt(n_challengers) reflects diminishing information gain
    from additional challenger models.

    Args:
        base_pd: PD from the champion model.
        challenger_pd: PD from the challenger model.
        n_challengers: Number of challenger models considered.

    Returns:
        Additive MoC in absolute PD terms.
    """
    if n_challengers < 1:
        n_challengers = 1

    gap = max(challenger_pd - base_pd, 0.0)
    moc = gap / math.sqrt(n_challengers)
    logger.debug(
        "Model uncertainty MoC: base=%.4f, challenger=%.4f, moc=%.6f",
        base_pd,
        challenger_pd,
        moc,
    )
    return moc


# ── Aggregation ───────────────────────────────────────────────────


def calculate_total_moc(
    base_pd: float,
    components: list[MoCComponent],
) -> MoCResult:
    """Aggregate all MoC components and apply to base PD.

    Total additive MoC is the sum of individual component adjustments
    (converted from basis points to absolute PD).  The adjusted PD is
    capped at 1.0 and floored at the Basel III minimum of 3 bps
    (CRE32.13).

    Args:
        base_pd: Unadjusted PD estimate.
        components: List of MoC adjustments.

    Returns:
        MoCResult with adjusted PD and component breakdown.
    """
    total_bps = sum(c.adjustment_bps for c in components)
    moc_additive = total_bps / 10_000.0  # bps -> absolute

    adjusted_pd = base_pd + moc_additive
    # Basel III PD floor 3 bps (CRE32.13)
    adjusted_pd = float(np.clip(adjusted_pd, 0.0003, 1.0))

    logger.info(
        "Total MoC: base_pd=%.4f, moc=%.4f (%.1f bps), adjusted_pd=%.4f",
        base_pd,
        moc_additive,
        total_bps,
        adjusted_pd,
    )

    return MoCResult(
        base_pd=base_pd,
        moc_additive=moc_additive,
        adjusted_pd=adjusted_pd,
        components=list(components),
    )


# ── Term Structure Application ────────────────────────────────────


def apply_moc_to_pd_curve(
    pd_curve: np.ndarray,
    moc_result: MoCResult,
) -> np.ndarray:
    """Apply MoC adjustment to entire PD term structure.

    Adds the absolute MoC to each tenor point and enforces the Basel III
    PD floor (3 bps) and ceiling (100 %).

    Args:
        pd_curve: Array of PD values by tenor.
        moc_result: MoCResult containing the additive MoC.

    Returns:
        Adjusted PD curve, clipped to [0.0003, 1.0].
    """
    pd_curve = np.asarray(pd_curve, dtype=np.float64)
    adjusted = pd_curve + moc_result.moc_additive
    return np.clip(adjusted, 0.0003, 1.0)
