"""Cure rate modeling for LGD estimation.

Models the probability that a defaulted exposure returns to performing status
("cures") without loss realization. Key component of workout LGD estimation.

References:
    - EBA GL/2017/16, Section 6.3.2 (treatment of cured defaults)
    - BCBS d350, Para 232-234
"""

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class CureRateResult:
    """Result of cure rate estimation."""

    overall_cure_rate: float
    time_to_cure_months_mean: float
    time_to_cure_months_median: float
    cure_rates_by_period: NDArray[np.float64]  # Monthly/quarterly cure rates
    n_defaults: int
    n_cured: int


# ── Core Cure Rate Estimation ─────────────────────────────────────


def estimate_cure_rate(
    default_outcomes: NDArray[np.int64],
    time_in_default_months: NDArray[np.float64] | None = None,
) -> CureRateResult:
    """Estimate cure rate from historical default outcomes.

    Args:
        default_outcomes: Binary array — 1 = cured, 0 = loss/write-off.
        time_in_default_months: Time spent in default before resolution.
            If provided, used to compute time-to-cure statistics and
            monthly cure rate profile.

    Returns:
        CureRateResult with aggregate and period-level cure statistics.
    """
    default_outcomes = np.asarray(default_outcomes, dtype=np.int64)
    n_defaults = len(default_outcomes)
    n_cured = int(np.sum(default_outcomes))

    if n_defaults == 0:
        logger.warning("Empty default_outcomes array; returning zero cure rate")
        return CureRateResult(
            overall_cure_rate=0.0,
            time_to_cure_months_mean=0.0,
            time_to_cure_months_median=0.0,
            cure_rates_by_period=np.array([], dtype=np.float64),
            n_defaults=0,
            n_cured=0,
        )

    overall_cure_rate = n_cured / n_defaults

    # Time-to-cure statistics
    if time_in_default_months is not None:
        time_in_default_months = np.asarray(time_in_default_months, dtype=np.float64)
        cured_mask = default_outcomes == 1
        if np.any(cured_mask):
            cure_times = time_in_default_months[cured_mask]
            mean_ttc = float(np.mean(cure_times))
            median_ttc = float(np.median(cure_times))
        else:
            mean_ttc = 0.0
            median_ttc = 0.0

        # Build monthly cure rate profile
        max_month = int(np.ceil(np.max(time_in_default_months))) if len(time_in_default_months) > 0 else 0
        max_month = max(max_month, 1)
        cure_rates_by_period = np.zeros(max_month, dtype=np.float64)

        for m in range(max_month):
            # Observations still in default at start of month m
            at_risk = np.sum(time_in_default_months >= m)
            # Cured during month m
            cured_in_month = np.sum(
                cured_mask & (time_in_default_months >= m) & (time_in_default_months < m + 1)
            )
            cure_rates_by_period[m] = float(cured_in_month / at_risk) if at_risk > 0 else 0.0
    else:
        mean_ttc = 0.0
        median_ttc = 0.0
        cure_rates_by_period = np.array([overall_cure_rate], dtype=np.float64)

    return CureRateResult(
        overall_cure_rate=overall_cure_rate,
        time_to_cure_months_mean=mean_ttc,
        time_to_cure_months_median=median_ttc,
        cure_rates_by_period=cure_rates_by_period,
        n_defaults=n_defaults,
        n_cured=n_cured,
    )


# ── Segmented Cure Rates ─────────────────────────────────────────


def cure_rate_by_segment(
    default_outcomes: NDArray[np.int64],
    segments: NDArray[np.int64],
    segment_labels: list[str] | None = None,
) -> dict[str, float]:
    """Segment-level cure rates (e.g., by product, rating grade).

    Args:
        default_outcomes: Binary array — 1 = cured, 0 = loss/write-off.
        segments: Integer segment identifiers (same length as outcomes).
        segment_labels: Optional human-readable labels for each segment ID.

    Returns:
        Dict mapping segment label to cure rate.
    """
    default_outcomes = np.asarray(default_outcomes, dtype=np.int64)
    segments = np.asarray(segments, dtype=np.int64)

    unique_segments = np.unique(segments)

    if segment_labels is not None and len(segment_labels) != len(unique_segments):
        logger.warning(
            "segment_labels length (%d) != unique segments (%d); using numeric labels",
            len(segment_labels),
            len(unique_segments),
        )
        segment_labels = None

    label_map: dict[int, str] = {}
    for idx, seg_id in enumerate(unique_segments):
        if segment_labels is not None:
            label_map[int(seg_id)] = segment_labels[idx]
        else:
            label_map[int(seg_id)] = str(seg_id)

    results: dict[str, float] = {}
    for seg_id in unique_segments:
        mask = segments == seg_id
        n_seg = int(np.sum(mask))
        n_cured = int(np.sum(default_outcomes[mask]))
        rate = n_cured / n_seg if n_seg > 0 else 0.0
        results[label_map[int(seg_id)]] = rate

    return results


# ── Macroeconomic Adjustment ──────────────────────────────────────


def macro_adjusted_cure_rate(
    base_cure_rate: float,
    gdp_growth_current: float,
    gdp_growth_baseline: float,
    sensitivity: float = 0.5,
) -> float:
    """Adjust cure rate for macroeconomic conditions.

    Cure rates tend to decline in economic downturns and increase in
    expansions.

    Formula:
        adj_rate = base_rate * (1 + sensitivity * (gdp_current - gdp_baseline) / |gdp_baseline|)

    The result is bounded to [0, 1].

    Args:
        base_cure_rate: Long-run average cure rate.
        gdp_growth_current: Current GDP growth rate (e.g., 0.02 = 2 %).
        gdp_growth_baseline: Baseline/long-run GDP growth rate.
        sensitivity: Elasticity of cure rate to GDP deviation (default 0.5).

    Returns:
        Macro-adjusted cure rate in [0, 1].
    """
    if abs(gdp_growth_baseline) < 1e-10:
        # Avoid division by near-zero; use absolute difference instead
        adjustment = sensitivity * (gdp_growth_current - gdp_growth_baseline)
    else:
        adjustment = sensitivity * (gdp_growth_current - gdp_growth_baseline) / abs(
            gdp_growth_baseline
        )

    adjusted = base_cure_rate * (1.0 + adjustment)
    return float(np.clip(adjusted, 0.0, 1.0))


# ── LGD Cure Adjustment ──────────────────────────────────────────


def lgd_with_cure_adjustment(
    workout_lgd: float,
    cure_rate: float,
    lgd_if_cured: float = 0.0,
) -> float:
    """Adjusted LGD incorporating cure probability.

    LGD_adjusted = (1 - cure_rate) * workout_lgd + cure_rate * lgd_if_cured

    Per EBA GL/2017/16 Section 6.3.2, cured exposures may still incur
    indirect costs (e.g. carrying costs during default), hence lgd_if_cured
    can be > 0.

    Args:
        workout_lgd: LGD for non-cured (workout) cases.
        cure_rate: Probability of cure.
        lgd_if_cured: LGD applicable to cured cases (default 0.0).

    Returns:
        Blended LGD in [0, 1].
    """
    cure_rate = float(np.clip(cure_rate, 0.0, 1.0))
    blended = (1.0 - cure_rate) * workout_lgd + cure_rate * lgd_if_cured
    return float(np.clip(blended, 0.0, 1.0))


# ── Cure Rate Term Structure ─────────────────────────────────────


def cure_rate_term_structure(
    monthly_cure_probabilities: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Build cumulative cure probability curve from monthly cure hazard rates.

    Given monthly conditional cure probabilities h(t) (probability of
    curing in month t given still in default at start of month t), the
    cumulative cure probability by month T is:

        P(cured by T) = 1 - prod_{t=0}^{T-1} (1 - h(t))

    This is analogous to a survival analysis framework where "cure" is
    the event of interest.

    Args:
        monthly_cure_probabilities: Monthly conditional cure hazard rates,
            each in [0, 1].

    Returns:
        Cumulative cure probability curve (same length as input).
    """
    monthly_cure_probabilities = np.asarray(monthly_cure_probabilities, dtype=np.float64)
    monthly_cure_probabilities = np.clip(monthly_cure_probabilities, 0.0, 1.0)

    # Survival (still-in-default) probability each month
    survival = np.cumprod(1.0 - monthly_cure_probabilities)

    # Cumulative cure = 1 - cumulative survival
    cumulative_cure = 1.0 - survival
    return np.asarray(cumulative_cure, dtype=np.float64)
