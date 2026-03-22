"""
PD backtesting framework.

Reference: BCBS WP14 (May 2005), EBA GL/2017/16.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .calibration import binomial_test, jeffreys_test, traffic_light_test

logger = logging.getLogger(__name__)


def pd_backtest_summary(
    predicted_pds: np.ndarray,
    observed_defaults: np.ndarray,
    rating_grades: np.ndarray | None = None,
) -> dict[str, float]:
    """Summary statistics for PD backtesting.

    Args:
        predicted_pds: Predicted PDs per exposure.
        observed_defaults: Binary default indicator (0/1).
        rating_grades: Optional rating grade labels for grouping.

    Returns:
        Dict with overall metrics.
    """
    predicted_pds = np.asarray(predicted_pds, dtype=np.float64)
    observed_defaults = np.asarray(observed_defaults, dtype=np.int64)

    n = len(predicted_pds)
    n_defaults = int(np.sum(observed_defaults))
    avg_pd = float(np.mean(predicted_pds))
    observed_dr = n_defaults / n if n > 0 else 0.0

    return {
        "n_observations": n,
        "n_defaults": n_defaults,
        "average_predicted_pd": avg_pd,
        "observed_default_rate": observed_dr,
        "ratio_observed_to_predicted": observed_dr / avg_pd if avg_pd > 0 else 0.0,
    }


@dataclass
class GradeBacktestResult:
    """Backtest result for a single rating grade.

    Attributes:
        grade: Rating grade label.
        n_observations: Number of exposures in the grade.
        n_defaults: Observed defaults.
        predicted_pd: Average predicted PD for the grade.
        observed_dr: Observed default rate.
        binomial: Result dict from the binomial test.
        traffic_light: Traffic-light colour (green/yellow/red).
        jeffreys: Result dict from the Jeffreys test.
    """

    grade: str
    n_observations: int
    n_defaults: int
    predicted_pd: float
    observed_dr: float
    binomial: dict[str, Any]
    traffic_light: str
    jeffreys: dict[str, Any]


@dataclass
class FullBacktestResult:
    """Aggregated backtest result across all rating grades.

    Attributes:
        summary: Overall summary statistics.
        grade_results: Per-grade backtest results.
        overall_traffic_light: Worst traffic-light colour across grades.
        overall_assessment: Human-readable overall assessment.
    """

    summary: dict[str, float]
    grade_results: list[GradeBacktestResult]
    overall_traffic_light: str
    overall_assessment: str


def pd_backtest_full(
    predicted_pds: np.ndarray,
    observed_defaults: np.ndarray,
    rating_grades: np.ndarray,
    confidence: float = 0.99,
) -> FullBacktestResult:
    """Full PD backtest: binomial, traffic-light, and Jeffreys per rating grade.

    Runs a comprehensive backtesting suite for each rating grade, combining
    frequentist (binomial), regulatory (traffic-light), and Bayesian (Jeffreys)
    approaches as recommended in BCBS WP14 and EBA GL/2017/16.

    Args:
        predicted_pds: Predicted PDs per exposure.
        observed_defaults: Binary default indicator (0/1).
        rating_grades: Rating grade labels per exposure (same length).
        confidence: Confidence level for binomial and Jeffreys tests.

    Returns:
        FullBacktestResult with per-grade detail and overall assessment.

    Raises:
        ValueError: If input arrays have mismatched lengths.
    """
    predicted_pds = np.asarray(predicted_pds, dtype=np.float64)
    observed_defaults = np.asarray(observed_defaults, dtype=np.int64)
    rating_grades = np.asarray(rating_grades)

    if not (len(predicted_pds) == len(observed_defaults) == len(rating_grades)):
        raise ValueError(
            "predicted_pds, observed_defaults, and rating_grades must have the same length."
        )

    summary = pd_backtest_summary(predicted_pds, observed_defaults, rating_grades)

    unique_grades = np.unique(rating_grades)
    grade_results: list[GradeBacktestResult] = []
    traffic_light_colours: list[str] = []

    for grade in unique_grades:
        mask = rating_grades == grade
        n_obs = int(np.sum(mask))
        n_def = int(np.sum(observed_defaults[mask]))
        avg_pd = float(np.mean(predicted_pds[mask]))
        obs_dr = n_def / n_obs if n_obs > 0 else 0.0

        binom_result = binomial_test(n_def, n_obs, avg_pd, confidence=confidence)
        tl_result = traffic_light_test(n_def, n_obs, avg_pd)
        jeff_result = jeffreys_test(n_def, n_obs, avg_pd, confidence=confidence)

        traffic_light_colours.append(tl_result)

        grade_results.append(
            GradeBacktestResult(
                grade=str(grade),
                n_observations=n_obs,
                n_defaults=n_def,
                predicted_pd=avg_pd,
                observed_dr=obs_dr,
                binomial=binom_result,
                traffic_light=tl_result,
                jeffreys=jeff_result,
            )
        )
        logger.debug(
            "Grade %s: n=%d, defaults=%d, TL=%s, binom_reject=%s",
            grade,
            n_obs,
            n_def,
            tl_result,
            binom_result.get("reject_h0"),
        )

    # Overall traffic light is the worst across grades
    _tl_order = {"green": 0, "yellow": 1, "red": 2}
    overall_tl = max(traffic_light_colours, key=lambda c: _tl_order.get(c, 0)) if traffic_light_colours else "green"

    # Derive human-readable assessment
    n_red = sum(1 for c in traffic_light_colours if c == "red")
    n_yellow = sum(1 for c in traffic_light_colours if c == "yellow")
    if n_red > 0:
        assessment = (
            f"FAIL: {n_red} grade(s) in red zone — material PD under-estimation detected. "
            "Recalibration required per EBA GL/2017/16 Art. 58."
        )
    elif n_yellow > 0:
        assessment = (
            f"WARNING: {n_yellow} grade(s) in yellow zone — potential PD under-estimation. "
            "Monitor closely; recalibration may be warranted."
        )
    else:
        assessment = "PASS: All grades in green zone — PD calibration adequate."

    logger.info("PD backtest overall: %s (traffic_light=%s)", assessment, overall_tl)

    return FullBacktestResult(
        summary=summary,
        grade_results=grade_results,
        overall_traffic_light=overall_tl,
        overall_assessment=assessment,
    )


@dataclass
class VintageResult:
    """Backtest result for a single vintage (time period).

    Attributes:
        period: Period label (e.g. '2023-Q1').
        n_observations: Number of exposures in the vintage.
        n_defaults: Observed defaults.
        predicted_pd: Average predicted PD.
        observed_dr: Observed default rate.
        traffic_light: Traffic-light colour.
        binomial: Binomial test result dict.
    """

    period: str
    n_observations: int
    n_defaults: int
    predicted_pd: float
    observed_dr: float
    traffic_light: str
    binomial: dict[str, Any]


@dataclass
class MultiPeriodBacktestResult:
    """Result of multi-period (time-series) backtesting.

    Attributes:
        vintage_results: Per-vintage backtest outcomes.
        n_periods: Number of distinct periods analysed.
        n_red: Count of periods with red traffic light.
        n_yellow: Count of periods with yellow traffic light.
        n_green: Count of periods with green traffic light.
        overall_assessment: Summary assessment across time series.
    """

    vintage_results: list[VintageResult]
    n_periods: int
    n_red: int
    n_yellow: int
    n_green: int
    overall_assessment: str


def multi_period_backtest(
    predicted_pds: np.ndarray,
    observed_defaults: np.ndarray,
    periods: np.ndarray,
    confidence: float = 0.99,
) -> MultiPeriodBacktestResult:
    """Time-series backtesting across vintages.

    Runs binomial and traffic-light tests for each distinct period (vintage),
    enabling trend analysis and detection of systematic calibration drift
    over time as recommended by EBA GL/2017/16 Section 6.3.

    Args:
        predicted_pds: Predicted PDs per exposure.
        observed_defaults: Binary default indicator (0/1).
        periods: Period label per exposure (e.g. '2023-Q1').
        confidence: Confidence level for the binomial test.

    Returns:
        MultiPeriodBacktestResult with per-vintage detail and trend assessment.

    Raises:
        ValueError: If input arrays have mismatched lengths.
    """
    predicted_pds = np.asarray(predicted_pds, dtype=np.float64)
    observed_defaults = np.asarray(observed_defaults, dtype=np.int64)
    periods = np.asarray(periods)

    if not (len(predicted_pds) == len(observed_defaults) == len(periods)):
        raise ValueError(
            "predicted_pds, observed_defaults, and periods must have the same length."
        )

    unique_periods = np.unique(periods)
    vintage_results: list[VintageResult] = []

    for period in sorted(unique_periods):
        mask = periods == period
        n_obs = int(np.sum(mask))
        n_def = int(np.sum(observed_defaults[mask]))
        avg_pd = float(np.mean(predicted_pds[mask]))
        obs_dr = n_def / n_obs if n_obs > 0 else 0.0

        binom_result = binomial_test(n_def, n_obs, avg_pd, confidence=confidence)
        tl = traffic_light_test(n_def, n_obs, avg_pd)

        vintage_results.append(
            VintageResult(
                period=str(period),
                n_observations=n_obs,
                n_defaults=n_def,
                predicted_pd=avg_pd,
                observed_dr=obs_dr,
                traffic_light=tl,
                binomial=binom_result,
            )
        )
        logger.debug("Period %s: n=%d, defaults=%d, TL=%s", period, n_obs, n_def, tl)

    n_red = sum(1 for v in vintage_results if v.traffic_light == "red")
    n_yellow = sum(1 for v in vintage_results if v.traffic_light == "yellow")
    n_green = sum(1 for v in vintage_results if v.traffic_light == "green")
    n_periods = len(vintage_results)

    # Assess trend: consecutive reds or yellows suggest systematic drift
    if n_red >= 2:
        assessment = (
            f"FAIL: {n_red}/{n_periods} periods in red zone — systematic PD under-estimation. "
            "Immediate recalibration required per EBA GL/2017/16 Art. 58."
        )
    elif n_red == 1:
        assessment = (
            f"WARNING: 1/{n_periods} period(s) in red zone. "
            "Investigate root cause and monitor subsequent periods."
        )
    elif n_yellow >= n_periods // 2 and n_periods > 1:
        assessment = (
            f"WARNING: {n_yellow}/{n_periods} periods in yellow zone — "
            "potential calibration drift. Consider recalibration."
        )
    else:
        assessment = (
            f"PASS: {n_green}/{n_periods} periods in green zone — "
            "PD calibration stable over time."
        )

    logger.info("Multi-period backtest: %s", assessment)

    return MultiPeriodBacktestResult(
        vintage_results=vintage_results,
        n_periods=n_periods,
        n_red=n_red,
        n_yellow=n_yellow,
        n_green=n_green,
        overall_assessment=assessment,
    )
