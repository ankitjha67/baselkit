"""Large Exposures framework (BCBS LEX).

Reference:
    - BCBS "Supervisory framework for measuring and controlling large
      exposures" (d283, April 2014; LEX in the consolidated framework),
      effective 1 January 2019.
    - EU CRR Part Four (Arts. 387-403), CRR3 amendments.

A "large exposure" is the sum of all exposure values to a single
counterparty (or a group of connected counterparties) that is equal to or
above 10% of the bank's eligible (Tier 1) capital base, and must be
reported (LEX30). The exposure to any single counterparty or connected
group is limited to 25% of Tier 1 capital (LEX20), tightened to 15% for
exposures between global systemically important banks (G-SIBs).

Exposure value is measured before risk weighting: the sum of on-balance
amounts, off-balance items after their credit conversion factor, the
counterparty-credit-risk EAD (SA-CCR), and securities-financing
exposures, net of eligible credit risk mitigation.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# LEX thresholds and limits, expressed as fractions of Tier 1 capital.
LE_REPORTING_THRESHOLD: float = 0.10  # LEX30 — report at/above 10%
LE_LIMIT: float = 0.25  # LEX20 — single counterparty / group limit
LE_GSIB_TO_GSIB_LIMIT: float = 0.15  # tighter G-SIB-to-G-SIB limit


def exposure_value(
    on_balance: float,
    off_balance_notional: float = 0.0,
    ccf: float = 1.0,
    derivative_ead: float = 0.0,
    sft_exposure: float = 0.0,
    eligible_crm: float = 0.0,
) -> float:
    """Large-exposure value for a counterparty (LEX10).

    Measured before risk weighting and floored at zero::

        value = max(on_balance + ccf * off_balance + derivative_ead
                    + sft_exposure - eligible_crm, 0)

    Off-balance-sheet items use a credit conversion factor with a 10%
    floor under the large-exposures framework (LEX30.4); callers should
    pass a CCF already subject to that floor.

    Args:
        on_balance: On-balance-sheet exposure amount.
        off_balance_notional: Off-balance-sheet nominal amount.
        ccf: Credit conversion factor applied to the off-balance nominal.
        derivative_ead: Counterparty-credit-risk EAD (e.g. SA-CCR).
        sft_exposure: Securities-financing-transaction exposure.
        eligible_crm: Eligible credit risk mitigation (substitution value
            of guarantees/collateral) reducing the exposure.

    Returns:
        Large-exposure value (non-negative).

    Raises:
        ValueError: If any input amount is negative or ccf is outside [0, 1].
    """
    if min(on_balance, off_balance_notional, derivative_ead, sft_exposure, eligible_crm) < 0.0:
        raise ValueError("exposure amounts must be non-negative")
    if not 0.0 <= ccf <= 1.0:
        raise ValueError("ccf must be in [0, 1]")

    gross = on_balance + ccf * off_balance_notional + derivative_ead + sft_exposure
    return max(gross - eligible_crm, 0.0)


def aggregate_connected_group(exposure_values: Sequence[float]) -> float:
    """Aggregate exposure values across a group of connected counterparties.

    Per LEX10.10-10.20, counterparties linked by a control relationship or
    economic interdependence are treated as a single counterparty; their
    exposure values are summed for the limit test.

    Args:
        exposure_values: Per-counterparty large-exposure values.

    Returns:
        Total group exposure value.

    Raises:
        ValueError: If any value is negative.
    """
    if any(v < 0.0 for v in exposure_values):
        raise ValueError("exposure values must be non-negative")
    return float(sum(exposure_values))


@dataclass(frozen=True)
class LargeExposureResult:
    """Outcome of a large-exposure limit assessment.

    Attributes:
        counterparty_id: Counterparty or connected-group identifier.
        exposure_value: Aggregate exposure value.
        tier1_capital: Tier 1 capital base.
        ratio_pct: Exposure as a percentage of Tier 1 capital.
        limit_pct: Applicable limit as a percentage of Tier 1 capital.
        is_large: True if the exposure is reportable (>= 10% of Tier 1).
        is_breach: True if the exposure exceeds the applicable limit.
        headroom: Remaining capacity to the limit (negative if breached).
    """

    counterparty_id: str
    exposure_value: float
    tier1_capital: float
    ratio_pct: float
    limit_pct: float
    is_large: bool
    is_breach: bool
    headroom: float


def assess_large_exposure(
    counterparty_id: str,
    exposure_value: float,
    tier1_capital: float,
    is_gsib_to_gsib: bool = False,
    reporting_threshold: float = LE_REPORTING_THRESHOLD,
) -> LargeExposureResult:
    """Assess a counterparty exposure against the large-exposure limits.

    Args:
        counterparty_id: Counterparty or connected-group identifier.
        exposure_value: Aggregate large-exposure value (see
            :func:`exposure_value`).
        tier1_capital: Tier 1 capital base. Must be strictly positive.
        is_gsib_to_gsib: True if this is an exposure from a G-SIB to
            another G-SIB, applying the tighter 15% limit.
        reporting_threshold: Fraction of Tier 1 at/above which the
            exposure is reportable (default 10%).

    Returns:
        A :class:`LargeExposureResult`.

    Raises:
        ValueError: If ``tier1_capital`` is not positive or
            ``exposure_value`` is negative.
    """
    if tier1_capital <= 0.0:
        raise ValueError("tier1_capital must be positive")
    if exposure_value < 0.0:
        raise ValueError("exposure_value must be non-negative")

    limit = LE_GSIB_TO_GSIB_LIMIT if is_gsib_to_gsib else LE_LIMIT
    ratio = exposure_value / tier1_capital
    limit_amount = limit * tier1_capital

    return LargeExposureResult(
        counterparty_id=counterparty_id,
        exposure_value=round(exposure_value, 6),
        tier1_capital=round(tier1_capital, 6),
        ratio_pct=round(ratio * 100.0, 4),
        limit_pct=round(limit * 100.0, 4),
        is_large=ratio >= reporting_threshold,
        is_breach=exposure_value > limit_amount,
        headroom=round(limit_amount - exposure_value, 6),
    )


@dataclass(frozen=True)
class LargeExposureReport:
    """Portfolio-level large-exposures report (LEX30).

    Attributes:
        tier1_capital: Tier 1 capital base.
        n_counterparties: Number of counterparties/groups assessed.
        large_exposures: Reportable exposures (>= reporting threshold),
            sorted by exposure value descending.
        breaches: Exposures exceeding their applicable limit.
        total_large_exposure_value: Sum of all reportable exposure values.
    """

    tier1_capital: float
    n_counterparties: int
    large_exposures: tuple[LargeExposureResult, ...]
    breaches: tuple[LargeExposureResult, ...]
    total_large_exposure_value: float


def large_exposures_report(
    exposures: Sequence[tuple[str, float]],
    tier1_capital: float,
    gsib_counterparties: Sequence[str] | None = None,
    reporting_threshold: float = LE_REPORTING_THRESHOLD,
) -> LargeExposureReport:
    """Build the supervisory large-exposures report for a portfolio.

    Args:
        exposures: Sequence of ``(counterparty_id, exposure_value)`` pairs,
            each value already aggregated across any connected group.
        tier1_capital: Tier 1 capital base.
        gsib_counterparties: Identifiers subject to the tighter
            G-SIB-to-G-SIB 15% limit (assumes the reporting bank is a
            G-SIB). Defaults to none.
        reporting_threshold: Reporting threshold as a fraction of Tier 1.

    Returns:
        A :class:`LargeExposureReport` listing reportable exposures and
        any limit breaches.

    Raises:
        ValueError: If ``tier1_capital`` is not positive.
    """
    if tier1_capital <= 0.0:
        raise ValueError("tier1_capital must be positive")

    gsib_set = set(gsib_counterparties or ())
    assessed = [
        assess_large_exposure(
            cid, value, tier1_capital,
            is_gsib_to_gsib=cid in gsib_set,
            reporting_threshold=reporting_threshold,
        )
        for cid, value in exposures
    ]

    large = sorted(
        (r for r in assessed if r.is_large),
        key=lambda r: r.exposure_value,
        reverse=True,
    )
    breaches = tuple(r for r in assessed if r.is_breach)

    return LargeExposureReport(
        tier1_capital=round(tier1_capital, 6),
        n_counterparties=len(assessed),
        large_exposures=tuple(large),
        breaches=breaches,
        total_large_exposure_value=round(sum(r.exposure_value for r in large), 6),
    )
