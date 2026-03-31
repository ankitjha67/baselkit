"""Revolving credit ECL calculation engine.

Computes ECL for revolving facilities with proper drawn/undrawn
decomposition per IFRS 7 B8E, behavioral life determination per
IFRS 9 B5.5.40, and probability-weighted scenario support.

The drawn ECL component is recognized as a **loss allowance**
(contra-asset reducing gross carrying amount).  The undrawn ECL
component is recognized as a **provision (liability)**.

References:
    - IFRS 9 paragraphs 5.5.1-5.5.20 (ECL measurement)
    - IFRS 9 paragraph B5.5.31 (drawdown expectations)
    - IFRS 7 paragraph B8E (drawn/undrawn presentation)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from creditriskengine.core.types import IFRS9Stage
from creditriskengine.ecl.ifrs9.ecl_calc import discount_factors


@dataclass
class RevolvingECLResult:
    """Result of a revolving credit ECL calculation.

    Attributes:
        total_ecl: Total expected credit loss.
        ecl_drawn: ECL on the drawn component (loss allowance,
            contra-asset per IFRS 7 B8E).
        ecl_undrawn: ECL on the undrawn component (provision,
            liability per IFRS 7 B8E).
        ecl_by_period: ECL contribution by period (for diagnostics).
        behavioral_life_months: Behavioral life used in the calculation.
        ccf_used: CCF applied to the undrawn component.
        ead_profile: Total EAD by period.
    """

    total_ecl: float
    ecl_drawn: float
    ecl_undrawn: float
    ecl_by_period: np.ndarray
    behavioral_life_months: int
    ccf_used: float
    ead_profile: np.ndarray


def calculate_revolving_ecl(
    stage: IFRS9Stage,
    drawn: float,
    undrawn: float,
    ccf: float,
    pd_12m: float,
    lgd: float,
    eir: float = 0.0,
    marginal_pds: np.ndarray | None = None,
    behavioral_life_months: int = 36,
    lgd_curve: np.ndarray | None = None,
    ead_drawn_curve: np.ndarray | None = None,
    ead_undrawn_curve: np.ndarray | None = None,
) -> RevolvingECLResult:
    """Calculate ECL for a revolving credit facility.

    Produces a decomposition of ECL into drawn and undrawn components
    for IFRS 7 B8E presentation.

    For **Stage 1**, ECL covers 12-month expected losses.
    For **Stage 2/3/POCI**, ECL covers the full behavioral life.

    The formula for each period *t*:

        ECL_drawn(t) = Marginal_PD(t) × LGD(t) × Drawn(t) × DF(t)
        ECL_undrawn(t) = Marginal_PD(t) × LGD(t) × CCF × Undrawn(t) × DF(t)

    Args:
        stage: IFRS 9 impairment stage.
        drawn: Current drawn balance.
        undrawn: Current undrawn commitment.
        ccf: Credit conversion factor for the undrawn portion.
        pd_12m: 12-month probability of default (used for Stage 1).
        lgd: Scalar LGD (overridden by lgd_curve if provided).
        eir: Effective interest rate for discounting.
        marginal_pds: Array of marginal PDs per period (required for
            Stage 2/3/POCI).  For Stage 1, if not provided, a single
            period using pd_12m is assumed.
        behavioral_life_months: Behavioral life in months per B5.5.40.
            Used as the horizon for Stage 2/3/POCI calculations.
            For Stage 1, capped at 12 months.
        lgd_curve: Optional per-period LGD array.
        ead_drawn_curve: Optional per-period drawn EAD array.
        ead_undrawn_curve: Optional per-period undrawn EAD array
            (pre-CCF; CCF is applied internally).

    Returns:
        :class:`RevolvingECLResult` with total, drawn, and undrawn ECL.

    Raises:
        ValueError: If marginal_pds is missing for lifetime stages.
    """
    # Determine horizon
    if stage == IFRS9Stage.STAGE_1:
        n_periods = min(12, behavioral_life_months)
    else:
        n_periods = behavioral_life_months

    # Build marginal PD array
    if marginal_pds is not None:
        mpd = np.asarray(marginal_pds, dtype=float)[:n_periods]
        if len(mpd) < n_periods:
            mpd = np.pad(mpd, (0, n_periods - len(mpd)))
    elif stage == IFRS9Stage.STAGE_1:
        mpd = np.array([pd_12m])
        n_periods = 1
    else:
        raise ValueError(
            "marginal_pds is required for Stage 2/3/POCI lifetime ECL."
        )

    # Build LGD array
    if lgd_curve is not None:
        lgd_arr = np.asarray(lgd_curve, dtype=float)[:n_periods]
        if len(lgd_arr) < n_periods:
            lgd_arr = np.pad(
                lgd_arr,
                (0, n_periods - len(lgd_arr)),
                constant_values=lgd_arr[-1],
            )
    else:
        lgd_arr = np.full(n_periods, lgd)

    # Build drawn/undrawn EAD arrays
    if ead_drawn_curve is not None:
        drawn_arr = np.asarray(ead_drawn_curve, dtype=float)[:n_periods]
        if len(drawn_arr) < n_periods:
            drawn_arr = np.pad(
                drawn_arr,
                (0, n_periods - len(drawn_arr)),
                constant_values=drawn_arr[-1],
            )
    else:
        drawn_arr = np.full(n_periods, drawn)

    if ead_undrawn_curve is not None:
        undrawn_arr = np.asarray(ead_undrawn_curve, dtype=float)[:n_periods]
        if len(undrawn_arr) < n_periods:
            undrawn_arr = np.pad(
                undrawn_arr,
                (0, n_periods - len(undrawn_arr)),
                constant_values=undrawn_arr[-1],
            )
    else:
        undrawn_arr = np.full(n_periods, undrawn)

    # Discount factors
    dfs = discount_factors(eir, n_periods)

    # ECL decomposition
    ecl_drawn_arr = mpd * lgd_arr * drawn_arr * dfs
    ecl_undrawn_arr = mpd * lgd_arr * (ccf * undrawn_arr) * dfs
    ecl_total_arr = ecl_drawn_arr + ecl_undrawn_arr

    total_ecl_drawn = float(np.sum(ecl_drawn_arr))
    total_ecl_undrawn = float(np.sum(ecl_undrawn_arr))

    ead_profile = drawn_arr + ccf * undrawn_arr

    return RevolvingECLResult(
        total_ecl=total_ecl_drawn + total_ecl_undrawn,
        ecl_drawn=total_ecl_drawn,
        ecl_undrawn=total_ecl_undrawn,
        ecl_by_period=ecl_total_arr,
        behavioral_life_months=behavioral_life_months,
        ccf_used=ccf,
        ead_profile=ead_profile,
    )


def revolving_ecl_scenario_weighted(
    scenarios: list[tuple[float, RevolvingECLResult]],
) -> RevolvingECLResult:
    """Compute probability-weighted ECL across macroeconomic scenarios.

    Per IFRS 9.5.5.17, ECL must reflect an unbiased, probability-weighted
    amount determined by evaluating a range of possible outcomes.

    Args:
        scenarios: List of ``(weight, result)`` tuples, where weight is
            the scenario probability and result is a
            :class:`RevolvingECLResult`.  Weights must sum to ~1.0.

    Returns:
        Probability-weighted :class:`RevolvingECLResult`.

    Raises:
        ValueError: If scenarios is empty or weights don't sum to ~1.0.
    """
    if not scenarios:
        raise ValueError("At least one scenario is required.")

    total_weight = sum(w for w, _ in scenarios)
    if abs(total_weight - 1.0) > 1e-6:
        raise ValueError(
            f"Scenario weights must sum to 1.0, got {total_weight:.6f}."
        )

    first_result = scenarios[0][1]
    n_periods = len(first_result.ecl_by_period)

    w_total = 0.0
    w_drawn = 0.0
    w_undrawn = 0.0
    w_by_period = np.zeros(n_periods)
    w_ead_profile = np.zeros(n_periods)

    for weight, result in scenarios:
        w_total += weight * result.total_ecl
        w_drawn += weight * result.ecl_drawn
        w_undrawn += weight * result.ecl_undrawn
        result_periods = result.ecl_by_period[:n_periods]
        w_by_period[: len(result_periods)] += weight * result_periods
        result_ead = result.ead_profile[:n_periods]
        w_ead_profile[: len(result_ead)] += weight * result_ead

    return RevolvingECLResult(
        total_ecl=w_total,
        ecl_drawn=w_drawn,
        ecl_undrawn=w_undrawn,
        ecl_by_period=w_by_period,
        behavioral_life_months=first_result.behavioral_life_months,
        ccf_used=first_result.ccf_used,
        ead_profile=w_ead_profile,
    )
