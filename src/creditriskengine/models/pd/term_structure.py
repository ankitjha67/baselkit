"""
PD term structure construction and manipulation.

Builds multi-year cumulative PD curves from constant hazard rates,
rating transition matrices, or observed PD points using interpolation.

References:
- EBA GL/2017/16: PD estimation
- Engelmann & Rauhmeier: The Basel II Risk Parameters (2nd ed.)
- Lando (2004): Credit Risk Modeling
"""

import logging

import numpy as np
from numpy.linalg import matrix_power

logger = logging.getLogger(__name__)


# -- Constant-Hazard Term Structure -----------------------------------------


def pd_term_structure_from_hazard(
    annual_pd: float,
    max_years: int = 30,
) -> np.ndarray:
    """Build a cumulative PD curve assuming constant hazard rate.

    Under constant hazard the survival probability decays geometrically:

        cum_pd(t) = 1 - (1 - annual_pd)^t

    Args:
        annual_pd: One-year probability of default in (0, 1).
        max_years: Number of years in the curve (default 30).

    Returns:
        Array of cumulative PDs for years 1 .. max_years.
    """
    if not 0.0 <= annual_pd <= 1.0:
        raise ValueError(f"annual_pd must be in [0, 1], got {annual_pd}")
    if max_years < 1:
        raise ValueError(f"max_years must be >= 1, got {max_years}")

    years = np.arange(1, max_years + 1, dtype=np.float64)
    cum_pd = 1.0 - (1.0 - annual_pd) ** years
    return cum_pd


# -- Transition-Matrix Term Structure ---------------------------------------


def pd_term_structure_from_transitions(
    transition_matrix: np.ndarray,
    initial_rating: int,
    max_years: int = 30,
) -> np.ndarray:
    """Build a cumulative PD curve from a rating transition matrix.

    The default state is assumed to be the **last** column (absorbing state).
    Cumulative PD at horizon *t* is obtained by raising the matrix to the
    *t*-th power and reading the default column for the initial rating:

        cum_pd(t) = (TM^t)[initial_rating, default_col]

    Args:
        transition_matrix: Annual transition matrix (N x N) with the last
            column/row representing the default (absorbing) state.
        initial_rating: Row index of the obligor's current rating grade.
        max_years: Length of the term structure (default 30).

    Returns:
        Array of cumulative PDs for years 1 .. max_years.
    """
    tm = np.asarray(transition_matrix, dtype=np.float64)
    if tm.ndim != 2 or tm.shape[0] != tm.shape[1]:
        raise ValueError("transition_matrix must be a square 2-D array")
    if initial_rating < 0 or initial_rating >= tm.shape[0]:
        raise ValueError(
            f"initial_rating {initial_rating} out of range [0, {tm.shape[0] - 1}]"
        )

    default_col = tm.shape[1] - 1
    cum_pds = np.zeros(max_years, dtype=np.float64)
    for t in range(1, max_years + 1):
        tm_t = matrix_power(tm, t)
        cum_pds[t - 1] = tm_t[initial_rating, default_col]
    return cum_pds


# -- Interpolation ----------------------------------------------------------


def interpolate_pd_term_structure(
    pd_curve: np.ndarray,
    target_years: np.ndarray,
) -> np.ndarray:
    """Log-linear interpolation of a PD term structure.

    Converts cumulative PDs to log-survival space, applies linear
    interpolation, and converts back:

        S(t) = 1 - cum_pd(t)
        log_S  is linearly interpolated
        cum_pd = 1 - exp(interpolated log_S)

    Args:
        pd_curve: Observed cumulative PDs at integer years 1 .. len(pd_curve).
        target_years: Year values at which to interpolate (may be fractional).

    Returns:
        Interpolated cumulative PDs at each target year.
    """
    pd_curve = np.asarray(pd_curve, dtype=np.float64)
    target_years = np.asarray(target_years, dtype=np.float64)

    # Known year grid: 1, 2, ..., len(pd_curve)
    known_years = np.arange(1, len(pd_curve) + 1, dtype=np.float64)

    # Clamp survival away from zero for log safety
    survival = np.clip(1.0 - pd_curve, 1e-15, 1.0)
    log_survival = np.log(survival)

    # Prepend year-0 anchor (survival = 1 => log_S = 0)
    known_years = np.concatenate([[0.0], known_years])
    log_survival = np.concatenate([[0.0], log_survival])

    interp_log_s = np.interp(target_years, known_years, log_survival)
    return 1.0 - np.exp(interp_log_s)


# -- Forward / Marginal PDs ------------------------------------------------


def forward_pd(cumulative_pds: np.ndarray) -> np.ndarray:
    """Extract forward (marginal conditional) PDs from a cumulative PD curve.

    The forward PD for period *t* is the probability of defaulting in
    period *t* given survival to the start of period *t*:

        fwd_pd(1) = cum_pd(1)
        fwd_pd(t) = (cum_pd(t) - cum_pd(t-1)) / (1 - cum_pd(t-1))   for t > 1

    Args:
        cumulative_pds: Array of cumulative PDs for periods 1 .. T.

    Returns:
        Array of forward PDs for each period.
    """
    cumulative_pds = np.asarray(cumulative_pds, dtype=np.float64)
    n = len(cumulative_pds)
    fwd = np.zeros(n, dtype=np.float64)

    fwd[0] = cumulative_pds[0]
    for t in range(1, n):
        denom = 1.0 - cumulative_pds[t - 1]
        if denom < 1e-15:
            fwd[t] = 0.0
        else:
            fwd[t] = (cumulative_pds[t] - cumulative_pds[t - 1]) / denom
    return fwd
