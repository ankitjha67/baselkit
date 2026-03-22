"""
Lifetime PD term structure construction.

Builds cumulative and marginal PD curves from annual PD inputs
or transition matrices for ECL calculation.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def cumulative_pd_from_annual(annual_pds: list[float]) -> np.ndarray:
    """Convert annual (conditional) PDs to cumulative PD curve.

    Formula:
        Cumulative_PD(t) = 1 - Product(k=1..t)(1 - PD_k)

    Args:
        annual_pds: List of annual conditional PDs for periods 1..T.

    Returns:
        Array of cumulative PDs for each period.
    """
    survival = np.cumprod([1.0 - pd for pd in annual_pds])
    return 1.0 - survival


def marginal_pd_from_cumulative(cumulative_pds: np.ndarray) -> np.ndarray:
    """Extract marginal (incremental) PDs from cumulative PD curve.

    Formula:
        Marginal_PD(t) = Cumulative_PD(t) - Cumulative_PD(t-1)

    Args:
        cumulative_pds: Array of cumulative PDs.

    Returns:
        Array of marginal PDs for each period.
    """
    marginal = np.diff(cumulative_pds, prepend=0.0)
    return np.maximum(marginal, 0.0)


def survival_probabilities(cumulative_pds: np.ndarray) -> np.ndarray:
    """Calculate survival probabilities from cumulative PDs.

    Formula:
        S(t) = 1 - Cumulative_PD(t)

    Args:
        cumulative_pds: Array of cumulative PDs.

    Returns:
        Array of survival probabilities.
    """
    return 1.0 - cumulative_pds


def flat_pd_term_structure(
    annual_pd: float,
    horizon_years: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a flat (constant hazard) PD term structure.

    Assumes constant annual conditional PD over the horizon.

    Args:
        annual_pd: Annual conditional PD (constant).
        horizon_years: Number of years for the term structure.

    Returns:
        Tuple of (cumulative_pds, marginal_pds).
    """
    annual_pds = [annual_pd] * horizon_years
    cum_pds = cumulative_pd_from_annual(annual_pds)
    marg_pds = marginal_pd_from_cumulative(cum_pds)
    return cum_pds, marg_pds


def lifetime_pd_from_rating_transitions(
    transition_matrix: np.ndarray,
    initial_rating: int,
    default_state: int,
    horizon_years: int,
) -> np.ndarray:
    """Compute lifetime cumulative PD from a rating transition matrix.

    Uses matrix exponentiation: P(t) = TM^t.
    Cumulative PD(t) = P(t)[initial_rating, default_state].

    Args:
        transition_matrix: Annual rating transition matrix (NxN).
        initial_rating: Index of the initial rating grade.
        default_state: Index of the default (absorbing) state.
        horizon_years: Number of years.

    Returns:
        Array of cumulative PDs for each year.
    """
    cum_pds = np.zeros(horizon_years)
    tm_power = np.eye(transition_matrix.shape[0])

    for t in range(horizon_years):
        tm_power = tm_power @ transition_matrix
        cum_pds[t] = tm_power[initial_rating, default_state]

    return cum_pds
