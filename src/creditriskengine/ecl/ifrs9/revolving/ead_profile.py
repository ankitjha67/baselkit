"""Revolving EAD term structure with drawn/undrawn decomposition.

Unlike amortizing loans where EAD monotonically decreases, revolving
facilities allow re-draws and balance fluctuations.  This module
generates separate drawn and undrawn EAD arrays for use in the ECL
formula and for IFRS 7 B8E presentation (loss allowance vs. provision).

References:
    - IFRS 9 paragraph B5.5.31 (consistency with drawdown expectations)
    - IFRS 7 paragraph B8E (drawn/undrawn provisioning split)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RevolvingEADProfile:
    """Period-by-period EAD decomposition for a revolving facility.

    Attributes:
        drawn: Drawn balance by period (in currency millions).
        undrawn: Undrawn commitment by period.
        ead: Total EAD by period (drawn + CCF x undrawn).
        ccf: The CCF applied to the undrawn component.
        limit: Credit limit (drawn + undrawn at period 0).
    """

    drawn: np.ndarray
    undrawn: np.ndarray
    ead: np.ndarray
    ccf: float
    limit: float


def revolving_ead_term_structure(
    drawn: float,
    undrawn: float,
    ccf: float,
    n_periods: int,
    repayment_rate: float = 0.0,
    redraw_rate: float = 0.0,
) -> RevolvingEADProfile:
    """Generate a revolving EAD term structure.

    Models the drawn balance dynamics of a revolving facility where
    the borrower may partially repay and re-draw within the credit limit.

    The net balance change per period is:

        Drawn(t) = Drawn(t-1) × (1 - repayment_rate + redraw_rate)

    capped at the credit limit.  The undrawn portion adjusts accordingly.

    Args:
        drawn: Current drawn balance.
        undrawn: Current undrawn commitment.
        ccf: Credit conversion factor for the undrawn portion.
        n_periods: Number of periods (months or years) to project.
        repayment_rate: Fraction of drawn balance repaid per period
            (e.g., 0.03 = 3% monthly repayment).
        redraw_rate: Fraction of drawn balance re-drawn per period
            (e.g., 0.01 = 1% monthly re-draw).  Must be less than
            or equal to repayment_rate for a net-repaying profile.

    Returns:
        :class:`RevolvingEADProfile` with per-period arrays.

    Raises:
        ValueError: If inputs are negative or n_periods < 1.
    """
    if drawn < 0 or undrawn < 0 or ccf < 0:
        raise ValueError("drawn, undrawn, and ccf must be non-negative.")
    if n_periods < 1:
        raise ValueError("n_periods must be at least 1.")

    limit = drawn + undrawn
    net_factor = 1.0 - repayment_rate + redraw_rate

    drawn_arr = np.empty(n_periods)
    undrawn_arr = np.empty(n_periods)

    current_drawn = drawn
    for t in range(n_periods):
        current_drawn = max(0.0, min(current_drawn * net_factor, limit))
        drawn_arr[t] = current_drawn
        undrawn_arr[t] = limit - current_drawn

    ead_arr = drawn_arr + ccf * undrawn_arr

    return RevolvingEADProfile(
        drawn=drawn_arr,
        undrawn=undrawn_arr,
        ead=ead_arr,
        ccf=ccf,
        limit=limit,
    )


def ead_drawn_undrawn_split(
    drawn: float,
    undrawn: float,
    ccf: float,
    n_periods: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return constant drawn and CCF-weighted undrawn EAD arrays.

    A simplified profile where balances remain flat across periods
    (no repayment/redraw dynamics).  Useful for quick ECL estimates
    and when detailed balance projections are unavailable.

    Args:
        drawn: Current drawn balance.
        undrawn: Current undrawn commitment.
        ccf: Credit conversion factor.
        n_periods: Number of projection periods.

    Returns:
        Tuple of (drawn_ead_array, undrawn_ead_array), each of length
        ``n_periods``.  Total EAD = drawn + undrawn at each period.
    """
    drawn_arr = np.full(n_periods, drawn, dtype=float)
    undrawn_arr = np.full(n_periods, ccf * undrawn, dtype=float)
    return drawn_arr, undrawn_arr
