"""
Roll-rate / Markov delinquency-bucket loss forecasting for retail.

Reference:
    - Thomas, Edelman & Crook (2002) — Credit Scoring and Its Applications.
    - SAS / Oracle multi-state Markov IFRS 9 methodologies.
    - Basel retail loss-forecasting practice (card / personal loans).

Models the flow of accounts across delinquency buckets
(Current → 30 → 60 → 90 → 120+ → Charge-off) using a Markov
transition matrix, and projects expected charge-off losses.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

logger = logging.getLogger(__name__)


class DelinquencyBucket(IntEnum):
    """Standard retail delinquency buckets (absorbing: CHARGE_OFF)."""

    CURRENT = 0
    DPD_30 = 1
    DPD_60 = 2
    DPD_90 = 3
    DPD_120_PLUS = 4
    CHARGE_OFF = 5


N_BUCKETS: int = len(DelinquencyBucket)


def roll_rate_matrix(
    flow_counts: np.ndarray,
) -> np.ndarray:
    """Estimate the roll-rate transition matrix from observed flows.

    Each row is normalised to sum to 1.0 (transition probabilities from
    a given bucket). CHARGE_OFF is absorbing — its row is forced to a
    self-transition.

    Args:
        flow_counts: (N_BUCKETS, N_BUCKETS) matrix of observed account
            counts flowing from bucket i (row) to bucket j (column).

    Returns:
        Row-stochastic transition matrix (N_BUCKETS, N_BUCKETS).

    Raises:
        ValueError: If the matrix is not (N_BUCKETS, N_BUCKETS).
    """
    flow_counts = np.asarray(flow_counts, dtype=np.float64)
    if flow_counts.shape != (N_BUCKETS, N_BUCKETS):
        raise ValueError(
            f"flow_counts must be ({N_BUCKETS}, {N_BUCKETS}), "
            f"got {flow_counts.shape}"
        )

    matrix = np.zeros((N_BUCKETS, N_BUCKETS))
    for i in range(N_BUCKETS):
        row_sum = flow_counts[i].sum()
        if row_sum > 0:
            matrix[i] = flow_counts[i] / row_sum
        else:
            matrix[i, i] = 1.0  # No observed flow → stay

    # Charge-off is absorbing
    matrix[DelinquencyBucket.CHARGE_OFF] = 0.0
    matrix[DelinquencyBucket.CHARGE_OFF, DelinquencyBucket.CHARGE_OFF] = 1.0

    return matrix


@dataclass(frozen=True)
class RollRateResult:
    """Result of a multi-period roll-rate projection.

    Attributes:
        balances_by_period: (n_periods+1, N_BUCKETS) matrix of projected
            balances per bucket at each period (row 0 = initial).
        cumulative_charge_off: Cumulative charge-off balance at the end.
        charge_off_rate: Cumulative charge-off / initial total balance.
    """

    balances_by_period: np.ndarray
    cumulative_charge_off: float
    charge_off_rate: float


def project_charge_off(
    initial_balances: np.ndarray,
    transition_matrix: np.ndarray,
    n_periods: int,
) -> RollRateResult:
    """Project balances and charge-off over multiple periods.

    Applies the Markov transition matrix repeatedly:
        b_{t+1} = b_t @ P

    Args:
        initial_balances: Initial balance per bucket (N_BUCKETS,).
        transition_matrix: Row-stochastic roll-rate matrix.
        n_periods: Number of periods to project.

    Returns:
        :class:`RollRateResult`.

    Raises:
        ValueError: If dimensions are inconsistent or n_periods < 1.
    """
    initial_balances = np.asarray(initial_balances, dtype=np.float64)
    if initial_balances.shape != (N_BUCKETS,):
        raise ValueError(f"initial_balances must be ({N_BUCKETS},)")
    if transition_matrix.shape != (N_BUCKETS, N_BUCKETS):
        raise ValueError(f"transition_matrix must be ({N_BUCKETS}, {N_BUCKETS})")
    if n_periods < 1:
        raise ValueError("n_periods must be >= 1")

    total_initial = initial_balances.sum()
    balances = np.zeros((n_periods + 1, N_BUCKETS))
    balances[0] = initial_balances

    for t in range(1, n_periods + 1):
        balances[t] = balances[t - 1] @ transition_matrix

    cumulative_co = float(balances[-1, DelinquencyBucket.CHARGE_OFF])
    co_rate = cumulative_co / total_initial if total_initial > 0 else 0.0

    return RollRateResult(
        balances_by_period=balances,
        cumulative_charge_off=cumulative_co,
        charge_off_rate=co_rate,
    )
