"""
CECL estimation methods: WARM, vintage analysis, DCF, PD/LGD.

Reference: ASC 326-20, FASB Staff Q&A, Interagency Policy Statement
on Allowances for Credit Losses (2019).
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def warm_method(
    ead: float,
    historical_loss_rate: float,
    remaining_life_years: float,
    qualitative_factor: float = 0.0,
) -> float:
    """Weighted Average Remaining Maturity (WARM) method.

    Formula:
        ECL = EAD * Loss_Rate * Remaining_Life * (1 + Q_factor)

    Simple method suitable for smaller institutions.

    Args:
        ead: Exposure at default.
        historical_loss_rate: Annualized historical loss rate.
        remaining_life_years: Weighted average remaining maturity.
        qualitative_factor: Q-factor multiplier adjustment.

    Returns:
        ECL amount.
    """
    adjusted_rate = historical_loss_rate * (1.0 + qualitative_factor)
    return ead * adjusted_rate * remaining_life_years


def vintage_analysis(
    vintage_loss_matrix: np.ndarray,
    current_balances: np.ndarray,
) -> float:
    """Vintage analysis method for CECL.

    Uses historical loss patterns by origination vintage and age
    to project remaining lifetime losses.

    Args:
        vintage_loss_matrix: Matrix of cumulative loss rates [vintage x age].
            Each row is a vintage, columns are ages.
            The last filled value per row is the current age.
        current_balances: Current outstanding balances per vintage.

    Returns:
        Total projected remaining ECL.
    """
    total_ecl = 0.0
    n_vintages = vintage_loss_matrix.shape[0]
    n_ages = vintage_loss_matrix.shape[1]

    for v in range(n_vintages):
        # Find current age (last non-zero entry)
        losses = vintage_loss_matrix[v]
        current_age = 0
        for a in range(n_ages):
            if losses[a] > 0 or a == 0:
                current_age = a

        current_cum_loss = losses[current_age]
        # Use the most mature vintage's ultimate loss rate as proxy
        ultimate_loss = np.max(vintage_loss_matrix[:, -1])
        remaining_loss = max(ultimate_loss - current_cum_loss, 0.0)
        total_ecl += current_balances[v] * remaining_loss

    return total_ecl


def dcf_method(
    contractual_cashflows: np.ndarray,
    expected_cashflows: np.ndarray,
    discount_rate: float,
) -> float:
    """Discounted Cash Flow (DCF) method for CECL.

    ECL = PV(contractual) - PV(expected)

    Args:
        contractual_cashflows: Contractual cash flows per period.
        expected_cashflows: Expected (loss-adjusted) cash flows per period.
        discount_rate: Effective interest rate.

    Returns:
        ECL amount (difference in present values).
    """
    periods = len(contractual_cashflows)
    t = np.arange(1, periods + 1, dtype=np.float64)
    dfs = 1.0 / (1.0 + discount_rate) ** t

    pv_contractual = float(np.sum(contractual_cashflows * dfs))
    pv_expected = float(np.sum(expected_cashflows * dfs))

    return max(pv_contractual - pv_expected, 0.0)
