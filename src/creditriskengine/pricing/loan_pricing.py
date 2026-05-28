"""
Risk-based loan pricing.

Reference:
    - McKinsey / industry RAROC-based loan pricing.
    - Cost-plus pricing with expected loss, capital, and funding.

The break-even spread covers expected loss, the cost of holding
economic capital at the hurdle rate, operating costs, and funding,
net of the return earned on held capital. The risk-based rate adds
the cost of funds to the break-even spread.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def break_even_spread(
    pd: float,
    lgd: float,
    ead: float,
    economic_capital: float,
    hurdle_rate: float,
    operating_cost: float = 0.0,
    capital_benefit_rate: float = 0.0,
) -> float:
    """Break-even spread (over cost of funds) for a facility.

    Required pre-funding revenue to achieve RAROC = hurdle_rate:
        revenue* = EL + opex + hurdle_rate * EC - capital_benefit
    Spread = revenue* / EAD

    Args:
        pd: Probability of default.
        lgd: Loss given default.
        ead: Exposure at default.
        economic_capital: Economic capital allocated.
        hurdle_rate: Cost of equity / hurdle rate (decimal).
        operating_cost: Allocated operating cost (currency).
        capital_benefit_rate: Return earned on held capital.

    Returns:
        Break-even spread as a decimal of EAD.

    Raises:
        ValueError: If EAD is non-positive.

    Reference:
        McKinsey RAROC-based pricing.
    """
    if ead <= 0:
        raise ValueError("ead must be positive")

    expected_loss = pd * lgd * ead
    capital_cost = hurdle_rate * economic_capital
    capital_benefit = capital_benefit_rate * economic_capital

    required_revenue = expected_loss + operating_cost + capital_cost - capital_benefit
    return required_revenue / ead


def risk_based_loan_rate(
    pd: float,
    lgd: float,
    ead: float,
    economic_capital: float,
    cost_of_funds: float,
    hurdle_rate: float,
    operating_cost: float = 0.0,
    capital_benefit_rate: float = 0.0,
) -> float:
    """All-in risk-based loan rate.

    rate = cost_of_funds + break_even_spread

    Args:
        pd: Probability of default.
        lgd: Loss given default.
        ead: Exposure at default.
        economic_capital: Economic capital allocated.
        cost_of_funds: Cost of funds (decimal).
        hurdle_rate: Cost of equity / hurdle rate.
        operating_cost: Allocated operating cost (currency).
        capital_benefit_rate: Return earned on held capital.

    Returns:
        All-in loan rate as a decimal.
    """
    spread = break_even_spread(
        pd=pd,
        lgd=lgd,
        ead=ead,
        economic_capital=economic_capital,
        hurdle_rate=hurdle_rate,
        operating_cost=operating_cost,
        capital_benefit_rate=capital_benefit_rate,
    )
    return cost_of_funds + spread
