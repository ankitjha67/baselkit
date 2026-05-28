"""
RAROC and Economic Value Added (EVA).

Reference:
    - Zaik, Walter, Kelling & James (1996) — RAROC at Bank of America.
    - McKinsey — economic capital in performance management.
    - BCBS 152 — range of practices in economic capital frameworks.

RAROC = (revenue - expected loss - operating cost + capital benefit)
        / economic capital

A facility creates value when RAROC exceeds the hurdle rate (the cost
of equity). EVA expresses the same idea in currency:
    EVA = risk-adjusted return - hurdle_rate * economic_capital
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RAROCResult:
    """RAROC calculation result.

    Attributes:
        risk_adjusted_return: Numerator (revenue - EL - opex + capital benefit).
        economic_capital: Denominator (economic/regulatory capital).
        raroc: Risk-adjusted return on capital (decimal).
        expected_loss: Expected loss component (PD * LGD * EAD).
    """

    risk_adjusted_return: float
    economic_capital: float
    raroc: float
    expected_loss: float


def raroc(
    revenue: float,
    pd: float,
    lgd: float,
    ead: float,
    economic_capital: float,
    operating_cost: float = 0.0,
    capital_benefit_rate: float = 0.0,
) -> RAROCResult:
    """Compute Risk-Adjusted Return on Capital.

    RAROC = (revenue - EL - opex + capital_benefit) / economic_capital
    EL    = PD * LGD * EAD
    capital_benefit = capital_benefit_rate * economic_capital
        (return earned on the capital held against the exposure)

    Args:
        revenue: Net interest + fee revenue from the facility.
        pd: Probability of default (1-year).
        lgd: Loss given default.
        ead: Exposure at default.
        economic_capital: Economic (or regulatory) capital allocated.
        operating_cost: Allocated operating cost.
        capital_benefit_rate: Risk-free return earned on held capital.

    Returns:
        :class:`RAROCResult`.

    Raises:
        ValueError: If economic_capital is non-positive.

    Reference:
        Zaik et al. (1996), McKinsey.
    """
    if economic_capital <= 0:
        raise ValueError("economic_capital must be positive")

    expected_loss = pd * lgd * ead
    capital_benefit = capital_benefit_rate * economic_capital
    risk_adjusted_return = (
        revenue - expected_loss - operating_cost + capital_benefit
    )
    value = risk_adjusted_return / economic_capital

    return RAROCResult(
        risk_adjusted_return=risk_adjusted_return,
        economic_capital=economic_capital,
        raroc=value,
        expected_loss=expected_loss,
    )


def economic_value_added(
    raroc_result: RAROCResult,
    hurdle_rate: float,
) -> float:
    """Economic Value Added (EVA) from a RAROC result.

    EVA = risk_adjusted_return - hurdle_rate * economic_capital

    Positive EVA means the facility earns above its cost of capital.

    Args:
        raroc_result: A computed :class:`RAROCResult`.
        hurdle_rate: Cost of equity / hurdle rate (decimal).

    Returns:
        EVA in currency units.
    """
    return (
        raroc_result.risk_adjusted_return
        - hurdle_rate * raroc_result.economic_capital
    )


def raroc_hurdle_check(
    raroc_result: RAROCResult,
    hurdle_rate: float,
) -> bool:
    """Check whether RAROC clears the hurdle rate.

    Args:
        raroc_result: A computed :class:`RAROCResult`.
        hurdle_rate: Cost of equity / hurdle rate (decimal).

    Returns:
        ``True`` if RAROC >= hurdle_rate (value-creating).
    """
    return raroc_result.raroc >= hurdle_rate
