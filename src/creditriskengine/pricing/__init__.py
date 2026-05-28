"""Risk-adjusted pricing and capital allocation.

Modules:
    raroc      : Risk-Adjusted Return on Capital and Economic Value Added.
    loan_pricing : Risk-based loan pricing and break-even spread.
    allocation : Portfolio capital allocation (Euler, ES-contribution).

References:
    - McKinsey / industry RAROC frameworks.
    - Tasche (2008) — capital allocation and Euler principle.
    - BCBS economic capital range-of-practices (BCBS 152).
"""

from creditriskengine.pricing.allocation import (
    euler_var_contributions,
    expected_shortfall_contributions,
    marginal_contributions,
)
from creditriskengine.pricing.loan_pricing import (
    break_even_spread,
    risk_based_loan_rate,
)
from creditriskengine.pricing.raroc import (
    economic_value_added,
    raroc,
    raroc_hurdle_check,
)

__all__ = [
    "raroc",
    "economic_value_added",
    "raroc_hurdle_check",
    "break_even_spread",
    "risk_based_loan_rate",
    "euler_var_contributions",
    "expected_shortfall_contributions",
    "marginal_contributions",
]
