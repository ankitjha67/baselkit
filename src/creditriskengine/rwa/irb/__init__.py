"""IRB (Internal Ratings-Based) approach for credit risk RWA.

Implements Foundation IRB and Advanced IRB calculators per BCBS d424,
CRE31-32.
"""

from creditriskengine.rwa.irb.advanced import AdvancedIRBCalculator
from creditriskengine.rwa.irb.formulas import (
    irb_capital_requirement_k,
    irb_risk_weight,
    maturity_adjustment,
)
from creditriskengine.rwa.irb.foundation import FoundationIRBCalculator

__all__ = [
    "FoundationIRBCalculator",
    "AdvancedIRBCalculator",
    "irb_risk_weight",
    "irb_capital_requirement_k",
    "maturity_adjustment",
]
