"""Ind AS 109 expected credit loss framework (Indian GAAP, converged with IFRS 9).

Wraps IFRS 9 ECL functions with India-specific defaults (RBI norms).
"""

from creditriskengine.ecl.ind_as109.ind_as_ecl import (
    assign_stage_ind_as,
    calculate_ecl_ind_as,
)

__all__ = [
    "assign_stage_ind_as",
    "calculate_ecl_ind_as",
]
