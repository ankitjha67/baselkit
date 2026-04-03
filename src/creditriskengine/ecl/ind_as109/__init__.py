"""Ind AS 109 expected credit loss framework (Indian GAAP, converged with IFRS 9).

Wraps IFRS 9 ECL functions with India-specific defaults (RBI norms),
including IRAC asset classification, provisioning floors, and
restructured account handling.
"""

from creditriskengine.ecl.ind_as109.ind_as_ecl import (
    IRACAssetClass,
    assign_stage_ind_as,
    calculate_ecl_ind_as,
    classify_irac,
    irac_to_ifrs9_stage,
    rbi_minimum_provision,
    restructured_account_stage,
)

__all__ = [
    "assign_stage_ind_as",
    "calculate_ecl_ind_as",
    "IRACAssetClass",
    "classify_irac",
    "irac_to_ifrs9_stage",
    "rbi_minimum_provision",
    "restructured_account_stage",
]
