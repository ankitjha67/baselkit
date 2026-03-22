"""IFRS 9 expected credit loss framework.

Provides staging, SICR assessment, lifetime PD term structures,
scenario weighting, and 12-month/lifetime ECL calculation.
"""

from creditriskengine.ecl.ifrs9.ecl_calc import calculate_ecl, ecl_12_month, ecl_lifetime
from creditriskengine.ecl.ifrs9.lifetime_pd import (
    cumulative_pd_from_annual,
    flat_pd_term_structure,
    marginal_pd_from_cumulative,
)
from creditriskengine.ecl.ifrs9.scenarios import Scenario, weighted_ecl
from creditriskengine.ecl.ifrs9.sicr import assess_sicr
from creditriskengine.ecl.ifrs9.staging import assign_stage
from creditriskengine.ecl.ifrs9.ttc_to_pit import ttc_to_pit_pd

__all__ = [
    "calculate_ecl",
    "ecl_12_month",
    "ecl_lifetime",
    "assign_stage",
    "assess_sicr",
    "cumulative_pd_from_annual",
    "marginal_pd_from_cumulative",
    "flat_pd_term_structure",
    "ttc_to_pit_pd",
    "Scenario",
    "weighted_ecl",
]
