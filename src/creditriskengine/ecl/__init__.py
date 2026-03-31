"""Expected Credit Loss (ECL) calculation package.

Supports IFRS 9, US CECL (ASC 326), Ind AS 109, and revolving credit
ECL with behavioral life, CCF models, and drawn/undrawn decomposition.
"""

from creditriskengine.ecl.cecl import cecl_loss_rate, cecl_pd_lgd
from creditriskengine.ecl.ifrs9 import assign_stage, calculate_ecl
from creditriskengine.ecl.ifrs9.revolving import (
    RevolvingECLResult,
    calculate_revolving_ecl,
)
from creditriskengine.ecl.ind_as109 import assign_stage_ind_as, calculate_ecl_ind_as

__all__ = [
    "calculate_ecl",
    "assign_stage",
    "cecl_pd_lgd",
    "cecl_loss_rate",
    "assign_stage_ind_as",
    "calculate_ecl_ind_as",
    # Revolving credit
    "calculate_revolving_ecl",
    "RevolvingECLResult",
]
