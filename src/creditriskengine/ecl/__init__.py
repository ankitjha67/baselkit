"""Expected Credit Loss (ECL) calculation package.

Supports IFRS 9, US CECL (ASC 326), and Ind AS 109 frameworks.
"""

from creditriskengine.ecl.cecl import cecl_loss_rate, cecl_pd_lgd
from creditriskengine.ecl.ifrs9 import assign_stage, calculate_ecl
from creditriskengine.ecl.ind_as109 import assign_stage_ind_as, calculate_ecl_ind_as

__all__ = [
    "calculate_ecl",
    "assign_stage",
    "cecl_pd_lgd",
    "cecl_loss_rate",
    "assign_stage_ind_as",
    "calculate_ecl_ind_as",
]
