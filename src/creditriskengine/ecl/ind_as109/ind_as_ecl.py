"""
Ind AS 109 ECL calculation — Indian Accounting Standard 109.

Ind AS 109 is fully converged with IFRS 9 for impairment.
The ECL methodology is identical to IFRS 9.

Key India-specific elements:
- Default definition: 90 DPD for ALL categories (RBI Master Circular)
- No additional unlikeliness-to-pay indicators for most banks
- RBI guidelines on Ind AS provisioning per RBI/2019-20/170

This module wraps the IFRS 9 ECL functions with India-specific defaults.
"""

import logging
from typing import Optional

import numpy as np

from creditriskengine.core.types import IFRS9Stage
from creditriskengine.ecl.ifrs9.ecl_calc import calculate_ecl, ecl_12_month, ecl_lifetime
from creditriskengine.ecl.ifrs9.staging import assign_stage

logger = logging.getLogger(__name__)

# RBI default: 90 DPD for all categories
RBI_DEFAULT_DPD_THRESHOLD: int = 90

# RBI SICR backstop: 30 DPD (aligned with IFRS 9)
RBI_SICR_DPD_BACKSTOP: int = 30


def assign_stage_ind_as(
    days_past_due: int,
    is_npa: bool = False,
    is_poci: bool = False,
    sicr_triggered: bool = False,
) -> IFRS9Stage:
    """Assign Ind AS 109 impairment stage with RBI-specific defaults.

    RBI classifies assets as NPA (Non-Performing Asset) at 90 DPD.
    NPA maps to Stage 3 (credit-impaired).

    Args:
        days_past_due: Days past due.
        is_npa: Whether classified as NPA per RBI IRAC norms.
        is_poci: Whether purchased/originated credit-impaired.
        sicr_triggered: Whether SICR has been triggered.

    Returns:
        IFRS9Stage.
    """
    is_defaulted = is_npa or days_past_due >= RBI_DEFAULT_DPD_THRESHOLD
    return assign_stage(
        days_past_due=days_past_due,
        is_credit_impaired=is_defaulted,
        is_defaulted=is_defaulted,
        is_poci=is_poci,
        sicr_triggered=sicr_triggered,
        dpd_backstop=RBI_SICR_DPD_BACKSTOP,
    )


def calculate_ecl_ind_as(
    stage: IFRS9Stage,
    pd_12m: float,
    lgd: float,
    ead: float,
    eir: float = 0.0,
    marginal_pds: Optional[np.ndarray] = None,
    lgd_curve: Optional[np.ndarray] = None,
    ead_curve: Optional[np.ndarray] = None,
) -> float:
    """Calculate ECL per Ind AS 109 (delegates to IFRS 9 calculation).

    Args:
        stage: Ind AS 109 impairment stage (same as IFRS 9).
        pd_12m: 12-month PD.
        lgd: Loss given default.
        ead: Exposure at default.
        eir: Effective interest rate.
        marginal_pds: Marginal PD curve for lifetime ECL.
        lgd_curve: Optional LGD term structure.
        ead_curve: Optional EAD term structure.

    Returns:
        ECL amount.
    """
    return calculate_ecl(
        stage=stage,
        pd_12m=pd_12m,
        lgd=lgd,
        ead=ead,
        eir=eir,
        marginal_pds=marginal_pds,
        lgd_curve=lgd_curve,
        ead_curve=ead_curve,
    )
