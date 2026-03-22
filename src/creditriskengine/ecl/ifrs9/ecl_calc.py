"""
IFRS 9 ECL computation — 12-month and lifetime.

12-month ECL (Stage 1):
    ECL_12m = PD_12m * LGD * EAD * DF

Lifetime ECL (Stage 2 and 3):
    ECL_lifetime = Sum(t=1..T) [Marginal_PD(t) * LGD(t) * EAD(t) * DF(t)]

Reference: IFRS 9.5.5.1-5.5.20, IFRS 9.B5.5.28-B5.5.29.
"""

import logging

import numpy as np

from creditriskengine.core.types import IFRS9Stage

logger = logging.getLogger(__name__)


def discount_factors(
    eir: float,
    periods: int,
) -> np.ndarray:
    """Calculate discount factors at the effective interest rate.

    DF(t) = 1 / (1 + EIR)^t

    Args:
        eir: Effective interest rate (annualized).
        periods: Number of periods.

    Returns:
        Array of discount factors for periods 1..T.
    """
    if eir <= -1.0:
        raise ValueError(f"EIR must be greater than -1, got {eir}")
    t = np.arange(1, periods + 1, dtype=np.float64)
    return 1.0 / (1.0 + eir) ** t


def ecl_12_month(
    pd_12m: float,
    lgd: float,
    ead: float,
    eir: float = 0.0,
) -> float:
    """Calculate 12-month ECL for Stage 1 exposures.

    Formula:
        ECL_12m = PD_12m * LGD * EAD * DF(1)

    Args:
        pd_12m: 12-month probability of default.
        lgd: Loss given default.
        ead: Exposure at default.
        eir: Effective interest rate for discounting.

    Returns:
        12-month ECL amount.
    """
    df = 1.0 / (1.0 + eir) if eir > 0 else 1.0
    ecl = pd_12m * lgd * ead * df
    logger.debug(
        "12m ECL: PD=%.4f LGD=%.2f EAD=%.2f DF=%.4f ECL=%.2f",
        pd_12m, lgd, ead, df, ecl,
    )
    return ecl


def ecl_lifetime(
    marginal_pds: np.ndarray,
    lgds: np.ndarray | float,
    eads: np.ndarray | float,
    eir: float = 0.0,
) -> float:
    """Calculate lifetime ECL for Stage 2/3 exposures.

    Formula:
        ECL = Sum(t=1..T) [Marginal_PD(t) * LGD(t) * EAD(t) * DF(t)]

    Args:
        marginal_pds: Array of marginal PDs for each period.
        lgds: LGD values (scalar or array per period).
        eads: EAD values (scalar or array per period).
        eir: Effective interest rate for discounting.

    Returns:
        Lifetime ECL amount.
    """
    periods = len(marginal_pds)
    dfs = discount_factors(eir, periods)

    if isinstance(lgds, (int, float)):
        lgds = np.full(periods, lgds)
    if isinstance(eads, (int, float)):
        eads = np.full(periods, eads)

    ecl = float(np.sum(marginal_pds * lgds * eads * dfs))
    logger.debug("Lifetime ECL: periods=%d ECL=%.2f", periods, ecl)
    return ecl


def calculate_ecl(
    stage: IFRS9Stage,
    pd_12m: float,
    lgd: float,
    ead: float,
    eir: float = 0.0,
    marginal_pds: np.ndarray | None = None,
    lgd_curve: np.ndarray | None = None,
    ead_curve: np.ndarray | None = None,
) -> float:
    """Unified ECL calculation dispatcher based on IFRS 9 stage.

    Stage 1: 12-month ECL
    Stage 2/3/POCI: Lifetime ECL

    Args:
        stage: IFRS 9 impairment stage.
        pd_12m: 12-month PD.
        lgd: Loss given default (scalar).
        ead: Exposure at default (scalar).
        eir: Effective interest rate.
        marginal_pds: Marginal PD curve (required for lifetime ECL).
        lgd_curve: Optional LGD term structure.
        ead_curve: Optional EAD term structure.

    Returns:
        ECL amount.
    """
    if stage == IFRS9Stage.STAGE_1:
        return ecl_12_month(pd_12m, lgd, ead, eir)

    # Stage 2, 3, POCI: lifetime ECL
    if marginal_pds is None:
        raise ValueError("marginal_pds required for lifetime ECL (Stage 2/3/POCI)")

    lgd_input = lgd_curve if lgd_curve is not None else lgd
    ead_input = ead_curve if ead_curve is not None else ead
    return ecl_lifetime(marginal_pds, lgd_input, ead_input, eir)
