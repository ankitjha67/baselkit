"""Market risk — FRTB integration point.

Provides the interface and simplified calculations for the Fundamental
Review of the Trading Book (FRTB) per BCBS d424, MAR.

This is an **integration point**, not a full FRTB implementation.
It covers:

- Sensitivities-Based Method (SbM) capital for credit spread risk
- Default Risk Charge (DRC) — simplified
- Residual Risk Add-On (RRAO)
- Aggregation into total market risk capital

SbM formula (MAR21)::

    K = sqrt(sum_i(WS_i^2) + 2 * sum_{i<j}(rho_ij * WS_i * WS_j))

where WS_i = risk_weight_i * sensitivity_i.

DRC simplified (MAR22)::

    DRC = sum(LGD * notional * risk_weight)

RRAO (MAR23)::

    RRAO = 1% of gross notional for exotic instruments
         + 0.1% of gross notional for other instruments bearing
           residual risk.
"""

import logging
import math
from enum import Enum
from typing import Final

logger = logging.getLogger(__name__)


# ============================================================
# FRTB approach enum
# ============================================================

class FRTBApproach(Enum):
    """FRTB approach for market risk capital."""

    STANDARDISED = "standardised"
    IMA = "internal_models_approach"


# ============================================================
# RRAO rates — MAR23
# ============================================================

RRAO_EXOTIC_RATE: Final[float] = 0.01   # 1 %
RRAO_OTHER_RATE: Final[float] = 0.001   # 0.1 %


# ============================================================
# Sensitivities-Based Method — credit spread risk
# ============================================================

def calculate_sbm_credit_spread(
    sensitivities: list[float],
    risk_weights: list[float],
    correlations: list[list[float]],
) -> float:
    """Compute SbM capital charge for credit spread risk.

    Implements the aggregation formula (MAR21)::

        K = sqrt(sum_i WS_i^2  +  2 * sum_{i<j} rho_ij * WS_i * WS_j)

    where ``WS_i = risk_weight_i * sensitivity_i``.

    Args:
        sensitivities: Net sensitivities per risk factor.
        risk_weights: Corresponding risk weights (same length).
        correlations: Square correlation matrix of size ``n x n``
            where ``n = len(sensitivities)``.  Diagonal entries are
            ignored (implicitly 1).

    Returns:
        SbM capital charge (non-negative).

    Raises:
        ValueError: On dimension mismatches or empty inputs.
    """
    n = len(sensitivities)
    if n == 0:
        raise ValueError("sensitivities must not be empty")
    if len(risk_weights) != n:
        raise ValueError(
            f"risk_weights length ({len(risk_weights)}) must match "
            f"sensitivities length ({n})"
        )
    if len(correlations) != n:
        raise ValueError(
            f"correlations must be {n}x{n}; got {len(correlations)} rows"
        )
    for idx, row in enumerate(correlations):
        if len(row) != n:
            raise ValueError(
                f"correlations row {idx} has length {len(row)}; expected {n}"
            )

    # Weighted sensitivities
    ws = [s * rw for s, rw in zip(sensitivities, risk_weights)]

    # Variance-covariance aggregation
    total = 0.0
    for i in range(n):
        total += ws[i] ** 2
        for j in range(i + 1, n):
            total += 2.0 * correlations[i][j] * ws[i] * ws[j]

    # Floor at zero before sqrt (can go negative with anti-correlations)
    k = math.sqrt(max(total, 0.0))

    logger.debug(
        "SbM credit spread capital = %.2f  (%d risk factors)", k, n,
    )
    return k


# ============================================================
# Default Risk Charge — simplified
# ============================================================

def calculate_drc(
    lgds: list[float],
    notionals: list[float],
    risk_weights: list[float],
) -> float:
    """Simplified Default Risk Charge (MAR22).

    DRC = sum_i(LGD_i * notional_i * risk_weight_i)

    Args:
        lgds: Loss-given-default per position (as decimals, e.g. 0.6).
        notionals: Notional amounts per position.
        risk_weights: DRC risk weights per position.

    Returns:
        Total DRC (non-negative).

    Raises:
        ValueError: On dimension mismatches or empty inputs.
    """
    n = len(lgds)
    if n == 0:
        raise ValueError("lgds must not be empty")
    if len(notionals) != n or len(risk_weights) != n:
        raise ValueError(
            f"All input lists must have the same length ({n}); got "
            f"notionals={len(notionals)}, risk_weights={len(risk_weights)}"
        )

    drc = sum(
        lgd * notional * rw
        for lgd, notional, rw in zip(lgds, notionals, risk_weights)
    )

    logger.debug("DRC = %.2f  (%d positions)", drc, n)
    return drc


# ============================================================
# Residual Risk Add-On
# ============================================================

def calculate_rrao(
    exotic_gross_notional: float = 0.0,
    other_gross_notional: float = 0.0,
) -> float:
    """Residual Risk Add-On per MAR23.

    RRAO = 1 % of exotic gross notional + 0.1 % of other gross notional.

    Args:
        exotic_gross_notional: Gross notional of exotic instruments.
        other_gross_notional: Gross notional of other instruments
            bearing residual risk.

    Returns:
        RRAO charge (non-negative).

    Raises:
        ValueError: If notionals are negative.
    """
    if exotic_gross_notional < 0:
        raise ValueError(
            f"exotic_gross_notional must be non-negative; got "
            f"{exotic_gross_notional}"
        )
    if other_gross_notional < 0:
        raise ValueError(
            f"other_gross_notional must be non-negative; got "
            f"{other_gross_notional}"
        )

    rrao = (
        RRAO_EXOTIC_RATE * exotic_gross_notional
        + RRAO_OTHER_RATE * other_gross_notional
    )
    logger.debug(
        "RRAO = %.2f  (exotic=%.2f, other=%.2f)",
        rrao, exotic_gross_notional, other_gross_notional,
    )
    return rrao


# ============================================================
# SA market risk — combined
# ============================================================

def calculate_sa_market_risk(
    credit_spread_sensitivities: list[float],
    risk_weights: list[float],
    correlations: list[list[float]],
    drc_lgds: list[float] | None = None,
    drc_notionals: list[float] | None = None,
    drc_risk_weights: list[float] | None = None,
    exotic_gross_notional: float = 0.0,
    other_gross_notional: float = 0.0,
) -> dict:
    """Calculate total SA market risk capital.

    Combines SbM (credit spread risk), DRC, and RRAO into a single
    result dictionary.

    Args:
        credit_spread_sensitivities: Sensitivities for SbM.
        risk_weights: Risk weights for SbM.
        correlations: Correlation matrix for SbM.
        drc_lgds: LGDs for DRC (optional).
        drc_notionals: Notionals for DRC (optional).
        drc_risk_weights: Risk weights for DRC (optional).
        exotic_gross_notional: Gross notional of exotics for RRAO.
        other_gross_notional: Gross notional of other for RRAO.

    Returns:
        Dictionary with ``sbm``, ``drc``, ``rrao``, ``total`` keys.
    """
    sbm = calculate_sbm_credit_spread(
        credit_spread_sensitivities, risk_weights, correlations,
    )

    drc = 0.0
    if drc_lgds is not None and drc_notionals is not None and drc_risk_weights is not None:
        drc = calculate_drc(drc_lgds, drc_notionals, drc_risk_weights)

    rrao = calculate_rrao(exotic_gross_notional, other_gross_notional)

    result = total_market_risk_capital(sbm, drc, rrao)
    return result


# ============================================================
# Total market risk capital
# ============================================================

def total_market_risk_capital(
    sbm: float,
    drc: float,
    rrao: float,
) -> dict:
    """Aggregate market risk capital components.

    Total = SbM + DRC + RRAO

    Args:
        sbm: Sensitivities-Based Method capital.
        drc: Default Risk Charge.
        rrao: Residual Risk Add-On.

    Returns:
        Dictionary with ``sbm``, ``drc``, ``rrao``, ``total`` keys.
    """
    total = sbm + drc + rrao
    logger.info(
        "Total market risk capital = %.2f  (SbM=%.2f, DRC=%.2f, RRAO=%.2f)",
        total, sbm, drc, rrao,
    )
    return {
        "sbm": sbm,
        "drc": drc,
        "rrao": rrao,
        "total": total,
    }
