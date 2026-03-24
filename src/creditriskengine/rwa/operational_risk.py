"""Operational risk — Standardised Measurement Approach (SMA).

Implements the Basel III SMA per BCBS d424, OPE25.

Key formula::

    ORC = BIC x ILM

where:
- BIC = Business Indicator Component (piecewise-linear function of BI)
- ILM = Internal Loss Multiplier (function of loss component and BIC)
- BI  = ILDC + SC + FC  (Business Indicator)

BIC buckets (OPE25.8-25.10):
- Bucket 1: BI <= 1bn  -> BIC = 0.12 x BI
- Bucket 2: 1bn < BI <= 30bn -> BIC = 120m + 0.15 x (BI - 1bn)
- Bucket 3: BI > 30bn -> BIC = 120m + 4350m + 0.18 x (BI - 30bn)

ILM (OPE25.11-25.13)::

    ILM = ln(exp(1) - 1 + (LC / BIC) ^ 0.8)

where LC = 15 x average annual operational risk losses over 10 years.

For jurisdictions that set ILM = 1 (e.g. EU CRR3), BIC is returned
directly as the capital charge.
"""

import logging
import math
from typing import Final

logger = logging.getLogger(__name__)

# ============================================================
# Regulatory thresholds (EUR)
# ============================================================

_BUCKET_1_LIMIT: Final[float] = 1_000_000_000.0   # EUR 1 bn
_BUCKET_2_LIMIT: Final[float] = 30_000_000_000.0   # EUR 30 bn

_ALPHA_1: Final[float] = 0.12  # Marginal coefficient, bucket 1
_ALPHA_2: Final[float] = 0.15  # Marginal coefficient, bucket 2
_ALPHA_3: Final[float] = 0.18  # Marginal coefficient, bucket 3

# Pre-computed BIC at bucket boundaries
_BIC_AT_1BN: Final[float] = _ALPHA_1 * _BUCKET_1_LIMIT          # 120 000 000
_BIC_AT_30BN: Final[float] = (
    _BIC_AT_1BN + _ALPHA_2 * (_BUCKET_2_LIMIT - _BUCKET_1_LIMIT)
)  # 120m + 4 350m = 4 470 000 000

# Loss component multiplier: LC = 15 x average annual loss
_LC_MULTIPLIER: Final[int] = 15


# ============================================================
# Business Indicator
# ============================================================

def calculate_bi(ildc: float, sc: float, fc: float) -> float:
    """Compute the Business Indicator (BI).

    BI = ILDC + SC + FC

    Each sub-component is taken as an absolute value per OPE25.5-25.7,
    but callers are expected to pass positive values representing the
    absolute-value aggregations described in the standard.

    Args:
        ildc: Interest, Lease and Dividend Component.
        sc: Services Component.
        fc: Financial Component.

    Returns:
        Business Indicator value (EUR).

    Raises:
        ValueError: If any component is negative.
    """
    if ildc < 0 or sc < 0 or fc < 0:
        raise ValueError(
            f"BI components must be non-negative; got "
            f"ildc={ildc}, sc={sc}, fc={fc}"
        )
    bi = ildc + sc + fc
    logger.debug("BI = %.2f  (ILDC=%.2f, SC=%.2f, FC=%.2f)", bi, ildc, sc, fc)
    return bi


# ============================================================
# Business Indicator Component
# ============================================================

def calculate_bic(bi: float) -> float:
    """Compute the Business Indicator Component (BIC).

    Piecewise-linear function of the Business Indicator (OPE25.8-25.10):

    - Bucket 1: BI <= 1 bn  -> 0.12 x BI
    - Bucket 2: 1 bn < BI <= 30 bn  -> 120m + 0.15 x (BI - 1 bn)
    - Bucket 3: BI > 30 bn  -> 4 470m + 0.18 x (BI - 30 bn)

    Args:
        bi: Business Indicator value (EUR).

    Returns:
        BIC value (EUR).

    Raises:
        ValueError: If *bi* is negative.
    """
    if bi < 0:
        raise ValueError(f"BI must be non-negative; got {bi}")

    if bi <= _BUCKET_1_LIMIT:
        bic = _ALPHA_1 * bi
    elif bi <= _BUCKET_2_LIMIT:
        bic = _BIC_AT_1BN + _ALPHA_2 * (bi - _BUCKET_1_LIMIT)
    else:
        bic = _BIC_AT_30BN + _ALPHA_3 * (bi - _BUCKET_2_LIMIT)

    logger.debug("BIC = %.2f for BI = %.2f", bic, bi)
    return bic


# ============================================================
# Internal Loss Multiplier
# ============================================================

def calculate_ilm(lc: float, bic: float) -> float:
    """Compute the Internal Loss Multiplier (ILM).

    Per OPE25.11-25.13::

        ILM = ln(exp(1) - 1 + (LC / BIC) ^ 0.8)

    Special cases:
    - If BIC is zero (bank with no business indicator), ILM = 1.0.
    - If LC is zero (no historical losses), the ratio term vanishes
      and ILM = ln(exp(1) - 1) ~ 0.541.

    Args:
        lc: Loss Component (15 x average annual operational risk loss).
        bic: Business Indicator Component.

    Returns:
        ILM (dimensionless multiplier).

    Raises:
        ValueError: If *lc* or *bic* is negative.
    """
    if lc < 0:
        raise ValueError(f"Loss component must be non-negative; got {lc}")
    if bic < 0:
        raise ValueError(f"BIC must be non-negative; got {bic}")

    if bic == 0.0:
        return 1.0

    ratio = lc / bic
    ilm = math.log(math.exp(1) - 1 + ratio ** 0.8)
    logger.debug(
        "ILM = %.4f  (LC=%.2f, BIC=%.2f, LC/BIC=%.4f)",
        ilm, lc, bic, ratio,
    )
    return ilm


# ============================================================
# SMA capital
# ============================================================

def sma_capital(
    bi: float,
    average_annual_loss: float | None = None,
    use_ilm: bool = True,
) -> dict:
    """Calculate operational risk capital under the SMA.

    Args:
        bi: Business Indicator value (EUR).
        average_annual_loss: Average annual operational risk loss over
            the preceding 10 years.  Required when *use_ilm* is True
            and the bank is in a jurisdiction that mandates internal
            loss data.  May be ``None`` if ILM is not used or if the
            bank opts not to supply loss data.
        use_ilm: Whether to apply the Internal Loss Multiplier.
            Set to ``False`` for jurisdictions where ILM = 1
            (e.g. EU CRR3).

    Returns:
        Dictionary with keys:

        - ``bi``: Business Indicator
        - ``bic``: Business Indicator Component
        - ``lc``: Loss Component (or ``None``)
        - ``ilm``: Internal Loss Multiplier applied (1.0 when disabled)
        - ``capital``: Operational risk capital charge (ORC)

    Raises:
        ValueError: If inputs are invalid.
    """
    bic = calculate_bic(bi)

    lc: float | None = None
    ilm = 1.0

    if use_ilm and average_annual_loss is not None:
        if average_annual_loss < 0:
            raise ValueError(
                f"average_annual_loss must be non-negative; got "
                f"{average_annual_loss}"
            )
        lc = _LC_MULTIPLIER * average_annual_loss
        ilm = calculate_ilm(lc, bic)

    capital = bic * ilm

    logger.info(
        "SMA capital = %.2f  (BI=%.2f, BIC=%.2f, ILM=%.4f, use_ilm=%s)",
        capital, bi, bic, ilm, use_ilm,
    )
    return {
        "bi": bi,
        "bic": bic,
        "lc": lc,
        "ilm": ilm,
        "capital": capital,
    }
