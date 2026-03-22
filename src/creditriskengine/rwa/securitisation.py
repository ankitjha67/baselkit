"""Securitisation risk weight approaches — BCBS d424, CRE40-44.

Implements:
- SEC-IRBA: Internal Ratings-Based Approach (CRE41)
- SEC-SA: Standardised Approach (CRE42)
- SEC-ERBA: External Ratings-Based Approach (CRE43)

Hierarchy: SEC-IRBA > SEC-ERBA > SEC-SA (CRE40.4)
"""

import math
import logging
from dataclasses import dataclass
from typing import Final, Optional

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)

# ============================================================
# Data classes
# ============================================================


@dataclass(frozen=True)
class SecuritisationTranche:
    """Represents a securitisation tranche.

    Attributes:
        tranche_id: Unique identifier.
        attachment_point: A — lower bound of loss allocation (0-1).
        detachment_point: D — upper bound of loss allocation (0-1).
        notional: Outstanding notional amount.
        external_rating: Credit quality step if externally rated.
        is_senior: Whether the tranche is the most senior.
        is_resecuritisation: Whether the tranche is a re-securitisation.
        maturity_years: Regulatory maturity (T), capped at 5 years.
    """

    tranche_id: str
    attachment_point: float
    detachment_point: float
    notional: float
    external_rating: Optional[int] = None
    is_senior: bool = False
    is_resecuritisation: bool = False
    maturity_years: float = 2.5


@dataclass(frozen=True)
class SecuritisationPool:
    """Underlying pool characteristics.

    Attributes:
        kirb: Capital requirement if pool held on balance sheet (IRB).
        ksa: Capital requirement under SA.
        pool_ead: Total EAD of the underlying pool.
        n_effective: Effective number of exposures in the pool.
        lgd_pool: Exposure-weighted average LGD of the pool.
        is_sts: EU STS (Simple, Transparent, Standardised) flag.
    """

    kirb: float
    ksa: float
    pool_ead: float
    n_effective: float
    lgd_pool: float = 0.50
    is_sts: bool = False


# ============================================================
# Supervisory parameters — CRE41.3 / CRE42.3
# ============================================================

# p parameter for KSSFA supervisory formula (CRE41.3, Table 1)
_P_SENIOR_WHOLESALE: Final[float] = 0.5
_P_NON_SENIOR_WHOLESALE: Final[float] = 1.0
_P_SENIOR_RETAIL: Final[float] = 0.5
_P_NON_SENIOR_RETAIL: Final[float] = 1.0

# Floor risk weight for SEC-IRBA and SEC-SA — CRE41.6
_RW_FLOOR: Final[float] = 0.15  # 15 %
_RW_FLOOR_STS: Final[float] = 0.10  # 10 % for STS (EU only)

# Absolute cap — CRE40.52
_RW_CAP: Final[float] = 12.50  # 1250 %

# SEC-SA parameters
_SA_W: Final[float] = 0.0  # W parameter — delinquent exposures ratio
_SA_P_FACTOR: Final[float] = 1.0


# ============================================================
# SEC-ERBA lookup tables — CRE43, Table 3
# ============================================================

SEC_ERBA_SENIOR_RW: Final[dict[int, float]] = {
    1: 0.15, 2: 0.15, 3: 0.25, 4: 0.30, 5: 0.40,
    6: 0.50, 7: 0.65, 8: 0.85, 9: 1.05, 10: 1.25,
    11: 1.70, 12: 2.50, 13: 3.50, 14: 5.00, 15: 6.50,
    16: 8.00, 17: 10.00,
}

SEC_ERBA_NON_SENIOR_RW: Final[dict[int, float]] = {
    1: 0.15, 2: 0.25, 3: 0.35, 4: 0.45, 5: 0.60,
    6: 0.80, 7: 1.00, 8: 1.20, 9: 1.40, 10: 1.70,
    11: 2.10, 12: 3.00, 13: 4.00, 14: 5.50, 15: 7.50,
    16: 9.50, 17: 12.50,
}

SEC_ERBA_NON_SENIOR_THIN_RW: Final[dict[int, float]] = {
    1: 0.25, 2: 0.35, 3: 0.50, 4: 0.65, 5: 0.85,
    6: 1.00, 7: 1.25, 8: 1.55, 9: 1.85, 10: 2.20,
    11: 2.80, 12: 3.50, 13: 4.50, 14: 6.50, 15: 8.50,
    16: 10.50, 17: 12.50,
}

# Thin tranche threshold (CRE43.4): D - A < 0.03
_THIN_TRANCHE_THRESHOLD: Final[float] = 0.03

# ERBA maturity adjustment coefficients (CRE43.5)
_ERBA_MATURITY_ADJ: Final[dict[int, tuple[float, float, float]]] = {
    # CQS: (a, b, c) where adjustment = a + b * MT + c * MT^2
    # Simplified: only the linear maturity factor from CRE43
    # Base RW * (1 + 0.4 * (MT - 1)) for non-senior short-term
}


# ============================================================
# Supervisory formula — KSSFA
# ============================================================

def _kssfa(a: float, attachment: float, detachment: float) -> float:
    """Supervisory formula KSSFA(a) per CRE41.2.

    KSSFA is the capital requirement for a tranche, computed as::

        KSSFA(a) = (exp(a * u) - exp(a * l)) / (a * (exp(a) - 1))

    where *l* and *u* are the attachment and detachment points
    adjusted for the capital requirement (``a`` parameter encodes
    the supervisory scaling).

    For the degenerate case where ``a`` is close to zero, the formula
    simplifies to ``(u - l)`` (linear interpolation).

    Args:
        a: Supervisory scaling parameter.
        attachment: Lower bound of tranche loss allocation (0-1).
        detachment: Upper bound of tranche loss allocation (0-1).

    Returns:
        Capital requirement fraction for the tranche.
    """
    if abs(a) < 1e-10:
        return detachment - attachment

    exp_a = math.exp(a)
    denominator = a * (exp_a - 1.0)
    if abs(denominator) < 1e-15:
        return detachment - attachment

    numerator = math.exp(a * detachment) - math.exp(a * attachment)
    return numerator / denominator


def _compute_p_parameter(
    kirb: float,
    lgd_pool: float,
    n_effective: float,
    is_senior: bool,
) -> float:
    """Compute supervisory parameter *p* for SEC-IRBA per CRE41.3.

    For SEC-IRBA::

        p = max(0.3, A + B * (1/N) + C * KIRB + D * LGD + E * MT)

    Simplified to the key structural form::

        p = max(0.3, 0.7 * is_senior_flag + (1 - is_senior_flag))

    The full formula uses:
        A = -(1 / (p * KIRB))

    Here we use the standard supervisory p values and adjust
    for pool granularity.

    Args:
        kirb: Pool-level IRB capital requirement.
        lgd_pool: Exposure-weighted average LGD.
        n_effective: Effective number of exposures.
        is_senior: Whether tranche is most senior.

    Returns:
        Supervisory parameter p.
    """
    if is_senior:
        p = _P_SENIOR_WHOLESALE
    else:
        p = _P_NON_SENIOR_WHOLESALE

    # Granularity adjustment — CRE41.3 footnote
    if n_effective > 0:
        granularity_adj = 1.0 / n_effective
        p = max(0.3, p * (1.0 - granularity_adj) + granularity_adj)

    return p


def _compute_a_parameter(
    p: float,
    capital_req: float,
    attachment: float,
    detachment: float,
) -> float:
    """Compute the ``a`` parameter for the KSSFA supervisory formula.

    Per CRE41.2::

        a = -(1 / (p * K))

    where K is the pool capital requirement (KIRB or KSA) and p is
    the supervisory parameter.

    Args:
        p: Supervisory parameter.
        capital_req: Pool capital requirement (KIRB or KSA).
        attachment: Tranche attachment point.
        detachment: Tranche detachment point.

    Returns:
        The ``a`` parameter.
    """
    if capital_req <= 0 or p <= 0:
        return 0.0
    return -1.0 / (p * capital_req)


# ============================================================
# SEC-IRBA — CRE41
# ============================================================

def sec_irba_risk_weight(
    tranche: SecuritisationTranche,
    pool: SecuritisationPool,
) -> float:
    """SEC-IRBA risk weight per CRE41.

    Formula::

        RW = max(floor, 12.5 * KSSFA(KIRB) / max(D - A, 1e-10))

    where:
    - KSSFA uses the supervisory formula with ``a = -(1 / (p * KIRB))``
    - Supervisory parameter ``p`` depends on seniority and pool type
    - If ``A >= KIRB``: tranche absorbs no expected loss, formula applies
    - If ``D <= KIRB``: tranche is fully below capital requirement,
      RW = 1250 %

    Reference: BCBS CRE41.1-41.6.

    Args:
        tranche: Tranche parameters.
        pool: Underlying pool parameters.

    Returns:
        Risk weight as a decimal (e.g. 0.15 for 15 %).
    """
    a_point = tranche.attachment_point
    d_point = tranche.detachment_point
    kirb = pool.kirb

    # Tranche entirely below KIRB — CRE41.4
    if d_point <= kirb:
        logger.debug(
            "Tranche '%s': D (%.4f) <= KIRB (%.4f) -> RW = 1250%%",
            tranche.tranche_id,
            d_point,
            kirb,
        )
        return _RW_CAP

    # Tranche straddles KIRB — adjust attachment to KIRB
    effective_a = max(a_point, kirb)

    # Compute supervisory parameter p
    p = _compute_p_parameter(
        kirb=kirb,
        lgd_pool=pool.lgd_pool,
        n_effective=pool.n_effective,
        is_senior=tranche.is_senior,
    )

    # Compute ``a`` parameter for KSSFA
    a_param = _compute_a_parameter(p, kirb, effective_a, d_point)

    # KSSFA capital
    k_tranche = _kssfa(a_param, effective_a, d_point)

    # Risk weight
    tranche_thickness = max(d_point - a_point, 1e-10)
    rw = 12.5 * k_tranche / tranche_thickness

    # Apply floor
    floor = _RW_FLOOR_STS if pool.is_sts else _RW_FLOOR
    rw = max(rw, floor)

    # Apply cap
    rw = min(rw, _RW_CAP)

    logger.debug(
        "SEC-IRBA tranche '%s': A=%.4f, D=%.4f, KIRB=%.4f, p=%.2f, "
        "a=%.4f, k=%.6f, RW=%.4f",
        tranche.tranche_id,
        a_point,
        d_point,
        kirb,
        p,
        a_param,
        k_tranche,
        rw,
    )
    return rw


# ============================================================
# SEC-SA — CRE42
# ============================================================

def sec_sa_risk_weight(
    tranche: SecuritisationTranche,
    pool: SecuritisationPool,
) -> float:
    """SEC-SA risk weight per CRE42.

    Uses the same KSSFA supervisory formula but with KSA instead of KIRB.

    The ``p`` parameter for SEC-SA is fixed:
    - ``p = 1.0`` for non-senior tranches
    - ``p = 0.5`` for senior tranches

    An additional delinquency parameter ``W`` increases KA::

        KA = (1 - W) * KSA + W * 0.5

    where W is the ratio of delinquent underlying exposures.

    Reference: BCBS CRE42.1-42.5.

    Args:
        tranche: Tranche parameters.
        pool: Underlying pool parameters.

    Returns:
        Risk weight as a decimal (e.g. 0.15 for 15 %).
    """
    a_point = tranche.attachment_point
    d_point = tranche.detachment_point
    ksa = pool.ksa

    # Adjusted KA for delinquencies
    # W defaults to 0 — caller should adjust pool.ksa if delinquencies exist
    ka = ksa

    # Tranche entirely below KA
    if d_point <= ka:
        logger.debug(
            "SEC-SA tranche '%s': D (%.4f) <= KSA (%.4f) -> RW = 1250%%",
            tranche.tranche_id,
            d_point,
            ka,
        )
        return _RW_CAP

    effective_a = max(a_point, ka)

    # p parameter
    p = 0.5 if tranche.is_senior else 1.0

    # Compute ``a`` parameter for KSSFA
    a_param = _compute_a_parameter(p, ka, effective_a, d_point)

    # KSSFA capital
    k_tranche = _kssfa(a_param, effective_a, d_point)

    # Risk weight
    tranche_thickness = max(d_point - a_point, 1e-10)
    rw = 12.5 * k_tranche / tranche_thickness

    # Apply floor
    floor = _RW_FLOOR_STS if pool.is_sts else _RW_FLOOR
    rw = max(rw, floor)

    # Apply cap
    rw = min(rw, _RW_CAP)

    logger.debug(
        "SEC-SA tranche '%s': A=%.4f, D=%.4f, KSA=%.4f, p=%.2f, RW=%.4f",
        tranche.tranche_id,
        a_point,
        d_point,
        ksa,
        p,
        rw,
    )
    return rw


# ============================================================
# SEC-ERBA — CRE43
# ============================================================

def sec_erba_risk_weight(tranche: SecuritisationTranche) -> float:
    """SEC-ERBA risk weight per CRE43.

    Risk weights are looked up from tables keyed by:
    - Credit quality step (external rating)
    - Seniority (senior vs non-senior)
    - Tranche thickness (thin vs thick)

    Maturity adjustment (CRE43.5): for tranches with maturity != 5 years,
    the base RW is interpolated between the 1-year and 5-year RWs.
    Simplified here as::

        RW_adjusted = RW_base * (1 + 0.4 * max(MT - 1, 0))

    capped at 5 years.

    Re-securitisation positions receive a 2x multiplier (CRE43.8).

    Reference: BCBS CRE43.1-43.8, Table 3.

    Args:
        tranche: Tranche parameters (must have ``external_rating`` set).

    Returns:
        Risk weight as a decimal.

    Raises:
        ValueError: If ``external_rating`` is not set on the tranche.
    """
    if tranche.external_rating is None:
        raise ValueError(
            f"Tranche '{tranche.tranche_id}' has no external rating; "
            "SEC-ERBA requires a rating."
        )

    cqs = tranche.external_rating
    thickness = tranche.detachment_point - tranche.attachment_point

    # Select lookup table
    if tranche.is_senior:
        table = SEC_ERBA_SENIOR_RW
    elif thickness < _THIN_TRANCHE_THRESHOLD:
        table = SEC_ERBA_NON_SENIOR_THIN_RW
    else:
        table = SEC_ERBA_NON_SENIOR_RW

    base_rw = table.get(cqs)
    if base_rw is None:
        logger.warning(
            "CQS %d not found in SEC-ERBA table for tranche '%s'; "
            "applying 1250%% cap",
            cqs,
            tranche.tranche_id,
        )
        return _RW_CAP

    # Maturity adjustment — CRE43.5
    # Linear interpolation between 1-year and 5-year base RW
    mt = min(max(tranche.maturity_years, 1.0), 5.0)
    rw = base_rw * (1.0 + 0.4 * (mt - 1.0))

    # Re-securitisation multiplier — CRE43.8
    if tranche.is_resecuritisation:
        rw *= 2.0

    # Cap
    rw = min(rw, _RW_CAP)

    logger.debug(
        "SEC-ERBA tranche '%s': CQS=%d, senior=%s, thickness=%.4f, "
        "MT=%.1f, base_rw=%.4f, final_rw=%.4f",
        tranche.tranche_id,
        cqs,
        tranche.is_senior,
        thickness,
        mt,
        base_rw,
        rw,
    )
    return rw


# ============================================================
# Risk weight cap — CRE40.52
# ============================================================

def sec_risk_weight_cap(
    tranche: SecuritisationTranche,
    pool: SecuritisationPool,
) -> float:
    """Maximum risk weight cap: RW <= 1250 %.

    Per CRE40.52, a bank may apply a maximum capital requirement
    equal to the pool-level capital requirement allocated to the
    tranche pro-rata::

        RW_max = pool_capital / tranche_notional * 12.5

    This avoids situations where the tranche RW exceeds the
    economic substance of the risk transfer.

    Args:
        tranche: Tranche parameters.
        pool: Underlying pool parameters.

    Returns:
        Capped risk weight as a decimal, no greater than 12.50 (1250 %).
    """
    if tranche.notional <= 0:
        return _RW_CAP

    # Pro-rata share of pool capital
    tranche_share = tranche.notional / max(pool.pool_ead, 1e-10)
    pool_capital = pool.kirb * pool.pool_ead
    max_capital = pool_capital * tranche_share

    rw_max = max_capital / max(tranche.notional, 1e-10) * 12.5
    return min(rw_max, _RW_CAP)


# ============================================================
# Approach assignment — CRE40.4
# ============================================================

def assign_securitisation_approach(
    tranche: SecuritisationTranche,
    pool: SecuritisationPool,
    has_irb_approval: bool = False,
) -> str:
    """Determine applicable approach per hierarchy CRE40.4.

    The hierarchy is:
    1. SEC-IRBA — if the bank has IRB approval for the underlying
       asset type and can compute KIRB.
    2. SEC-ERBA — if the tranche has an external rating and the
       jurisdiction permits use of external ratings.
    3. SEC-SA — fallback using SA capital requirements (KSA).

    Args:
        tranche: Tranche parameters.
        pool: Underlying pool parameters.
        has_irb_approval: Whether the bank has IRB approval for the pool.

    Returns:
        One of ``"SEC-IRBA"``, ``"SEC-ERBA"``, or ``"SEC-SA"``.
    """
    if has_irb_approval and pool.kirb > 0:
        logger.info(
            "Tranche '%s': SEC-IRBA applicable (IRB approval, KIRB=%.4f)",
            tranche.tranche_id,
            pool.kirb,
        )
        return "SEC-IRBA"

    if tranche.external_rating is not None:
        logger.info(
            "Tranche '%s': SEC-ERBA applicable (CQS=%d)",
            tranche.tranche_id,
            tranche.external_rating,
        )
        return "SEC-ERBA"

    logger.info(
        "Tranche '%s': SEC-SA fallback (KSA=%.4f)",
        tranche.tranche_id,
        pool.ksa,
    )
    return "SEC-SA"
