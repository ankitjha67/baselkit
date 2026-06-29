"""Securitisation risk weight approaches — BCBS d424, CRE40-44.

Implements:
- SEC-IRBA: Internal Ratings-Based Approach (CRE41)
- SEC-SA: Standardised Approach (CRE42)
- SEC-ERBA: External Ratings-Based Approach (CRE43)

Hierarchy: SEC-IRBA > SEC-ERBA > SEC-SA (CRE40.4)
"""

import logging
import math
from dataclasses import dataclass
from typing import Final

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
    external_rating: int | None = None
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
        is_retail: True for retail underlying pools (drives the SEC-IRBA
            supervisory parameter ``p`` per CRE44.13).
    """

    kirb: float
    ksa: float
    pool_ead: float
    n_effective: float
    lgd_pool: float = 0.50
    is_sts: bool = False
    is_retail: bool = False


# ============================================================
# Supervisory parameters — CRE41.3 / CRE42.3
# ============================================================

# SEC-SA fixed p parameter (CRE42.3): 0.5 senior, 1.0 non-senior.
_P_SENIOR_WHOLESALE: Final[float] = 0.5
_P_NON_SENIOR_WHOLESALE: Final[float] = 1.0
_P_SENIOR_RETAIL: Final[float] = 0.5
_P_NON_SENIOR_RETAIL: Final[float] = 1.0

# SEC-IRBA supervisory parameter p (CRE44.13):
#   p = max(0.3, A + B/N + C*KIRB + D*LGD + E*MT)
# Coefficients keyed by (is_retail, is_senior, is_granular). Granularity
# (N >= 25) only differentiates wholesale rows; retail rows ignore it.
_P_IRBA_FLOOR: Final[float] = 0.30
_P_IRBA_GRANULARITY_THRESHOLD: Final[float] = 25.0
# (A, B, C, D, E)
_P_IRBA_COEFFS: Final[dict[tuple[bool, bool, bool], tuple[float, float, float, float, float]]] = {
    # Wholesale, senior, granular (N >= 25)
    (False, True, True): (0.0, 3.56, -1.85, 0.55, 0.07),
    # Wholesale, senior, non-granular (N < 25)
    (False, True, False): (0.11, 2.61, -2.91, 0.68, 0.07),
    # Wholesale, non-senior, granular
    (False, False, True): (0.16, 2.87, -1.03, 0.21, 0.07),
    # Wholesale, non-senior, non-granular
    (False, False, False): (0.22, 2.35, -2.46, 0.48, 0.07),
    # Retail, senior (granularity not applicable)
    (True, True, True): (0.0, 0.0, -7.48, 0.71, 0.24),
    (True, True, False): (0.0, 0.0, -7.48, 0.71, 0.24),
    # Retail, non-senior
    (True, False, True): (0.0, 0.0, -5.78, 0.55, 0.27),
    (True, False, False): (0.0, 0.0, -5.78, 0.55, 0.27),
}

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

# CRE43 Table 2 — long-term external ratings. Each entry is the risk
# weight at tranche maturity MT = 1 year and MT = 5 years; intermediate
# maturities are linearly interpolated (CRE43.5). CQS 1 = AAA ... 17 = CCC;
# anything below CCC- (or an unmapped CQS) takes the 1250 % cap.
SEC_ERBA_SENIOR_RW: Final[dict[int, tuple[float, float]]] = {
    1: (0.15, 0.20), 2: (0.15, 0.30), 3: (0.25, 0.40), 4: (0.30, 0.45),
    5: (0.40, 0.50), 6: (0.50, 0.65), 7: (0.60, 0.70), 8: (0.75, 0.90),
    9: (0.90, 1.05), 10: (1.20, 1.40), 11: (1.40, 1.60), 12: (1.60, 1.80),
    13: (2.00, 2.25), 14: (2.50, 2.80), 15: (3.10, 3.40), 16: (3.80, 4.20),
    17: (4.60, 5.05),
}

# Non-senior table — these are the thin-tranche (T -> 0) values; thicker
# non-senior tranches are reduced by the CRE43.6 thickness adjustment.
SEC_ERBA_NON_SENIOR_RW: Final[dict[int, tuple[float, float]]] = {
    1: (0.15, 0.70), 2: (0.15, 0.90), 3: (0.30, 1.20), 4: (0.40, 1.40),
    5: (0.60, 1.60), 6: (0.80, 1.80), 7: (1.20, 2.10), 8: (1.70, 2.60),
    9: (2.20, 3.10), 10: (3.30, 4.20), 11: (4.70, 5.80), 12: (6.20, 7.60),
    13: (7.50, 8.60), 14: (9.00, 9.50), 15: (10.50, 10.50),
    16: (11.30, 11.30), 17: (12.50, 12.50),
}

# Risk-weight floor for SEC-ERBA (CRE43.6).
_ERBA_RW_FLOOR: Final[float] = 0.15


# ============================================================
# Supervisory formula — KSSFA
# ============================================================

def _kssfa(a: float, lower: float, u: float) -> float:
    """Supervisory formula KSSFA per CRE44.14.

        KSSFA(KIRB) = (exp(a * u) - exp(a * l)) / (a * (u - l))

    where, per CRE44.14:
        a = -1 / (p * KIRB)
        u = D - KIRB
        l = max(A - KIRB, 0)

    For the degenerate case where ``a`` or ``(u - l)`` is close to zero
    the formula collapses to ``(u - l)`` (linear interpolation).

    Args:
        a: Supervisory scaling parameter -1/(p*K).
        lower: Lower KIRB-adjusted bound l = max(A - KIRB, 0).
        u: Upper KIRB-adjusted bound D - KIRB.

    Returns:
        Capital requirement per unit of securitisation exposure.
    """
    span = u - lower
    if abs(a) < 1e-10 or abs(a * span) < 1e-15:
        return span

    numerator = math.exp(a * u) - math.exp(a * lower)
    return numerator / (a * span)


def _compute_p_parameter(
    kirb: float,
    lgd_pool: float,
    n_effective: float,
    is_senior: bool,
    maturity_years: float,
    is_retail: bool,
) -> float:
    """Compute supervisory parameter *p* for SEC-IRBA per CRE44.13.

    Full five-coefficient formula::

        p = max(0.3, A + B * (1/N) + C * KIRB + D * LGD + E * MT)

    where the coefficients (A, B, C, D, E) are read from the CRE44.13
    table keyed on exposure type (retail/wholesale), seniority, and — for
    wholesale only — granularity (granular when N >= 25).

    Args:
        kirb: Pool-level IRB capital requirement (decimal).
        lgd_pool: Exposure-weighted average pool LGD (decimal).
        n_effective: Effective number of exposures N.
        is_senior: Whether the tranche is the most senior.
        maturity_years: Tranche maturity MT in years (1-5 per CRE44.16).
        is_retail: Whether the underlying pool is retail.

    Returns:
        Supervisory parameter p, floored at 0.30.
    """
    is_granular = n_effective >= _P_IRBA_GRANULARITY_THRESHOLD
    a, b, c, d, e = _P_IRBA_COEFFS[(is_retail, is_senior, is_granular)]

    # MT is bounded to [1, 5] years per CRE44.16.
    mt = min(max(maturity_years, 1.0), 5.0)
    n_term = b / n_effective if n_effective > 0 else 0.0

    p = a + n_term + c * kirb + d * lgd_pool + e * mt
    return max(_P_IRBA_FLOOR, p)


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


def _ssfa_risk_weight(
    attachment: float,
    detachment: float,
    ka: float,
    p: float,
    floor: float,
) -> float:
    """Supervisory-formula risk weight per CRE44.14-44.15.

    Shared by SEC-IRBA (ka = KIRB) and SEC-SA (ka = KA). Implements the
    full three-region assembly:

    * ``D <= KA`` -> 1250 % (tranche fully below the capital requirement).
    * ``A >= KA`` -> ``RW = 12.5 * KSSFA`` (tranche fully above KA).
    * ``A < KA < D`` -> exposure-weighted blend of 1250 % on the
      below-KA slice and ``12.5 * KSSFA`` on the above-KA slice.

    Args:
        attachment: Tranche attachment point A (0-1).
        detachment: Tranche detachment point D (0-1).
        ka: Pool capital requirement KIRB (SEC-IRBA) or KA (SEC-SA).
        p: Supervisory parameter p.
        floor: Applicable risk-weight floor (decimal).

    Returns:
        Risk weight as a decimal multiple (e.g. 0.15 = 15 %, 12.5 = 1250 %).
    """
    if detachment <= ka:
        return _RW_CAP
    if ka <= 0.0 or p <= 0.0:
        # Pool carries no capital requirement: the SSFA collapses and the
        # tranche takes the floor.
        return min(max(floor, floor), _RW_CAP)

    a_param = _compute_a_parameter(p, ka, attachment, detachment)
    u = detachment - ka
    lower = max(attachment - ka, 0.0)
    kssfa = _kssfa(a_param, lower, u)

    if attachment >= ka:
        rw = 12.5 * kssfa
    else:
        thickness = detachment - attachment
        below_share = (ka - attachment) / thickness
        above_share = (detachment - ka) / thickness
        rw = below_share * _RW_CAP + above_share * 12.5 * kssfa

    return min(max(rw, floor), _RW_CAP)


# ============================================================
# SEC-IRBA — CRE41
# ============================================================

def sec_irba_risk_weight(
    tranche: SecuritisationTranche,
    pool: SecuritisationPool,
) -> float:
    """SEC-IRBA risk weight per CRE44.

    Uses the full supervisory formula (SSFA), with:
    - ``p`` from the CRE44.13 five-coefficient table
      ``p = max(0.3, A + B/N + C*KIRB + D*LGD + E*MT)``;
    - ``KSSFA = (e^{a*u} - e^{a*l}) / (a*(u-l))`` with
      ``a = -1/(p*KIRB)``, ``u = D - KIRB``, ``l = max(A - KIRB, 0)``;
    - the three-region risk-weight assembly (CRE44.15): 1250 % below
      KIRB, ``12.5 * KSSFA`` above KIRB, exposure-weighted when the
      tranche straddles KIRB.

    Reference: BCBS CRE44.13-44.16.

    Args:
        tranche: Tranche parameters.
        pool: Underlying pool parameters.

    Returns:
        Risk weight as a decimal (e.g. 0.15 for 15 %).
    """
    a_point = tranche.attachment_point
    d_point = tranche.detachment_point
    kirb = pool.kirb

    # Supervisory parameter p per CRE44.13.
    p = _compute_p_parameter(
        kirb=kirb,
        lgd_pool=pool.lgd_pool,
        n_effective=pool.n_effective,
        is_senior=tranche.is_senior,
        maturity_years=tranche.maturity_years,
        is_retail=pool.is_retail,
    )

    floor = _RW_FLOOR_STS if pool.is_sts else _RW_FLOOR
    rw = _ssfa_risk_weight(a_point, d_point, kirb, p, floor)

    logger.debug(
        "SEC-IRBA tranche '%s': A=%.4f, D=%.4f, KIRB=%.4f, p=%.4f, RW=%.4f",
        tranche.tranche_id, a_point, d_point, kirb, p, rw,
    )
    return rw


# ============================================================
# SEC-SA — CRE42
# ============================================================

def sec_sa_risk_weight(
    tranche: SecuritisationTranche,
    pool: SecuritisationPool,
    delinquency_ratio: float = 0.0,
) -> float:
    """SEC-SA risk weight per CRE42.

    Uses the same SSFA supervisory formula but on ``KA`` (derived from
    KSA) instead of KIRB. The ``p`` parameter is fixed at 0.5 (senior) or
    1.0 (non-senior). The delinquency adjustment (CRE42.2) raises the
    capital requirement::

        KA = (1 - W) * KSA + W * 0.5

    where W is the share of underlying exposures 90+ days past due, in
    bankruptcy/foreclosure, or otherwise delinquent.

    Reference: BCBS CRE42.1-42.5.

    Args:
        tranche: Tranche parameters.
        pool: Underlying pool parameters.
        delinquency_ratio: W, the share of delinquent underlying
            exposures, in [0, 1] (default 0.0).

    Returns:
        Risk weight as a decimal (e.g. 0.15 for 15 %).

    Raises:
        ValueError: If ``delinquency_ratio`` is outside [0, 1].
    """
    if not 0.0 <= delinquency_ratio <= 1.0:
        raise ValueError("delinquency_ratio (W) must be in [0, 1]")

    a_point = tranche.attachment_point
    d_point = tranche.detachment_point

    # KA with the CRE42.2 delinquency adjustment.
    ka = (1.0 - delinquency_ratio) * pool.ksa + delinquency_ratio * 0.5

    p = 0.5 if tranche.is_senior else 1.0
    floor = _RW_FLOOR_STS if pool.is_sts else _RW_FLOOR
    rw = _ssfa_risk_weight(a_point, d_point, ka, p, floor)

    logger.debug(
        "SEC-SA tranche '%s': A=%.4f, D=%.4f, KA=%.4f (W=%.4f), p=%.2f, RW=%.4f",
        tranche.tranche_id, a_point, d_point, ka, delinquency_ratio, p, rw,
    )
    return rw


# ============================================================
# SEC-ERBA — CRE43
# ============================================================

def sec_erba_risk_weight(tranche: SecuritisationTranche) -> float:
    """SEC-ERBA risk weight per CRE43.

    Full implementation of CRE43.4-43.8:

    1. Look up the (1-year, 5-year) risk weights for the tranche's credit
       quality step and seniority from CRE43 Table 2.
    2. **Maturity adjustment** (CRE43.5): linearly interpolate between the
       1-year and 5-year risk weights for the tranche maturity MT (bounded
       to [1, 5] years)::

           RW = RW_1y + (RW_5y - RW_1y) * (MT - 1) / (5 - 1)

    3. **Thickness adjustment** (CRE43.6), non-senior only: the table holds
       the thin-tranche values, so thicker tranches are reduced::

           RW = RW * (1 - min(D - A, 0.5))

    4. Floor at 15 % (CRE43.6), re-securitisation x2 (CRE43.8), 1250 % cap.

    Reference: BCBS CRE43.4-43.8, Table 2.

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

    table = SEC_ERBA_SENIOR_RW if tranche.is_senior else SEC_ERBA_NON_SENIOR_RW
    entry = table.get(cqs)
    if entry is None:
        logger.warning(
            "CQS %d not found in SEC-ERBA table for tranche '%s'; "
            "applying 1250%% cap",
            cqs, tranche.tranche_id,
        )
        return _RW_CAP

    rw_1y, rw_5y = entry

    # Maturity adjustment — CRE43.5 (linear interpolation over [1, 5] years).
    mt = min(max(tranche.maturity_years, 1.0), 5.0)
    rw = rw_1y + (rw_5y - rw_1y) * (mt - 1.0) / 4.0

    # Thickness adjustment for non-senior tranches — CRE43.6.
    if not tranche.is_senior:
        rw *= 1.0 - min(thickness, 0.5)

    # Floor (CRE43.6).
    rw = max(rw, _ERBA_RW_FLOOR)

    # Re-securitisation multiplier — CRE43.8.
    if tranche.is_resecuritisation:
        rw *= 2.0

    # 1250 % cap.
    rw = min(rw, _RW_CAP)

    logger.debug(
        "SEC-ERBA tranche '%s': CQS=%d, senior=%s, thickness=%.4f, "
        "MT=%.1f, RW_1y=%.4f, RW_5y=%.4f, final_rw=%.4f",
        tranche.tranche_id, cqs, tranche.is_senior, thickness, mt,
        rw_1y, rw_5y, rw,
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
