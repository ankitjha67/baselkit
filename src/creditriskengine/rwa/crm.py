"""Credit Risk Mitigation (CRM) — BCBS d424, CRE22.

Implements the CRM framework for the standardised approach:

- **Supervisory haircuts** (CRE22.40–22.56): look-up table for eligible
  financial collateral, including maturity buckets and currency-mismatch
  add-on.
- **Comprehensive approach** (CRE22.57–22.77): adjusts both exposure and
  collateral by supervisory haircuts to derive the adjusted exposure E*.
- **Simple approach** (CRE22.35–22.39): substitutes the risk weight of
  eligible collateral for the covered portion, subject to a 20 % floor
  (0 % for cash / government securities denominated in domestic currency).
- **Guarantee / credit-derivative substitution** (CRE22.78–22.93):
  replaces the obligor risk weight with the guarantor risk weight for the
  covered portion.
- **Maturity mismatch adjustment** (CRE22.33): scales the credit
  protection value when the residual maturity of the protection is shorter
  than that of the exposure.

All public functions are pure — no side effects, no shared mutable state.
"""

import logging
from typing import Final

logger = logging.getLogger(__name__)

__all__ = [
    "supervisory_haircut",
    "comprehensive_approach",
    "simple_approach",
    "guarantee_substitution",
    "maturity_mismatch_adjustment",
    "HAIRCUT_TABLE",
    "CURRENCY_MISMATCH_HAIRCUT",
]

# ============================================================
# Constants — CRE22.40 supervisory haircut table
# ============================================================

CURRENCY_MISMATCH_HAIRCUT: Final[float] = 0.08
"""8 % add-on when collateral is denominated in a different currency
than the exposure (CRE22.52)."""

# Haircut table keyed by ``(collateral_type, credit_quality_step)``
# where CQS is 1-based (1 = AAA/AA-).  Values are dicts mapping a
# *residual-maturity bucket label* to the haircut percentage expressed
# as a float (e.g. 0.005 = 0.5 %).
#
# Maturity buckets:
#   "le1"  : <= 1 year
#   "1to5" : > 1 year and <= 5 years
#   "gt5"  : > 5 years
#
# For asset types where maturity is irrelevant (cash, equities, gold)
# the single key ``"any"`` is used.

HAIRCUT_TABLE: Final[dict[tuple[str, int | None], dict[str, float]]] = {
    # --- Cash (CRE22.40) ---
    ("cash", None): {"any": 0.0},

    # --- Sovereign / government bonds (CRE22.41–22.43) ---
    # CQS 1 (AAA to AA-)
    ("sovereign_bond", 1): {"le1": 0.005, "1to5": 0.02, "gt5": 0.04},
    # CQS 2-3 (A+ to BBB-)
    ("sovereign_bond", 2): {"le1": 0.01, "1to5": 0.03, "gt5": 0.06},
    ("sovereign_bond", 3): {"le1": 0.01, "1to5": 0.03, "gt5": 0.06},

    # --- Corporate bonds / other eligible bonds (CRE22.44–22.46) ---
    # CQS 1 (AAA to AA-)
    ("corporate_bond", 1): {"le1": 0.01, "1to5": 0.04, "gt5": 0.08},
    # CQS 2-3 (A+ to BBB-)
    ("corporate_bond", 2): {"le1": 0.02, "1to5": 0.06, "gt5": 0.12},
    ("corporate_bond", 3): {"le1": 0.02, "1to5": 0.06, "gt5": 0.12},

    # --- Equities (CRE22.49) ---
    ("main_index_equity", None): {"any": 0.15},
    ("other_equity", None): {"any": 0.25},

    # --- Gold (CRE22.50) ---
    ("gold", None): {"any": 0.15},
}


def _maturity_bucket(residual_maturity_years: float) -> str:
    """Return the haircut-table maturity bucket label.

    Args:
        residual_maturity_years: Residual maturity in years.

    Returns:
        One of ``"le1"``, ``"1to5"``, or ``"gt5"``.
    """
    if residual_maturity_years <= 1.0:
        return "le1"
    if residual_maturity_years <= 5.0:
        return "1to5"
    return "gt5"


# ============================================================
# Public API
# ============================================================

def supervisory_haircut(
    collateral_type: str,
    residual_maturity_years: float = 0.0,
    credit_quality_step: int | None = None,
    currency_mismatch: bool = False,
) -> float:
    """Look up the supervisory haircut for eligible collateral.

    Implements the haircut table in **CRE22.40–22.56**.

    Args:
        collateral_type: One of ``"cash"``, ``"sovereign_bond"``,
            ``"corporate_bond"``, ``"main_index_equity"``,
            ``"other_equity"``, ``"gold"``.
        residual_maturity_years: Residual maturity of the collateral
            instrument in years.  Only relevant for bonds.
        credit_quality_step: External credit quality step (1–3).
            Required for bond collateral, ignored otherwise.
        currency_mismatch: ``True`` when the collateral currency
            differs from the exposure currency (CRE22.52).

    Returns:
        The combined supervisory haircut as a decimal fraction
        (e.g. 0.08 = 8 %).

    Raises:
        ValueError: If the collateral type / CQS combination is not
            found in the supervisory table.
    """
    key = (collateral_type, credit_quality_step)
    bucket_map = HAIRCUT_TABLE.get(key)
    if bucket_map is None:
        raise ValueError(
            f"No supervisory haircut for collateral_type={collateral_type!r}, "
            f"credit_quality_step={credit_quality_step!r}. "
            "Check CRE22.40–22.56 for eligible collateral types."
        )

    if "any" in bucket_map:
        haircut = bucket_map["any"]
    else:
        bucket = _maturity_bucket(residual_maturity_years)
        haircut = bucket_map[bucket]

    if currency_mismatch:
        haircut += CURRENCY_MISMATCH_HAIRCUT

    logger.debug(
        "supervisory_haircut: type=%s, cqs=%s, mat=%.2f, fx_mm=%s → %.4f",
        collateral_type,
        credit_quality_step,
        residual_maturity_years,
        currency_mismatch,
        haircut,
    )
    return haircut


def comprehensive_approach(
    exposure: float,
    collateral_value: float,
    collateral_type: str,
    residual_maturity_years: float = 0.0,
    credit_quality_step: int | None = None,
    currency_mismatch: bool = False,
    exposure_haircut: float = 0.0,
) -> dict[str, float]:
    """Compute adjusted exposure E* under the comprehensive approach.

    Implements **CRE22.57–22.77**::

        E* = max(0, E × (1 + He) - C × (1 - Hc - Hfx))

    Args:
        exposure: Gross exposure value (E).
        collateral_value: Market value of eligible collateral (C).
        collateral_type: Collateral type for haircut look-up.
        residual_maturity_years: Residual maturity of the collateral.
        credit_quality_step: CQS of the collateral issuer (bonds).
        currency_mismatch: ``True`` when a currency mismatch exists.
        exposure_haircut: Supervisory haircut for the exposure
            instrument (He); typically non-zero only for repo-style
            transactions (CRE22.59).

    Returns:
        Dict with keys:

        - ``adjusted_exposure``: E* after haircuts.
        - ``collateral_haircut``: Hc used.
        - ``fx_haircut``: Hfx used (0 if no currency mismatch).
        - ``exposure_haircut``: He used.
        - ``gross_exposure``: Original exposure.
        - ``collateral_value``: Original collateral value.
    """
    hc = supervisory_haircut(
        collateral_type=collateral_type,
        residual_maturity_years=residual_maturity_years,
        credit_quality_step=credit_quality_step,
        currency_mismatch=False,  # fx handled separately below
    )
    hfx = CURRENCY_MISMATCH_HAIRCUT if currency_mismatch else 0.0

    e_star = max(0.0, exposure * (1.0 + exposure_haircut) - collateral_value * (1.0 - hc - hfx))

    logger.debug(
        "comprehensive_approach: E=%.2f, C=%.2f, He=%.4f, Hc=%.4f, Hfx=%.4f → E*=%.2f",
        exposure,
        collateral_value,
        exposure_haircut,
        hc,
        hfx,
        e_star,
    )
    return {
        "adjusted_exposure": e_star,
        "collateral_haircut": hc,
        "fx_haircut": hfx,
        "exposure_haircut": exposure_haircut,
        "gross_exposure": exposure,
        "collateral_value": collateral_value,
    }


def simple_approach(
    exposure: float,
    collateral_value: float,
    exposure_rw: float,
    collateral_rw: float,
    is_cash_or_zero_haircut: bool = False,
) -> dict[str, float]:
    """Apply the simple approach for CRM.

    Under the simple approach (**CRE22.35–22.39**) the risk weight of the
    collateral replaces the exposure risk weight for the collateralised
    portion.  The collateral risk weight is floored at 20 %, except for
    cash collateral and certain government-security collateral where
    0 % applies.

    Args:
        exposure: Gross exposure value.
        collateral_value: Market value of eligible collateral, capped
            internally at the exposure value.
        exposure_rw: Risk weight of the original exposure (%).
        collateral_rw: Risk weight applicable to the collateral
            instrument (%).  Will be floored to 20 % (or 0 % if
            *is_cash_or_zero_haircut* is ``True``).
        is_cash_or_zero_haircut: ``True`` for cash collateral or
            sovereign bonds denominated in domestic currency where a
            0 % haircut applies (CRE22.37).

    Returns:
        Dict with keys:

        - ``rwa``: Total risk-weighted assets.
        - ``covered_portion``: The collateral-covered portion of the exposure.
        - ``uncovered_portion``: The remaining uncovered portion.
        - ``effective_collateral_rw``: The (floored) risk weight applied
          to the covered portion.
        - ``exposure_rw``: Risk weight applied to the uncovered portion.
    """
    covered = min(collateral_value, exposure)
    uncovered = exposure - covered

    if is_cash_or_zero_haircut:
        floor = 0.0
    else:
        floor = 20.0

    effective_collateral_rw = max(collateral_rw, floor)

    rwa = uncovered * (exposure_rw / 100.0) + covered * (effective_collateral_rw / 100.0)

    logger.debug(
        "simple_approach: E=%.2f, C=%.2f, RW_e=%.1f%%, RW_c=%.1f%% (floor=%.1f%%) → RWA=%.2f",
        exposure,
        collateral_value,
        exposure_rw,
        effective_collateral_rw,
        floor,
        rwa,
    )
    return {
        "rwa": rwa,
        "covered_portion": covered,
        "uncovered_portion": uncovered,
        "effective_collateral_rw": effective_collateral_rw,
        "exposure_rw": exposure_rw,
    }


def guarantee_substitution(
    exposure_rw: float,
    guarantor_rw: float,
    coverage_ratio: float,
) -> dict[str, float]:
    """Substitution approach for guarantees and credit derivatives.

    Implements **CRE22.78–22.93**::

        RW_effective = g × RW_guarantor + (1 - g) × RW_obligor

    Args:
        exposure_rw: Risk weight of the original obligor (%).
        guarantor_rw: Risk weight of the guarantor / protection
            seller (%).
        coverage_ratio: Fraction of the exposure covered by the
            guarantee (0.0–1.0).

    Returns:
        Dict with keys:

        - ``effective_rw``: Blended risk weight (%).
        - ``exposure_rw``: Original obligor risk weight (%).
        - ``guarantor_rw``: Guarantor risk weight (%).
        - ``coverage_ratio``: Coverage ratio used.

    Raises:
        ValueError: If *coverage_ratio* is outside [0, 1].
    """
    if not 0.0 <= coverage_ratio <= 1.0:
        raise ValueError(
            f"coverage_ratio must be in [0, 1], got {coverage_ratio}"
        )

    effective_rw = coverage_ratio * guarantor_rw + (1.0 - coverage_ratio) * exposure_rw

    logger.debug(
        "guarantee_substitution: RW_obligor=%.1f%%, RW_guarantor=%.1f%%, g=%.4f → RW_eff=%.2f%%",
        exposure_rw,
        guarantor_rw,
        coverage_ratio,
        effective_rw,
    )
    return {
        "effective_rw": effective_rw,
        "exposure_rw": exposure_rw,
        "guarantor_rw": guarantor_rw,
        "coverage_ratio": coverage_ratio,
    }


def maturity_mismatch_adjustment(
    collateral_value: float,
    collateral_maturity: float,
    exposure_maturity: float,
) -> float:
    """Adjust collateral value for maturity mismatch.

    Implements **CRE22.33**::

        Pa = P × max(0, (t - 0.25) / (T - 0.25))

    where *P* is the unadjusted collateral (or protection) value,
    *t* is the residual maturity of the credit protection in years,
    and *T* is the residual maturity of the exposure in years.

    A maturity mismatch exists when the residual maturity of the
    credit protection is less than that of the underlying exposure.
    Protection with a residual maturity of less than one year where
    the underlying has a residual maturity of more than one year is
    recognised only if the protection maturity exceeds three months
    (0.25 years).

    Args:
        collateral_value: Unadjusted value of the credit protection (P).
        collateral_maturity: Residual maturity of the protection in
            years (t).
        exposure_maturity: Residual maturity of the exposure in
            years (T).

    Returns:
        Adjusted protection value Pa.  Returns the unadjusted value
        when there is no maturity mismatch (t >= T).  Returns 0 when
        the protection maturity is 0.25 years or less and the exposure
        maturity exceeds 0.25 years.
    """
    t = collateral_maturity
    big_t = exposure_maturity

    # No mismatch — full recognition.
    if t >= big_t:
        logger.debug(
            "maturity_mismatch_adjustment: no mismatch (t=%.2f >= T=%.2f), Pa=P=%.2f",
            t,
            big_t,
            collateral_value,
        )
        return collateral_value

    # Protection maturity <= 3 months — no recognition.
    if t <= 0.25:
        logger.debug(
            "maturity_mismatch_adjustment: t=%.2f <= 0.25, Pa=0.0",
            t,
        )
        return 0.0

    # At this point t > 0.25 and t < big_t, so big_t > 0.25 is guaranteed
    # and the denominator (big_t - 0.25) is always positive.
    adjusted = collateral_value * max(0.0, (t - 0.25) / (big_t - 0.25))

    logger.debug(
        "maturity_mismatch_adjustment: P=%.2f, t=%.2f, T=%.2f → Pa=%.2f",
        collateral_value,
        t,
        big_t,
        adjusted,
    )
    return adjusted
