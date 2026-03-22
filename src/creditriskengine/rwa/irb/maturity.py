"""
Maturity adjustment for IRB approach — BCBS d424, CRE31.7, CRE32.44-32.47.

Provides helpers for determining the effective maturity and whether
the maturity adjustment factor applies for a given asset class.
The core maturity adjustment formula is in :mod:`formulas`.
"""

import logging

from creditriskengine.rwa.irb.formulas import maturity_adjustment

logger = logging.getLogger(__name__)

__all__ = [
    "effective_maturity_airb",
    "effective_maturity_firb",
    "maturity_adjustment",
    "needs_maturity_adjustment",
]

# Asset classes subject to the maturity adjustment (CRE31.7)
_MATURITY_ADJUSTED_CLASSES: frozenset[str] = frozenset(
    {"corporate", "sovereign", "bank"}
)

# F-IRB fixed maturity value (CRE32.47)
_FIRB_FIXED_MATURITY: float = 2.5

# A-IRB maturity floor and cap (CRE32.44-32.46)
_AIRB_MATURITY_FLOOR: float = 1.0
_AIRB_MATURITY_CAP: float = 5.0


def effective_maturity_firb() -> float:
    """Return the F-IRB fixed maturity of 2.5 years.

    Under Foundation IRB, all exposures use a fixed effective maturity of
    2.5 years unless national supervisors allow the use of actual maturity
    (BCBS CRE32.47).

    Returns:
        2.5 (years).
    """
    return _FIRB_FIXED_MATURITY


def effective_maturity_airb(maturity_years: float) -> float:
    """Compute the A-IRB effective maturity, subject to floor and cap.

    Under Advanced IRB, the bank-estimated effective maturity M is:
    - Floored at 1 year (BCBS CRE32.44)
    - Capped at 5 years (BCBS CRE32.46)

    Exception: certain short-term exposures may have M < 1 year under
    national discretion (CRE32.45). This function does not model those
    exceptions; callers should handle them upstream.

    Args:
        maturity_years: Bank-estimated effective maturity in years.

    Returns:
        Effective maturity M, floored at 1.0 and capped at 5.0.
    """
    m = max(_AIRB_MATURITY_FLOOR, min(maturity_years, _AIRB_MATURITY_CAP))
    logger.debug(
        "A-IRB effective maturity: input=%.2f -> clamped=%.2f",
        maturity_years,
        m,
    )
    return m


def needs_maturity_adjustment(asset_class: str) -> bool:
    """Determine whether the maturity adjustment applies.

    The maturity adjustment factor (BCBS CRE31.7) applies only to
    corporate, sovereign, and bank exposures. Retail exposures do
    NOT receive a maturity adjustment (BCBS CRE31.8-31.10).

    Args:
        asset_class: IRB asset class string (e.g. ``'corporate'``,
            ``'residential_mortgage'``).

    Returns:
        ``True`` if the maturity adjustment should be applied.
    """
    return asset_class in _MATURITY_ADJUSTED_CLASSES
