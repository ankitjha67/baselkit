"""
Asset correlation functions for IRB approach — BCBS d424, CRE31.5-31.10.

This module provides a unified routing function :func:`get_asset_correlation`
that dispatches to the correct correlation formula based on IRB asset class.
Individual correlation functions are re-exported from :mod:`formulas` for
convenience.
"""

import logging

from creditriskengine.rwa.irb.formulas import (
    asset_correlation_corporate,
    asset_correlation_other_retail,
    asset_correlation_qrre,
    asset_correlation_residential_mortgage,
    sme_firm_size_adjustment,
)

logger = logging.getLogger(__name__)

__all__ = [
    "asset_correlation_corporate",
    "asset_correlation_other_retail",
    "asset_correlation_qrre",
    "asset_correlation_residential_mortgage",
    "get_asset_correlation",
    "sme_firm_size_adjustment",
]

# Asset classes that use the corporate/sovereign/bank correlation formula
_WHOLESALE_CLASSES: frozenset[str] = frozenset(
    {"corporate", "sovereign", "bank"}
)


def get_asset_correlation(
    asset_class: str,
    pd: float,
    turnover_eur_millions: float | None = None,
) -> float:
    """Route to the correct correlation function by IRB asset class.

    This is the primary entry-point for obtaining the asset correlation R
    used in the IRB capital requirement formula (BCBS CRE31.4).

    Mapping (BCBS CRE31.5-31.10):
        - corporate / sovereign / bank -> :func:`asset_correlation_corporate`
          with optional SME firm-size adjustment (CRE31.6).
        - residential_mortgage -> fixed 0.15 (CRE31.8).
        - qrre -> fixed 0.04 (CRE31.9).
        - other_retail -> :func:`asset_correlation_other_retail` (CRE31.10).

    Args:
        asset_class: One of ``'corporate'``, ``'sovereign'``, ``'bank'``,
            ``'residential_mortgage'``, ``'qrre'``, ``'other_retail'``.
        pd: Probability of Default (annualized, in [0.0003, 1.0]).
        turnover_eur_millions: Annual sales in EUR millions. Only used for
            SME firm-size adjustment on corporate exposures (CRE31.6).
            Pass ``None`` for non-SME or non-corporate exposures.

    Returns:
        Asset correlation R (always > 0).

    Raises:
        ValueError: If *asset_class* is not recognised.
    """
    if asset_class in _WHOLESALE_CLASSES:
        r = asset_correlation_corporate(pd)

        # SME firm-size adjustment only applies to corporate (CRE31.6)
        if asset_class == "corporate" and turnover_eur_millions is not None:
            adjustment = sme_firm_size_adjustment(turnover_eur_millions)
            r += adjustment
            r = max(r, 0.0)
            logger.debug(
                "SME adjustment: turnover=%.1fM, adj=%.4f, R_final=%.4f",
                turnover_eur_millions,
                adjustment,
                r,
            )

        return r

    if asset_class == "residential_mortgage":
        return asset_correlation_residential_mortgage(pd)

    if asset_class == "qrre":
        return asset_correlation_qrre(pd)

    if asset_class == "other_retail":
        return asset_correlation_other_retail(pd)

    raise ValueError(
        f"Unknown asset class '{asset_class}'. "
        f"Expected one of: corporate, sovereign, bank, "
        f"residential_mortgage, qrre, other_retail."
    )
