"""Behavioral life determination for revolving credit under IFRS 9.

Implements the three-factor framework from IFRS 9 paragraph B5.5.40
to determine the period over which ECL should be measured for revolving
products subject to the paragraph 5.5.20 exception.

Per IASB February 2017 staff paper (Agenda ref 12B) and PwC guidance,
the **shortest** of the three B5.5.40 factors determines the exposure
period.  Credit risk management actions only shorten the life for the
portion of the portfolio expected to be subject to those actions.

References:
    - IFRS 9 paragraphs 5.5.20, B5.5.39-40
    - IASB February 2017 staff paper, Agenda ref 12B
    - PwC "In Depth" November 2017, FAQ 1-3
    - April 2015 ITG discussion
"""

from __future__ import annotations

import numpy as np

from creditriskengine.ecl.ifrs9.revolving.product_config import (
    PRODUCT_CONFIGS,
)
from creditriskengine.ecl.ifrs9.revolving.types import RevolvingProductType


def determine_behavioral_life(
    historical_life_months: float | None = None,
    time_to_default_months: float | None = None,
    crm_action_months: float | None = None,
    product_type: RevolvingProductType | None = None,
) -> int:
    """Determine the behavioral life for a revolving facility.

    Applies the three-factor framework from IFRS 9 B5.5.40:

    (a) Historical exposure period on similar instruments
    (b) Time for defaults to occur after significant credit deterioration
    (c) Expected credit risk management actions (limit reduction/removal)

    The **shortest** non-None factor is used per IASB guidance.  If no
    factors are provided, falls back to the product-type default.

    Args:
        historical_life_months: Average observed account life on similar
            instruments (B5.5.40(a)).
        time_to_default_months: Average time from SICR to default on
            similar instruments (B5.5.40(b)).
        crm_action_months: Expected time until credit risk management
            actions are taken (limit reduction, cancellation) once
            credit risk increases (B5.5.40(c)).
        product_type: Falls back to product-type default if no factors
            are provided.

    Returns:
        Behavioral life in months (integer, at least 1).

    Raises:
        ValueError: If no factors and no product_type are provided.
    """
    factors = [
        f
        for f in (
            historical_life_months,
            time_to_default_months,
            crm_action_months,
        )
        if f is not None and f > 0
    ]

    if factors:
        return max(1, int(np.ceil(min(factors))))

    if product_type is not None:
        config = PRODUCT_CONFIGS.get(product_type)
        if config is not None:
            return config.default_behavioral_life_months

    raise ValueError(
        "At least one B5.5.40 factor or a product_type must be provided."
    )


def segment_behavioral_life(
    base_life_months: int,
    risk_segment_lives: dict[str, int] | None = None,
) -> dict[str, int]:
    """Assign behavioral lives by risk segment.

    Higher-risk segments may have shorter behavioral lives because
    credit risk management actions (limit reduction, cancellation)
    are expected to be taken earlier.  Lower-risk segments retain
    the base life or longer.

    Args:
        base_life_months: Default behavioral life for the portfolio.
        risk_segment_lives: Optional mapping of segment name to
            overridden behavioral life in months.  Segments not
            specified receive the base life.

    Returns:
        Dict mapping segment name to behavioral life in months.
        Always includes a ``"base"`` segment.
    """
    result: dict[str, int] = {"base": base_life_months}
    if risk_segment_lives:
        for segment, life in risk_segment_lives.items():
            result[segment] = max(1, life)
    return result


def effective_life_months(
    product_type: RevolvingProductType,
    is_draw_period: bool = True,
) -> int:
    """Return effective life considering draw/repayment phases.

    For products with distinct draw and repayment periods (e.g., HELOCs),
    returns the draw-period life during the draw phase and falls back
    to the full behavioral life during repayment (when CCF = 0 and the
    product behaves like an amortizing loan).

    Args:
        product_type: Revolving product classification.
        is_draw_period: True if currently in draw period (default).

    Returns:
        Effective behavioral life in months.
    """
    config = PRODUCT_CONFIGS.get(product_type)
    if config is None:
        raise KeyError(f"No config for product type: {product_type!r}")

    if config.has_draw_period and config.draw_period_months is not None:
        if is_draw_period:
            return config.draw_period_months
        return config.default_behavioral_life_months

    return config.default_behavioral_life_months
