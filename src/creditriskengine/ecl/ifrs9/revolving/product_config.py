"""Default product configurations for revolving credit types.

Each configuration captures regulatory and empirical defaults for a
product type, drawn from the research literature and supervisory guidance.

References:
    - PwC "In Depth" November 2017 (revolving ECL FAQs)
    - Araten & Jacobs (2001) on LEQ methodology
    - Tong et al. (2016) on credit card CCF distribution
    - BCBS d424 CRE32.22-32.24 (supervisory LGD)
"""

from __future__ import annotations

from dataclasses import dataclass

from creditriskengine.ecl.ifrs9.revolving.types import RevolvingProductType


@dataclass(frozen=True)
class RevolvingProductConfig:
    """Default parameters for a revolving product type.

    Attributes:
        product_type: The revolving product classification.
        default_behavioral_life_months: Central estimate of behavioral
            life per B5.5.40, in months.
        behavioral_life_range: (min, max) plausible range in months.
        typical_ccf_range: (low, high) empirical CCF range.
        typical_lgd: Default LGD assumption for the product type.
        is_collectively_managed: True if managed on a portfolio basis
            (IFRS 9 para 5.5.20 applies); False if individually managed
            (para 5.5.19 applies, contractual period used).
        has_draw_period: True if the product has a distinct draw vs.
            repayment phase (e.g., HELOCs).
        draw_period_months: Length of the draw period, if applicable.
    """

    product_type: RevolvingProductType
    default_behavioral_life_months: int
    behavioral_life_range: tuple[int, int]
    typical_ccf_range: tuple[float, float]
    typical_lgd: float
    is_collectively_managed: bool
    has_draw_period: bool = False
    draw_period_months: int | None = None


PRODUCT_CONFIGS: dict[RevolvingProductType, RevolvingProductConfig] = {
    RevolvingProductType.CREDIT_CARD: RevolvingProductConfig(
        product_type=RevolvingProductType.CREDIT_CARD,
        default_behavioral_life_months=36,
        behavioral_life_range=(24, 60),
        typical_ccf_range=(0.50, 0.95),
        typical_lgd=0.85,
        is_collectively_managed=True,
    ),
    RevolvingProductType.OVERDRAFT: RevolvingProductConfig(
        product_type=RevolvingProductType.OVERDRAFT,
        default_behavioral_life_months=24,
        behavioral_life_range=(12, 36),
        typical_ccf_range=(0.50, 1.00),
        typical_lgd=0.70,
        is_collectively_managed=True,
    ),
    RevolvingProductType.HELOC: RevolvingProductConfig(
        product_type=RevolvingProductType.HELOC,
        default_behavioral_life_months=120,
        behavioral_life_range=(60, 180),
        typical_ccf_range=(0.30, 0.75),
        typical_lgd=0.30,
        is_collectively_managed=True,
        has_draw_period=True,
        draw_period_months=120,
    ),
    RevolvingProductType.CORPORATE_REVOLVER: RevolvingProductConfig(
        product_type=RevolvingProductType.CORPORATE_REVOLVER,
        default_behavioral_life_months=48,
        behavioral_life_range=(36, 60),
        typical_ccf_range=(0.50, 0.75),
        typical_lgd=0.40,
        is_collectively_managed=False,
    ),
    RevolvingProductType.WORKING_CAPITAL: RevolvingProductConfig(
        product_type=RevolvingProductType.WORKING_CAPITAL,
        default_behavioral_life_months=24,
        behavioral_life_range=(12, 36),
        typical_ccf_range=(0.40, 0.70),
        typical_lgd=0.45,
        is_collectively_managed=False,
    ),
    RevolvingProductType.MARGIN_LENDING: RevolvingProductConfig(
        product_type=RevolvingProductType.MARGIN_LENDING,
        default_behavioral_life_months=6,
        behavioral_life_range=(1, 12),
        typical_ccf_range=(0.60, 0.90),
        typical_lgd=0.25,
        is_collectively_managed=False,
    ),
}


def get_product_config(
    product_type: RevolvingProductType,
) -> RevolvingProductConfig:
    """Return the default configuration for a revolving product type.

    Args:
        product_type: The revolving product classification.

    Returns:
        Frozen dataclass with default parameters.

    Raises:
        KeyError: If the product type has no configuration.
    """
    if product_type not in PRODUCT_CONFIGS:
        raise KeyError(
            f"No configuration for product type: {product_type!r}"
        )
    return PRODUCT_CONFIGS[product_type]
