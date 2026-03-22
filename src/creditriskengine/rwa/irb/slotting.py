"""
Supervisory slotting approach for specialised lending — BCBS d424, CRE34.1-34.7.

When banks cannot meet the IRB requirements for estimating PD on
specialised lending exposures, they must use the supervisory slotting
criteria approach, which maps exposures to one of five risk categories
with prescribed risk weights.

Key regulatory references:
    - Slotting categories and risk weights: CRE34.2 (Table 1)
    - HVCRE risk weights: CRE34.3 (Table 2)
    - National discretion for "Strong": CRE34.4
    - CRR3 Art. 153(5): EU discretion for preferential "Strong" weight
"""

import logging
from enum import StrEnum

from creditriskengine.core.types import IRBSpecialisedLendingType

logger = logging.getLogger(__name__)

__all__ = [
    "SlottingCategory",
    "assign_slotting_category",
    "slotting_risk_weight",
]


class SlottingCategory(StrEnum):
    """Supervisory slotting risk categories per BCBS CRE34.2.

    Exposures are assessed against a set of qualitative and quantitative
    criteria and assigned to one of five categories.
    """

    STRONG = "strong"
    GOOD = "good"
    SATISFACTORY = "satisfactory"
    WEAK = "weak"
    DEFAULT = "default"


# ============================================================
# RISK WEIGHT TABLES — BCBS CRE34.2-34.3
# ============================================================

# Non-HVCRE specialised lending risk weights (CRE34.2, Table 1)
_RW_NON_HVCRE: dict[SlottingCategory, float] = {
    SlottingCategory.STRONG: 70.0,
    SlottingCategory.GOOD: 90.0,
    SlottingCategory.SATISFACTORY: 115.0,
    SlottingCategory.WEAK: 250.0,
    SlottingCategory.DEFAULT: 0.0,
}

# HVCRE specialised lending risk weights (CRE34.3, Table 2)
_RW_HVCRE: dict[SlottingCategory, float] = {
    SlottingCategory.STRONG: 95.0,
    SlottingCategory.GOOD: 120.0,
    SlottingCategory.SATISFACTORY: 140.0,
    SlottingCategory.WEAK: 250.0,
    SlottingCategory.DEFAULT: 0.0,
}

# National discretion: preferential "Strong" weights (CRE34.4, CRR3 Art. 153(5))
_RW_NON_HVCRE_PREFERENTIAL: dict[SlottingCategory, float] = {
    SlottingCategory.STRONG: 50.0,
    SlottingCategory.GOOD: 70.0,
    SlottingCategory.SATISFACTORY: 115.0,
    SlottingCategory.WEAK: 250.0,
    SlottingCategory.DEFAULT: 0.0,
}

_RW_HVCRE_PREFERENTIAL: dict[SlottingCategory, float] = {
    SlottingCategory.STRONG: 70.0,
    SlottingCategory.GOOD: 95.0,
    SlottingCategory.SATISFACTORY: 140.0,
    SlottingCategory.WEAK: 250.0,
    SlottingCategory.DEFAULT: 0.0,
}


def slotting_risk_weight(
    category: SlottingCategory,
    sl_type: IRBSpecialisedLendingType | None = None,
    use_preferential: bool = False,
) -> float:
    """Look up the supervisory slotting risk weight.

    Risk weights per BCBS CRE34.2-34.4:

    Non-HVCRE (standard / preferential):
        Strong:       70% / 50%
        Good:         90% / 70%
        Satisfactory: 115% / 115%
        Weak:         250% / 250%
        Default:      0% / 0%

    HVCRE (standard / preferential):
        Strong:       95% / 70%
        Good:         120% / 95%
        Satisfactory: 140% / 140%
        Weak:         250% / 250%
        Default:      0% / 0%

    Default category: risk weight is 0% because losses are expected to
    be covered by specific provisions.

    Args:
        category: The supervisory slotting category.
        sl_type: Specialised lending sub-type. If
            :attr:`IRBSpecialisedLendingType.HIGH_VOLATILITY_CRE`, the
            higher HVCRE risk weights apply.
        use_preferential: If ``True``, apply national-discretion
            preferential weights for Strong/Good (CRE34.4).

    Returns:
        Risk weight as a percentage (e.g. 70.0 means 70%).
    """
    is_hvcre = sl_type == IRBSpecialisedLendingType.HIGH_VOLATILITY_CRE

    if is_hvcre:
        table = _RW_HVCRE_PREFERENTIAL if use_preferential else _RW_HVCRE
    else:
        table = _RW_NON_HVCRE_PREFERENTIAL if use_preferential else _RW_NON_HVCRE

    rw = table[category]

    logger.debug(
        "Slotting RW: category=%s type=%s preferential=%s -> RW=%.1f%%",
        category.value,
        sl_type.value if sl_type else "non-HVCRE",
        use_preferential,
        rw,
    )
    return rw


def assign_slotting_category(
    financial_strength: str,
    political_and_legal: str,
    transaction_characteristics: str,
    asset_characteristics: str,
    sponsor_strength: str,
) -> SlottingCategory:
    """Assign a slotting category based on qualitative assessment criteria.

    The supervisory slotting criteria (BCBS CRE34.5-34.7) assess exposures
    across multiple dimensions. Each dimension is rated on the same scale
    as :class:`SlottingCategory` (strong/good/satisfactory/weak).

    This function uses a conservative aggregation: the overall category
    is the *worst* (highest-risk) of the individual dimension ratings.
    Supervisors may allow more nuanced approaches.

    Args:
        financial_strength: Rating for financial strength of the project/asset.
        political_and_legal: Rating for political and legal environment.
        transaction_characteristics: Rating for transaction/asset structure.
        asset_characteristics: Rating for quality and condition of the asset.
        sponsor_strength: Rating for strength of the sponsor/developer.

    Returns:
        The overall :class:`SlottingCategory`.

    Raises:
        ValueError: If any dimension rating is not a valid slotting category.
    """
    dimensions = {
        "financial_strength": financial_strength,
        "political_and_legal": political_and_legal,
        "transaction_characteristics": transaction_characteristics,
        "asset_characteristics": asset_characteristics,
        "sponsor_strength": sponsor_strength,
    }

    # Ordered from best to worst
    category_order: list[SlottingCategory] = [
        SlottingCategory.STRONG,
        SlottingCategory.GOOD,
        SlottingCategory.SATISFACTORY,
        SlottingCategory.WEAK,
        SlottingCategory.DEFAULT,
    ]
    rank: dict[str, int] = {cat.value: idx for idx, cat in enumerate(category_order)}

    worst_rank = 0
    for dim_name, dim_value in dimensions.items():
        dim_lower = dim_value.lower().strip()
        if dim_lower not in rank:
            raise ValueError(
                f"Invalid slotting rating '{dim_value}' for dimension "
                f"'{dim_name}'. Must be one of: "
                f"{', '.join(c.value for c in category_order)}."
            )
        dim_rank = rank[dim_lower]
        if dim_rank > worst_rank:
            worst_rank = dim_rank

    result = category_order[worst_rank]

    logger.debug(
        "Slotting assignment: dimensions=%s -> category=%s",
        dimensions,
        result.value,
    )
    return result
