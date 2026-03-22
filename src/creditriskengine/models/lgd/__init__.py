"""LGD (Loss Given Default) modeling framework.

Provides workout LGD estimation, downturn adjustment, LGD term structures,
and cure rate modeling.
"""

from creditriskengine.models.lgd.cure_rate import (
    CureRateResult,
    estimate_cure_rate,
    lgd_with_cure_adjustment,
    macro_adjusted_cure_rate,
)
from creditriskengine.models.lgd.lgd_model import (
    apply_lgd_floor,
    downturn_lgd,
    lgd_term_structure,
    workout_lgd,
)

__all__ = [
    "workout_lgd",
    "downturn_lgd",
    "lgd_term_structure",
    "apply_lgd_floor",
    "CureRateResult",
    "estimate_cure_rate",
    "lgd_with_cure_adjustment",
    "macro_adjusted_cure_rate",
]
