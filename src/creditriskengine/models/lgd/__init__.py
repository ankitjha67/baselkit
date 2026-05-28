"""LGD (Loss Given Default) modeling framework.

Provides workout LGD estimation, downturn adjustment, LGD term structures,
and cure rate modeling.
"""

from creditriskengine.models.lgd.beta_regression import (
    beta_lgd_mean,
    downturn_lgd_quantile,
    fit_beta_lgd,
)
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
from creditriskengine.models.lgd.recovery_curves import (
    RecoveryCurveFit,
    RecoveryCurveType,
    cumulative_recovery_fraction,
    discounted_workout_lgd,
    fit_recovery_curve,
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
    # Recovery curves
    "RecoveryCurveType",
    "RecoveryCurveFit",
    "fit_recovery_curve",
    "cumulative_recovery_fraction",
    "discounted_workout_lgd",
    # Beta regression
    "fit_beta_lgd",
    "beta_lgd_mean",
    "downturn_lgd_quantile",
]
