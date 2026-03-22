"""
LGD (Loss Given Default) modeling framework.

Provides workout LGD estimation, downturn LGD adjustment,
and LGD curve generation.

References:
- EBA GL/2017/16: LGD estimation
- BCBS d424: CRE36 (LGD under IRB)
- BCBS CRE32.22-32.24: Supervisory LGD values
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Supervisory LGD values per BCBS CRE32.22-32.24 (F-IRB)
SUPERVISORY_LGD_SENIOR_UNSECURED: float = 0.45
SUPERVISORY_LGD_SUBORDINATED: float = 0.75

# Minimum LGD floors per CRE32.25 (A-IRB)
LGD_FLOOR_UNSECURED: float = 0.25
LGD_FLOOR_SECURED_FINANCIAL: float = 0.0
LGD_FLOOR_SECURED_RECEIVABLES: float = 0.10
LGD_FLOOR_SECURED_CRE_RRE: float = 0.10
LGD_FLOOR_SECURED_OTHER: float = 0.15


def workout_lgd(
    ead_at_default: float,
    total_recoveries: float,
    total_costs: float,
    discount_rate: float = 0.0,
    time_to_recovery_years: float = 0.0,
) -> float:
    """Calculate workout LGD from realized recovery data.

    LGD = 1 - (Recoveries - Costs) / EAD, discounted to default date.

    Args:
        ead_at_default: Exposure at the time of default.
        total_recoveries: Total gross recoveries.
        total_costs: Total workout/collection costs.
        discount_rate: Annual discount rate for time value.
        time_to_recovery_years: Average time from default to recovery.

    Returns:
        Workout LGD in [0, 1].
    """
    if ead_at_default <= 0:
        return 1.0
    net_recovery = total_recoveries - total_costs
    if discount_rate > 0 and time_to_recovery_years > 0:
        discount_factor = 1.0 / (1.0 + discount_rate) ** time_to_recovery_years
        net_recovery *= discount_factor
    lgd = 1.0 - net_recovery / ead_at_default
    return float(np.clip(lgd, 0.0, 1.0))


def downturn_lgd(
    base_lgd: float,
    downturn_add_on: float | None = None,
    method: str = "additive",
) -> float:
    """Estimate downturn LGD per EBA GL/2017/16 Section 6.3.

    Methods:
        - "additive": LGD_dt = base_LGD + add_on
        - "haircut": LGD_dt = 1 - (1 - base_LGD) × (1 - add_on)
        - "regulatory": Use BCBS formula max(base_LGD, 0.10 + 0.40 × base_LGD)

    Args:
        base_lgd: Through-the-cycle / long-run average LGD.
        downturn_add_on: Downturn add-on (for additive/haircut).
        method: Estimation method.

    Returns:
        Downturn LGD, clipped to [0, 1].
    """
    if method == "regulatory":
        dt_lgd = max(base_lgd, 0.10 + 0.40 * base_lgd)
    elif method == "haircut":
        add_on = downturn_add_on if downturn_add_on is not None else 0.08
        dt_lgd = 1.0 - (1.0 - base_lgd) * (1.0 - add_on)
    else:  # additive
        add_on = downturn_add_on if downturn_add_on is not None else 0.08
        dt_lgd = base_lgd + add_on

    return float(np.clip(dt_lgd, 0.0, 1.0))


def lgd_term_structure(
    base_lgd: float,
    n_periods: int,
    recovery_curve: np.ndarray | None = None,
) -> np.ndarray:
    """Generate LGD term structure over multiple periods.

    If a recovery curve is provided, LGD(t) = 1 - cumulative_recovery(t).
    Otherwise returns flat LGD.

    Args:
        base_lgd: Base LGD estimate.
        n_periods: Number of periods.
        recovery_curve: Optional cumulative recovery rates by period.

    Returns:
        Array of LGD values by period.
    """
    if recovery_curve is not None:
        recovery_curve = np.asarray(recovery_curve, dtype=np.float64)
        lgds = 1.0 - recovery_curve[:n_periods]
        # Pad if recovery_curve shorter than n_periods
        if len(lgds) < n_periods:
            pad_value = lgds[-1] if len(lgds) > 0 else base_lgd
            lgds = np.concatenate([lgds, np.full(n_periods - len(lgds), pad_value)])
        return np.clip(lgds, 0.0, 1.0)
    return np.full(n_periods, base_lgd)


def apply_lgd_floor(
    lgd: float,
    is_secured: bool = False,
    collateral_type: str = "unsecured",
) -> float:
    """Apply Basel III A-IRB LGD floors per CRE32.25.

    Args:
        lgd: Estimated LGD.
        is_secured: Whether exposure is secured.
        collateral_type: Type of collateral.

    Returns:
        Floored LGD.
    """
    if not is_secured:
        floor = LGD_FLOOR_UNSECURED
    elif collateral_type == "financial":
        floor = LGD_FLOOR_SECURED_FINANCIAL
    elif collateral_type == "receivables":
        floor = LGD_FLOOR_SECURED_RECEIVABLES
    elif collateral_type in ("cre", "rre"):
        floor = LGD_FLOOR_SECURED_CRE_RRE
    else:
        floor = LGD_FLOOR_SECURED_OTHER
    return max(lgd, floor)


# ── Sklearn-compatible Estimator ──────────────────────────────────

from sklearn.base import BaseEstimator, RegressorMixin  # noqa: E402


class LGDModel(BaseEstimator, RegressorMixin):  # type: ignore[misc]
    """Sklearn-compatible LGD model.

    Wraps workout LGD estimation with fit/predict interface.

    Parameters:
        method: LGD estimation method ('workout', 'downturn', 'regulatory').
        downturn_method: Downturn LGD method ('additive', 'haircut', 'regulatory_formula').
        downturn_add_on: Additive stress for downturn LGD.
        collateral_type: For regulatory LGD floor application.
    """

    def __init__(
        self,
        method: str = "workout",
        downturn_method: str = "regulatory_formula",
        downturn_add_on: float = 0.10,
        collateral_type: str = "unsecured",
    ) -> None:
        self.method = method
        self.downturn_method = downturn_method
        self.downturn_add_on = downturn_add_on
        self.collateral_type = collateral_type
        self.mean_lgd_: float | None = None
        self.is_fitted_: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LGDModel":  # noqa: N803
        """Fit LGD model. X = features, y = realized LGDs.

        Stores mean LGD for baseline predictions.
        """
        y = np.asarray(y, dtype=np.float64)
        self.mean_lgd_ = float(np.mean(y))
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        """Predict LGD values. Returns mean LGD (baseline model).

        Override in subclasses for more sophisticated models.
        """
        assert self.is_fitted_, "Call fit() first"
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        base = np.full(n, self.mean_lgd_)
        if self.method == "downturn":
            base = np.array([
                downturn_lgd(lgd, self.downturn_add_on, self.downturn_method)
                for lgd in base
            ])
        is_secured = self.collateral_type != "unsecured"
        base = np.array([
            apply_lgd_floor(lgd, is_secured, self.collateral_type)
            for lgd in base
        ])
        return base
