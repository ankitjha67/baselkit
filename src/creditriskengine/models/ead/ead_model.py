"""
EAD (Exposure at Default) modeling framework.

Provides CCF estimation, regulatory EAD calculation, and
EAD term structure generation for off-balance sheet exposures.

References:
- BCBS d424: CRE31 (EAD under IRB), CRE20 (EAD under SA)
- CRE32.26-32.32: CCF parameters
- EBA GL/2017/16: EAD estimation for A-IRB
"""

import logging

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

logger = logging.getLogger(__name__)

# Supervisory CCFs for F-IRB per BCBS CRE32.29-32.32
SUPERVISORY_CCFS: dict[str, float] = {
    "committed_unconditionally_cancellable": 0.40,
    "committed_other": 0.75,
    "transaction_related_contingencies": 0.50,
    "trade_related_contingencies": 0.20,
    "note_issuance_facilities": 0.75,
    "direct_credit_substitutes": 1.00,
}

# Basel III / CRR3 Standardized Approach CCFs (BCBS d424 Table 2)
# For SA capital calculation -- distinct from F-IRB supervisory CCFs above.
SA_CCFS: dict[str, float] = {
    "unconditionally_cancellable": 0.10,
    "committed_any_maturity": 0.40,
    "nif_ruf": 0.50,
    "transaction_related": 0.50,
    "trade_related": 0.20,
    "direct_credit_substitutes": 1.00,
}

# A-IRB CCF floors per CRE32.33
CCF_FLOOR_AIRB: float = 0.50


def calculate_ead(
    drawn_amount: float,
    undrawn_commitment: float,
    ccf: float,
) -> float:
    """Calculate Exposure at Default.

    EAD = Drawn + CCF × Undrawn

    Args:
        drawn_amount: Current outstanding balance.
        undrawn_commitment: Undrawn committed amount.
        ccf: Credit Conversion Factor.

    Returns:
        EAD value.
    """
    return drawn_amount + ccf * undrawn_commitment


def estimate_ccf(
    ead_at_default: float,
    drawn_at_reference: float,
    limit: float,
) -> float:
    """Estimate realized CCF from default observation.

    CCF = (EAD - Drawn_ref) / (Limit - Drawn_ref)

    Args:
        ead_at_default: Actual EAD observed at default.
        drawn_at_reference: Drawn amount at reference date (12m before default).
        limit: Total committed limit.

    Returns:
        Realized CCF, clipped to [0, 1].
    """
    undrawn = limit - drawn_at_reference
    if undrawn <= 0:
        return 1.0
    ccf = (ead_at_default - drawn_at_reference) / undrawn
    return float(np.clip(ccf, 0.0, 1.0))


def get_supervisory_ccf(facility_type: str) -> float:
    """Get the F-IRB supervisory CCF for a facility type.

    Args:
        facility_type: Facility type key (see SUPERVISORY_CCFS).

    Returns:
        Supervisory CCF value.

    Raises:
        ValueError: If facility type is unknown.
    """
    ccf = SUPERVISORY_CCFS.get(facility_type)
    if ccf is None:
        raise ValueError(
            f"Unknown facility type: {facility_type}. "
            f"Valid types: {list(SUPERVISORY_CCFS.keys())}"
        )
    return ccf


def apply_ccf_floor(ccf: float, approach: str = "airb") -> float:
    """Apply regulatory CCF floor.

    A-IRB floor per CRE32.33: CCF >= 50% for revolving.
    F-IRB: supervisory values are fixed, no floor needed.

    Args:
        ccf: Estimated CCF.
        approach: "airb" or "firb".

    Returns:
        Floored CCF.
    """
    if approach == "airb":
        return max(ccf, CCF_FLOOR_AIRB)
    return ccf


def ead_term_structure(
    drawn_amount: float,
    undrawn_commitment: float,
    ccf: float,
    n_periods: int,
    amortization_rate: float = 0.0,
) -> np.ndarray:
    """Generate EAD term structure with optional amortization.

    Args:
        drawn_amount: Current drawn amount.
        undrawn_commitment: Undrawn commitment.
        ccf: Credit conversion factor.
        n_periods: Number of periods.
        amortization_rate: Annual amortization rate for the drawn amount.

    Returns:
        Array of EAD values by period.
    """
    eads = np.empty(n_periods, dtype=np.float64)
    for t in range(n_periods):
        remaining_drawn = drawn_amount * (1.0 - amortization_rate) ** t
        ead = remaining_drawn + ccf * undrawn_commitment
        eads[t] = max(ead, 0.0)
    return eads


# ── Sklearn-compatible Estimator ──────────────────────────────────



class EADModel(BaseEstimator, RegressorMixin):  # type: ignore[misc]
    """Sklearn-compatible EAD model.

    Wraps EAD/CCF estimation with fit/predict interface.

    Parameters:
        ccf_method: CCF estimation method ('supervisory', 'estimated').
        facility_type: For supervisory CCF lookup.
    """

    def __init__(
        self,
        ccf_method: str = "supervisory",
        facility_type: str = "committed_other",
    ) -> None:
        self.ccf_method = ccf_method
        self.facility_type = facility_type
        self.mean_ccf_: float | None = None
        self.is_fitted_: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EADModel":  # noqa: N803
        """Fit EAD model. X = [drawn, undrawn], y = realized EAD."""
        y = np.asarray(y, dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)  # noqa: N806
        # Estimate mean CCF from data
        if X.shape[1] >= 2:
            undrawn = X[:, 1]
            drawn = X[:, 0]
            mask = undrawn > 0
            if mask.any():
                ccfs = (y[mask] - drawn[mask]) / undrawn[mask]
                self.mean_ccf_ = float(np.clip(np.mean(ccfs), 0, 1))
            else:
                self.mean_ccf_ = 0.75  # Default
        else:
            self.mean_ccf_ = 0.75
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        """Predict EAD. X columns: [drawn, undrawn]."""
        assert self.is_fitted_, "Call fit() first"
        X = np.asarray(X, dtype=np.float64)  # noqa: N806
        if self.ccf_method == "supervisory":
            ccf = get_supervisory_ccf(self.facility_type)
        else:
            ccf = self.mean_ccf_ or 0.75
        return np.array([calculate_ead(row[0], row[1], ccf) for row in X])
