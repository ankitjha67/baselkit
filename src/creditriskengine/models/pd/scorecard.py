"""
PD (Probability of Default) modeling framework.

Provides logistic regression scorecard, master scale construction,
and PD calibration using the anchor point method.

References:
- BCBS d350: Regulatory treatment of accounting provisions
- EBA GL/2017/16: PD estimation, LGD estimation
- Engelmann & Rauhmeier: The Basel II Risk Parameters (2nd ed.)
"""

import logging
import math

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)


# ── Logistic Regression Scorecard ──────────────────────────────────


def logistic_score(
    coefficients: np.ndarray,
    features: np.ndarray,
    intercept: float = 0.0,
) -> np.ndarray:
    """Compute logistic regression log-odds scores.

    Score = intercept + sum(coef_i * feature_i)

    Args:
        coefficients: Model coefficients (length = n_features).
        features: Feature matrix (n_obs × n_features).
        intercept: Model intercept.

    Returns:
        Array of log-odds scores.
    """
    features = np.asarray(features, dtype=np.float64)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    return np.asarray(features @ coefficients + intercept)


def score_to_pd(scores: np.ndarray) -> np.ndarray:
    """Convert log-odds scores to PD using the logistic function.

    PD = 1 / (1 + exp(-score))

    Args:
        scores: Log-odds scores.

    Returns:
        Array of PDs in [0, 1].
    """
    scores = np.asarray(scores, dtype=np.float64)
    return np.asarray(1.0 / (1.0 + np.exp(-scores)))


def pd_to_score(
    pds: np.ndarray,
    base_score: float = 600.0,
    base_odds: float = 50.0,
    pdo: float = 20.0,
) -> np.ndarray:
    """Convert PDs to scorecard points using the industry standard formula.

    Score = base_score - pdo/ln(2) * ln(odds)
    where odds = (1-PD)/PD

    Args:
        pds: Probability of default values.
        base_score: Score at which odds = base_odds.
        base_odds: Odds at the base score.
        pdo: Points to double the odds.

    Returns:
        Array of scorecard points.
    """
    pds = np.asarray(pds, dtype=np.float64)
    pds = np.clip(pds, 1e-10, 1.0 - 1e-10)
    odds = (1.0 - pds) / pds
    factor = pdo / math.log(2)
    offset = base_score - factor * math.log(base_odds)
    return np.asarray(offset + factor * np.log(odds))


# ── Master Scale / Rating Grade Construction ───────────────────────


def build_master_scale(
    grade_boundaries: list[float],
    grade_labels: list[str] | None = None,
) -> list[dict[str, object]]:
    """Build a master scale mapping PD ranges to rating grades.

    Args:
        grade_boundaries: Sorted ascending PD boundary values.
            N boundaries produce N-1 grades.
            First boundary is the lower PD bound, last is the upper.
        grade_labels: Optional labels for each grade.

    Returns:
        List of dicts with grade, pd_lower, pd_upper, pd_midpoint.
    """
    if len(grade_boundaries) < 2:
        raise ValueError("Need at least 2 boundaries to define 1 grade")

    n_grades = len(grade_boundaries) - 1
    if grade_labels is None:
        grade_labels = [f"Grade_{i + 1}" for i in range(n_grades)]

    if len(grade_labels) != n_grades:
        raise ValueError(
            f"Expected {n_grades} labels, got {len(grade_labels)}"
        )

    scale = []
    for i in range(n_grades):
        lo = grade_boundaries[i]
        hi = grade_boundaries[i + 1]
        mid = math.sqrt(lo * hi) if lo > 0 else hi / 2.0
        scale.append({
            "grade": grade_labels[i],
            "pd_lower": lo,
            "pd_upper": hi,
            "pd_midpoint": mid,
        })
    return scale


def assign_rating_grade(
    pd: float,
    master_scale: list[dict[str, object]],
) -> str:
    """Assign a rating grade based on PD and master scale.

    Args:
        pd: Probability of default.
        master_scale: Master scale from build_master_scale().

    Returns:
        Grade label.
    """
    for grade in master_scale:
        if grade["pd_lower"] <= pd < grade["pd_upper"]:  # type: ignore[operator]
            return str(grade["grade"])
    # Default to last grade if PD >= last upper boundary
    return str(master_scale[-1]["grade"])


# ── PD Calibration ─────────────────────────────────────────────────


def calibrate_pd_anchor_point(
    central_tendency: float,
    raw_pds: np.ndarray,
) -> np.ndarray:
    """Calibrate PDs using the anchor-point method.

    Scales raw (model) PDs so their portfolio-weighted average
    matches the long-run central tendency.

    Formula:
        calibrated_PD_i = raw_PD_i × (central_tendency / mean(raw_PDs))

    Args:
        central_tendency: Long-run average default rate.
        raw_pds: Raw/uncalibrated PD estimates.

    Returns:
        Calibrated PD array, clipped to [PD_floor, 1.0].
    """
    raw_pds = np.asarray(raw_pds, dtype=np.float64)
    avg = float(np.mean(raw_pds))
    if avg < 1e-15:
        return np.full_like(raw_pds, central_tendency)
    scaling = central_tendency / avg
    calibrated = raw_pds * scaling
    # Basel III PD floor 3 bps per CRE32.13
    return np.clip(calibrated, 0.0003, 1.0)


def calibrate_pd_bayesian(
    prior_pd: float,
    observed_defaults: int,
    n_observations: int,
    weight: float = 0.5,
) -> float:
    """Bayesian PD calibration combining prior with observed data.

    PD_calibrated = weight × prior_PD + (1-weight) × observed_DR

    Args:
        prior_pd: Prior (long-run) PD estimate.
        observed_defaults: Number of observed defaults.
        n_observations: Number of observations.
        weight: Weight on prior (0 = pure data, 1 = pure prior).

    Returns:
        Calibrated PD.
    """
    if n_observations <= 0:
        return prior_pd
    observed_dr = observed_defaults / n_observations
    return weight * prior_pd + (1.0 - weight) * observed_dr


def vasicek_single_factor_pd(
    pd_ttc: float,
    rho: float,
    confidence: float = 0.999,
) -> float:
    """Conditional PD under the Vasicek single-factor model.

    PD_conditional = Phi( (Phi^-1(PD) + sqrt(rho)*Phi^-1(confidence)) / sqrt(1-rho) )

    This is the stressed/downturn PD used in regulatory capital.

    Args:
        pd_ttc: Through-the-cycle PD.
        rho: Asset correlation.
        confidence: Confidence level (default 99.9%).

    Returns:
        Conditional (stressed) PD.
    """
    if pd_ttc <= 0:
        return 0.0
    if pd_ttc >= 1:
        return 1.0
    g_pd = norm.ppf(pd_ttc)
    g_conf = norm.ppf(confidence)
    numerator = g_pd + math.sqrt(rho) * g_conf
    denominator = math.sqrt(1.0 - rho)
    return float(norm.cdf(numerator / denominator))


# ── Sklearn-compatible Estimator ──────────────────────────────────

from sklearn.base import (  # noqa: E402
    BaseEstimator,
    ClassifierMixin,
)


class ScorecardBuilder(BaseEstimator, ClassifierMixin):  # type: ignore[misc]
    """Sklearn-compatible PD scorecard builder.

    Implements fit/predict interface for logistic regression-based PD models.
    Follows sklearn estimator patterns per project spec.

    Parameters:
        base_score: Base scorecard score (default 600).
        pdo: Points to double the odds (default 20).
        base_odds: Odds at base score (default 50).
    """

    def __init__(
        self,
        base_score: float = 600.0,
        pdo: float = 20.0,
        base_odds: float = 50.0,
    ) -> None:
        self.base_score = base_score
        self.pdo = pdo
        self.base_odds = base_odds
        # Fitted attributes (set in fit())
        self.intercept_: float | None = None
        self.coefficients_: np.ndarray | None = None
        self.master_scale_: list[dict[str, float]] | None = None
        self.is_fitted_: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ScorecardBuilder":  # noqa: N803
        """Fit logistic regression and build master scale.

        Uses sklearn LogisticRegression internally.
        """
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        lr.fit(X, y)
        self.intercept_ = float(lr.intercept_[0])
        self.coefficients_ = lr.coef_[0].copy()
        self.is_fitted_ = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        """Predict PD (probability of default) for each observation."""
        assert self.is_fitted_, "Call fit() first"
        assert self.coefficients_ is not None and self.intercept_ is not None
        scores = logistic_score(self.coefficients_, X, self.intercept_)
        # Handle both 1D single-sample and 2D cases
        if scores.ndim == 0:
            scores = np.array([float(scores)])
        pds = score_to_pd(scores)
        return np.column_stack([1 - pds, pds])

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        """Predict default (1) or non-default (0)."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def score_points(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        """Convert to scorecard points."""
        assert self.is_fitted_, "Call fit() first"
        assert self.coefficients_ is not None and self.intercept_ is not None
        scores = logistic_score(self.coefficients_, X, self.intercept_)
        if scores.ndim == 0:
            scores = np.array([float(scores)])
        pds = score_to_pd(scores)
        return pd_to_score(pds, self.base_score, self.base_odds, self.pdo)
