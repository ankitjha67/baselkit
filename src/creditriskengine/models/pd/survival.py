"""
Survival analysis for lifetime PD modeling.

Reference:
    - IFRS 9.B5.5.28-29 (lifetime ECL term structure).
    - Cox (1972) — Proportional hazards regression.
    - Kaplan & Meier (1958) — Non-parametric survival estimation.
    - Nelson (1972), Aalen (1978) — Cumulative hazard estimation.

Survival analysis is the canonical method for IFRS 9 lifetime PD on
long-tenor exposures (mortgages, project finance) where transition-
matrix approaches break down. Provides:
    - Kaplan-Meier survival estimator
    - Nelson-Aalen cumulative hazard
    - Weibull / exponential parametric survival
    - Cox proportional hazards regression
    - Discrete-time hazard → cumulative/marginal PD term structure
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def kaplan_meier(
    durations: np.ndarray,
    event_observed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the survival function via Kaplan-Meier.

    S(t) = prod_{t_i <= t} (1 - d_i / n_i)

    where d_i = events at time t_i, n_i = at-risk just before t_i.

    Args:
        durations: Observed time-to-event or censoring for each subject.
        event_observed: 1 if default observed, 0 if censored.

    Returns:
        Tuple of (unique_times, survival_probabilities), both sorted
        by time ascending.

    Reference:
        Kaplan & Meier (1958).
    """
    durations = np.asarray(durations, dtype=np.float64)
    event_observed = np.asarray(event_observed, dtype=np.int64)

    unique_times = np.unique(durations[event_observed == 1])
    if len(unique_times) == 0:
        return np.array([0.0]), np.array([1.0])

    survival = []
    s = 1.0
    for t in unique_times:
        n_at_risk = np.sum(durations >= t)
        d_events = np.sum((durations == t) & (event_observed == 1))
        if n_at_risk > 0:
            s *= 1.0 - d_events / n_at_risk
        survival.append(s)

    return unique_times, np.array(survival)


def nelson_aalen(
    durations: np.ndarray,
    event_observed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the cumulative hazard via Nelson-Aalen.

    H(t) = sum_{t_i <= t} (d_i / n_i)

    Args:
        durations: Observed time-to-event or censoring.
        event_observed: 1 if default observed, 0 if censored.

    Returns:
        Tuple of (unique_times, cumulative_hazard).

    Reference:
        Nelson (1972), Aalen (1978).
    """
    durations = np.asarray(durations, dtype=np.float64)
    event_observed = np.asarray(event_observed, dtype=np.int64)

    unique_times = np.unique(durations[event_observed == 1])
    if len(unique_times) == 0:
        return np.array([0.0]), np.array([0.0])

    cumulative = []
    h = 0.0
    for t in unique_times:
        n_at_risk = np.sum(durations >= t)
        d_events = np.sum((durations == t) & (event_observed == 1))
        if n_at_risk > 0:
            h += d_events / n_at_risk
        cumulative.append(h)

    return unique_times, np.array(cumulative)


def weibull_survival(
    t: np.ndarray | float,
    shape: float,
    scale: float,
) -> np.ndarray:
    """Weibull survival function.

    S(t) = exp(-(t / scale)^shape)

    Args:
        t: Time(s) at which to evaluate survival.
        shape: Weibull shape parameter k (>0). k>1 = increasing hazard,
            k<1 = decreasing hazard, k=1 = exponential (constant hazard).
        scale: Weibull scale parameter lambda (>0).

    Returns:
        Survival probability/probabilities.

    Raises:
        ValueError: If shape or scale is non-positive.
    """
    if shape <= 0 or scale <= 0:
        raise ValueError("shape and scale must be positive")
    t_arr = np.asarray(t, dtype=np.float64)
    return np.exp(-((np.maximum(t_arr, 0.0) / scale) ** shape))


def discrete_hazard_to_pd_curve(
    hazard_rates: np.ndarray,
) -> dict[str, np.ndarray]:
    """Convert a discrete-time hazard sequence to PD term structure.

    Given per-period hazard rates h_t (conditional default probability
    in period t given survival to t), compute:
        survival(t)    = prod_{s<=t} (1 - h_s)
        cumulative(t)  = 1 - survival(t)
        marginal(t)    = survival(t-1) - survival(t)

    Args:
        hazard_rates: Array of per-period conditional default
            probabilities, each in [0, 1].

    Returns:
        Dict with keys ``survival``, ``cumulative_pd``, ``marginal_pd``.

    Raises:
        ValueError: If any hazard is outside [0, 1].

    Reference:
        IFRS 9.B5.5.28-29.
    """
    hazard_rates = np.asarray(hazard_rates, dtype=np.float64)
    if np.any(hazard_rates < 0) or np.any(hazard_rates > 1):
        raise ValueError("hazard rates must be in [0, 1]")

    survival = np.cumprod(1.0 - hazard_rates)
    cumulative_pd = 1.0 - survival

    survival_prev = np.concatenate([[1.0], survival[:-1]])
    marginal_pd = survival_prev - survival

    return {
        "survival": survival,
        "cumulative_pd": cumulative_pd,
        "marginal_pd": marginal_pd,
    }


class CoxPH:
    """Cox proportional hazards regression.

    Fits the partial likelihood (Breslow tie handling) to estimate
    covariate coefficients, then estimates the baseline cumulative
    hazard for survival prediction.

    Model:
        h(t | x) = h_0(t) * exp(beta . x)

    Reference:
        Cox (1972).
    """

    def __init__(self) -> None:
        self.coefficients: np.ndarray | None = None
        self.baseline_times_: np.ndarray | None = None
        self.baseline_cumhaz_: np.ndarray | None = None

    def _neg_log_partial_likelihood(
        self,
        beta: np.ndarray,
        x: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray,
    ) -> float:
        """Breslow negative log partial likelihood."""
        risk_scores = x @ beta
        ll = 0.0
        event_times = np.unique(durations[events == 1])
        for t in event_times:
            at_risk = durations >= t
            dying = (durations == t) & (events == 1)
            ll += np.sum(risk_scores[dying])
            risk_sum = np.sum(np.exp(risk_scores[at_risk]))
            d = np.sum(dying)
            ll -= d * np.log(risk_sum)
        return -ll

    def fit(
        self,
        x: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray,
    ) -> CoxPH:
        """Fit the Cox model via partial likelihood maximisation.

        Args:
            x: Covariate matrix (n_samples, n_features).
            durations: Time-to-event or censoring per sample.
            events: 1 if default observed, 0 if censored.

        Returns:
            Self (fitted).
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        durations = np.asarray(durations, dtype=np.float64)
        events = np.asarray(events, dtype=np.int64)

        n_features = x.shape[1]
        beta0 = np.zeros(n_features)

        result = minimize(
            self._neg_log_partial_likelihood,
            beta0,
            args=(x, durations, events),
            method="BFGS",
        )
        self.coefficients = result.x

        # Breslow baseline cumulative hazard
        event_times = np.unique(durations[events == 1])
        risk_scores = np.exp(x @ self.coefficients)
        cumhaz = []
        h = 0.0
        for t in event_times:
            at_risk = durations >= t
            d = np.sum((durations == t) & (events == 1))
            denom = np.sum(risk_scores[at_risk])
            if denom > 0:
                h += d / denom
            cumhaz.append(h)
        self.baseline_times_ = event_times
        self.baseline_cumhaz_ = np.array(cumhaz)

        return self

    def predict_survival(
        self,
        x: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """Predict survival probability at time t for given covariates.

        S(t | x) = exp(-H_0(t) * exp(beta . x))

        Args:
            x: Covariate matrix (n_samples, n_features).
            t: Time at which to evaluate survival.

        Returns:
            Survival probabilities (n_samples,).

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if (
            self.coefficients is None
            or self.baseline_cumhaz_ is None
            or self.baseline_times_ is None
        ):
            raise RuntimeError("CoxPH model must be fitted before prediction")

        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        # Baseline cumulative hazard at time t (step function)
        idx = int(self.baseline_times_.searchsorted(t, side="right")) - 1
        h0_t = float(self.baseline_cumhaz_[idx]) if idx >= 0 else 0.0

        risk = np.exp(x @ self.coefficients)
        return np.asarray(np.exp(-h0_t * risk))
