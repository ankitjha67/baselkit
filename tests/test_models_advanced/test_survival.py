"""Tests for survival analysis PD modeling."""

from __future__ import annotations

import numpy as np
import pytest

from creditriskengine.models.pd import (
    CoxPH,
    discrete_hazard_to_pd_curve,
    kaplan_meier,
    nelson_aalen,
    weibull_survival,
)


class TestKaplanMeier:
    def test_no_censoring_decreasing(self) -> None:
        durations = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        events = np.array([1, 1, 1, 1, 1])
        times, surv = kaplan_meier(durations, events)
        assert len(times) == 5
        # Survival is monotonically non-increasing
        assert np.all(np.diff(surv) <= 1e-9)
        assert surv[-1] == pytest.approx(0.0)

    def test_with_censoring(self) -> None:
        durations = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        events = np.array([1, 0, 1, 0, 1])
        times, surv = kaplan_meier(durations, events)
        assert np.all(surv <= 1.0)
        assert np.all(surv >= 0.0)

    def test_no_events(self) -> None:
        durations = np.array([1.0, 2.0, 3.0])
        events = np.array([0, 0, 0])
        times, surv = kaplan_meier(durations, events)
        assert surv[0] == 1.0

    def test_first_step(self) -> None:
        # 5 at risk, 1 event at t=1 → S(1) = 1 - 1/5 = 0.8
        durations = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        events = np.array([1, 1, 1, 1, 1])
        times, surv = kaplan_meier(durations, events)
        assert surv[0] == pytest.approx(0.8)


class TestNelsonAalen:
    def test_increasing(self) -> None:
        durations = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        events = np.array([1, 1, 1, 1, 1])
        times, cumhaz = nelson_aalen(durations, events)
        assert np.all(np.diff(cumhaz) >= -1e-9)

    def test_first_step(self) -> None:
        durations = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        events = np.array([1, 1, 1, 1, 1])
        times, cumhaz = nelson_aalen(durations, events)
        assert cumhaz[0] == pytest.approx(0.2)  # 1/5

    def test_no_events_returns_zero(self) -> None:
        durations = np.array([1.0, 2.0, 3.0])
        events = np.array([0, 0, 0])
        times, cumhaz = nelson_aalen(durations, events)
        np.testing.assert_array_equal(times, np.array([0.0]))
        np.testing.assert_array_equal(cumhaz, np.array([0.0]))


class TestWeibullSurvival:
    def test_at_scale_is_exp_neg1(self) -> None:
        # At t = scale, S = exp(-1) regardless of shape
        s = weibull_survival(2.0, shape=1.5, scale=2.0)
        assert s == pytest.approx(np.exp(-1.0))

    def test_exponential_special_case(self) -> None:
        # shape=1 → exponential: S(t) = exp(-t/scale)
        s = weibull_survival(1.0, shape=1.0, scale=2.0)
        assert s == pytest.approx(np.exp(-0.5))

    def test_decreasing_in_time(self) -> None:
        s1 = weibull_survival(1.0, 1.5, 2.0)
        s2 = weibull_survival(3.0, 1.5, 2.0)
        assert s2 < s1

    def test_invalid_params(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            weibull_survival(1.0, shape=0, scale=2.0)
        with pytest.raises(ValueError, match="must be positive"):
            weibull_survival(1.0, shape=1.5, scale=0)

    def test_array_input(self) -> None:
        s = weibull_survival(np.array([1.0, 2.0, 3.0]), 1.0, 2.0)
        assert len(s) == 3


class TestDiscreteHazardToPD:
    def test_basic(self) -> None:
        hazards = np.array([0.1, 0.1, 0.1])
        result = discrete_hazard_to_pd_curve(hazards)
        # survival: 0.9, 0.81, 0.729
        np.testing.assert_allclose(result["survival"], [0.9, 0.81, 0.729])
        np.testing.assert_allclose(
            result["cumulative_pd"], [0.1, 0.19, 0.271]
        )

    def test_marginal_sums_to_cumulative(self) -> None:
        hazards = np.array([0.05, 0.08, 0.10, 0.12])
        result = discrete_hazard_to_pd_curve(hazards)
        assert np.sum(result["marginal_pd"]) == pytest.approx(
            result["cumulative_pd"][-1]
        )

    def test_invalid_hazard_raises(self) -> None:
        with pytest.raises(ValueError, match="must be in"):
            discrete_hazard_to_pd_curve(np.array([0.1, 1.5]))


class TestCoxPH:
    def test_fit_and_predict(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x = rng.normal(0, 1, (n, 2))
        # Higher x[:,0] → shorter survival
        true_beta = np.array([0.8, -0.3])
        hazard = np.exp(x @ true_beta)
        durations = rng.exponential(1.0 / hazard)
        events = np.ones(n, dtype=int)

        model = CoxPH().fit(x, durations, events)
        assert model.coefficients is not None
        assert len(model.coefficients) == 2
        # First coefficient should be positive (higher hazard)
        assert model.coefficients[0] > 0

    def test_predict_survival_in_unit_interval(self) -> None:
        rng = np.random.default_rng(7)
        n = 100
        x = rng.normal(0, 1, (n, 1))
        durations = rng.exponential(1.0, n)
        events = np.ones(n, dtype=int)
        model = CoxPH().fit(x, durations, events)
        surv = model.predict_survival(x[:5], t=1.0)
        assert np.all(surv >= 0.0)
        assert np.all(surv <= 1.0)

    def test_predict_before_fit_raises(self) -> None:
        model = CoxPH()
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict_survival(np.array([[1.0]]), t=1.0)

    def test_fit_and_predict_with_1d_covariate(self) -> None:
        # 1-D covariate arrays should be reshaped to (n, 1) internally.
        rng = np.random.default_rng(11)
        n = 100
        x = rng.normal(0, 1, n)  # 1-D
        durations = rng.exponential(1.0, n)
        events = np.ones(n, dtype=int)
        model = CoxPH().fit(x, durations, events)
        assert model.coefficients is not None
        assert len(model.coefficients) == 1
        # Predict with a 1-D covariate vector as well.
        surv = model.predict_survival(x[:5], t=1.0)
        assert np.all(surv >= 0.0)
        assert np.all(surv <= 1.0)
