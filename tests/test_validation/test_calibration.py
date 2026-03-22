"""Tests for calibration statistical tests."""

import numpy as np
import pytest

from creditriskengine.validation.calibration import (
    binomial_test,
    brier_score,
    hosmer_lemeshow_test,
    jeffreys_test,
    spiegelhalter_test,
    traffic_light_test,
)


class TestBinomialTest:
    def test_well_calibrated(self) -> None:
        # 10 defaults out of 1000 with predicted PD=1% → consistent
        result = binomial_test(10, 1000, 0.01)
        assert not result["reject_h0"]

    def test_severe_underestimation(self) -> None:
        # 50 defaults out of 1000 with predicted PD=1% → reject
        result = binomial_test(50, 1000, 0.01)
        assert result["reject_h0"]
        assert result["z_stat"] > 0

    def test_zero_observations(self) -> None:
        result = binomial_test(0, 0, 0.01)
        assert result["p_value"] == 1.0
        assert not result["reject_h0"]

    def test_zero_variance_pd(self) -> None:
        """Line 51: std < 1e-15 early return when predicted_pd is 0 or 1."""
        result = binomial_test(0, 100, 0.0)
        assert result["z_stat"] == 0.0
        assert result["p_value"] == 1.0
        assert result["reject_h0"] is False


class TestHosmerLemeshow:
    def test_perfect_calibration(self) -> None:
        observed = np.array([10.0, 20.0, 30.0])
        predicted = np.array([0.01, 0.02, 0.03])
        counts = np.array([1000.0, 1000.0, 1000.0])
        result = hosmer_lemeshow_test(observed, predicted, counts)
        assert not result["reject_h0"]

    def test_poor_calibration(self) -> None:
        observed = np.array([50.0, 50.0, 50.0])
        predicted = np.array([0.01, 0.01, 0.01])
        counts = np.array([1000.0, 1000.0, 1000.0])
        result = hosmer_lemeshow_test(observed, predicted, counts)
        assert result["hl_stat"] > 0
        assert result["reject_h0"]


class TestSpiegelhalter:
    def test_well_calibrated(self) -> None:
        rng = np.random.default_rng(42)
        n = 1000
        preds = rng.uniform(0.01, 0.10, n)
        outcomes = (rng.uniform(0, 1, n) < preds).astype(float)
        result = spiegelhalter_test(outcomes, preds)
        assert "z_stat" in result
        assert "p_value" in result

    def test_empty(self) -> None:
        result = spiegelhalter_test(np.array([]), np.array([]))
        assert not result["reject_h0"]

    def test_zero_variance_brier(self) -> None:
        """Line 133: var_brier < 1e-15 early return (all predictions 0 or 1)."""
        # When all predictions are exactly 0.0 or 1.0, (1-2*y_pred)^2 * y_pred*(1-y_pred) = 0
        y_true = np.array([0.0, 0.0, 1.0, 1.0])
        y_pred = np.array([0.0, 0.0, 1.0, 1.0])
        result = spiegelhalter_test(y_true, y_pred)
        assert result["z_stat"] == 0.0
        assert result["p_value"] == 1.0
        assert result["reject_h0"] is False


class TestTrafficLight:
    def test_green(self) -> None:
        assert traffic_light_test(10, 1000, 0.01) == "green"

    def test_red(self) -> None:
        assert traffic_light_test(100, 1000, 0.01) == "red"

    def test_empty(self) -> None:
        assert traffic_light_test(0, 0, 0.01) == "green"

    def test_yellow(self) -> None:
        """Line 177: yellow return when n_defaults between p_95 and p_9999."""
        from scipy import stats

        n = 1000
        pd_val = 0.01
        p_95 = int(stats.binom.ppf(0.95, n, pd_val))
        p_9999 = int(stats.binom.ppf(0.9999, n, pd_val))
        n_defaults = p_95 + 1  # just above green
        assert n_defaults <= p_9999, "Sanity: must be in yellow zone"
        assert traffic_light_test(n_defaults, n, pd_val) == "yellow"


class TestJeffreys:
    def test_pd_within_interval(self) -> None:
        result = jeffreys_test(10, 1000, 0.01)
        assert result["pd_within_interval"]

    def test_pd_outside_interval(self) -> None:
        result = jeffreys_test(50, 1000, 0.01)
        assert not result["pd_within_interval"]

    def test_posterior_mean_reasonable(self) -> None:
        result = jeffreys_test(10, 1000, 0.01)
        assert 0.005 < result["posterior_mean"] < 0.02


class TestBrierScore:
    def test_perfect_predictions(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.0, 0.0, 1.0, 1.0])
        assert brier_score(y_true, y_pred) == pytest.approx(0.0)

    def test_worst_predictions(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1.0, 1.0, 0.0, 0.0])
        assert brier_score(y_true, y_pred) == pytest.approx(1.0)
