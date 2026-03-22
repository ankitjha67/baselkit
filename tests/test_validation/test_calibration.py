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
    def test_well_calibrated(self):
        # 10 defaults out of 1000 with predicted PD=1% → consistent
        result = binomial_test(10, 1000, 0.01)
        assert result["reject_h0"] == False

    def test_severe_underestimation(self):
        # 50 defaults out of 1000 with predicted PD=1% → reject
        result = binomial_test(50, 1000, 0.01)
        assert result["reject_h0"] == True
        assert result["z_stat"] > 0

    def test_zero_observations(self):
        result = binomial_test(0, 0, 0.01)
        assert result["p_value"] == 1.0
        assert result["reject_h0"] == False


class TestHosmerLemeshow:
    def test_perfect_calibration(self):
        observed = np.array([10.0, 20.0, 30.0])
        predicted = np.array([0.01, 0.02, 0.03])
        counts = np.array([1000.0, 1000.0, 1000.0])
        result = hosmer_lemeshow_test(observed, predicted, counts)
        assert result["reject_h0"] == False

    def test_poor_calibration(self):
        observed = np.array([50.0, 50.0, 50.0])
        predicted = np.array([0.01, 0.01, 0.01])
        counts = np.array([1000.0, 1000.0, 1000.0])
        result = hosmer_lemeshow_test(observed, predicted, counts)
        assert result["hl_stat"] > 0
        assert result["reject_h0"] == True


class TestSpiegelhalter:
    def test_well_calibrated(self):
        rng = np.random.default_rng(42)
        n = 1000
        preds = rng.uniform(0.01, 0.10, n)
        outcomes = (rng.uniform(0, 1, n) < preds).astype(float)
        result = spiegelhalter_test(outcomes, preds)
        assert "z_stat" in result
        assert "p_value" in result

    def test_empty(self):
        result = spiegelhalter_test(np.array([]), np.array([]))
        assert result["reject_h0"] == False


class TestTrafficLight:
    def test_green(self):
        assert traffic_light_test(10, 1000, 0.01) == "green"

    def test_red(self):
        assert traffic_light_test(100, 1000, 0.01) == "red"

    def test_empty(self):
        assert traffic_light_test(0, 0, 0.01) == "green"


class TestJeffreys:
    def test_pd_within_interval(self):
        result = jeffreys_test(10, 1000, 0.01)
        assert result["pd_within_interval"] == True

    def test_pd_outside_interval(self):
        result = jeffreys_test(50, 1000, 0.01)
        assert result["pd_within_interval"] == False

    def test_posterior_mean_reasonable(self):
        result = jeffreys_test(10, 1000, 0.01)
        assert 0.005 < result["posterior_mean"] < 0.02


class TestBrierScore:
    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.0, 0.0, 1.0, 1.0])
        assert brier_score(y_true, y_pred) == pytest.approx(0.0)

    def test_worst_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1.0, 1.0, 0.0, 0.0])
        assert brier_score(y_true, y_pred) == pytest.approx(1.0)
