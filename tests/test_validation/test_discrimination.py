"""Tests for model validation discrimination metrics."""

from unittest.mock import patch

import numpy as np
import pytest

from creditriskengine.validation.discrimination import (
    accuracy_ratio,
    auroc,
    cap_curve,
    divergence,
    gini_coefficient,
    information_value,
    ks_statistic,
    somers_d,
)


class TestAUROC:
    def test_perfect_model(self) -> None:
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_score = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        assert auroc(y_true, y_score) == pytest.approx(1.0, abs=1e-10)

    def test_random_model(self) -> None:
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=10000)
        y_score = rng.random(10000)
        auc = auroc(y_true, y_score)
        assert 0.48 < auc < 0.52

    def test_no_positives(self) -> None:
        assert auroc(np.zeros(10), np.random.rand(10)) == 0.5

    def test_no_negatives(self) -> None:
        assert auroc(np.ones(10), np.random.rand(10)) == 0.5


class TestGiniCoefficient:
    def test_perfect(self) -> None:
        y_true = np.array([1, 1, 0, 0])
        y_score = np.array([0.9, 0.8, 0.2, 0.1])
        assert gini_coefficient(y_true, y_score) == pytest.approx(1.0, abs=1e-10)

    def test_relation_to_auroc(self) -> None:
        rng = np.random.default_rng(123)
        y_true = rng.integers(0, 2, size=500)
        y_score = rng.random(500)
        auc = auroc(y_true, y_score)
        gini = gini_coefficient(y_true, y_score)
        assert gini == pytest.approx(2 * auc - 1, abs=1e-10)


class TestKSStatistic:
    def test_perfect_separation(self) -> None:
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_score = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        ks = ks_statistic(y_true, y_score)
        assert ks == pytest.approx(1.0, abs=0.01)

    def test_no_data_returns_zero(self) -> None:
        assert ks_statistic(np.ones(5), np.random.rand(5)) == 0.0

    def test_ks_in_unit_interval(self) -> None:
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=200)
        y_score = rng.random(200)
        ks = ks_statistic(y_true, y_score)
        assert 0.0 <= ks <= 1.0


class TestDivergence:
    def test_zero_for_identical(self) -> None:
        y_true = np.array([1, 1, 0, 0])
        y_score = np.array([0.5, 0.5, 0.5, 0.5])
        assert divergence(y_true, y_score) == 0.0

    def test_positive_for_separation(self) -> None:
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_score = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        assert divergence(y_true, y_score) > 0.0

    def test_insufficient_samples_returns_zero(self) -> None:
        # Only one default observation (len(defaults) < 2)
        y_true = np.array([1, 0, 0, 0])
        y_score = np.array([0.9, 0.3, 0.2, 0.1])
        assert divergence(y_true, y_score) == 0.0


class TestInformationValue:
    def test_no_signal(self) -> None:
        rng = np.random.default_rng(42)
        feature = rng.random(1000)
        target = rng.integers(0, 2, size=1000)
        iv = information_value(feature, target)
        assert iv < 0.1  # Should be near zero for random

    def test_no_defaults_returns_zero(self) -> None:
        assert information_value(np.random.rand(100), np.zeros(100)) == 0.0

    def test_exception_in_binning_returns_zero(self) -> None:
        feature = np.array([1.0, 2.0, 3.0, 4.0])
        target = np.array([0, 1, 0, 1])
        with patch("numpy.percentile", side_effect=ValueError("bad bins")):
            iv = information_value(feature, target, bins=10)
        assert iv == 0.0


class TestCapCurve:
    def test_basic(self) -> None:
        y_true = np.array([1, 1, 0, 0])
        y_score = np.array([0.9, 0.8, 0.2, 0.1])
        frac_pop, frac_def = cap_curve(y_true, y_score)
        assert len(frac_pop) == 4
        assert len(frac_def) == 4
        assert frac_pop[-1] == pytest.approx(1.0)
        assert frac_def[-1] == pytest.approx(1.0)

    def test_no_defaults(self) -> None:
        y_true = np.array([0, 0, 0])
        y_score = np.array([0.5, 0.3, 0.1])
        frac_pop, frac_def = cap_curve(y_true, y_score)
        # n_defaults == 0, so frac_defaults = cum_defaults (no division)
        assert frac_pop[-1] == pytest.approx(1.0)


class TestAccuracyRatio:
    def test_equals_gini(self) -> None:
        y_true = np.array([1, 1, 0, 0])
        y_score = np.array([0.9, 0.8, 0.2, 0.1])
        ar = accuracy_ratio(y_true, y_score)
        gini = gini_coefficient(y_true, y_score)
        assert ar == pytest.approx(gini)


class TestSomersD:
    def test_equals_gini(self) -> None:
        y_true = np.array([1, 1, 0, 0])
        y_score = np.array([0.9, 0.8, 0.2, 0.1])
        sd = somers_d(y_true, y_score)
        gini = gini_coefficient(y_true, y_score)
        assert sd == pytest.approx(gini)
