"""Tests for model validation discrimination metrics."""

import numpy as np
import pytest

from creditriskengine.validation.discrimination import (
    auroc,
    divergence,
    gini_coefficient,
    information_value,
    ks_statistic,
)


class TestAUROC:
    def test_perfect_model(self):
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_score = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        assert auroc(y_true, y_score) == pytest.approx(1.0, abs=1e-10)

    def test_random_model(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=10000)
        y_score = rng.random(10000)
        auc = auroc(y_true, y_score)
        assert 0.48 < auc < 0.52

    def test_no_positives(self):
        assert auroc(np.zeros(10), np.random.rand(10)) == 0.5

    def test_no_negatives(self):
        assert auroc(np.ones(10), np.random.rand(10)) == 0.5


class TestGiniCoefficient:
    def test_perfect(self):
        y_true = np.array([1, 1, 0, 0])
        y_score = np.array([0.9, 0.8, 0.2, 0.1])
        assert gini_coefficient(y_true, y_score) == pytest.approx(1.0, abs=1e-10)

    def test_relation_to_auroc(self):
        rng = np.random.default_rng(123)
        y_true = rng.integers(0, 2, size=500)
        y_score = rng.random(500)
        auc = auroc(y_true, y_score)
        gini = gini_coefficient(y_true, y_score)
        assert gini == pytest.approx(2 * auc - 1, abs=1e-10)


class TestKSStatistic:
    def test_perfect_separation(self):
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_score = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        ks = ks_statistic(y_true, y_score)
        assert ks == pytest.approx(1.0, abs=0.01)

    def test_no_data_returns_zero(self):
        assert ks_statistic(np.ones(5), np.random.rand(5)) == 0.0

    def test_ks_in_unit_interval(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=200)
        y_score = rng.random(200)
        ks = ks_statistic(y_true, y_score)
        assert 0.0 <= ks <= 1.0


class TestDivergence:
    def test_zero_for_identical(self):
        y_true = np.array([1, 1, 0, 0])
        y_score = np.array([0.5, 0.5, 0.5, 0.5])
        assert divergence(y_true, y_score) == 0.0

    def test_positive_for_separation(self):
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_score = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        assert divergence(y_true, y_score) > 0.0


class TestInformationValue:
    def test_no_signal(self):
        rng = np.random.default_rng(42)
        feature = rng.random(1000)
        target = rng.integers(0, 2, size=1000)
        iv = information_value(feature, target)
        assert iv < 0.1  # Should be near zero for random

    def test_no_defaults_returns_zero(self):
        assert information_value(np.random.rand(100), np.zeros(100)) == 0.0
