"""Tests for stability monitoring metrics."""

import numpy as np
import pytest

from creditriskengine.validation.stability import (
    characteristic_stability_index,
    herfindahl_index,
    migration_matrix_stability,
    population_stability_index,
)


class TestPSI:
    def test_identical_distributions(self) -> None:
        data = np.random.default_rng(42).normal(0, 1, 1000)
        psi = population_stability_index(data, data)
        assert psi == pytest.approx(0.0, abs=0.01)

    def test_shifted_distribution(self) -> None:
        rng = np.random.default_rng(42)
        expected = rng.normal(0, 1, 1000)
        actual = rng.normal(1.0, 1, 1000)  # shifted mean
        psi = population_stability_index(actual, expected)
        assert psi > 0.10  # significant shift

    def test_precomputed_proportions(self) -> None:
        actual = np.array([0.1, 0.2, 0.3, 0.4])
        expected = np.array([0.1, 0.2, 0.3, 0.4])
        psi = population_stability_index(actual, expected, precomputed=True)
        assert psi == pytest.approx(0.0, abs=1e-6)

    def test_precomputed_different(self) -> None:
        actual = np.array([0.05, 0.15, 0.30, 0.50])
        expected = np.array([0.25, 0.25, 0.25, 0.25])
        psi = population_stability_index(actual, expected, precomputed=True)
        assert psi > 0.0


class TestCSI:
    def test_delegates_to_psi(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 500)
        assert characteristic_stability_index(data, data) == pytest.approx(
            population_stability_index(data, data), abs=1e-6
        )


class TestHerfindahl:
    def test_perfect_concentration(self) -> None:
        shares = np.array([1.0])
        assert herfindahl_index(shares) == pytest.approx(1.0)

    def test_perfect_diversification(self) -> None:
        n = 100
        shares = np.full(n, 1.0 / n)
        assert herfindahl_index(shares) == pytest.approx(1.0 / n)

    def test_two_equal_shares(self) -> None:
        shares = np.array([0.5, 0.5])
        assert herfindahl_index(shares) == pytest.approx(0.5)


class TestMigrationMatrixStability:
    def test_identical_matrices(self) -> None:
        m = np.eye(3)
        result = migration_matrix_stability(m, m)
        assert result["frobenius_norm"] == pytest.approx(0.0)
        assert result["max_abs_diff"] == pytest.approx(0.0)

    def test_different_matrices(self) -> None:
        m1 = np.eye(3)
        m2 = np.array([[0.9, 0.1, 0.0], [0.0, 0.9, 0.1], [0.0, 0.0, 1.0]])
        result = migration_matrix_stability(m1, m2)
        assert result["frobenius_norm"] > 0
        assert result["max_abs_diff"] == pytest.approx(0.1)
