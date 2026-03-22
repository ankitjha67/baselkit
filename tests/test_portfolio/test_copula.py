"""Tests for Gaussian copula Monte Carlo simulation."""

import numpy as np
import pytest

from creditriskengine.portfolio.copula import (
    credit_var,
    expected_shortfall,
    loss_distribution_stats,
    simulate_multi_factor,
    simulate_single_factor,
)


class TestSingleFactorSimulation:
    def test_shape(self):
        pds = np.array([0.02, 0.03, 0.01])
        lgds = np.array([0.45, 0.40, 0.50])
        eads = np.array([100.0, 200.0, 150.0])
        losses = simulate_single_factor(pds, lgds, eads, rho=0.15, n_simulations=1000, seed=42)
        assert losses.shape == (1000,)

    def test_losses_non_negative(self):
        pds = np.array([0.02, 0.03])
        lgds = np.array([0.45, 0.40])
        eads = np.array([100.0, 200.0])
        losses = simulate_single_factor(pds, lgds, eads, rho=0.15, n_simulations=1000, seed=42)
        assert np.all(losses >= 0)

    def test_mean_near_expected_loss(self):
        n = 50
        pds = np.full(n, 0.02)
        lgds = np.full(n, 0.45)
        eads = np.full(n, 100.0)
        losses = simulate_single_factor(pds, lgds, eads, rho=0.15, n_simulations=50_000, seed=123)
        el = np.sum(pds * lgds * eads)
        assert np.mean(losses) == pytest.approx(el, rel=0.1)

    def test_invalid_rho_raises(self):
        pds = np.array([0.02])
        lgds = np.array([0.45])
        eads = np.array([100.0])
        with pytest.raises(ValueError, match="rho must be in"):
            simulate_single_factor(pds, lgds, eads, rho=1.0, seed=42)


class TestMultiFactorSimulation:
    def test_shape(self):
        pds = np.array([0.02, 0.03])
        lgds = np.array([0.45, 0.40])
        eads = np.array([100.0, 200.0])
        loadings = np.array([[0.3, 0.1], [0.1, 0.3]])
        losses = simulate_multi_factor(pds, lgds, eads, loadings, n_simulations=500, seed=42)
        assert losses.shape == (500,)


class TestCreditVaR:
    def test_exceeds_mean(self):
        losses = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        var = credit_var(losses, 0.99)
        assert var >= np.mean(losses)


class TestExpectedShortfall:
    def test_exceeds_var(self):
        rng = np.random.default_rng(42)
        losses = rng.exponential(10.0, 10_000)
        var = credit_var(losses, 0.99)
        es = expected_shortfall(losses, 0.99)
        assert es >= var


class TestLossDistributionStats:
    def test_keys(self):
        losses = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = loss_distribution_stats(losses, total_ead=100.0)
        assert "mean_loss" in result
        assert "var_999" in result
        assert "es_999" in result
        assert "skewness" in result
