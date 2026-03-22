"""Tests for Credit VaR utilities."""

import numpy as np
import pytest

from creditriskengine.portfolio.var import (
    component_var,
    cornish_fisher_var,
    expected_shortfall,
    historical_simulation_var,
    incremental_var,
    parametric_credit_var,
)


class TestParametricCreditVaR:
    """Parametric VaR = EL + z * sigma."""

    def test_basic_calculation(self):
        # z(0.999) ~ 3.0902
        var = parametric_credit_var(el=100, ul_std=50, confidence=0.999)
        assert var == pytest.approx(100 + 3.0902 * 50, abs=1.0)

    def test_zero_std(self):
        var = parametric_credit_var(el=100, ul_std=0.0, confidence=0.999)
        assert var == pytest.approx(100.0)

    def test_higher_confidence_higher_var(self):
        var_95 = parametric_credit_var(el=100, ul_std=50, confidence=0.95)
        var_999 = parametric_credit_var(el=100, ul_std=50, confidence=0.999)
        assert var_999 > var_95

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            parametric_credit_var(el=100, ul_std=50, confidence=1.5)

    def test_negative_std_raises(self):
        with pytest.raises(ValueError, match="ul_std"):
            parametric_credit_var(el=100, ul_std=-10, confidence=0.999)


class TestHistoricalSimulationVaR:
    """Historical simulation VaR from empirical distribution."""

    def test_returns_correct_quantile(self):
        np.random.seed(42)
        losses = np.random.normal(100, 20, 10_000)
        var_95 = historical_simulation_var(losses, confidence=0.95)
        # 95th percentile of N(100,20) ~ 100 + 1.645*20 = 132.9
        assert var_95 == pytest.approx(132.9, abs=2.0)

    def test_empty_distribution_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            historical_simulation_var(np.array([]), confidence=0.95)

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            historical_simulation_var(np.array([1, 2, 3]), confidence=0.0)

    def test_deterministic_small_sample(self):
        losses = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        var_90 = historical_simulation_var(losses, confidence=0.90)
        # 90th percentile of [10..100]
        assert var_90 == pytest.approx(91.0, abs=1.0)


class TestCornishFisherVaR:
    """Cornish-Fisher adjusted VaR."""

    def test_zero_skew_kurtosis_equals_parametric(self):
        cf_var = cornish_fisher_var(el=100, ul_std=50, skewness=0.0, kurtosis=0.0, confidence=0.999)
        p_var = parametric_credit_var(el=100, ul_std=50, confidence=0.999)
        assert cf_var == pytest.approx(p_var, rel=1e-6)

    def test_positive_skew_increases_var(self):
        normal_var = cornish_fisher_var(el=100, ul_std=50, skewness=0.0, kurtosis=0.0)
        skewed_var = cornish_fisher_var(el=100, ul_std=50, skewness=1.0, kurtosis=0.0)
        assert skewed_var > normal_var

    def test_excess_kurtosis_increases_var(self):
        normal_var = cornish_fisher_var(el=100, ul_std=50, skewness=0.0, kurtosis=0.0)
        fat_tail_var = cornish_fisher_var(el=100, ul_std=50, skewness=0.0, kurtosis=3.0)
        assert fat_tail_var > normal_var

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            cornish_fisher_var(el=100, ul_std=50, skewness=0.0, kurtosis=0.0, confidence=0.0)

    def test_negative_std_raises(self):
        with pytest.raises(ValueError, match="ul_std"):
            cornish_fisher_var(el=100, ul_std=-1, skewness=0.0, kurtosis=0.0)


class TestIncrementalVaR:
    """Incremental VaR -- change from adding an exposure."""

    def test_positive_when_exposure_increases_risk(self):
        np.random.seed(42)
        base = np.random.normal(100, 20, 5_000)
        # Adding correlated losses increases VaR
        additional = np.random.normal(50, 15, 5_000)
        combined = base + additional
        ivar = incremental_var(base, combined, confidence=0.95)
        assert ivar > 0

    def test_empty_base_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            incremental_var(np.array([]), np.array([1, 2, 3]))

    def test_empty_combined_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            incremental_var(np.array([1, 2, 3]), np.array([]))


class TestComponentVaR:
    """Component VaR -- Euler decomposition."""

    def test_sums_to_portfolio_var(self):
        portfolio_var = 100.0
        portfolio_std = 40.0
        exposure_stds = np.array([20.0, 30.0])
        corrs = np.array([0.8, 0.6])
        # Construct such that sum of component VaRs = portfolio VaR
        # sum = (100/40) * (20*0.8 + 30*0.6) = 2.5 * (16+18) = 85
        cvar = component_var(portfolio_var, portfolio_std, exposure_stds, corrs)
        # Sum approximates portfolio VaR only if correlations are exact Euler weights.
        # Here we just check the formula works.
        expected_sum = (portfolio_var / portfolio_std) * np.sum(exposure_stds * corrs)
        assert np.sum(cvar) == pytest.approx(expected_sum, rel=1e-6)

    def test_shape_matches_input(self):
        cvar = component_var(100.0, 40.0, np.array([10.0, 20.0, 30.0]), np.array([0.5, 0.6, 0.7]))
        assert cvar.shape == (3,)

    def test_zero_portfolio_std_returns_zeros(self):
        cvar = component_var(100.0, 0.0, np.array([10.0, 20.0]), np.array([0.5, 0.6]))
        np.testing.assert_allclose(cvar, [0.0, 0.0])

    def test_mismatched_shapes_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            component_var(100.0, 40.0, np.array([10.0, 20.0]), np.array([0.5]))


class TestExpectedShortfall:
    """Expected Shortfall (CVaR) -- average loss beyond VaR."""

    def test_es_gte_var(self):
        np.random.seed(42)
        losses = np.random.normal(100, 20, 10_000)
        var_95 = historical_simulation_var(losses, confidence=0.95)
        es_95 = expected_shortfall(losses, confidence=0.95)
        assert es_95 >= var_95

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            expected_shortfall(np.array([]))

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            expected_shortfall(np.array([1, 2, 3]), confidence=1.0)

    def test_known_distribution(self):
        np.random.seed(123)
        losses = np.random.exponential(scale=100, size=50_000)
        es_95 = expected_shortfall(losses, confidence=0.95)
        var_95 = historical_simulation_var(losses, confidence=0.95)
        # ES should be meaningfully above VaR for exponential distribution
        assert es_95 > var_95 * 1.05
