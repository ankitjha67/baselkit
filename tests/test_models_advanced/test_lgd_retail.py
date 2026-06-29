"""Tests for recovery curves, beta LGD, and roll-rate retail forecasting."""

from __future__ import annotations

import numpy as np
import pytest

from creditriskengine.models.lgd import (
    RecoveryCurveType,
    beta_lgd_mean,
    cumulative_recovery_fraction,
    discounted_workout_lgd,
    downturn_lgd_quantile,
    fit_beta_lgd,
    fit_recovery_curve,
)
from creditriskengine.models.retail import (
    DelinquencyBucket,
    project_charge_off,
    roll_rate_matrix,
)
from creditriskengine.models.retail.roll_rate import N_BUCKETS

# ============================================================================
# Recovery curves
# ============================================================================


class TestRecoveryCurves:
    def test_fit_weibull(self) -> None:
        rng = np.random.default_rng(42)
        times = rng.weibull(1.5, 500) * 2.0 + 0.01
        fit = fit_recovery_curve(times, RecoveryCurveType.WEIBULL)
        assert fit.curve_type == RecoveryCurveType.WEIBULL
        assert fit.shape > 0
        assert fit.scale > 0
        assert fit.mean_recovery_time > 0

    def test_fit_lognormal(self) -> None:
        rng = np.random.default_rng(42)
        times = rng.lognormal(0.5, 0.5, 500)
        fit = fit_recovery_curve(times, RecoveryCurveType.LOGNORMAL)
        assert fit.curve_type == RecoveryCurveType.LOGNORMAL
        assert fit.mean_recovery_time > 0

    def test_fit_gamma(self) -> None:
        rng = np.random.default_rng(42)
        times = rng.gamma(2.0, 1.0, 500)
        fit = fit_recovery_curve(times, RecoveryCurveType.GAMMA)
        assert fit.curve_type == RecoveryCurveType.GAMMA

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty and positive"):
            fit_recovery_curve(np.array([]))

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty and positive"):
            fit_recovery_curve(np.array([1.0, -2.0]))

    def test_cumulative_fraction_monotonic(self) -> None:
        rng = np.random.default_rng(42)
        times = rng.weibull(1.5, 500) * 2.0 + 0.01
        fit = fit_recovery_curve(times)
        f1 = cumulative_recovery_fraction(fit, 1.0)
        f5 = cumulative_recovery_fraction(fit, 5.0)
        assert 0.0 <= f1 <= f5 <= 1.0

    def test_cumulative_at_zero(self) -> None:
        rng = np.random.default_rng(42)
        times = rng.weibull(1.5, 100) + 0.01
        fit = fit_recovery_curve(times)
        assert cumulative_recovery_fraction(fit, 0.0) == 0.0

    def test_discounted_workout_lgd(self) -> None:
        rng = np.random.default_rng(42)
        times = rng.weibull(1.5, 500) * 2.0 + 0.01
        fit = fit_recovery_curve(times)
        # EAD 100, nominal recovery 60, 5% discount
        lgd = discounted_workout_lgd(
            exposure_at_default=100.0,
            total_nominal_recovery=60.0,
            fit=fit,
            discount_rate=0.05,
        )
        # Discounting makes LGD higher than the nominal 40%
        assert 0.40 < lgd < 1.0

    def test_discounted_lgd_zero_ead(self) -> None:
        rng = np.random.default_rng(42)
        fit = fit_recovery_curve(rng.weibull(1.5, 100) + 0.01)
        assert discounted_workout_lgd(0.0, 50.0, fit, 0.05) == 0.0


# ============================================================================
# Beta LGD
# ============================================================================


class TestBetaLGD:
    def test_fit_recovers_mean(self) -> None:
        rng = np.random.default_rng(42)
        lgd_obs = rng.beta(2.0, 3.0, 1000)
        alpha, beta = fit_beta_lgd(lgd_obs)
        fitted_mean = beta_lgd_mean(alpha, beta)
        assert fitted_mean == pytest.approx(np.mean(lgd_obs), abs=0.02)

    def test_mean_formula(self) -> None:
        assert beta_lgd_mean(2.0, 3.0) == pytest.approx(0.4)

    def test_downturn_above_mean(self) -> None:
        alpha, beta = 2.0, 3.0
        mean = beta_lgd_mean(alpha, beta)
        downturn = downturn_lgd_quantile(alpha, beta, 0.90)
        assert downturn > mean

    def test_higher_confidence_higher_downturn(self) -> None:
        d90 = downturn_lgd_quantile(2.0, 3.0, 0.90)
        d99 = downturn_lgd_quantile(2.0, 3.0, 0.99)
        assert d99 > d90

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            fit_beta_lgd(np.array([]))

    def test_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="must be in"):
            fit_beta_lgd(np.array([0.5, 1.5]))

    def test_invalid_params_raise(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            beta_lgd_mean(0, 3.0)
        with pytest.raises(ValueError, match="must be positive"):
            downturn_lgd_quantile(2.0, 0, 0.9)


# ============================================================================
# Roll-rate retail
# ============================================================================


class TestRollRate:
    def _sample_flows(self) -> np.ndarray:
        # 6x6 flow counts (Current → ... → Charge-off)
        flows = np.zeros((N_BUCKETS, N_BUCKETS))
        flows[0] = [900, 100, 0, 0, 0, 0]      # Current
        flows[1] = [400, 200, 400, 0, 0, 0]    # 30 DPD
        flows[2] = [100, 100, 100, 700, 0, 0]  # 60 DPD
        flows[3] = [50, 0, 50, 100, 800, 0]    # 90 DPD
        flows[4] = [0, 0, 0, 0, 200, 800]      # 120+ DPD
        flows[5] = [0, 0, 0, 0, 0, 1000]       # Charge-off
        return flows

    def test_matrix_row_stochastic(self) -> None:
        matrix = roll_rate_matrix(self._sample_flows())
        row_sums = matrix.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(N_BUCKETS))

    def test_charge_off_absorbing(self) -> None:
        matrix = roll_rate_matrix(self._sample_flows())
        co = int(DelinquencyBucket.CHARGE_OFF)
        assert matrix[co, co] == 1.0
        assert matrix[co, :co].sum() == 0.0

    def test_wrong_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="flow_counts must be"):
            roll_rate_matrix(np.zeros((3, 3)))

    def test_projection_increases_charge_off(self) -> None:
        matrix = roll_rate_matrix(self._sample_flows())
        initial = np.array([10000.0, 0, 0, 0, 0, 0])
        result = project_charge_off(initial, matrix, n_periods=12)
        assert result.cumulative_charge_off > 0
        assert 0.0 <= result.charge_off_rate <= 1.0

    def test_projection_conserves_mass(self) -> None:
        matrix = roll_rate_matrix(self._sample_flows())
        initial = np.array([10000.0, 0, 0, 0, 0, 0])
        result = project_charge_off(initial, matrix, n_periods=6)
        # Total balance conserved across buckets each period
        for t in range(result.balances_by_period.shape[0]):
            assert result.balances_by_period[t].sum() == pytest.approx(10000.0)

    def test_invalid_periods_raises(self) -> None:
        matrix = roll_rate_matrix(self._sample_flows())
        initial = np.array([10000.0, 0, 0, 0, 0, 0])
        with pytest.raises(ValueError, match="n_periods must be"):
            project_charge_off(initial, matrix, n_periods=0)

    def test_empty_row_stays(self) -> None:
        # A bucket with no observed flow should self-transition
        flows = np.zeros((N_BUCKETS, N_BUCKETS))
        flows[0] = [900, 100, 0, 0, 0, 0]
        matrix = roll_rate_matrix(flows)
        # Bucket 2 had no flows → stays
        assert matrix[2, 2] == 1.0


# ============================================================================
# Coverage edge cases
# ============================================================================


class TestRecoveryCurveCdfFamilies:
    def test_cumulative_fraction_lognormal(self) -> None:
        rng = np.random.default_rng(7)
        times = rng.lognormal(0.5, 0.5, 500)
        fit = fit_recovery_curve(times, RecoveryCurveType.LOGNORMAL)
        f = cumulative_recovery_fraction(fit, 2.0)
        assert 0.0 <= f <= 1.0

    def test_cumulative_fraction_gamma(self) -> None:
        rng = np.random.default_rng(7)
        times = rng.gamma(2.0, 1.0, 500)
        fit = fit_recovery_curve(times, RecoveryCurveType.GAMMA)
        f = cumulative_recovery_fraction(fit, 2.0)
        assert 0.0 <= f <= 1.0


class TestBetaLGDEdgeCases:
    def test_fit_degenerate_zero_variance(self) -> None:
        # All identical observations → variance 0 → concentration fallback
        alpha, beta = fit_beta_lgd(np.array([0.4, 0.4, 0.4, 0.4]))
        # Mean reproduced as alpha / (alpha + beta)
        assert beta_lgd_mean(alpha, beta) == pytest.approx(0.4)
        assert alpha == pytest.approx(0.4 * 100.0)
        assert beta == pytest.approx(0.6 * 100.0)

    def test_fit_single_observation(self) -> None:
        # len == 1 → variance defaults to 0 → fallback branch
        alpha, beta = fit_beta_lgd(np.array([0.25]))
        assert beta_lgd_mean(alpha, beta) == pytest.approx(0.25)

    def test_downturn_quantile_invalid_confidence(self) -> None:
        with pytest.raises(ValueError, match="confidence_level must be in"):
            downturn_lgd_quantile(2.0, 3.0, confidence_level=1.5)


class TestProjectChargeOffShapeValidation:
    @staticmethod
    def _sample_matrix() -> np.ndarray:
        flows = np.zeros((N_BUCKETS, N_BUCKETS))
        for i in range(N_BUCKETS):
            flows[i, i] = 100.0
        return roll_rate_matrix(flows)

    def test_wrong_initial_balances_shape(self) -> None:
        matrix = self._sample_matrix()
        with pytest.raises(ValueError, match="initial_balances must be"):
            project_charge_off(np.zeros(3), matrix, n_periods=3)

    def test_wrong_transition_matrix_shape(self) -> None:
        initial = np.zeros(N_BUCKETS)
        initial[0] = 1000.0
        with pytest.raises(ValueError, match="transition_matrix must be"):
            project_charge_off(initial, np.zeros((3, 3)), n_periods=3)
