"""Tests for Margin of Conservatism (MoC) -- ECB Guide to Internal Models, Chapter 7."""

import numpy as np
import pytest

from creditriskengine.models.pd.margin_of_conservatism import (
    MoCCategory,
    MoCComponent,
    MoCResult,
    apply_moc_to_pd_curve,
    calculate_total_moc,
    data_representativeness_moc,
    estimation_error_moc,
    model_uncertainty_moc,
)


class TestEstimationErrorMoC:
    """Category A: sampling/estimation error MoC."""

    def test_returns_positive_value(self) -> None:
        moc = estimation_error_moc(pd_estimate=0.02, n_observations=1000, n_defaults=20)
        assert moc > 0

    def test_larger_sample_smaller_moc(self) -> None:
        moc_small = estimation_error_moc(pd_estimate=0.02, n_observations=100, n_defaults=2)
        moc_large = estimation_error_moc(pd_estimate=0.02, n_observations=10_000, n_defaults=200)
        assert moc_small > moc_large

    def test_zero_observations_returns_pd(self) -> None:
        moc = estimation_error_moc(pd_estimate=0.05, n_observations=0, n_defaults=0)
        assert moc == pytest.approx(0.05)

    def test_result_bounded(self) -> None:
        moc = estimation_error_moc(pd_estimate=0.50, n_observations=50, n_defaults=25)
        # MoC should not exceed 1 - pd_estimate (since upper bound capped at 1.0)
        assert moc <= 0.50 + 0.01  # small tolerance

    def test_high_confidence_larger_moc(self) -> None:
        moc_75 = estimation_error_moc(
            pd_estimate=0.02, n_observations=500, n_defaults=10, confidence_level=0.75
        )
        moc_95 = estimation_error_moc(
            pd_estimate=0.02, n_observations=500, n_defaults=10, confidence_level=0.95
        )
        assert moc_95 > moc_75


class TestDataRepresentativenessMoC:
    """Category B: data representativeness MoC."""

    def test_zero_psi_only_dr_diff(self) -> None:
        moc = data_representativeness_moc(
            development_default_rate=0.03,
            application_default_rate=0.05,
            psi=0.0,
        )
        assert moc == pytest.approx(0.02, abs=1e-6)

    def test_moderate_psi_adds_25bps(self) -> None:
        moc = data_representativeness_moc(
            development_default_rate=0.03,
            application_default_rate=0.03,
            psi=0.15,
        )
        assert moc == pytest.approx(0.0025, abs=1e-6)

    def test_high_psi_adds_50bps(self) -> None:
        moc = data_representativeness_moc(
            development_default_rate=0.03,
            application_default_rate=0.03,
            psi=0.30,
        )
        assert moc == pytest.approx(0.0050, abs=1e-6)

    def test_low_psi_no_penalty(self) -> None:
        moc = data_representativeness_moc(
            development_default_rate=0.03,
            application_default_rate=0.03,
            psi=0.05,
        )
        assert moc == pytest.approx(0.0, abs=1e-6)

    def test_higher_psi_higher_moc(self) -> None:
        moc_low = data_representativeness_moc(0.03, 0.03, psi=0.05)
        moc_high = data_representativeness_moc(0.03, 0.03, psi=0.30)
        assert moc_high > moc_low


class TestModelUncertaintyMoC:
    """Category C: model risk MoC."""

    def test_positive_gap(self) -> None:
        moc = model_uncertainty_moc(base_pd=0.02, challenger_pd=0.04)
        assert moc == pytest.approx(0.02, abs=1e-6)

    def test_zero_gap_when_challenger_lower(self) -> None:
        moc = model_uncertainty_moc(base_pd=0.05, challenger_pd=0.03)
        assert moc == pytest.approx(0.0, abs=1e-6)

    def test_proportional_to_gap(self) -> None:
        moc_small = model_uncertainty_moc(base_pd=0.02, challenger_pd=0.03)
        moc_large = model_uncertainty_moc(base_pd=0.02, challenger_pd=0.06)
        assert moc_large > moc_small

    def test_multiple_challengers_shrinks_moc(self) -> None:
        moc_1 = model_uncertainty_moc(base_pd=0.02, challenger_pd=0.04, n_challengers=1)
        moc_4 = model_uncertainty_moc(base_pd=0.02, challenger_pd=0.04, n_challengers=4)
        assert moc_1 > moc_4
        # With 4 challengers: gap/sqrt(4) = gap/2
        assert moc_4 == pytest.approx(moc_1 / 2.0, rel=1e-6)

    def test_zero_challengers_treated_as_one(self) -> None:
        moc = model_uncertainty_moc(base_pd=0.02, challenger_pd=0.04, n_challengers=0)
        expected = model_uncertainty_moc(base_pd=0.02, challenger_pd=0.04, n_challengers=1)
        assert moc == pytest.approx(expected, rel=1e-6)


class TestCalculateTotalMoC:
    """Aggregation of MoC components."""

    def test_basic_aggregation(self) -> None:
        components = [
            MoCComponent(category=MoCCategory.ESTIMATION_ERROR, description="A", adjustment_bps=50),
            MoCComponent(category=MoCCategory.DATA_DEFICIENCY, description="B", adjustment_bps=25),
        ]
        result = calculate_total_moc(base_pd=0.02, components=components)
        assert isinstance(result, MoCResult)
        # 75 bps = 0.0075
        assert result.moc_additive == pytest.approx(0.0075, abs=1e-6)
        assert result.adjusted_pd == pytest.approx(0.02 + 0.0075, abs=1e-6)

    def test_pd_floor_applied(self) -> None:
        components = []
        result = calculate_total_moc(base_pd=0.0001, components=components)
        # Basel III PD floor: 0.0003
        assert result.adjusted_pd == pytest.approx(0.0003, abs=1e-6)

    def test_pd_capped_at_1(self) -> None:
        components = [
            MoCComponent(
                category=MoCCategory.ESTIMATION_ERROR,
                description="Huge", adjustment_bps=50_000,
            ),
        ]
        result = calculate_total_moc(base_pd=0.50, components=components)
        assert result.adjusted_pd == pytest.approx(1.0, abs=1e-6)

    def test_components_preserved(self) -> None:
        components = [
            MoCComponent(category=MoCCategory.ESTIMATION_ERROR, description="A", adjustment_bps=30),
        ]
        result = calculate_total_moc(base_pd=0.02, components=components)
        assert len(result.components) == 1
        assert result.components[0].adjustment_bps == 30


class TestApplyMoCToPDCurve:
    """Apply MoC to PD term structure."""

    def test_shifts_all_values_up(self) -> None:
        curve = np.array([0.01, 0.02, 0.03, 0.05])
        moc_result = MoCResult(base_pd=0.01, moc_additive=0.005, adjusted_pd=0.015, components=[])
        adjusted = apply_moc_to_pd_curve(curve, moc_result)
        np.testing.assert_allclose(adjusted, [0.015, 0.025, 0.035, 0.055])

    def test_floor_applied(self) -> None:
        curve = np.array([0.0001, 0.0002])
        moc_result = MoCResult(
            base_pd=0.0001, moc_additive=0.00005,
            adjusted_pd=0.0003, components=[],
        )
        adjusted = apply_moc_to_pd_curve(curve, moc_result)
        assert all(p >= 0.0003 for p in adjusted)

    def test_cap_at_1(self) -> None:
        curve = np.array([0.95, 0.98])
        moc_result = MoCResult(base_pd=0.95, moc_additive=0.10, adjusted_pd=1.0, components=[])
        adjusted = apply_moc_to_pd_curve(curve, moc_result)
        assert all(p <= 1.0 for p in adjusted)

    def test_shape_preserved(self) -> None:
        curve = np.array([0.01, 0.02, 0.03])
        moc_result = MoCResult(base_pd=0.01, moc_additive=0.005, adjusted_pd=0.015, components=[])
        adjusted = apply_moc_to_pd_curve(curve, moc_result)
        assert adjusted.shape == curve.shape
