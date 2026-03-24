"""Tests for PD term structure construction and manipulation."""

import numpy as np
import pytest

from creditriskengine.models.pd.term_structure import (
    forward_pd,
    interpolate_pd_term_structure,
    pd_term_structure_from_hazard,
    pd_term_structure_from_transitions,
)

# -- pd_term_structure_from_hazard ------------------------------------------


class TestPdTermStructureFromHazard:
    def test_basic_curve(self):
        """Cumulative PD grows monotonically toward 1."""
        curve = pd_term_structure_from_hazard(0.02, max_years=10)
        assert len(curve) == 10
        # Year 1: 1 - (1-0.02)^1 = 0.02
        assert curve[0] == pytest.approx(0.02, abs=1e-10)
        # Year 2: 1 - (0.98)^2
        assert curve[1] == pytest.approx(1.0 - 0.98**2, abs=1e-10)
        # Monotonically increasing
        assert np.all(np.diff(curve) > 0)

    def test_zero_pd(self):
        """Zero PD produces an all-zero curve."""
        curve = pd_term_structure_from_hazard(0.0, max_years=5)
        np.testing.assert_array_equal(curve, np.zeros(5))

    def test_one_pd(self):
        """PD=1 produces an all-ones curve."""
        curve = pd_term_structure_from_hazard(1.0, max_years=5)
        np.testing.assert_array_almost_equal(curve, np.ones(5))

    def test_long_horizon_approaches_one(self):
        """Over a long horizon cumulative PD approaches 1."""
        curve = pd_term_structure_from_hazard(0.05, max_years=200)
        assert curve[-1] > 0.999

    def test_invalid_pd_raises(self):
        with pytest.raises(ValueError):
            pd_term_structure_from_hazard(-0.1)
        with pytest.raises(ValueError):
            pd_term_structure_from_hazard(1.5)

    def test_invalid_years_raises(self):
        with pytest.raises(ValueError):
            pd_term_structure_from_hazard(0.01, max_years=0)


# -- pd_term_structure_from_transitions -------------------------------------


class TestPdTermStructureFromTransitions:
    @pytest.fixture()
    def simple_tm(self):
        """3-state transition matrix: grade 0, grade 1, default."""
        return np.array([
            [0.90, 0.05, 0.05],
            [0.10, 0.80, 0.10],
            [0.00, 0.00, 1.00],
        ])

    def test_basic_curve(self, simple_tm):
        curve = pd_term_structure_from_transitions(simple_tm, initial_rating=0, max_years=5)
        assert len(curve) == 5
        # Year 1: default probability for grade 0
        assert curve[0] == pytest.approx(0.05, abs=1e-10)
        # Monotonically increasing (default is absorbing)
        assert np.all(np.diff(curve) >= 0)

    def test_default_state_stays_defaulted(self, simple_tm):
        """Starting in default state gives cum_pd = 1 for all years."""
        curve = pd_term_structure_from_transitions(simple_tm, initial_rating=2, max_years=5)
        np.testing.assert_array_almost_equal(curve, np.ones(5))

    def test_approaches_one(self, simple_tm):
        """Over long horizon, cumulative PD approaches 1."""
        curve = pd_term_structure_from_transitions(simple_tm, initial_rating=0, max_years=200)
        assert curve[-1] > 0.99

    def test_invalid_rating_raises(self, simple_tm):
        with pytest.raises(ValueError):
            pd_term_structure_from_transitions(simple_tm, initial_rating=5)

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            pd_term_structure_from_transitions(np.ones((2, 3)), initial_rating=0)


# -- interpolate_pd_term_structure ------------------------------------------


class TestInterpolatePdTermStructure:
    def test_integer_years_match(self):
        """Interpolation at integer years reproduces the original curve."""
        curve = pd_term_structure_from_hazard(0.03, max_years=10)
        target = np.arange(1, 11, dtype=float)
        interp = interpolate_pd_term_structure(curve, target)
        np.testing.assert_array_almost_equal(interp, curve, decimal=10)

    def test_fractional_years(self):
        """Interpolated values lie between neighbours."""
        curve = pd_term_structure_from_hazard(0.05, max_years=10)
        target = np.array([1.5, 3.5, 7.5])
        interp = interpolate_pd_term_structure(curve, target)
        # 1.5-year PD should lie between year-1 and year-2 PDs
        assert curve[0] < interp[0] < curve[1]
        assert curve[2] < interp[1] < curve[3]

    def test_year_zero(self):
        """Interpolation at year 0 gives PD = 0."""
        curve = pd_term_structure_from_hazard(0.02, max_years=5)
        interp = interpolate_pd_term_structure(curve, np.array([0.0]))
        assert interp[0] == pytest.approx(0.0, abs=1e-12)


# -- forward_pd -------------------------------------------------------------


class TestForwardPd:
    def test_constant_hazard_gives_constant_forward(self):
        """Under constant hazard, forward PDs should all be equal."""
        annual_pd = 0.03
        curve = pd_term_structure_from_hazard(annual_pd, max_years=20)
        fwd = forward_pd(curve)
        np.testing.assert_array_almost_equal(fwd, np.full(20, annual_pd), decimal=10)

    def test_reconstruction(self):
        """Forward PDs can reconstruct the original cumulative curve."""
        curve = pd_term_structure_from_hazard(0.05, max_years=10)
        fwd = forward_pd(curve)
        # Reconstruct: survival = product(1 - fwd_pd)
        survival = np.cumprod(1.0 - fwd)
        reconstructed = 1.0 - survival
        np.testing.assert_array_almost_equal(reconstructed, curve, decimal=10)

    def test_first_element(self):
        """First forward PD equals first cumulative PD."""
        curve = np.array([0.02, 0.05, 0.10])
        fwd = forward_pd(curve)
        assert fwd[0] == pytest.approx(0.02, abs=1e-12)

    def test_increasing_cumulative(self):
        """Forward PDs are non-negative for a non-decreasing cumulative curve."""
        curve = np.array([0.01, 0.03, 0.06, 0.10, 0.15])
        fwd = forward_pd(curve)
        assert np.all(fwd >= -1e-12)

    def test_fully_defaulted(self):
        """When cumulative PD hits 1, subsequent forward PDs are 0."""
        curve = np.array([0.5, 1.0, 1.0, 1.0])
        fwd = forward_pd(curve)
        assert fwd[0] == pytest.approx(0.5)
        assert fwd[1] == pytest.approx(1.0)
        # After full default, forward PD is 0 (no survivors)
        assert fwd[2] == pytest.approx(0.0)
        assert fwd[3] == pytest.approx(0.0)
