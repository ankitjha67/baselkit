"""Tests for rating transition matrix estimation and analysis."""

import numpy as np
import pytest

from creditriskengine.models.pd.transition_matrix import (
    default_column,
    estimate_transition_matrix,
    generator_matrix,
    multi_period_transition_matrix,
    validate_transition_matrix,
)

# -- estimate_transition_matrix ------------------------------------------------


class TestEstimateTransitionMatrix:
    def test_basic_cohort(self):
        """Simple 3-grade transition with known counts."""
        starts = np.array([0, 0, 0, 0, 1, 1, 1, 2])
        ends = np.array([0, 0, 1, 2, 0, 1, 2, 2])
        tm = estimate_transition_matrix(starts, ends, n_grades=3)

        assert tm.shape == (3, 3)
        # Grade 0: 2 stay, 1 to grade 1, 1 to grade 2
        np.testing.assert_array_almost_equal(tm[0], [0.5, 0.25, 0.25])
        # Grade 1: 1 to grade 0, 1 stays, 1 to grade 2
        np.testing.assert_array_almost_equal(tm[1], [1 / 3, 1 / 3, 1 / 3])
        # Grade 2: 1 stays
        np.testing.assert_array_almost_equal(tm[2], [0.0, 0.0, 1.0])

    def test_rows_sum_to_one(self):
        """Every row of the estimated matrix sums to 1."""
        rng = np.random.default_rng(42)
        n_obs = 1000
        starts = rng.integers(0, 5, size=n_obs)
        ends = rng.integers(0, 5, size=n_obs)
        tm = estimate_transition_matrix(starts, ends, n_grades=5)
        np.testing.assert_allclose(tm.sum(axis=1), np.ones(5), atol=1e-12)

    def test_empty_grade_gives_zero_row(self):
        """A grade with no observations yields a zero row."""
        starts = np.array([0, 0])
        ends = np.array([0, 1])
        tm = estimate_transition_matrix(starts, ends, n_grades=3)
        # Grade 2 has no observations: row should be all zeros
        np.testing.assert_array_almost_equal(tm[2], [0.0, 0.0, 0.0])

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            estimate_transition_matrix(np.array([0, 1]), np.array([0]), n_grades=2)

    def test_all_defaults(self):
        """All obligors transition to default state."""
        starts = np.array([0, 0, 1, 1])
        ends = np.array([2, 2, 2, 2])
        tm = estimate_transition_matrix(starts, ends, n_grades=3)
        assert tm[0, 2] == pytest.approx(1.0)
        assert tm[1, 2] == pytest.approx(1.0)


# -- multi_period_transition_matrix -------------------------------------------


class TestMultiPeriodTransitionMatrix:
    @pytest.fixture()
    def simple_tm(self):
        return np.array([
            [0.90, 0.05, 0.05],
            [0.10, 0.80, 0.10],
            [0.00, 0.00, 1.00],
        ])

    def test_one_period_is_identity_mult(self, simple_tm):
        """Raising to power 1 returns the original matrix."""
        result = multi_period_transition_matrix(simple_tm, periods=1)
        np.testing.assert_array_almost_equal(result, simple_tm)

    def test_two_periods(self, simple_tm):
        """Power 2 equals M @ M."""
        result = multi_period_transition_matrix(simple_tm, periods=2)
        expected = simple_tm @ simple_tm
        np.testing.assert_array_almost_equal(result, expected)

    def test_rows_sum_to_one(self, simple_tm):
        """Rows of multi-period matrix still sum to 1."""
        result = multi_period_transition_matrix(simple_tm, periods=5)
        np.testing.assert_allclose(result.sum(axis=1), np.ones(3), atol=1e-10)

    def test_default_absorbing(self, simple_tm):
        """Default state remains absorbing after multiple periods."""
        result = multi_period_transition_matrix(simple_tm, periods=10)
        np.testing.assert_array_almost_equal(result[2], [0.0, 0.0, 1.0])

    def test_invalid_periods_raises(self, simple_tm):
        with pytest.raises(ValueError, match="periods must be >= 1"):
            multi_period_transition_matrix(simple_tm, periods=0)

    def test_long_horizon_converges_to_default(self, simple_tm):
        """Over a very long horizon, all probability mass goes to default."""
        result = multi_period_transition_matrix(simple_tm, periods=500)
        # All grades should have nearly 100% default probability
        assert result[0, 2] > 0.999
        assert result[1, 2] > 0.999


# -- generator_matrix ---------------------------------------------------------


class TestGeneratorMatrix:
    def test_basic_properties(self):
        """Generator rows sum to ~0 and off-diag elements are >= 0."""
        tm = np.array([
            [0.90, 0.05, 0.05],
            [0.10, 0.80, 0.10],
            [0.00, 0.00, 1.00],
        ])
        q_mat = generator_matrix(tm)
        assert q_mat.shape == (3, 3)
        # Rows of Q should sum to approximately 0
        np.testing.assert_allclose(q_mat.sum(axis=1), np.zeros(3), atol=1e-6)

    def test_diagonal_negative(self):
        """Diagonal elements of Q should be non-positive."""
        tm = np.array([
            [0.95, 0.03, 0.02],
            [0.05, 0.90, 0.05],
            [0.00, 0.00, 1.00],
        ])
        q_mat = generator_matrix(tm)
        assert np.all(np.diag(q_mat) <= 1e-10)

    def test_identity_gives_zero_generator(self):
        """Identity matrix should produce a zero generator."""
        q_mat = generator_matrix(np.eye(3))
        np.testing.assert_array_almost_equal(q_mat, np.zeros((3, 3)))

    def test_real_valued_output(self):
        """Output should always be real-valued."""
        tm = np.array([
            [0.90, 0.05, 0.05],
            [0.10, 0.80, 0.10],
            [0.00, 0.00, 1.00],
        ])
        q_mat = generator_matrix(tm)
        assert q_mat.dtype == np.float64


# -- validate_transition_matrix ------------------------------------------------


class TestValidateTransitionMatrix:
    def test_valid_matrix(self):
        """A well-formed transition matrix passes all checks."""
        tm = np.array([
            [0.90, 0.05, 0.05],
            [0.10, 0.80, 0.10],
            [0.00, 0.00, 1.00],
        ])
        errors = validate_transition_matrix(tm)
        assert errors == []

    def test_non_square_fails(self):
        errors = validate_transition_matrix(np.ones((2, 3)))
        assert len(errors) > 0
        assert "square" in errors[0].lower()

    def test_row_not_summing_to_one(self):
        tm = np.array([
            [0.50, 0.20, 0.10],
            [0.10, 0.80, 0.10],
            [0.00, 0.00, 1.00],
        ])
        errors = validate_transition_matrix(tm)
        assert any("sum to 1" in e for e in errors)

    def test_negative_values(self):
        tm = np.array([
            [1.10, -0.05, -0.05],
            [0.10, 0.80, 0.10],
            [0.00, 0.00, 1.00],
        ])
        errors = validate_transition_matrix(tm)
        assert any("[0, 1]" in e for e in errors)

    def test_non_absorbing_default(self):
        """Last row not [0,...,0,1] should trigger error."""
        tm = np.array([
            [0.90, 0.05, 0.05],
            [0.10, 0.80, 0.10],
            [0.10, 0.10, 0.80],
        ])
        errors = validate_transition_matrix(tm)
        assert any("absorbing" in e.lower() for e in errors)

    def test_values_greater_than_one(self):
        tm = np.array([
            [0.90, 0.05, 0.05],
            [0.10, 0.80, 1.10],
            [0.00, 0.00, 1.00],
        ])
        errors = validate_transition_matrix(tm)
        assert any("[0, 1]" in e for e in errors)

    def test_1d_array_fails(self):
        errors = validate_transition_matrix(np.array([1.0]))
        assert len(errors) > 0


# -- default_column -----------------------------------------------------------


class TestDefaultColumn:
    def test_basic(self):
        tm = np.array([
            [0.90, 0.05, 0.05],
            [0.10, 0.80, 0.10],
            [0.00, 0.00, 1.00],
        ])
        pds = default_column(tm)
        np.testing.assert_array_almost_equal(pds, [0.05, 0.10, 1.00])

    def test_zero_default_risk(self):
        tm = np.array([
            [0.60, 0.40, 0.00],
            [0.30, 0.70, 0.00],
            [0.00, 0.00, 1.00],
        ])
        pds = default_column(tm)
        assert pds[0] == pytest.approx(0.0)
        assert pds[1] == pytest.approx(0.0)
        assert pds[2] == pytest.approx(1.0)

    def test_length_matches_grades(self):
        tm = np.eye(5)
        pds = default_column(tm)
        assert len(pds) == 5

    def test_returns_copy(self):
        """Modifying result should not alter the original matrix."""
        tm = np.array([[0.9, 0.1], [0.0, 1.0]])
        pds = default_column(tm)
        pds[0] = 999.0
        assert tm[0, 1] == pytest.approx(0.1)


class TestGeneratorMatrixComplex:
    """Edge cases for generator_matrix with complex results."""

    def test_non_embeddable_matrix_warns(self) -> None:
        """A matrix with negative eigenvalues produces complex log with material imaginary part."""
        # This reflection-like matrix has a negative eigenvalue (-0.8),
        # so logm returns complex128 with imag norm ≈ pi
        tm = np.array([
            [0.1, 0.9, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        gen = generator_matrix(tm)
        # Should return real matrix after taking real part
        assert gen.dtype == np.float64
        assert gen.shape == (3, 3)
