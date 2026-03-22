"""Tests for PD modeling: scorecard, master scale, calibration."""

import math

import numpy as np
import pytest

from creditriskengine.models.pd.scorecard import (
    ScorecardBuilder,
    assign_rating_grade,
    build_master_scale,
    calibrate_pd_anchor_point,
    calibrate_pd_bayesian,
    logistic_score,
    pd_to_score,
    score_to_pd,
    vasicek_single_factor_pd,
)


class TestLogisticScore:
    def test_basic(self) -> None:
        coefficients = np.array([0.5, -0.3])
        features = np.array([[1.0, 2.0], [3.0, 4.0]])
        scores = logistic_score(coefficients, features, intercept=0.1)
        expected = np.array([0.5 * 1.0 - 0.3 * 2.0 + 0.1, 0.5 * 3.0 - 0.3 * 4.0 + 0.1])
        np.testing.assert_allclose(scores, expected)

    def test_single_feature(self) -> None:
        scores = logistic_score(np.array([1.0]), np.array([[2.0]]))
        assert scores[0] == pytest.approx(2.0)


class TestScoreToPD:
    def test_zero_score_gives_half(self) -> None:
        pds = score_to_pd(np.array([0.0]))
        assert pds[0] == pytest.approx(0.5)

    def test_positive_score_high_pd(self) -> None:
        pds = score_to_pd(np.array([10.0]))
        assert pds[0] > 0.99

    def test_negative_score_low_pd(self) -> None:
        pds = score_to_pd(np.array([-10.0]))
        assert pds[0] < 0.01

    def test_monotonic(self) -> None:
        scores = np.array([-5.0, -2.0, 0.0, 2.0, 5.0])
        pds = score_to_pd(scores)
        assert all(pds[i] < pds[i + 1] for i in range(len(pds) - 1))


class TestPDToScore:
    def test_roundtrip_relative(self) -> None:
        pds = np.array([0.01, 0.05, 0.10, 0.50])
        scores = pd_to_score(pds)
        # Higher PD → lower score
        assert scores[0] > scores[1] > scores[2] > scores[3]

    def test_base_score_at_base_odds(self) -> None:
        # PD where odds = base_odds = 50 → PD = 1/(1+50) ≈ 0.01961
        pd_at_base = 1.0 / (1.0 + 50.0)
        scores = pd_to_score(np.array([pd_at_base]), base_score=600.0, base_odds=50.0)
        assert scores[0] == pytest.approx(600.0, abs=0.1)


class TestMasterScale:
    def test_basic(self) -> None:
        boundaries = [0.0003, 0.001, 0.005, 0.01, 0.05, 0.10, 0.20, 1.0]
        scale = build_master_scale(boundaries)
        assert len(scale) == 7
        assert scale[0]["pd_lower"] == 0.0003
        assert scale[-1]["pd_upper"] == 1.0

    def test_custom_labels(self) -> None:
        scale = build_master_scale([0.0, 0.01, 0.05], grade_labels=["A", "B"])
        assert scale[0]["grade"] == "A"
        assert scale[1]["grade"] == "B"

    def test_too_few_boundaries(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            build_master_scale([0.01])

    def test_label_count_mismatch(self) -> None:
        with pytest.raises(ValueError, match="Expected 2"):
            build_master_scale([0.0, 0.01, 0.05], grade_labels=["A"])

    def test_midpoint_geometric(self) -> None:
        scale = build_master_scale([0.01, 0.04])
        expected_mid = math.sqrt(0.01 * 0.04)
        assert scale[0]["pd_midpoint"] == pytest.approx(expected_mid)


class TestAssignRatingGrade:
    def test_basic(self) -> None:
        scale = build_master_scale([0.0, 0.01, 0.05, 1.0], ["A", "B", "C"])
        assert assign_rating_grade(0.005, scale) == "A"
        assert assign_rating_grade(0.02, scale) == "B"
        assert assign_rating_grade(0.50, scale) == "C"

    def test_at_boundary(self) -> None:
        scale = build_master_scale([0.0, 0.01, 0.05], ["A", "B"])
        assert assign_rating_grade(0.01, scale) == "B"

    def test_above_max(self) -> None:
        scale = build_master_scale([0.0, 0.01, 0.05], ["A", "B"])
        assert assign_rating_grade(0.90, scale) == "B"


class TestCalibratePDAnchorPoint:
    def test_scales_to_central_tendency(self) -> None:
        raw_pds = np.array([0.01, 0.02, 0.03, 0.04])
        calibrated = calibrate_pd_anchor_point(0.05, raw_pds)
        assert float(np.mean(calibrated)) == pytest.approx(0.05)

    def test_pd_floor_applied(self) -> None:
        raw_pds = np.array([0.0001, 0.0002])
        calibrated = calibrate_pd_anchor_point(0.0002, raw_pds)
        assert all(p >= 0.0003 for p in calibrated)

    def test_zero_raw_pds(self) -> None:
        raw_pds = np.array([0.0, 0.0])
        calibrated = calibrate_pd_anchor_point(0.02, raw_pds)
        np.testing.assert_allclose(calibrated, [0.02, 0.02])


class TestCalibratePDBayesian:
    def test_pure_prior(self) -> None:
        assert calibrate_pd_bayesian(0.05, 10, 1000, weight=1.0) == pytest.approx(0.05)

    def test_pure_data(self) -> None:
        assert calibrate_pd_bayesian(0.05, 10, 1000, weight=0.0) == pytest.approx(0.01)

    def test_balanced(self) -> None:
        result = calibrate_pd_bayesian(0.05, 10, 1000, weight=0.5)
        assert result == pytest.approx(0.5 * 0.05 + 0.5 * 0.01)

    def test_no_observations(self) -> None:
        assert calibrate_pd_bayesian(0.05, 0, 0) == pytest.approx(0.05)


class TestVasicekSingleFactorPD:
    def test_stressed_pd_higher(self) -> None:
        stressed = vasicek_single_factor_pd(0.02, rho=0.15)
        assert stressed > 0.02

    def test_zero_pd(self) -> None:
        assert vasicek_single_factor_pd(0.0, 0.15) == 0.0

    def test_one_pd(self) -> None:
        assert vasicek_single_factor_pd(1.0, 0.15) == 1.0

    def test_higher_rho_higher_stressed_pd(self) -> None:
        low_rho = vasicek_single_factor_pd(0.02, rho=0.10)
        high_rho = vasicek_single_factor_pd(0.02, rho=0.30)
        assert high_rho > low_rho

    def test_higher_confidence_higher_pd(self) -> None:
        pd_99 = vasicek_single_factor_pd(0.02, rho=0.15, confidence=0.99)
        pd_999 = vasicek_single_factor_pd(0.02, rho=0.15, confidence=0.999)
        assert pd_999 > pd_99

    def test_typical_corporate(self) -> None:
        # Typical corporate: PD=1%, R=0.15, 99.9% confidence
        stressed = vasicek_single_factor_pd(0.01, rho=0.15, confidence=0.999)
        assert 0.0 < stressed < 1.0
        assert stressed > 0.01


class TestPDToScoreExtended:
    """Additional scorecard point tests."""

    def test_custom_parameters(self) -> None:
        pds = np.array([0.02])
        scores = pd_to_score(pds, base_score=500.0, base_odds=20.0, pdo=30.0)
        assert len(scores) == 1
        assert np.isfinite(scores[0])

    def test_extreme_low_pd_clips(self) -> None:
        pds = np.array([0.0])  # exactly 0 -> should be clipped
        scores = pd_to_score(pds)
        assert np.isfinite(scores[0])

    def test_extreme_high_pd_clips(self) -> None:
        pds = np.array([1.0])  # exactly 1 -> should be clipped
        scores = pd_to_score(pds)
        assert np.isfinite(scores[0])


class TestMasterScaleExtended:
    """Additional master scale tests."""

    def test_midpoint_zero_lower(self) -> None:
        scale = build_master_scale([0.0, 0.04])
        # When lower = 0, midpoint = hi / 2.0
        assert scale[0]["pd_midpoint"] == pytest.approx(0.02)

    def test_default_labels(self) -> None:
        scale = build_master_scale([0.0, 0.01, 0.05])
        assert scale[0]["grade"] == "Grade_1"
        assert scale[1]["grade"] == "Grade_2"


class TestScorecardBuilder:
    """Test sklearn-compatible PD scorecard builder."""

    @pytest.fixture
    def training_data(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(42)
        n = 500
        X = rng.normal(0, 1, (n, 3))  # noqa: N806
        logits = 0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.2 * X[:, 2] - 1.0
        prob = 1.0 / (1.0 + np.exp(-logits))
        y = (rng.random(n) < prob).astype(int)
        return X, y

    def test_fit(self, training_data: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = training_data  # noqa: N806
        model = ScorecardBuilder()
        model.fit(X, y)
        assert model.is_fitted_ is True
        assert model.intercept_ is not None
        assert model.coefficients_ is not None
        assert len(model.coefficients_) == 3

    def test_predict_proba(self, training_data: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = training_data  # noqa: N806
        model = ScorecardBuilder()
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        # Probabilities should sum to 1
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)
        # All between 0 and 1
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

    def test_predict(self, training_data: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = training_data  # noqa: N806
        model = ScorecardBuilder()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_score_points(self, training_data: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = training_data  # noqa: N806
        model = ScorecardBuilder(base_score=600.0, pdo=20.0, base_odds=50.0)
        model.fit(X, y)
        points = model.score_points(X)
        assert len(points) == len(X)
        assert np.all(np.isfinite(points))

    def test_predict_before_fit_raises(self) -> None:
        model = ScorecardBuilder()
        with pytest.raises(AssertionError):
            model.predict_proba(np.array([[1.0, 2.0, 3.0]]))

    def test_predict_proba_single_sample(  # noqa: E501
        self, training_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Cover line 297: scores.ndim == 0 branch for single sample."""
        X, y = training_data  # noqa: N806
        model = ScorecardBuilder()
        model.fit(X, y)
        # Single sample -> logistic_score may return scalar
        proba = model.predict_proba(X[:1])
        assert proba.shape == (1, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)

    def test_predict_proba_scalar_ndim0(self, training_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Cover line 297: scores.ndim == 0 when passing 1D input (single observation)."""
        X, y = training_data  # noqa: N806
        model = ScorecardBuilder()
        model.fit(X, y)
        # Pass a 1D array (single sample without batch dim) -> @ produces scalar ndim==0
        proba = model.predict_proba(X[0])
        assert proba.shape == (1, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)

    def test_score_points_single_sample(self, training_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Cover line 312: scores.ndim == 0 branch for single sample."""
        X, y = training_data  # noqa: N806
        model = ScorecardBuilder(base_score=600.0, pdo=20.0, base_odds=50.0)
        model.fit(X, y)
        points = model.score_points(X[:1])
        assert len(points) == 1
        assert np.all(np.isfinite(points))

    def test_score_points_scalar_ndim0(self, training_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Cover line 312: scores.ndim == 0 when passing 1D input (single observation)."""
        X, y = training_data  # noqa: N806
        model = ScorecardBuilder(base_score=600.0, pdo=20.0, base_odds=50.0)
        model.fit(X, y)
        # Pass 1D array -> @ with coefficients produces scalar ndim==0
        points = model.score_points(X[0])
        assert len(points) == 1
        assert np.all(np.isfinite(points))
