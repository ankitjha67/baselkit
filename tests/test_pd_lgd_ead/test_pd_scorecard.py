"""Tests for PD modeling: scorecard, master scale, calibration."""

import math

import numpy as np
import pytest

from creditriskengine.models.pd.scorecard import (
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
    def test_basic(self):
        coefficients = np.array([0.5, -0.3])
        features = np.array([[1.0, 2.0], [3.0, 4.0]])
        scores = logistic_score(coefficients, features, intercept=0.1)
        expected = np.array([0.5 * 1.0 - 0.3 * 2.0 + 0.1, 0.5 * 3.0 - 0.3 * 4.0 + 0.1])
        np.testing.assert_allclose(scores, expected)

    def test_single_feature(self):
        scores = logistic_score(np.array([1.0]), np.array([[2.0]]))
        assert scores[0] == pytest.approx(2.0)


class TestScoreToPD:
    def test_zero_score_gives_half(self):
        pds = score_to_pd(np.array([0.0]))
        assert pds[0] == pytest.approx(0.5)

    def test_positive_score_high_pd(self):
        pds = score_to_pd(np.array([10.0]))
        assert pds[0] > 0.99

    def test_negative_score_low_pd(self):
        pds = score_to_pd(np.array([-10.0]))
        assert pds[0] < 0.01

    def test_monotonic(self):
        scores = np.array([-5.0, -2.0, 0.0, 2.0, 5.0])
        pds = score_to_pd(scores)
        assert all(pds[i] < pds[i + 1] for i in range(len(pds) - 1))


class TestPDToScore:
    def test_roundtrip_relative(self):
        pds = np.array([0.01, 0.05, 0.10, 0.50])
        scores = pd_to_score(pds)
        # Higher PD → lower score
        assert scores[0] > scores[1] > scores[2] > scores[3]

    def test_base_score_at_base_odds(self):
        # PD where odds = base_odds = 50 → PD = 1/(1+50) ≈ 0.01961
        pd_at_base = 1.0 / (1.0 + 50.0)
        scores = pd_to_score(np.array([pd_at_base]), base_score=600.0, base_odds=50.0)
        assert scores[0] == pytest.approx(600.0, abs=0.1)


class TestMasterScale:
    def test_basic(self):
        boundaries = [0.0003, 0.001, 0.005, 0.01, 0.05, 0.10, 0.20, 1.0]
        scale = build_master_scale(boundaries)
        assert len(scale) == 7
        assert scale[0]["pd_lower"] == 0.0003
        assert scale[-1]["pd_upper"] == 1.0

    def test_custom_labels(self):
        scale = build_master_scale([0.0, 0.01, 0.05], grade_labels=["A", "B"])
        assert scale[0]["grade"] == "A"
        assert scale[1]["grade"] == "B"

    def test_too_few_boundaries(self):
        with pytest.raises(ValueError, match="at least 2"):
            build_master_scale([0.01])

    def test_label_count_mismatch(self):
        with pytest.raises(ValueError, match="Expected 2"):
            build_master_scale([0.0, 0.01, 0.05], grade_labels=["A"])

    def test_midpoint_geometric(self):
        scale = build_master_scale([0.01, 0.04])
        expected_mid = math.sqrt(0.01 * 0.04)
        assert scale[0]["pd_midpoint"] == pytest.approx(expected_mid)


class TestAssignRatingGrade:
    def test_basic(self):
        scale = build_master_scale([0.0, 0.01, 0.05, 1.0], ["A", "B", "C"])
        assert assign_rating_grade(0.005, scale) == "A"
        assert assign_rating_grade(0.02, scale) == "B"
        assert assign_rating_grade(0.50, scale) == "C"

    def test_at_boundary(self):
        scale = build_master_scale([0.0, 0.01, 0.05], ["A", "B"])
        assert assign_rating_grade(0.01, scale) == "B"

    def test_above_max(self):
        scale = build_master_scale([0.0, 0.01, 0.05], ["A", "B"])
        assert assign_rating_grade(0.90, scale) == "B"


class TestCalibratePDAnchorPoint:
    def test_scales_to_central_tendency(self):
        raw_pds = np.array([0.01, 0.02, 0.03, 0.04])
        calibrated = calibrate_pd_anchor_point(0.05, raw_pds)
        assert float(np.mean(calibrated)) == pytest.approx(0.05)

    def test_pd_floor_applied(self):
        raw_pds = np.array([0.0001, 0.0002])
        calibrated = calibrate_pd_anchor_point(0.0002, raw_pds)
        assert all(p >= 0.0003 for p in calibrated)

    def test_zero_raw_pds(self):
        raw_pds = np.array([0.0, 0.0])
        calibrated = calibrate_pd_anchor_point(0.02, raw_pds)
        np.testing.assert_allclose(calibrated, [0.02, 0.02])


class TestCalibratePDBayesian:
    def test_pure_prior(self):
        assert calibrate_pd_bayesian(0.05, 10, 1000, weight=1.0) == pytest.approx(0.05)

    def test_pure_data(self):
        assert calibrate_pd_bayesian(0.05, 10, 1000, weight=0.0) == pytest.approx(0.01)

    def test_balanced(self):
        result = calibrate_pd_bayesian(0.05, 10, 1000, weight=0.5)
        assert result == pytest.approx(0.5 * 0.05 + 0.5 * 0.01)

    def test_no_observations(self):
        assert calibrate_pd_bayesian(0.05, 0, 0) == pytest.approx(0.05)


class TestVasicekSingleFactorPD:
    def test_stressed_pd_higher(self):
        stressed = vasicek_single_factor_pd(0.02, rho=0.15)
        assert stressed > 0.02

    def test_zero_pd(self):
        assert vasicek_single_factor_pd(0.0, 0.15) == 0.0

    def test_one_pd(self):
        assert vasicek_single_factor_pd(1.0, 0.15) == 1.0

    def test_higher_rho_higher_stressed_pd(self):
        low_rho = vasicek_single_factor_pd(0.02, rho=0.10)
        high_rho = vasicek_single_factor_pd(0.02, rho=0.30)
        assert high_rho > low_rho
