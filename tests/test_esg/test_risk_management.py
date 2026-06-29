"""Tests for EBA/GL/2025/01 ESG risk management toolkit."""

import pytest

from creditriskengine.esg.risk_management import (
    ESGFactor,
    ESGRiskDriver,
    MaterialityLevel,
    MaterialityMethod,
    TimeHorizon,
    assess_esg_materiality,
    recommended_method,
    transition_plan_alignment,
)


class TestRecommendedMethod:
    def test_short_term_exposure_based(self) -> None:
        assert recommended_method(TimeHorizon.SHORT) == MaterialityMethod.EXPOSURE_BASED

    def test_medium_term_sector_based(self) -> None:
        assert recommended_method(TimeHorizon.MEDIUM) == MaterialityMethod.SECTOR_BASED

    def test_long_term_scenario_based(self) -> None:
        assert recommended_method(TimeHorizon.LONG) == MaterialityMethod.SCENARIO_BASED


class TestESGRiskDriver:
    def test_severity_is_likelihood_times_impact(self) -> None:
        d = ESGRiskDriver(ESGFactor.ENV_PHYSICAL, 100.0, likelihood=0.4, impact=0.5)
        assert d.severity == pytest.approx(0.20)

    def test_invalid_negative_exposure(self) -> None:
        with pytest.raises(ValueError, match="exposure_amount"):
            ESGRiskDriver(ESGFactor.SOCIAL, -1.0, 0.5, 0.5)

    def test_invalid_likelihood(self) -> None:
        with pytest.raises(ValueError, match="likelihood"):
            ESGRiskDriver(ESGFactor.SOCIAL, 1.0, 1.5, 0.5)

    def test_invalid_impact(self) -> None:
        with pytest.raises(ValueError, match="impact"):
            ESGRiskDriver(ESGFactor.SOCIAL, 1.0, 0.5, 1.5)


class TestAssessESGMateriality:
    def test_score_is_exposure_weighted_severity(self) -> None:
        # Single driver on half the book, severity 0.5*0.4=0.20
        # weight = 500/1000 = 0.5 -> score = 0.5 * 0.20 = 0.10
        drivers = [
            ESGRiskDriver(ESGFactor.ENV_TRANSITION, 500.0, 0.5, 0.4),
        ]
        res = assess_esg_materiality(drivers, total_exposure=1000.0)
        assert res.score == pytest.approx(0.10)

    def test_material_when_at_threshold(self) -> None:
        drivers = [ESGRiskDriver(ESGFactor.ENV_TRANSITION, 500.0, 0.5, 0.4)]
        res = assess_esg_materiality(
            drivers, total_exposure=1000.0, materiality_threshold=0.10
        )
        assert res.level == MaterialityLevel.MATERIAL

    def test_not_material_below_threshold(self) -> None:
        drivers = [ESGRiskDriver(ESGFactor.GOVERNANCE, 100.0, 0.1, 0.1)]
        res = assess_esg_materiality(drivers, total_exposure=1000.0)
        assert res.level == MaterialityLevel.NOT_MATERIAL

    def test_method_follows_horizon(self) -> None:
        drivers = [ESGRiskDriver(ESGFactor.ENV_PHYSICAL, 100.0, 0.2, 0.2)]
        res = assess_esg_materiality(
            drivers, total_exposure=1000.0, horizon=TimeHorizon.LONG
        )
        assert res.method == MaterialityMethod.SCENARIO_BASED
        assert res.horizon == TimeHorizon.LONG

    def test_breakdown_sums_to_score(self) -> None:
        drivers = [
            ESGRiskDriver(ESGFactor.ENV_PHYSICAL, 300.0, 0.6, 0.5),
            ESGRiskDriver(ESGFactor.SOCIAL, 200.0, 0.4, 0.3),
        ]
        res = assess_esg_materiality(drivers, total_exposure=1000.0)
        assert sum(res.score_by_factor.values()) == pytest.approx(res.score)

    def test_score_capped_at_one(self) -> None:
        # Overlapping drivers whose exposures exceed the book cannot push
        # materiality above a full write-down.
        drivers = [
            ESGRiskDriver(ESGFactor.ENV_TRANSITION, 1000.0, 1.0, 1.0),
            ESGRiskDriver(ESGFactor.ENV_PHYSICAL, 1000.0, 1.0, 1.0),
        ]
        res = assess_esg_materiality(drivers, total_exposure=1000.0)
        assert res.score == pytest.approx(1.0)
        assert res.exposure_at_risk == pytest.approx(1000.0)

    def test_zero_severity_not_counted_in_exposure_at_risk(self) -> None:
        drivers = [ESGRiskDriver(ESGFactor.GOVERNANCE, 500.0, 0.0, 0.9)]
        res = assess_esg_materiality(drivers, total_exposure=1000.0)
        assert res.exposure_at_risk == pytest.approx(0.0)

    def test_invalid_total_exposure(self) -> None:
        with pytest.raises(ValueError, match="total_exposure"):
            assess_esg_materiality([], total_exposure=0.0)

    def test_invalid_threshold(self) -> None:
        with pytest.raises(ValueError, match="materiality_threshold"):
            assess_esg_materiality([], total_exposure=1.0, materiality_threshold=2.0)


class TestTransitionPlanAlignment:
    def test_reduction_on_track(self) -> None:
        # Base 100 in 2020 -> target 0 by 2050. By 2035 (halfway), expected 50.
        # Actual 45 is ahead of plan.
        s = transition_plan_alignment(
            current_value=45.0,
            base_year_value=100.0,
            target_value=0.0,
            base_year=2020,
            target_year=2050,
            current_year=2035,
        )
        assert s.expected_value == pytest.approx(50.0)
        assert s.on_track is True
        assert s.gap == pytest.approx(-5.0)

    def test_reduction_behind_plan(self) -> None:
        s = transition_plan_alignment(
            current_value=60.0,
            base_year_value=100.0,
            target_value=0.0,
            base_year=2020,
            target_year=2050,
            current_year=2035,
        )
        assert s.on_track is False
        assert s.gap == pytest.approx(10.0)
        # Achieved 40 of required 50 -> 80%
        assert s.alignment_pct == pytest.approx(80.0)

    def test_required_annual_change(self) -> None:
        s = transition_plan_alignment(
            current_value=90.0,
            base_year_value=100.0,
            target_value=40.0,
            base_year=2020,
            target_year=2030,
            current_year=2022,
        )
        # (40 - 100) / 10 = -6 per year
        assert s.required_annual_change == pytest.approx(-6.0)

    def test_increase_path_gar_uplift(self) -> None:
        # GAR uplift target: 10% (2024) -> 50% (2034). Halfway expect 30%.
        s = transition_plan_alignment(
            current_value=35.0,
            base_year_value=10.0,
            target_value=50.0,
            base_year=2024,
            target_year=2034,
            current_year=2029,
        )
        assert s.expected_value == pytest.approx(30.0)
        assert s.on_track is True  # ahead on a growth path

    def test_base_year_no_change_required(self) -> None:
        s = transition_plan_alignment(
            current_value=100.0,
            base_year_value=100.0,
            target_value=0.0,
            base_year=2020,
            target_year=2050,
            current_year=2020,
        )
        assert s.alignment_pct == pytest.approx(100.0)
        assert s.on_track is True

    def test_invalid_target_year(self) -> None:
        with pytest.raises(ValueError, match="target_year must be after"):
            transition_plan_alignment(1.0, 1.0, 0.0, 2030, 2030, 2030)

    def test_current_year_out_of_window(self) -> None:
        with pytest.raises(ValueError, match="current_year must be within"):
            transition_plan_alignment(1.0, 1.0, 0.0, 2020, 2050, 2055)
