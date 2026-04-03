"""Tests for scenario governance metadata and sensitivity analysis."""

from datetime import UTC, datetime

import pytest

from creditriskengine.ecl.ifrs9.scenarios import (
    Scenario,
    ScenarioSetMetadata,
    SensitivityResult,
    scenario_sensitivity_analysis,
    validate_scenario_governance,
    weighted_ecl,
)


class TestScenarioSetMetadata:
    def test_creation(self) -> None:
        scenarios = [
            Scenario("base", 0.50, 100.0),
            Scenario("upside", 0.20, 50.0),
            Scenario("downside", 0.30, 200.0),
        ]
        meta = ScenarioSetMetadata(
            scenarios=scenarios,
            approved_by="Model Risk Committee",
            approval_date=datetime(2025, 1, 15, tzinfo=UTC),
            next_review_date=datetime(2025, 4, 15, tzinfo=UTC),
            methodology="Expert panel + GDP consensus",
            data_sources="IMF WEO Oct 2024",
        )
        assert meta.approved_by == "Model Risk Committee"
        assert len(meta.scenarios) == 3


class TestValidateScenarioGovernance:
    def test_fully_compliant(self) -> None:
        scenarios = [
            Scenario("base", 0.50, 100.0),
            Scenario("upside", 0.20, 50.0),
            Scenario("downside", 0.30, 200.0),
        ]
        meta = ScenarioSetMetadata(
            scenarios=scenarios,
            approved_by="MRC",
            approval_date=datetime(2025, 1, 1, tzinfo=UTC),
            next_review_date=datetime(2025, 4, 1, tzinfo=UTC),
            methodology="GDP-linked weight calibration",
        )
        warnings = validate_scenario_governance(meta)
        assert warnings == []

    def test_empty_scenarios(self) -> None:
        meta = ScenarioSetMetadata(scenarios=[])
        warnings = validate_scenario_governance(meta)
        assert any("no scenarios" in w.lower() for w in warnings)

    def test_fewer_than_three_scenarios(self) -> None:
        meta = ScenarioSetMetadata(
            scenarios=[Scenario("base", 0.6, 100.0), Scenario("down", 0.4, 200.0)],
            approved_by="MRC",
            approval_date=datetime(2025, 1, 1, tzinfo=UTC),
            next_review_date=datetime(2025, 4, 1, tzinfo=UTC),
            methodology="Simple",
        )
        warnings = validate_scenario_governance(meta)
        assert any("fewer than 3" in w.lower() for w in warnings)

    def test_weights_dont_sum_to_one(self) -> None:
        meta = ScenarioSetMetadata(
            scenarios=[
                Scenario("base", 0.50, 100.0),
                Scenario("up", 0.20, 50.0),
                Scenario("down", 0.20, 200.0),
            ],
            approved_by="MRC",
            approval_date=datetime(2025, 1, 1, tzinfo=UTC),
            next_review_date=datetime(2025, 4, 1, tzinfo=UTC),
            methodology="Test",
        )
        warnings = validate_scenario_governance(meta)
        assert any("weights sum" in w.lower() for w in warnings)

    def test_missing_approval(self) -> None:
        scenarios = [
            Scenario("base", 0.50, 100.0),
            Scenario("up", 0.20, 50.0),
            Scenario("down", 0.30, 200.0),
        ]
        meta = ScenarioSetMetadata(
            scenarios=scenarios,
            methodology="Test",
        )
        warnings = validate_scenario_governance(meta)
        assert any("approval authority" in w.lower() for w in warnings)
        assert any("approval date" in w.lower() for w in warnings)

    def test_missing_review_date(self) -> None:
        scenarios = [
            Scenario("base", 0.50, 100.0),
            Scenario("up", 0.20, 50.0),
            Scenario("down", 0.30, 200.0),
        ]
        meta = ScenarioSetMetadata(
            scenarios=scenarios,
            approved_by="MRC",
            approval_date=datetime(2025, 1, 1, tzinfo=UTC),
            methodology="Test",
        )
        warnings = validate_scenario_governance(meta)
        assert any("review date" in w.lower() for w in warnings)

    def test_review_before_approval(self) -> None:
        scenarios = [
            Scenario("base", 0.50, 100.0),
            Scenario("up", 0.20, 50.0),
            Scenario("down", 0.30, 200.0),
        ]
        meta = ScenarioSetMetadata(
            scenarios=scenarios,
            approved_by="MRC",
            approval_date=datetime(2025, 6, 1, tzinfo=UTC),
            next_review_date=datetime(2025, 1, 1, tzinfo=UTC),
            methodology="Test",
        )
        warnings = validate_scenario_governance(meta)
        assert any("review date must be after" in w.lower() for w in warnings)

    def test_missing_methodology(self) -> None:
        scenarios = [
            Scenario("base", 0.50, 100.0),
            Scenario("up", 0.20, 50.0),
            Scenario("down", 0.30, 200.0),
        ]
        meta = ScenarioSetMetadata(
            scenarios=scenarios,
            approved_by="MRC",
            approval_date=datetime(2025, 1, 1, tzinfo=UTC),
            next_review_date=datetime(2025, 4, 1, tzinfo=UTC),
        )
        warnings = validate_scenario_governance(meta)
        assert any("methodology" in w.lower() for w in warnings)


class TestScenarioSensitivityAnalysis:
    def test_basic_sensitivity(self) -> None:
        scenarios = [
            Scenario("base", 0.50, 100.0),
            Scenario("upside", 0.20, 50.0),
            Scenario("downside", 0.30, 200.0),
        ]
        result = scenario_sensitivity_analysis(scenarios, shift_size=0.10)
        assert isinstance(result, SensitivityResult)
        assert result.base_ecl == pytest.approx(weighted_ecl(scenarios))
        assert "base" in result.shifted_results
        assert "upside" in result.shifted_results
        assert "downside" in result.shifted_results

    def test_downside_most_sensitive(self) -> None:
        scenarios = [
            Scenario("base", 0.50, 100.0),
            Scenario("upside", 0.20, 50.0),
            Scenario("downside", 0.30, 300.0),
        ]
        result = scenario_sensitivity_analysis(scenarios, shift_size=0.10)
        # Downside has the highest ECL, so increasing its weight should
        # produce the largest deviation
        assert result.max_sensitivity_scenario == "downside"
        assert result.max_sensitivity_pct > 0

    def test_shifted_weights_sum_to_one(self) -> None:
        scenarios = [
            Scenario("base", 0.50, 100.0),
            Scenario("upside", 0.20, 50.0),
            Scenario("downside", 0.30, 200.0),
        ]
        # The function calls weighted_ecl internally which validates
        # weights sum to 1.0 — no ValueError means weights are correct
        result = scenario_sensitivity_analysis(scenarios, shift_size=0.10)
        assert len(result.shifted_results) == 3

    def test_empty_scenarios_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one scenario"):
            scenario_sensitivity_analysis([], shift_size=0.10)

    def test_invalid_shift_size(self) -> None:
        scenarios = [Scenario("base", 1.0, 100.0)]
        with pytest.raises(ValueError, match="shift_size must be in"):
            scenario_sensitivity_analysis(scenarios, shift_size=0.0)
        with pytest.raises(ValueError, match="shift_size must be in"):
            scenario_sensitivity_analysis(scenarios, shift_size=1.0)

    def test_single_scenario(self) -> None:
        scenarios = [Scenario("base", 1.0, 100.0)]
        result = scenario_sensitivity_analysis(scenarios, shift_size=0.10)
        # Single scenario: weight is already 1.0, capped at 1.0
        assert result.shifted_results["base"] == pytest.approx(100.0)

    def test_equal_ecl_produces_zero_sensitivity(self) -> None:
        scenarios = [
            Scenario("base", 0.50, 100.0),
            Scenario("upside", 0.25, 100.0),
            Scenario("downside", 0.25, 100.0),
        ]
        result = scenario_sensitivity_analysis(scenarios, shift_size=0.10)
        # All ECLs equal → no sensitivity to weight changes
        assert result.max_sensitivity_pct == pytest.approx(0.0)
