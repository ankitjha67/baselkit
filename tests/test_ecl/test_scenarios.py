"""Tests for IFRS 9 scenario weighting."""

import pytest

from creditriskengine.ecl.ifrs9.scenarios import Scenario, standard_scenario_weights, weighted_ecl


class TestWeightedECL:
    def test_basic(self) -> None:
        scenarios = [
            Scenario("base", 0.50, 100.0),
            Scenario("upside", 0.20, 50.0),
            Scenario("downside", 0.30, 200.0),
        ]
        ecl = weighted_ecl(scenarios)
        assert ecl == pytest.approx(0.50 * 100 + 0.20 * 50 + 0.30 * 200)

    def test_weights_must_sum_to_one(self) -> None:
        scenarios = [
            Scenario("base", 0.50, 100.0),
            Scenario("down", 0.30, 200.0),
        ]
        with pytest.raises(ValueError, match="must sum to 1.0"):
            weighted_ecl(scenarios)

    def test_single_scenario(self) -> None:
        ecl = weighted_ecl([Scenario("base", 1.0, 42.0)])
        assert ecl == pytest.approx(42.0)


class TestStandardScenarioWeights:
    def test_weights_sum_to_one(self) -> None:
        weights = standard_scenario_weights()
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_has_expected_keys(self) -> None:
        weights = standard_scenario_weights()
        assert "base" in weights
        assert "downside" in weights
