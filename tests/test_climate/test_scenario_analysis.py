"""Tests for the climate scenario-analysis engine (EBA/GL/2025/02, NGFS)."""

import pytest

from creditriskengine.climate.ngfs_scenarios import get_ngfs_scenario
from creditriskengine.climate.physical_risk import PhysicalHazard
from creditriskengine.climate.scenario_analysis import (
    ClimateExposure,
    project_climate_ecl,
    scenario_carbon_price,
)


class TestClimateExposureValidation:
    def test_valid(self) -> None:
        exp = ClimateExposure("E1", 1000.0, 0.02, 0.45)
        assert exp.ead == 1000.0

    def test_negative_ead(self) -> None:
        with pytest.raises(ValueError, match="ead must be non-negative"):
            ClimateExposure("E1", -1.0, 0.02, 0.45)

    def test_invalid_pd(self) -> None:
        with pytest.raises(ValueError, match="pd must be in"):
            ClimateExposure("E1", 1000.0, 1.5, 0.45)

    def test_invalid_lgd(self) -> None:
        with pytest.raises(ValueError, match="lgd must be in"):
            ClimateExposure("E1", 1000.0, 0.02, 1.5)


class TestScenarioCarbonPrice:
    def test_before_2030_holds_2030(self) -> None:
        s = get_ngfs_scenario("net_zero_2050")
        assert scenario_carbon_price(s, 2025) == pytest.approx(140.0)

    def test_after_2050_holds_2050(self) -> None:
        s = get_ngfs_scenario("net_zero_2050")
        assert scenario_carbon_price(s, 2060) == pytest.approx(725.0)

    def test_interpolation_midpoint(self) -> None:
        s = get_ngfs_scenario("net_zero_2050")
        # 2040 is halfway: (140 + 725) / 2 = 432.5
        assert scenario_carbon_price(s, 2040) == pytest.approx(432.5)


class TestProjectClimateEcl:
    def test_empty_raises(self) -> None:
        s = get_ngfs_scenario("net_zero_2050")
        with pytest.raises(ValueError, match="at least one exposure"):
            project_climate_ecl([], s)

    def test_no_climate_drivers_no_uplift(self) -> None:
        # No EBITDA (transition off) and no hazard (physical off) -> ECL flat.
        s = get_ngfs_scenario("net_zero_2050")
        exps = [ClimateExposure("E1", 1_000_000.0, 0.02, 0.45)]
        res = project_climate_ecl(exps, s, horizon_year=2030)
        assert res.baseline_ecl == pytest.approx(res.stressed_ecl)
        assert res.ecl_uplift == pytest.approx(0.0)
        assert res.ecl_uplift_pct == pytest.approx(0.0)

    def test_transition_uplift(self) -> None:
        # Carbon-intensive energy name under Net Zero 2050 -> PD uplift.
        s = get_ngfs_scenario("net_zero_2050")
        exps = [
            ClimateExposure(
                "E1", 1_000_000.0, 0.02, 0.45,
                sector="energy", scope1_emissions_tco2e=50_000.0, ebitda=10_000_000.0,
            )
        ]
        res = project_climate_ecl(exps, s, horizon_year=2050)
        assert res.stressed_ecl > res.baseline_ecl
        assert res.transition_ecl_uplift > 0.0
        assert res.physical_ecl_uplift == pytest.approx(0.0)
        assert res.by_exposure[0].transition_pd_multiplier > 1.0

    def test_physical_uplift(self) -> None:
        # Flood-exposed name under a high-physical-severity scenario.
        s = get_ngfs_scenario("fragmented_world")  # physical severity "high"
        exps = [
            ClimateExposure(
                "E1", 1_000_000.0, 0.02, 0.45, physical_hazard=PhysicalHazard.FLOOD,
            )
        ]
        res = project_climate_ecl(exps, s, horizon_year=2035)
        assert res.stressed_ecl > res.baseline_ecl
        assert res.physical_ecl_uplift > 0.0
        assert res.by_exposure[0].physical_pd_multiplier > 1.0
        assert res.by_exposure[0].stressed_lgd > 0.45

    def test_combined_channels_and_aggregation(self) -> None:
        s = get_ngfs_scenario("delayed_transition")
        exps = [
            ClimateExposure(
                "E1", 500_000.0, 0.03, 0.40,
                sector="energy", scope1_emissions_tco2e=30_000.0, ebitda=5_000_000.0,
                physical_hazard=PhysicalHazard.STORM,
            ),
            ClimateExposure("E2", 200_000.0, 0.01, 0.50),
        ]
        res = project_climate_ecl(exps, s, horizon_year=2040)
        assert res.n_exposures == 2
        assert len(res.by_exposure) == 2
        assert res.ecl_uplift == pytest.approx(res.stressed_ecl - res.baseline_ecl)
        assert res.stressed_ecl > res.baseline_ecl

    def test_pd_capped_at_one(self) -> None:
        # Extreme transition stress cannot push stressed PD above 1.0.
        s = get_ngfs_scenario("net_zero_2050")
        exps = [
            ClimateExposure(
                "E1", 1_000.0, 0.9, 0.9,
                sector="energy", scope1_emissions_tco2e=1_000_000.0, ebitda=1_000.0,
            )
        ]
        res = project_climate_ecl(exps, s, horizon_year=2050)
        assert res.by_exposure[0].stressed_pd <= 1.0

    def test_zero_baseline_ecl_pct_is_zero(self) -> None:
        s = get_ngfs_scenario("current_policies")
        exps = [ClimateExposure("E1", 0.0, 0.02, 0.45)]
        res = project_climate_ecl(exps, s)
        assert res.baseline_ecl == pytest.approx(0.0)
        assert res.ecl_uplift_pct == pytest.approx(0.0)
