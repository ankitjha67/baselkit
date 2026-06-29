"""Tests for creditriskengine.climate package."""

from __future__ import annotations

import pytest

from creditriskengine.climate import (
    CryptoAssetGroup,
    NGFSScenario,
    PCAFScore,
    crypto_asset_rwa,
    financed_emissions,
    get_ngfs_scenario,
    green_asset_ratio,
    list_ngfs_scenarios,
    physical_risk_lgd_haircut,
    transition_risk_pd_multiplier,
)
from creditriskengine.climate.financed_emissions import weighted_data_quality_score
from creditriskengine.climate.physical_risk import (
    PhysicalHazard,
    physical_risk_pd_multiplier,
)
from creditriskengine.climate.transition_risk import (
    CBAM_SECTORS,
    SECTOR_ELASTICITY,
    is_cbam_sector,
)

# ============================================================================
# NGFS Scenarios
# ============================================================================


class TestNGFSScenarios:
    def test_list_returns_six_scenarios(self) -> None:
        scenarios = list_ngfs_scenarios()
        assert len(scenarios) == 6
        assert all(isinstance(s, NGFSScenario) for s in scenarios)

    def test_get_net_zero(self) -> None:
        s = get_ngfs_scenario("net_zero_2050")
        assert s.name == "Net Zero 2050"
        assert s.category == "orderly"
        assert s.temperature_2100_c == 1.5

    def test_get_current_policies(self) -> None:
        s = get_ngfs_scenario("current_policies")
        assert s.category == "hot_house"
        assert s.temperature_2100_c == 3.0
        assert s.physical_risk_severity == "very_high"

    def test_unknown_scenario_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown NGFS scenario"):
            get_ngfs_scenario("nonexistent")

    def test_all_categories_present(self) -> None:
        categories = {s.category for s in list_ngfs_scenarios()}
        assert categories == {"orderly", "disorderly", "hot_house"}

    def test_carbon_prices_increase_with_ambition(self) -> None:
        nz = get_ngfs_scenario("net_zero_2050")
        cp = get_ngfs_scenario("current_policies")
        assert nz.carbon_price_2050_usd > cp.carbon_price_2050_usd

    def test_physical_risk_worse_in_hot_house(self) -> None:
        cp = get_ngfs_scenario("current_policies")
        nz = get_ngfs_scenario("net_zero_2050")
        assert cp.physical_risk_severity == "very_high"
        assert nz.physical_risk_severity == "low"


# ============================================================================
# Physical Risk
# ============================================================================


class TestPhysicalRisk:
    def test_flood_high_lgd_haircut(self) -> None:
        assert physical_risk_lgd_haircut(PhysicalHazard.FLOOD, "high") == 0.15

    def test_flood_low_no_haircut(self) -> None:
        assert physical_risk_lgd_haircut(PhysicalHazard.FLOOD, "low") == 0.0

    def test_wildfire_very_high(self) -> None:
        assert physical_risk_lgd_haircut(PhysicalHazard.WILDFIRE, "very_high") == 0.40

    def test_invalid_severity_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown hazard/severity"):
            physical_risk_lgd_haircut(PhysicalHazard.FLOOD, "extreme")

    def test_pd_multiplier_invalid_severity_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown hazard/severity"):
            physical_risk_pd_multiplier(PhysicalHazard.FLOOD, "extreme")

    def test_pd_multiplier_no_impact_at_low(self) -> None:
        assert physical_risk_pd_multiplier(PhysicalHazard.FLOOD, "low") == 1.0

    def test_pd_multiplier_increases_with_severity(self) -> None:
        medium = physical_risk_pd_multiplier(PhysicalHazard.FLOOD, "medium")
        high = physical_risk_pd_multiplier(PhysicalHazard.FLOOD, "high")
        assert medium < high

    def test_all_hazards_have_all_severities(self) -> None:
        for hazard in PhysicalHazard:
            for severity in ("low", "medium", "high", "very_high"):
                haircut = physical_risk_lgd_haircut(hazard, severity)
                mult = physical_risk_pd_multiplier(hazard, severity)
                assert haircut >= 0
                assert mult >= 1.0


# ============================================================================
# Transition Risk
# ============================================================================


class TestTransitionRisk:
    def test_zero_emissions_returns_1(self) -> None:
        assert transition_risk_pd_multiplier(0, 100, 1_000_000) == 1.0

    def test_oil_gas_high_carbon(self) -> None:
        mult = transition_risk_pd_multiplier(
            scope1_emissions_tco2e=100_000,
            carbon_price_usd=100,
            ebitda=10_000_000,
            sector="oil_gas",
        )
        # 100000 * 100 / 10M = 1.0 impact; 2.5 elasticity -> 1 + 2.5 = 3.5
        assert mult == pytest.approx(3.5)

    def test_technology_low_impact(self) -> None:
        mult = transition_risk_pd_multiplier(
            scope1_emissions_tco2e=1_000,
            carbon_price_usd=100,
            ebitda=50_000_000,
            sector="technology",
        )
        # 1000 * 100 / 50M = 0.002; 0.1 elasticity -> 1.0002
        assert mult == pytest.approx(1.0002)

    def test_capped_at_5(self) -> None:
        mult = transition_risk_pd_multiplier(
            scope1_emissions_tco2e=1_000_000,
            carbon_price_usd=1000,
            ebitda=100_000,
            sector="coal_mining",
        )
        assert mult == 5.0

    def test_floored_at_1(self) -> None:
        mult = transition_risk_pd_multiplier(0, 100, 1_000_000)
        assert mult == 1.0

    def test_zero_ebitda_returns_1(self) -> None:
        assert transition_risk_pd_multiplier(100, 100, 0) == 1.0

    def test_cbam_sectors(self) -> None:
        assert is_cbam_sector("steel")
        assert is_cbam_sector("cement")
        assert is_cbam_sector("aluminium")
        assert not is_cbam_sector("technology")
        assert len(CBAM_SECTORS) == 7

    def test_all_sectors_have_elasticity(self) -> None:
        assert len(SECTOR_ELASTICITY) >= 16


# ============================================================================
# Financed Emissions
# ============================================================================


class TestFinancedEmissions:
    def test_basic_calculation(self) -> None:
        # 50M outstanding / 200M EVIC = 0.25 attribution × 100k tCO2e = 25k
        fe = financed_emissions(50_000_000, 200_000_000, 100_000)
        assert fe == pytest.approx(25_000.0)

    def test_zero_outstanding(self) -> None:
        assert financed_emissions(0, 200_000_000, 100_000) == 0.0

    def test_zero_evic(self) -> None:
        assert financed_emissions(50_000_000, 0, 100_000) == 0.0

    def test_pcaf_scores(self) -> None:
        assert int(PCAFScore.VERIFIED) == 1
        assert int(PCAFScore.SECTOR_AVERAGE) == 5

    def test_weighted_data_quality(self) -> None:
        items = [
            (1000.0, PCAFScore.VERIFIED),
            (2000.0, PCAFScore.SECTOR_AVERAGE),
        ]
        score = weighted_data_quality_score(items)
        # (1000*1 + 2000*5) / 3000 = 11000/3000 ≈ 3.67
        assert score == pytest.approx(11000 / 3000)

    def test_weighted_quality_empty(self) -> None:
        assert weighted_data_quality_score([]) == 0.0


# ============================================================================
# Green Asset Ratio
# ============================================================================


class TestGreenAssetRatio:
    def test_basic_gar(self) -> None:
        result = green_asset_ratio(
            taxonomy_aligned=20_000,
            total_assets=1_000_000,
            sovereign_and_central_bank=200_000,
        )
        # Covered = 1M - 200k = 800k; GAR = 20k/800k = 2.5%
        assert result.gar_pct == pytest.approx(2.5)
        assert result.covered_assets_total == pytest.approx(800_000)

    def test_zero_covered_assets(self) -> None:
        result = green_asset_ratio(
            taxonomy_aligned=10_000,
            total_assets=100_000,
            sovereign_and_central_bank=100_000,
        )
        assert result.gar_pct == 0.0

    def test_excludes_trading_book(self) -> None:
        r1 = green_asset_ratio(10_000, 100_000, trading_book=0)
        r2 = green_asset_ratio(10_000, 100_000, trading_book=50_000)
        assert r2.gar_pct > r1.gar_pct  # Same numerator, smaller denominator


# ============================================================================
# Crypto-Asset Capital (BCBS SCO60)
# ============================================================================


class TestCryptoAssetCapital:
    def test_group_1a_uses_underlying_rw(self) -> None:
        result = crypto_asset_rwa(100_000, CryptoAssetGroup.GROUP_1A, underlying_rw_pct=20.0)
        assert result.risk_weight_pct == 20.0
        assert result.rwa == pytest.approx(20_000.0)

    def test_group_1b_includes_infra_addon(self) -> None:
        result = crypto_asset_rwa(100_000, CryptoAssetGroup.GROUP_1B, underlying_rw_pct=100.0)
        assert result.infrastructure_addon == pytest.approx(2_500.0)
        assert result.rwa == pytest.approx(102_500.0)

    def test_group_2b_1250_rw(self) -> None:
        result = crypto_asset_rwa(100_000, CryptoAssetGroup.GROUP_2B)
        assert result.risk_weight_pct == 1250.0
        assert result.rwa == pytest.approx(1_250_000.0)

    def test_group_2_soft_limit_breach(self) -> None:
        result = crypto_asset_rwa(
            exposure=100_000,
            group=CryptoAssetGroup.GROUP_2A,
            tier1_capital=5_000_000,
            total_group2_exposure=75_000,  # 1.5% of T1 > 1% soft
        )
        assert result.limit_breach is True
        assert result.limit_breach_level == "soft"

    def test_group_2_hard_limit_breach(self) -> None:
        result = crypto_asset_rwa(
            exposure=100_000,
            group=CryptoAssetGroup.GROUP_2A,
            tier1_capital=5_000_000,
            total_group2_exposure=150_000,  # 3% of T1 > 2% hard
        )
        assert result.limit_breach is True
        assert result.limit_breach_level == "hard"
        assert result.risk_weight_pct == 1250.0  # Reclassified to 2b

    def test_group_2_within_limits(self) -> None:
        result = crypto_asset_rwa(
            exposure=100_000,
            group=CryptoAssetGroup.GROUP_2A,
            tier1_capital=50_000_000,
            total_group2_exposure=100_000,  # 0.2% < 1%
        )
        assert result.limit_breach is False
        assert result.limit_breach_level == "none"
