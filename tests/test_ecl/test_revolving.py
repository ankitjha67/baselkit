"""Comprehensive tests for IFRS 9 revolving credit ECL module.

Covers:
- Enumeration types and product configurations
- Behavioral life determination (B5.5.40 three-factor framework)
- CCF models (regulatory SA, F-IRB, behavioral, EADF, PIT adjustment)
- Revolving EAD term structures with drawn/undrawn decomposition
- ECL calculation with IFRS 7 B8E drawn/undrawn split
- Multi-scenario probability weighting
- Multi-jurisdiction provision floors
- End-to-end integration with worked examples from research
"""

from __future__ import annotations

import numpy as np
import pytest

from creditriskengine.core.types import IFRS9Stage, Jurisdiction
from creditriskengine.ecl.ifrs9.revolving.behavioral_life import (
    determine_behavioral_life,
    effective_life_months,
    segment_behavioral_life,
)
from creditriskengine.ecl.ifrs9.revolving.ccf import (
    airb_ccf_floor,
    apply_ccf_with_floor,
    behavioral_ccf,
    ccf_pit_adjustment,
    eadf_ccf,
    regulatory_ccf_firb,
    regulatory_ccf_sa,
)
from creditriskengine.ecl.ifrs9.revolving.ead_profile import (
    RevolvingEADProfile,
    ead_drawn_undrawn_split,
    revolving_ead_term_structure,
)
from creditriskengine.ecl.ifrs9.revolving.ecl_revolving import (
    RevolvingECLResult,
    calculate_revolving_ecl,
    revolving_ecl_scenario_weighted,
)
from creditriskengine.ecl.ifrs9.revolving.product_config import (
    PRODUCT_CONFIGS,
    get_product_config,
)
from creditriskengine.ecl.ifrs9.revolving.provision_floors import (
    apply_provision_floor,
    get_provision_floors,
)
from creditriskengine.ecl.ifrs9.revolving.types import (
    BehavioralLifeMethod,
    CCFMethod,
    RevolvingProductType,
)

# =====================================================================
# 1. Type / Enum Tests
# =====================================================================


class TestTypes:
    def test_product_types(self) -> None:
        assert len(RevolvingProductType) == 6
        assert RevolvingProductType.CREDIT_CARD.value == "credit_card"
        assert RevolvingProductType.HELOC.value == "heloc"

    def test_ccf_methods(self) -> None:
        assert len(CCFMethod) == 4
        assert CCFMethod.REGULATORY_SA.value == "regulatory_sa"
        assert CCFMethod.EADF.value == "eadf"

    def test_behavioral_life_methods(self) -> None:
        assert len(BehavioralLifeMethod) == 3


# =====================================================================
# 2. Product Config Tests
# =====================================================================


class TestProductConfig:
    def test_all_product_types_have_configs(self) -> None:
        for pt in RevolvingProductType:
            assert pt in PRODUCT_CONFIGS

    def test_credit_card_config(self) -> None:
        cfg = get_product_config(RevolvingProductType.CREDIT_CARD)
        assert cfg.default_behavioral_life_months == 36
        assert cfg.behavioral_life_range == (24, 60)
        assert cfg.typical_ccf_range == (0.50, 0.95)
        assert cfg.typical_lgd == 0.85
        assert cfg.is_collectively_managed is True
        assert cfg.has_draw_period is False

    def test_configs_loaded_from_yaml(self) -> None:
        """Product configs are loaded from revolving_products.yml."""
        from creditriskengine.ecl.ifrs9.revolving.product_config import (
            load_revolving_product_configs,
        )

        configs = load_revolving_product_configs()
        assert len(configs) == 6
        for pt in RevolvingProductType:
            assert pt in configs

    def test_provision_floors_loaded_from_yaml(self) -> None:
        """Provision floors are loaded from provision_floors.yml."""
        from creditriskengine.ecl.ifrs9.revolving.provision_floors import (
            load_provision_floors,
        )

        floors = load_provision_floors()
        assert len(floors) >= 5
        jurisdictions = {f.jurisdiction for f in floors}
        assert Jurisdiction.UAE in jurisdictions
        assert Jurisdiction.INDIA in jurisdictions

    def test_heloc_has_draw_period(self) -> None:
        cfg = get_product_config(RevolvingProductType.HELOC)
        assert cfg.has_draw_period is True
        assert cfg.draw_period_months == 120

    def test_corporate_revolver_individually_managed(self) -> None:
        cfg = get_product_config(RevolvingProductType.CORPORATE_REVOLVER)
        assert cfg.is_collectively_managed is False

    def test_config_is_frozen(self) -> None:
        cfg = get_product_config(RevolvingProductType.CREDIT_CARD)
        with pytest.raises(AttributeError):
            cfg.default_behavioral_life_months = 99  # type: ignore[misc]


# =====================================================================
# 3. Behavioral Life Tests
# =====================================================================


class TestBehavioralLife:
    def test_shortest_factor_wins(self) -> None:
        life = determine_behavioral_life(
            historical_life_months=48,
            time_to_default_months=36,
            crm_action_months=24,
        )
        assert life == 24

    def test_single_factor(self) -> None:
        life = determine_behavioral_life(historical_life_months=60)
        assert life == 60

    def test_fallback_to_product_default(self) -> None:
        life = determine_behavioral_life(
            product_type=RevolvingProductType.CREDIT_CARD
        )
        assert life == 36

    def test_no_factors_raises(self) -> None:
        with pytest.raises(ValueError, match="B5.5.40"):
            determine_behavioral_life()

    def test_ceiling_rounding(self) -> None:
        life = determine_behavioral_life(historical_life_months=36.3)
        assert life == 37

    def test_minimum_one_month(self) -> None:
        life = determine_behavioral_life(crm_action_months=0.5)
        assert life == 1

    def test_zero_factors_ignored(self) -> None:
        life = determine_behavioral_life(
            historical_life_months=0,
            crm_action_months=24,
        )
        assert life == 24

    def test_segment_behavioral_life(self) -> None:
        result = segment_behavioral_life(
            36, {"high_risk": 18, "low_risk": 48}
        )
        assert result["base"] == 36
        assert result["high_risk"] == 18
        assert result["low_risk"] == 48

    def test_segment_minimum_one(self) -> None:
        result = segment_behavioral_life(36, {"tiny": 0})
        assert result["tiny"] == 1

    def test_effective_life_heloc_draw_period(self) -> None:
        life = effective_life_months(RevolvingProductType.HELOC, True)
        assert life == 120

    def test_effective_life_heloc_repayment(self) -> None:
        life = effective_life_months(RevolvingProductType.HELOC, False)
        cfg = get_product_config(RevolvingProductType.HELOC)
        assert life == cfg.default_behavioral_life_months

    def test_effective_life_non_draw_product(self) -> None:
        life = effective_life_months(RevolvingProductType.CREDIT_CARD)
        assert life == 36


# =====================================================================
# 4. CCF Tests
# =====================================================================


class TestCCF:
    # --- Regulatory SA (parametrized) ---
    @pytest.mark.parametrize(
        ("product", "jurisdiction", "transitional", "expected"),
        [
            (RevolvingProductType.CREDIT_CARD, Jurisdiction.BCBS, False, 0.10),
            (RevolvingProductType.OVERDRAFT, Jurisdiction.BCBS, False, 0.10),
            (RevolvingProductType.CREDIT_CARD, Jurisdiction.AUSTRALIA, False, 0.40),
            (RevolvingProductType.CREDIT_CARD, Jurisdiction.EU, True, 0.0),
            (RevolvingProductType.CORPORATE_REVOLVER, Jurisdiction.BCBS, False, 0.40),
            (RevolvingProductType.HELOC, Jurisdiction.BCBS, False, 0.40),
            (RevolvingProductType.WORKING_CAPITAL, Jurisdiction.BCBS, False, 0.40),
        ],
        ids=[
            "card-bcbs-10pct",
            "overdraft-bcbs-10pct",
            "card-apra-40pct",
            "card-eu-crr3-transitional-0pct",
            "corp-revolver-bcbs-40pct",
            "heloc-bcbs-40pct",
            "working-capital-bcbs-40pct",
        ],
    )
    def test_sa_ccf(
        self,
        product: RevolvingProductType,
        jurisdiction: Jurisdiction,
        transitional: bool,
        expected: float,
    ) -> None:
        ccf = regulatory_ccf_sa(product, jurisdiction, transitional)
        assert ccf == pytest.approx(expected)

    # --- Regulatory F-IRB (parametrized) ---
    @pytest.mark.parametrize(
        ("product", "expected"),
        [
            (RevolvingProductType.CREDIT_CARD, 0.40),
            (RevolvingProductType.OVERDRAFT, 0.40),
            (RevolvingProductType.CORPORATE_REVOLVER, 0.75),
            (RevolvingProductType.HELOC, 0.75),
        ],
        ids=["card-40pct", "overdraft-40pct", "corp-75pct", "heloc-75pct"],
    )
    def test_firb_ccf(
        self, product: RevolvingProductType, expected: float,
    ) -> None:
        assert regulatory_ccf_firb(product) == pytest.approx(expected)

    # --- A-IRB floor (parametrized) ---
    @pytest.mark.parametrize(
        ("product", "jurisdiction", "expected"),
        [
            (RevolvingProductType.CREDIT_CARD, Jurisdiction.BCBS, 0.05),
            (RevolvingProductType.CREDIT_CARD, Jurisdiction.AUSTRALIA, 0.20),
            (RevolvingProductType.CORPORATE_REVOLVER, Jurisdiction.BCBS, 0.20),
        ],
        ids=["card-bcbs-5pct", "card-apra-20pct", "corp-bcbs-20pct"],
    )
    def test_airb_floor(
        self,
        product: RevolvingProductType,
        jurisdiction: Jurisdiction,
        expected: float,
    ) -> None:
        assert airb_ccf_floor(product, jurisdiction) == pytest.approx(expected)

    # --- Behavioral CCF ---
    def test_behavioral_ccf_basic(self) -> None:
        ead = np.array([9200.0, 8500.0, 9800.0])
        drawn = np.array([6000.0, 5000.0, 7000.0])
        undrawn = np.array([4000.0, 5000.0, 3000.0])
        ccf = behavioral_ccf(ead, drawn, undrawn)
        expected = np.mean(
            (ead - drawn) / undrawn
        )
        assert ccf == pytest.approx(expected, abs=1e-6)

    def test_behavioral_ccf_clipped(self) -> None:
        ead = np.array([15000.0])
        drawn = np.array([6000.0])
        undrawn = np.array([4000.0])
        ccf = behavioral_ccf(ead, drawn, undrawn)
        assert ccf == 1.0

    def test_behavioral_ccf_all_zero_undrawn(self) -> None:
        ccf = behavioral_ccf(
            np.array([5000.0]), np.array([5000.0]), np.array([0.0])
        )
        assert ccf == 0.0

    def test_behavioral_ccf_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            behavioral_ccf(np.array([]), np.array([]), np.array([]))

    # --- EADF ---
    def test_eadf_basic(self) -> None:
        ead = np.array([8000.0, 9000.0])
        limit = np.array([10000.0, 10000.0])
        ccf = eadf_ccf(ead, limit)
        assert ccf == pytest.approx(0.85)

    def test_eadf_clipped(self) -> None:
        ccf = eadf_ccf(np.array([12000.0]), np.array([10000.0]))
        assert ccf == 1.0

    # --- PIT adjustment ---
    def test_pit_adjustment_neutral(self) -> None:
        pit = ccf_pit_adjustment(0.50, z_factor=0.0)
        assert pit == pytest.approx(0.50)

    def test_pit_adjustment_stress(self) -> None:
        pit = ccf_pit_adjustment(0.50, z_factor=1.0, sensitivity=0.5)
        assert pit == pytest.approx(0.75)

    def test_pit_adjustment_clipped(self) -> None:
        pit = ccf_pit_adjustment(0.80, z_factor=2.0, sensitivity=0.5)
        assert pit == 1.0

    # --- Floor application ---
    def test_apply_floor_behavioral(self) -> None:
        floored = apply_ccf_with_floor(
            0.03,
            RevolvingProductType.CREDIT_CARD,
            CCFMethod.BEHAVIORAL,
        )
        assert floored == pytest.approx(0.05)

    def test_apply_floor_regulatory_no_change(self) -> None:
        floored = apply_ccf_with_floor(
            0.10,
            RevolvingProductType.CREDIT_CARD,
            CCFMethod.REGULATORY_SA,
        )
        assert floored == 0.10


# =====================================================================
# 5. EAD Profile Tests
# =====================================================================


class TestEADProfile:
    def test_flat_profile(self) -> None:
        profile = revolving_ead_term_structure(
            drawn=6000, undrawn=4000, ccf=0.80, n_periods=12
        )
        assert isinstance(profile, RevolvingEADProfile)
        assert len(profile.drawn) == 12
        np.testing.assert_allclose(profile.drawn, 6000.0)
        np.testing.assert_allclose(profile.undrawn, 4000.0)
        np.testing.assert_allclose(profile.ead, 6000 + 0.80 * 4000)
        assert profile.limit == 10000.0

    def test_net_repaying_profile(self) -> None:
        profile = revolving_ead_term_structure(
            drawn=6000, undrawn=4000, ccf=0.80, n_periods=12,
            repayment_rate=0.05, redraw_rate=0.02,
        )
        assert profile.drawn[0] < 6000.0
        assert profile.drawn[-1] < profile.drawn[0]
        assert all(d >= 0 for d in profile.drawn)

    def test_drawn_capped_at_limit(self) -> None:
        profile = revolving_ead_term_structure(
            drawn=6000, undrawn=4000, ccf=0.80, n_periods=6,
            repayment_rate=0.0, redraw_rate=0.10,
        )
        assert all(d <= 10000.0 for d in profile.drawn)

    def test_negative_inputs_raise(self) -> None:
        with pytest.raises(ValueError):
            revolving_ead_term_structure(-1, 4000, 0.80, 12)

    def test_split_flat(self) -> None:
        drawn_arr, undrawn_arr = ead_drawn_undrawn_split(
            6000, 4000, 0.80, 5
        )
        assert len(drawn_arr) == 5
        assert len(undrawn_arr) == 5
        np.testing.assert_allclose(drawn_arr, 6000.0)
        np.testing.assert_allclose(undrawn_arr, 0.80 * 4000.0)


# =====================================================================
# 6. Revolving ECL Engine Tests
# =====================================================================


class TestRevolvingECL:
    def test_stage1_basic(self) -> None:
        result = calculate_revolving_ecl(
            stage=IFRS9Stage.STAGE_1,
            drawn=6000.0,
            undrawn=4000.0,
            ccf=0.80,
            pd_12m=0.03,
            lgd=0.85,
            eir=0.18,
        )
        assert isinstance(result, RevolvingECLResult)
        assert result.total_ecl > 0
        assert result.ecl_drawn > 0
        assert result.ecl_undrawn > 0
        assert result.total_ecl == pytest.approx(
            result.ecl_drawn + result.ecl_undrawn
        )

    def test_stage1_drawn_undrawn_ratio(self) -> None:
        result = calculate_revolving_ecl(
            stage=IFRS9Stage.STAGE_1,
            drawn=6000.0,
            undrawn=4000.0,
            ccf=0.80,
            pd_12m=0.03,
            lgd=0.85,
        )
        drawn_ead = 6000.0
        undrawn_ead = 0.80 * 4000.0
        expected_ratio = drawn_ead / (drawn_ead + undrawn_ead)
        actual_ratio = result.ecl_drawn / result.total_ecl
        assert actual_ratio == pytest.approx(expected_ratio, abs=0.01)

    def test_stage2_requires_marginal_pds(self) -> None:
        with pytest.raises(ValueError, match="marginal_pds"):
            calculate_revolving_ecl(
                stage=IFRS9Stage.STAGE_2,
                drawn=6000.0,
                undrawn=4000.0,
                ccf=0.80,
                pd_12m=0.03,
                lgd=0.85,
            )

    def test_stage2_lifetime(self) -> None:
        marginal_pds = np.array([
            0.030, 0.025, 0.020, 0.015, 0.010,
        ])
        result = calculate_revolving_ecl(
            stage=IFRS9Stage.STAGE_2,
            drawn=6000.0,
            undrawn=4000.0,
            ccf=0.80,
            pd_12m=0.03,
            lgd=0.85,
            eir=0.18,
            marginal_pds=marginal_pds,
            behavioral_life_months=5,
        )
        assert result.total_ecl > 0
        assert len(result.ecl_by_period) == 5
        assert result.behavioral_life_months == 5

    def test_cliff_effect(self) -> None:
        """Stage 2 ECL should be significantly higher than Stage 1."""
        # 36-month behavioral life with monthly marginal PDs
        marginal_pds = np.full(36, 0.0025)  # ~3% annual PD
        s1 = calculate_revolving_ecl(
            stage=IFRS9Stage.STAGE_1,
            drawn=6000.0, undrawn=4000.0, ccf=0.80,
            pd_12m=0.03, lgd=0.85, eir=0.015,
            marginal_pds=marginal_pds,
            behavioral_life_months=36,
        )
        s2 = calculate_revolving_ecl(
            stage=IFRS9Stage.STAGE_2,
            drawn=6000.0, undrawn=4000.0, ccf=0.80,
            pd_12m=0.03, lgd=0.85, eir=0.015,
            marginal_pds=marginal_pds,
            behavioral_life_months=36,
        )
        # S2 uses 36 months vs S1's 12, so S2 >> S1
        assert s2.total_ecl > s1.total_ecl * 2.0

    def test_zero_undrawn(self) -> None:
        result = calculate_revolving_ecl(
            stage=IFRS9Stage.STAGE_1,
            drawn=10000.0, undrawn=0.0, ccf=0.80,
            pd_12m=0.02, lgd=0.85,
        )
        assert result.ecl_undrawn == pytest.approx(0.0)
        assert result.total_ecl == result.ecl_drawn

    def test_with_lgd_curve(self) -> None:
        marginal_pds = np.array([0.03, 0.025, 0.02])
        lgd_curve = np.array([0.85, 0.80, 0.75])
        result = calculate_revolving_ecl(
            stage=IFRS9Stage.STAGE_2,
            drawn=6000.0, undrawn=4000.0, ccf=0.80,
            pd_12m=0.03, lgd=0.85,
            marginal_pds=marginal_pds,
            behavioral_life_months=3,
            lgd_curve=lgd_curve,
        )
        assert result.total_ecl > 0

    def test_ccf_sensitivity(self) -> None:
        """Per research: 50% -> 75% CCF should increase ECL ~21%."""
        kwargs = dict(
            stage=IFRS9Stage.STAGE_1,
            drawn=20_000_000.0,
            undrawn=30_000_000.0,
            pd_12m=0.005,
            lgd=0.40,
            eir=0.05,
        )
        r50 = calculate_revolving_ecl(ccf=0.50, **kwargs)
        r75 = calculate_revolving_ecl(ccf=0.75, **kwargs)
        pct_increase = (r75.total_ecl - r50.total_ecl) / r50.total_ecl
        assert pct_increase > 0.15


# =====================================================================
# 7. Scenario Weighting Tests
# =====================================================================


class TestScenarioWeighting:
    def test_single_scenario(self) -> None:
        result = calculate_revolving_ecl(
            stage=IFRS9Stage.STAGE_1,
            drawn=6000.0, undrawn=4000.0, ccf=0.80,
            pd_12m=0.03, lgd=0.85,
        )
        weighted = revolving_ecl_scenario_weighted([(1.0, result)])
        assert weighted.total_ecl == pytest.approx(result.total_ecl)

    def test_three_scenario_weighting(self) -> None:
        base = calculate_revolving_ecl(
            stage=IFRS9Stage.STAGE_1,
            drawn=6000.0, undrawn=4000.0, ccf=0.80,
            pd_12m=0.03, lgd=0.85,
        )
        adverse = calculate_revolving_ecl(
            stage=IFRS9Stage.STAGE_1,
            drawn=6000.0, undrawn=4000.0, ccf=0.80,
            pd_12m=0.045, lgd=0.85,
        )
        favorable = calculate_revolving_ecl(
            stage=IFRS9Stage.STAGE_1,
            drawn=6000.0, undrawn=4000.0, ccf=0.80,
            pd_12m=0.021, lgd=0.85,
        )
        weighted = revolving_ecl_scenario_weighted([
            (0.50, base), (0.30, adverse), (0.20, favorable),
        ])
        expected = (
            0.50 * base.total_ecl
            + 0.30 * adverse.total_ecl
            + 0.20 * favorable.total_ecl
        )
        assert weighted.total_ecl == pytest.approx(expected, rel=1e-6)
        assert weighted.ecl_drawn > 0
        assert weighted.ecl_undrawn > 0

    def test_weights_must_sum_to_one(self) -> None:
        result = calculate_revolving_ecl(
            stage=IFRS9Stage.STAGE_1,
            drawn=6000.0, undrawn=4000.0, ccf=0.80,
            pd_12m=0.03, lgd=0.85,
        )
        with pytest.raises(ValueError, match="sum to 1.0"):
            revolving_ecl_scenario_weighted([(0.50, result), (0.30, result)])

    def test_empty_scenarios_raises(self) -> None:
        with pytest.raises(ValueError):
            revolving_ecl_scenario_weighted([])


# =====================================================================
# 8. Provision Floor Tests
# =====================================================================


class TestProvisionFloors:
    @pytest.mark.parametrize(
        ("jurisdiction", "stage", "expected_rate"),
        [
            (Jurisdiction.UAE, None, 0.015),
            (Jurisdiction.INDIA, IFRS9Stage.STAGE_1, 0.01),
            (Jurisdiction.INDIA, IFRS9Stage.STAGE_2, 0.05),
            (Jurisdiction.SINGAPORE, None, 0.01),
            (Jurisdiction.SAUDI_ARABIA, None, 0.01),
        ],
        ids=["cbuae-1.5pct", "rbi-s1-1pct", "rbi-s2-5pct", "mas-1pct", "sama-1pct"],
    )
    def test_floor_exists(
        self,
        jurisdiction: Jurisdiction,
        stage: IFRS9Stage | None,
        expected_rate: float,
    ) -> None:
        floors = get_provision_floors(jurisdiction, stage)
        assert any(f.floor_rate == expected_rate for f in floors)

    def test_apply_floor_binding(self) -> None:
        ecl = apply_provision_floor(
            ecl=50.0,
            ead=10000.0,
            jurisdiction=Jurisdiction.INDIA,
            stage=IFRS9Stage.STAGE_1,
        )
        assert ecl == 100.0  # 1% of 10,000

    def test_apply_floor_non_binding(self) -> None:
        ecl = apply_provision_floor(
            ecl=200.0,
            ead=10000.0,
            jurisdiction=Jurisdiction.INDIA,
            stage=IFRS9Stage.STAGE_1,
        )
        assert ecl == 200.0

    def test_crwa_floor_requires_crwa(self) -> None:
        ecl = apply_provision_floor(
            ecl=50.0,
            ead=10000.0,
            jurisdiction=Jurisdiction.UAE,
            stage=IFRS9Stage.STAGE_1,
        )
        assert ecl == 50.0  # No CRWA provided, can't apply

    def test_crwa_floor_with_crwa(self) -> None:
        ecl = apply_provision_floor(
            ecl=50.0,
            ead=10000.0,
            jurisdiction=Jurisdiction.UAE,
            stage=IFRS9Stage.STAGE_1,
            crwa=100000.0,
        )
        assert ecl == 1500.0  # 1.5% of 100,000

    def test_no_floor_for_bcbs(self) -> None:
        floors = get_provision_floors(Jurisdiction.BCBS)
        assert len(floors) == 0


# =====================================================================
# 9. Integration Tests
# =====================================================================


class TestIntegration:
    def test_credit_card_full_workflow(self) -> None:
        """Research Example 1: Retail credit card Stage 1 vs Stage 2."""
        from creditriskengine.ecl.ifrs9.revolving import (
            apply_provision_floor,
            calculate_revolving_ecl,
            determine_behavioral_life,
            regulatory_ccf_sa,
            revolving_ecl_scenario_weighted,
        )

        # Determine behavioral life
        life = determine_behavioral_life(
            product_type=RevolvingProductType.CREDIT_CARD
        )
        assert life == 36

        # Get CCF
        ccf = regulatory_ccf_sa(RevolvingProductType.CREDIT_CARD)
        assert ccf == 0.10

        # Use behavioral CCF for ECL (higher than regulatory)
        ecl_ccf = 0.80

        # PD term structure (36 months = 3 years)
        marginal_pds = np.full(36, 0.0025)  # ~3% annual

        # Stage 1
        s1 = calculate_revolving_ecl(
            stage=IFRS9Stage.STAGE_1,
            drawn=6000.0, undrawn=4000.0, ccf=ecl_ccf,
            pd_12m=0.03, lgd=0.85, eir=0.015,
            marginal_pds=marginal_pds,
            behavioral_life_months=36,
        )
        assert s1.ecl_drawn > 0
        assert s1.ecl_undrawn > 0

        # Stage 2
        s2 = calculate_revolving_ecl(
            stage=IFRS9Stage.STAGE_2,
            drawn=6000.0, undrawn=4000.0, ccf=ecl_ccf,
            pd_12m=0.03, lgd=0.85, eir=0.015,
            marginal_pds=marginal_pds,
            behavioral_life_months=36,
        )

        # Cliff effect: Stage 2 >> Stage 1
        assert s2.total_ecl > s1.total_ecl * 2.0

        # Scenario weighting
        adverse = calculate_revolving_ecl(
            stage=IFRS9Stage.STAGE_2,
            drawn=6000.0, undrawn=4000.0, ccf=ecl_ccf,
            pd_12m=0.045, lgd=0.85, eir=0.015,
            marginal_pds=marginal_pds * 1.5,
            behavioral_life_months=36,
        )
        favorable = calculate_revolving_ecl(
            stage=IFRS9Stage.STAGE_2,
            drawn=6000.0, undrawn=4000.0, ccf=ecl_ccf,
            pd_12m=0.021, lgd=0.85, eir=0.015,
            marginal_pds=marginal_pds * 0.7,
            behavioral_life_months=36,
        )
        weighted = revolving_ecl_scenario_weighted([
            (0.50, s2), (0.30, adverse), (0.20, favorable),
        ])
        assert weighted.total_ecl > 0

        # Apply RBI provision floor
        floored = apply_provision_floor(
            ecl=weighted.total_ecl,
            ead=6000 + ecl_ccf * 4000,
            jurisdiction=Jurisdiction.INDIA,
            stage=IFRS9Stage.STAGE_2,
        )
        assert floored >= 0.05 * (6000 + ecl_ccf * 4000)

    def test_corporate_revolver_workflow(self) -> None:
        """Research Example 2: $50M corporate revolver."""
        from creditriskengine.ecl.ifrs9.revolving import (
            calculate_revolving_ecl,
            regulatory_ccf_sa,
        )

        ccf = regulatory_ccf_sa(RevolvingProductType.CORPORATE_REVOLVER)
        assert ccf == 0.40

        result = calculate_revolving_ecl(
            stage=IFRS9Stage.STAGE_1,
            drawn=20_000_000.0,
            undrawn=30_000_000.0,
            ccf=0.50,
            pd_12m=0.005,
            lgd=0.40,
            eir=0.05,
        )
        assert result.ecl_drawn > 0
        assert result.ecl_undrawn > 0
        assert result.total_ecl == pytest.approx(
            result.ecl_drawn + result.ecl_undrawn
        )

    def test_apra_ccf_conservatism(self) -> None:
        """APRA's 40% CCF vs Basel's 10% for credit cards."""
        from creditriskengine.ecl.ifrs9.revolving import regulatory_ccf_sa

        bcbs = regulatory_ccf_sa(
            RevolvingProductType.CREDIT_CARD,
            Jurisdiction.BCBS,
        )
        apra = regulatory_ccf_sa(
            RevolvingProductType.CREDIT_CARD,
            Jurisdiction.AUSTRALIA,
        )
        assert apra == 4 * bcbs

    def test_exposure_object_integration(self) -> None:
        """The module works with Exposure objects via convenience fn."""
        from creditriskengine.core.exposure import Exposure
        from creditriskengine.core.types import CreditRiskApproach
        from creditriskengine.ecl.ifrs9.revolving.ecl_revolving import (
            revolving_ecl_from_exposure,
        )

        exposure = Exposure(
            exposure_id="CC-001",
            counterparty_id="CUST-001",
            ead=9200.0,
            drawn_amount=6000.0,
            undrawn_commitment=4000.0,
            jurisdiction=Jurisdiction.BCBS,
            approach=CreditRiskApproach.SA,
            ifrs9_stage=IFRS9Stage.STAGE_1,
            current_pd=0.03,
            lgd=0.85,
            effective_interest_rate=0.18,
            is_revolving=True,
            credit_limit=10000.0,
            behavioral_life_months=36,
            ccf=0.80,
        )

        result = revolving_ecl_from_exposure(exposure)
        assert result.total_ecl > 0
        assert result.ecl_drawn > 0
        assert result.ecl_undrawn > 0
        assert result.total_ecl == pytest.approx(
            result.ecl_drawn + result.ecl_undrawn
        )

    def test_exposure_missing_ccf_raises(self) -> None:
        """Exposure without CCF raises ValueError."""
        from creditriskengine.core.exposure import Exposure
        from creditriskengine.core.types import CreditRiskApproach
        from creditriskengine.ecl.ifrs9.revolving.ecl_revolving import (
            revolving_ecl_from_exposure,
        )

        exposure = Exposure(
            exposure_id="CC-002",
            counterparty_id="CUST-002",
            ead=9200.0,
            drawn_amount=6000.0,
            undrawn_commitment=4000.0,
            jurisdiction=Jurisdiction.BCBS,
            approach=CreditRiskApproach.SA,
            ifrs9_stage=IFRS9Stage.STAGE_1,
            current_pd=0.03,
            lgd=0.85,
            is_revolving=True,
        )
        with pytest.raises(ValueError, match="ccf"):
            revolving_ecl_from_exposure(exposure)

    def test_parent_package_exports(self) -> None:
        """Revolving module is importable from parent packages."""
        from creditriskengine.ecl import calculate_revolving_ecl as fn1
        from creditriskengine.ecl.ifrs9 import calculate_revolving_ecl as fn2

        assert fn1 is fn2

    def test_ccf_single_source_of_truth(self) -> None:
        """Revolving CCF delegates to ead_model.py, not its own tables."""
        from creditriskengine.models.ead.ead_model import get_sa_ccf

        # Both paths should return the same value
        ead_ccf = get_sa_ccf("unconditionally_cancellable", "australia")
        revolving_ccf = regulatory_ccf_sa(
            RevolvingProductType.CREDIT_CARD, Jurisdiction.AUSTRALIA,
        )
        assert ead_ccf == revolving_ccf == 0.40
