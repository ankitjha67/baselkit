"""Tests for all gap/divergence fixes — spec compliance audit."""

import numpy as np
import pytest

# ============================================================
# Gap 1: Re-export module imports
# ============================================================


class TestReExportModulesPD:
    """Verify PD re-export modules resolve correctly."""

    def test_logistic_module(self) -> None:
        from creditriskengine.models.pd.logistic import logistic_score, score_to_pd
        scores = logistic_score(np.array([1.0]), np.array([[2.0]]), 0.5)
        assert scores[0] == pytest.approx(2.5)
        pds = score_to_pd(scores)
        assert 0.0 < pds[0] < 1.0

    def test_calibration_module(self) -> None:
        from creditriskengine.models.pd.calibration import (
            calibrate_pd_anchor_point,
            calibrate_pd_bayesian,
        )
        raw = np.array([0.02, 0.04])
        calibrated = calibrate_pd_anchor_point(0.03, raw)
        assert calibrated.mean() == pytest.approx(0.03, rel=0.01)

        pd_bay = calibrate_pd_bayesian(0.02, 5, 100, weight=0.5)
        assert 0.0 < pd_bay < 1.0

    def test_ttc_calibration_module(self) -> None:
        from creditriskengine.models.pd.ttc_calibration import vasicek_single_factor_pd
        cpd = vasicek_single_factor_pd(0.01, 0.15)
        assert cpd > 0.01  # stressed PD should be higher

    def test_rating_scale_module(self) -> None:
        from creditriskengine.models.pd.rating_scale import (
            assign_rating_grade,
            build_master_scale,
        )
        scale = build_master_scale([0.0, 0.01, 0.05, 1.0])
        assert len(scale) == 3
        grade = assign_rating_grade(0.005, scale)
        assert grade == "Grade_1"


class TestReExportModulesLGD:
    """Verify LGD re-export modules resolve correctly."""

    def test_workout_lgd_module(self) -> None:
        from creditriskengine.models.lgd.workout_lgd import workout_lgd
        lgd = workout_lgd(ead_at_default=1000.0, total_recoveries=600.0, total_costs=50.0)
        assert lgd == pytest.approx(0.45, abs=0.01)

    def test_downturn_lgd_module(self) -> None:
        from creditriskengine.models.lgd.downturn_lgd import downturn_lgd
        dt = downturn_lgd(0.30, downturn_add_on=0.10, method="additive")
        assert dt == pytest.approx(0.40)

    def test_regulatory_lgd_module(self) -> None:
        from creditriskengine.models.lgd.regulatory_lgd import apply_lgd_floor
        floored = apply_lgd_floor(0.05, collateral_type="unsecured")
        assert floored >= 0.25


class TestReExportModulesEAD:
    """Verify EAD re-export modules resolve correctly."""

    def test_ccf_module(self) -> None:
        from creditriskengine.models.ead.ccf import estimate_ccf
        ccf = estimate_ccf(
            ead_at_default=90.0,
            drawn_at_reference=60.0,
            limit=100.0,
        )
        assert ccf == pytest.approx(0.75, abs=0.01)

    def test_regulatory_ccf_module(self) -> None:
        from creditriskengine.models.ead.regulatory_ccf import (
            apply_ccf_floor,
            get_supervisory_ccf,
        )
        ccf = get_supervisory_ccf("committed_other")
        assert ccf == 0.75
        floored = apply_ccf_floor(0.30, approach="airb")
        assert floored >= 0.30  # floor applies to revolving at formula level

    def test_ead_estimation_module(self) -> None:
        from creditriskengine.models.ead.ead_estimation import ead_term_structure
        ts = ead_term_structure(
            drawn_amount=100.0, undrawn_commitment=50.0, ccf=0.75, n_periods=3
        )
        assert len(ts) == 3


class TestReExportModulesConcentration:
    """Verify concentration re-export modules resolve correctly."""

    def test_hhi_module(self) -> None:
        from creditriskengine.models.concentration.hhi import single_name_concentration
        result = single_name_concentration(np.array([100.0, 200.0, 300.0]))
        assert "hhi" in result
        assert result["hhi"] > 0

    def test_granularity_module(self) -> None:
        from creditriskengine.models.concentration.granularity import (
            granularity_adjustment,
        )
        ga = granularity_adjustment(
            eads=np.array([1e6, 2e6]),
            pds=np.array([0.02, 0.03]),
            lgds=np.array([0.40, 0.35]),
            rho=0.15,
        )
        assert ga > 0

    def test_sector_module(self) -> None:
        from creditriskengine.models.concentration.sector import sector_concentration
        result = sector_concentration(
            eads=np.array([100.0, 200.0]),
            sector_labels=np.array(["A", "B"]),
        )
        assert "sector_hhi" in result


# ============================================================
# Gap 2: QRRE Transactor Scalar
# ============================================================


class TestQRRETransactorScalar:
    """BCBS CRE31.9 footnote 15: QRRE transactors get 0.75× RW."""

    def test_transactor_scalar_applied(self) -> None:
        from creditriskengine.rwa.irb.formulas import irb_risk_weight
        rw_revolver = irb_risk_weight(pd=0.02, lgd=0.80, asset_class="qrre")
        rw_transactor = irb_risk_weight(
            pd=0.02, lgd=0.80, asset_class="qrre", is_qrre_transactor=True
        )
        assert rw_transactor == pytest.approx(rw_revolver * 0.75, rel=1e-6)

    def test_transactor_flag_no_effect_on_non_qrre(self) -> None:
        from creditriskengine.rwa.irb.formulas import irb_risk_weight
        rw = irb_risk_weight(pd=0.01, lgd=0.45, asset_class="corporate")
        rw_flag = irb_risk_weight(
            pd=0.01, lgd=0.45, asset_class="corporate", is_qrre_transactor=True
        )
        assert rw == pytest.approx(rw_flag, rel=1e-6)


# ============================================================
# Gap 3-5: CRR3 YAML Verification
# ============================================================


class TestCRR3YAMLFixes:
    """Verify CRR3 YAML structure and values."""

    @pytest.fixture()
    def crr3_config(self) -> dict:
        import yaml
        from pathlib import Path
        yaml_path = Path(__file__).resolve().parents[1] / (
            "src/creditriskengine/regulatory/eu/crr3.yml"
        )
        with open(yaml_path) as f:
            return yaml.safe_load(f)

    def test_pd_floor_is_3_bps(self, crr3_config: dict) -> None:
        """Gap 4: PD floor should be 3 bps per CRE32.13."""
        assert crr3_config["credit_risk"]["irb_approach"]["pd_floor_bps"] == 3

    def test_lgd_supervisory_secured_values(self, crr3_config: dict) -> None:
        """Gap 5: LGD supervisory values for secured should be 0.10/0.15."""
        lgd_sup = crr3_config["credit_risk"]["irb_approach"]["lgd_supervisory"]
        assert lgd_sup["secured_receivables"] == 0.10
        assert lgd_sup["secured_commercial_real_estate"] == 0.10
        assert lgd_sup["secured_residential_real_estate"] == 0.10
        assert lgd_sup["secured_other_collateral"] == 0.15

    def test_lgd_floors_correct(self, crr3_config: dict) -> None:
        """Gap 5: LGD floors should match CRE32.25."""
        lgd_floors = crr3_config["credit_risk"]["irb_approach"]["lgd_floors"]
        assert lgd_floors["secured_receivables"] == 0.10
        assert lgd_floors["secured_residential_real_estate"] == 0.10
        assert lgd_floors["secured_commercial_real_estate"] == 0.10
        assert lgd_floors["secured_other_collateral"] == 0.15

    def test_residential_re_ltv_buckets(self, crr3_config: dict) -> None:
        """Gap 3: Full 7-bucket LTV table for residential RE."""
        rre = crr3_config["credit_risk"]["standardised_approach"]["risk_weights"][
            "residential_re"
        ]
        not_cf = rre["not_dependent_on_cashflows"]
        assert len(not_cf) == 7
        assert not_cf[0]["risk_weight"] == 0.20
        assert not_cf[-1]["risk_weight"] == 0.70

    def test_residential_re_cashflow_dependent_buckets(self, crr3_config: dict) -> None:
        """Gap 3: Cashflow-dependent RRE table."""
        rre = crr3_config["credit_risk"]["standardised_approach"]["risk_weights"][
            "residential_re"
        ]
        cf = rre["dependent_on_cashflows"]
        assert len(cf) == 7
        assert cf[0]["risk_weight"] == 0.30
        assert cf[-1]["risk_weight"] == 1.05

    def test_commercial_re_structure(self, crr3_config: dict) -> None:
        """Gap 3: Commercial RE with min(60%, counterparty_rw) logic."""
        cre = crr3_config["credit_risk"]["standardised_approach"]["risk_weights"][
            "commercial_re"
        ]
        not_cf = cre["not_dependent_on_cashflows"]
        # First bucket: LTV <= 60%, note about min(60%, counterparty_rw)
        assert not_cf[0]["risk_weight"] == 0.60
        assert "min(60%, counterparty_rw)" in not_cf[0].get("note", "")
        # High LTV: counterparty_rw (null in YAML)
        assert not_cf[2]["risk_weight"] is None

    def test_airb_restrictions_with_revenue_threshold(self, crr3_config: dict) -> None:
        """Gap 3: AIRB restrictions include revenue threshold."""
        airb = crr3_config["credit_risk"]["irb_approach"]["airb_restrictions"]
        assert airb["revenue_threshold_eur"] == 500000000
        assert "large_corporates_revenue_above_500m_eur" in airb["excluded_classes"]

    def test_supporting_factors(self, crr3_config: dict) -> None:
        """Gap 3: SME and infrastructure supporting factors."""
        sme = crr3_config["credit_risk"]["standardised_approach"]["sme"]
        assert sme["sme_supporting_factor_threshold_eur"] == 2500000
        assert sme["sme_supporting_factor_below"] == pytest.approx(0.7619)
        infra = crr3_config["credit_risk"]["standardised_approach"]["infrastructure"]
        assert infra["infrastructure_supporting_factor"] == 0.75


# ============================================================
# Gap 6: UK PRA Loan-Splitting
# ============================================================


class TestUKPRALoanSplitting:
    """UK PRA loan-splitting for residential real estate (PS9/24)."""

    def test_basic_split_high_ltv(self) -> None:
        from creditriskengine.rwa.standardized.credit_risk_sa import (
            uk_pra_loan_splitting_rre,
        )
        result = uk_pra_loan_splitting_rre(
            loan_amount=200_000, property_value=250_000
        )
        # LTV = 80%
        assert result["ltv"] == pytest.approx(0.80)
        # Secured = 55% × 250k = 137,500
        assert result["secured_amount"] == pytest.approx(137_500.0)
        # Unsecured = 200k - 137.5k = 62,500
        assert result["unsecured_amount"] == pytest.approx(62_500.0)
        # Secured tranche LTV = 137500/250000 = 0.55, falls in [0.50, 0.60] → 25%
        assert result["secured_rw"] == 25.0
        # Unsecured = 100% (default counterparty RW)
        assert result["unsecured_rw"] == 100.0
        # Blended = (137500×25 + 62500×100) / 200000
        expected_blended = (137_500 * 25.0 + 62_500 * 100.0) / 200_000
        assert result["blended_rw"] == pytest.approx(expected_blended, rel=1e-6)

    def test_low_ltv_fully_secured(self) -> None:
        from creditriskengine.rwa.standardized.credit_risk_sa import (
            uk_pra_loan_splitting_rre,
        )
        # Loan 100k on 400k property → LTV=25%, fully within secured tranche
        result = uk_pra_loan_splitting_rre(
            loan_amount=100_000, property_value=400_000
        )
        assert result["secured_amount"] == 100_000.0
        assert result["unsecured_amount"] == 0.0
        assert result["blended_rw"] == result["secured_rw"]

    def test_zero_property_value(self) -> None:
        from creditriskengine.rwa.standardized.credit_risk_sa import (
            uk_pra_loan_splitting_rre,
        )
        result = uk_pra_loan_splitting_rre(
            loan_amount=100_000, property_value=0.0
        )
        assert result["unsecured_amount"] == 100_000.0
        assert result["blended_rw"] == 100.0

    def test_zero_loan_amount(self) -> None:
        from creditriskengine.rwa.standardized.credit_risk_sa import (
            uk_pra_loan_splitting_rre,
        )
        result = uk_pra_loan_splitting_rre(
            loan_amount=0.0, property_value=250_000
        )
        assert result["blended_rw"] == 100.0  # counterparty_rw fallback

    def test_custom_counterparty_rw(self) -> None:
        from creditriskengine.rwa.standardized.credit_risk_sa import (
            uk_pra_loan_splitting_rre,
        )
        result = uk_pra_loan_splitting_rre(
            loan_amount=200_000,
            property_value=250_000,
            counterparty_rw=75.0,
        )
        assert result["unsecured_rw"] == 75.0
        assert result["blended_rw"] < 75.0  # Blended should be lower

    def test_cashflow_dependent_flag(self) -> None:
        from creditriskengine.rwa.standardized.credit_risk_sa import (
            uk_pra_loan_splitting_rre,
        )
        result_normal = uk_pra_loan_splitting_rre(
            loan_amount=200_000, property_value=250_000
        )
        result_cf = uk_pra_loan_splitting_rre(
            loan_amount=200_000,
            property_value=250_000,
            is_cashflow_dependent=True,
        )
        # Cashflow-dependent should have higher secured RW
        assert result_cf["secured_rw"] > result_normal["secured_rw"]


class TestUKPRAYAMLLoanSplitting:
    """Verify UK PRA YAML has loan-splitting configuration."""

    def test_yaml_has_loan_splitting(self) -> None:
        import yaml
        from pathlib import Path
        yaml_path = Path(__file__).resolve().parents[1] / (
            "src/creditriskengine/regulatory/uk/pra_basel31.yml"
        )
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        rre = config["credit_risk"]["standardised_approach"]["residential_re"]
        assert rre["approach"] == "loan_splitting"
        assert rre["loan_splitting"]["threshold_ltv"] == 0.55

    def test_uk_lgd_supervisory_fixed(self) -> None:
        """Gap 5: UK PRA LGD supervisory values also corrected."""
        import yaml
        from pathlib import Path
        yaml_path = Path(__file__).resolve().parents[1] / (
            "src/creditriskengine/regulatory/uk/pra_basel31.yml"
        )
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        lgd_sup = config["credit_risk"]["irb_approach"]["lgd_supervisory"]
        assert lgd_sup["secured_receivables"] == 0.10
        assert lgd_sup["secured_other_collateral"] == 0.15


# ============================================================
# Gap 7: BoE ACS Stress Testing
# ============================================================


class TestBoEACSStressTest:
    """Bank of England Annual Cyclical Scenario stress test."""

    @pytest.fixture()
    def acs_scenario(self) -> "MacroScenario":
        from creditriskengine.portfolio.stress_testing import MacroScenario
        return MacroScenario(
            name="ACS 2025",
            horizon_years=5,
            variables={
                "gdp_growth": np.array([-0.04, -0.03, -0.01, 0.01, 0.02]),
                "unemployment": np.array([0.08, 0.10, 0.09, 0.07, 0.05]),
                "house_price_index": np.array([-0.20, -0.15, -0.05, 0.0, 0.03]),
            },
            severity="severely_adverse",
        )

    def test_minimum_5_year_horizon(self) -> None:
        from creditriskengine.portfolio.stress_testing import (
            BoEACSStressTest,
            MacroScenario,
        )
        scenario = MacroScenario(name="Short", horizon_years=4)
        with pytest.raises(ValueError, match="minimum 5-year"):
            BoEACSStressTest(scenario, horizon_years=4)

    def test_run_returns_expected_keys(self, acs_scenario) -> None:
        from creditriskengine.portfolio.stress_testing import BoEACSStressTest
        boe = BoEACSStressTest(acs_scenario)
        result = boe.run(
            np.array([0.02, 0.05]),
            np.array([0.40, 0.30]),
            np.array([1e6, 2e6]),
        )
        assert "cumulative_el" in result
        assert "cet1_trajectory" in result
        assert "min_cet1_ratio" in result
        assert "cet1_hurdle_breach" in result
        assert result["horizon_years"] == 5
        assert result["scenario"] == "ACS 2025"
        assert len(result["cet1_trajectory"]) == 5

    def test_stressed_el_exceeds_baseline(self, acs_scenario) -> None:
        from creditriskengine.portfolio.stress_testing import BoEACSStressTest
        boe = BoEACSStressTest(acs_scenario)
        result = boe.run(
            np.array([0.02, 0.03]),
            np.array([0.40, 0.35]),
            np.array([1e6, 1e6]),
        )
        assert result["cumulative_el"] > result["baseline_el"]

    def test_cet1_hurdle_breach_detection(self, acs_scenario) -> None:
        from creditriskengine.portfolio.stress_testing import BoEACSStressTest
        # Start with low capital so hurdle is breached
        boe = BoEACSStressTest(acs_scenario)
        result = boe.run(
            np.array([0.10, 0.15]),  # high PDs
            np.array([0.60, 0.70]),  # high LGDs
            np.array([1e7, 1e7]),    # large exposures
            initial_cet1_ratio=0.05,  # only 5% CET1
        )
        # With heavy losses, CET1 should breach the 4.5% hurdle
        assert result["cet1_hurdle_breach"] is True

    def test_no_breach_with_strong_capital(self, acs_scenario) -> None:
        from creditriskengine.portfolio.stress_testing import BoEACSStressTest
        boe = BoEACSStressTest(acs_scenario)
        result = boe.run(
            np.array([0.01]),
            np.array([0.30]),
            np.array([1e5]),
            initial_cet1_ratio=0.15,  # very strong starting CET1
        )
        assert result["cet1_hurdle_breach"] is False

    def test_translate_pd_stress_dual_factor(self, acs_scenario) -> None:
        from creditriskengine.portfolio.stress_testing import BoEACSStressTest
        boe = BoEACSStressTest(acs_scenario)
        mults = boe.translate_macro_to_pd_stress()
        assert len(mults) == 5
        # All multipliers should be >= 1 (adverse scenario)
        assert all(m >= 1.0 for m in mults)

    def test_translate_lgd_stress(self, acs_scenario) -> None:
        from creditriskengine.portfolio.stress_testing import BoEACSStressTest
        boe = BoEACSStressTest(acs_scenario)
        adds = boe.translate_macro_to_lgd_stress()
        assert len(adds) == 5
        # Negative HPI should give positive LGD add-ons
        assert adds[0] > 0

    def test_custom_hurdle_rates(self) -> None:
        from creditriskengine.portfolio.stress_testing import (
            BoEACSStressTest,
            MacroScenario,
        )
        scenario = MacroScenario(
            name="Custom",
            horizon_years=5,
            variables={
                "gdp_growth": np.array([0.02, 0.02, 0.02, 0.02, 0.02]),
            },
        )
        boe = BoEACSStressTest(
            scenario,
            cet1_hurdle_pct=0.06,
            leverage_hurdle_pct=0.04,
        )
        assert boe.cet1_hurdle_pct == 0.06
        assert boe.leverage_hurdle_pct == 0.04

    def test_import_from_portfolio_init(self) -> None:
        from creditriskengine.portfolio import BoEACSStressTest
        assert BoEACSStressTest is not None

    def test_total_rwa_defaults_to_sum_eads(self, acs_scenario) -> None:
        from creditriskengine.portfolio.stress_testing import BoEACSStressTest
        boe = BoEACSStressTest(acs_scenario)
        eads = np.array([1e6, 2e6])
        result = boe.run(
            np.array([0.02, 0.05]),
            np.array([0.40, 0.30]),
            eads,
        )
        # Should not error; total_rwa defaults to sum(base_eads)
        assert result["cumulative_el"] > 0


# ============================================================
# Gap 8: SA Bank SCRA through dispatcher
# ============================================================


class TestSABankSCRADispatch:
    """Verify SCRA works through the assign_sa_risk_weight dispatcher."""

    def test_scra_grade_a_via_dispatcher(self) -> None:
        from creditriskengine.core.types import SAExposureClass
        from creditriskengine.rwa.standardized.credit_risk_sa import (
            assign_sa_risk_weight,
        )
        rw = assign_sa_risk_weight(
            SAExposureClass.BANK, scra_grade="A"
        )
        assert rw == 40.0

    def test_scra_grade_b_via_dispatcher(self) -> None:
        from creditriskengine.core.types import SAExposureClass
        from creditriskengine.rwa.standardized.credit_risk_sa import (
            assign_sa_risk_weight,
        )
        rw = assign_sa_risk_weight(
            SAExposureClass.BANK, scra_grade="B"
        )
        assert rw == 75.0

    def test_scra_grade_c_via_dispatcher(self) -> None:
        from creditriskengine.core.types import SAExposureClass
        from creditriskengine.rwa.standardized.credit_risk_sa import (
            assign_sa_risk_weight,
        )
        rw = assign_sa_risk_weight(
            SAExposureClass.BANK, scra_grade="C"
        )
        assert rw == 150.0

    def test_short_term_bank_via_dispatcher(self) -> None:
        """Gap 8: is_short_term now passes through the dispatcher."""
        from creditriskengine.core.types import CreditQualityStep, SAExposureClass
        from creditriskengine.rwa.standardized.credit_risk_sa import (
            assign_sa_risk_weight,
        )
        rw = assign_sa_risk_weight(
            SAExposureClass.BANK,
            cqs=CreditQualityStep.CQS_4,
            is_short_term=True,
        )
        # CQS_4 short-term → 50%
        assert rw == 50.0

    def test_short_term_bank_normal_vs_short(self) -> None:
        from creditriskengine.core.types import CreditQualityStep, SAExposureClass
        from creditriskengine.rwa.standardized.credit_risk_sa import (
            assign_sa_risk_weight,
        )
        rw_normal = assign_sa_risk_weight(
            SAExposureClass.BANK, cqs=CreditQualityStep.CQS_3
        )
        rw_short = assign_sa_risk_weight(
            SAExposureClass.BANK,
            cqs=CreditQualityStep.CQS_3,
            is_short_term=True,
        )
        # CQS_3 normal → 50%, short-term → 20%
        assert rw_normal == 50.0
        assert rw_short == 20.0
