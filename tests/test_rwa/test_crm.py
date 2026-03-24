"""Tests for Credit Risk Mitigation — BCBS d424, CRE22."""

import math

import pytest

from creditriskengine.rwa.crm import (
    CURRENCY_MISMATCH_HAIRCUT,
    HAIRCUT_TABLE,
    comprehensive_approach,
    guarantee_substitution,
    maturity_mismatch_adjustment,
    simple_approach,
    supervisory_haircut,
)


# ============================================================
# supervisory_haircut
# ============================================================

class TestSupervisoryHaircut:
    """CRE22.40–22.56 supervisory haircut look-up."""

    # --- Cash ---
    def test_cash_zero_haircut(self) -> None:
        assert supervisory_haircut("cash") == 0.0

    def test_cash_with_currency_mismatch(self) -> None:
        h = supervisory_haircut("cash", currency_mismatch=True)
        assert h == pytest.approx(CURRENCY_MISMATCH_HAIRCUT)

    # --- Sovereign bonds ---
    def test_sovereign_cqs1_le1(self) -> None:
        assert supervisory_haircut("sovereign_bond", 0.5, credit_quality_step=1) == pytest.approx(0.005)

    def test_sovereign_cqs1_1to5(self) -> None:
        assert supervisory_haircut("sovereign_bond", 3.0, credit_quality_step=1) == pytest.approx(0.02)

    def test_sovereign_cqs1_gt5(self) -> None:
        assert supervisory_haircut("sovereign_bond", 7.0, credit_quality_step=1) == pytest.approx(0.04)

    def test_sovereign_cqs2_le1(self) -> None:
        assert supervisory_haircut("sovereign_bond", 1.0, credit_quality_step=2) == pytest.approx(0.01)

    def test_sovereign_cqs2_1to5(self) -> None:
        assert supervisory_haircut("sovereign_bond", 4.0, credit_quality_step=2) == pytest.approx(0.03)

    def test_sovereign_cqs2_gt5(self) -> None:
        assert supervisory_haircut("sovereign_bond", 10.0, credit_quality_step=2) == pytest.approx(0.06)

    def test_sovereign_cqs3_same_as_cqs2(self) -> None:
        for mat in [0.5, 3.0, 7.0]:
            h2 = supervisory_haircut("sovereign_bond", mat, credit_quality_step=2)
            h3 = supervisory_haircut("sovereign_bond", mat, credit_quality_step=3)
            assert h2 == h3

    # --- Corporate bonds ---
    def test_corporate_cqs1_le1(self) -> None:
        assert supervisory_haircut("corporate_bond", 0.5, credit_quality_step=1) == pytest.approx(0.01)

    def test_corporate_cqs1_1to5(self) -> None:
        assert supervisory_haircut("corporate_bond", 3.0, credit_quality_step=1) == pytest.approx(0.04)

    def test_corporate_cqs1_gt5(self) -> None:
        assert supervisory_haircut("corporate_bond", 7.0, credit_quality_step=1) == pytest.approx(0.08)

    def test_corporate_cqs2_le1(self) -> None:
        assert supervisory_haircut("corporate_bond", 0.5, credit_quality_step=2) == pytest.approx(0.02)

    def test_corporate_cqs2_1to5(self) -> None:
        assert supervisory_haircut("corporate_bond", 3.0, credit_quality_step=2) == pytest.approx(0.06)

    def test_corporate_cqs2_gt5(self) -> None:
        assert supervisory_haircut("corporate_bond", 7.0, credit_quality_step=2) == pytest.approx(0.12)

    # --- Equities ---
    def test_main_index_equity(self) -> None:
        assert supervisory_haircut("main_index_equity") == pytest.approx(0.15)

    def test_other_equity(self) -> None:
        assert supervisory_haircut("other_equity") == pytest.approx(0.25)

    # --- Gold ---
    def test_gold(self) -> None:
        assert supervisory_haircut("gold") == pytest.approx(0.15)

    # --- Currency mismatch add-on ---
    def test_currency_mismatch_addon_bond(self) -> None:
        base = supervisory_haircut("corporate_bond", 3.0, credit_quality_step=1)
        with_fx = supervisory_haircut("corporate_bond", 3.0, credit_quality_step=1, currency_mismatch=True)
        assert with_fx == pytest.approx(base + 0.08)

    def test_currency_mismatch_addon_equity(self) -> None:
        base = supervisory_haircut("main_index_equity")
        with_fx = supervisory_haircut("main_index_equity", currency_mismatch=True)
        assert with_fx == pytest.approx(base + 0.08)

    def test_currency_mismatch_addon_gold(self) -> None:
        with_fx = supervisory_haircut("gold", currency_mismatch=True)
        assert with_fx == pytest.approx(0.15 + 0.08)

    # --- Invalid collateral type ---
    def test_invalid_collateral_type_raises(self) -> None:
        with pytest.raises(ValueError, match="No supervisory haircut"):
            supervisory_haircut("real_estate")

    def test_bond_without_cqs_raises(self) -> None:
        with pytest.raises(ValueError, match="No supervisory haircut"):
            supervisory_haircut("sovereign_bond", 3.0)

    def test_bond_invalid_cqs_raises(self) -> None:
        with pytest.raises(ValueError, match="No supervisory haircut"):
            supervisory_haircut("sovereign_bond", 3.0, credit_quality_step=5)

    # --- Maturity boundary tests ---
    def test_maturity_boundary_exactly_1(self) -> None:
        """Exactly 1 year falls in the le1 bucket."""
        h = supervisory_haircut("sovereign_bond", 1.0, credit_quality_step=1)
        assert h == pytest.approx(0.005)

    def test_maturity_boundary_just_over_1(self) -> None:
        h = supervisory_haircut("sovereign_bond", 1.01, credit_quality_step=1)
        assert h == pytest.approx(0.02)

    def test_maturity_boundary_exactly_5(self) -> None:
        """Exactly 5 years falls in the 1to5 bucket."""
        h = supervisory_haircut("sovereign_bond", 5.0, credit_quality_step=1)
        assert h == pytest.approx(0.02)

    def test_maturity_boundary_just_over_5(self) -> None:
        h = supervisory_haircut("sovereign_bond", 5.01, credit_quality_step=1)
        assert h == pytest.approx(0.04)


# ============================================================
# comprehensive_approach
# ============================================================

class TestComprehensiveApproach:
    """CRE22.57–22.77 comprehensive approach."""

    def test_basic_calculation(self) -> None:
        """E* = max(0, E*(1+He) - C*(1-Hc-Hfx))."""
        result = comprehensive_approach(
            exposure=1_000_000,
            collateral_value=500_000,
            collateral_type="cash",
        )
        # Cash Hc=0, no fx, He=0 → E* = 1M - 500k = 500k
        assert result["adjusted_exposure"] == pytest.approx(500_000)

    def test_full_cash_collateral(self) -> None:
        result = comprehensive_approach(
            exposure=1_000_000,
            collateral_value=1_000_000,
            collateral_type="cash",
        )
        assert result["adjusted_exposure"] == pytest.approx(0.0)

    def test_over_collateralised_floors_at_zero(self) -> None:
        result = comprehensive_approach(
            exposure=1_000_000,
            collateral_value=2_000_000,
            collateral_type="cash",
        )
        assert result["adjusted_exposure"] == pytest.approx(0.0)

    def test_zero_collateral(self) -> None:
        result = comprehensive_approach(
            exposure=1_000_000,
            collateral_value=0,
            collateral_type="cash",
        )
        assert result["adjusted_exposure"] == pytest.approx(1_000_000)

    def test_with_collateral_haircut(self) -> None:
        result = comprehensive_approach(
            exposure=1_000_000,
            collateral_value=600_000,
            collateral_type="sovereign_bond",
            residual_maturity_years=3.0,
            credit_quality_step=1,
        )
        # Hc = 0.02, E* = 1M - 600k*(1-0.02) = 1M - 588k = 412k
        assert result["adjusted_exposure"] == pytest.approx(412_000)
        assert result["collateral_haircut"] == pytest.approx(0.02)

    def test_with_currency_mismatch(self) -> None:
        result = comprehensive_approach(
            exposure=1_000_000,
            collateral_value=600_000,
            collateral_type="cash",
            currency_mismatch=True,
        )
        # Hc=0, Hfx=0.08 → E* = 1M - 600k*(1-0-0.08) = 1M - 552k = 448k
        assert result["adjusted_exposure"] == pytest.approx(448_000)
        assert result["fx_haircut"] == pytest.approx(0.08)

    def test_with_exposure_haircut(self) -> None:
        """Repo-style: exposure also gets a haircut."""
        result = comprehensive_approach(
            exposure=1_000_000,
            collateral_value=1_000_000,
            collateral_type="cash",
            exposure_haircut=0.02,
        )
        # E* = 1M*(1+0.02) - 1M*(1-0-0) = 1.02M - 1M = 20k
        assert result["adjusted_exposure"] == pytest.approx(20_000)

    def test_with_all_haircuts(self) -> None:
        result = comprehensive_approach(
            exposure=1_000_000,
            collateral_value=800_000,
            collateral_type="corporate_bond",
            residual_maturity_years=3.0,
            credit_quality_step=1,
            currency_mismatch=True,
            exposure_haircut=0.01,
        )
        # Hc=0.04, Hfx=0.08, He=0.01
        # E* = 1M*(1.01) - 800k*(1-0.04-0.08) = 1010k - 800k*0.88 = 1010k - 704k = 306k
        assert result["adjusted_exposure"] == pytest.approx(306_000)

    def test_result_keys(self) -> None:
        result = comprehensive_approach(
            exposure=100, collateral_value=50, collateral_type="cash",
        )
        expected_keys = {
            "adjusted_exposure", "collateral_haircut", "fx_haircut",
            "exposure_haircut", "gross_exposure", "collateral_value",
        }
        assert set(result.keys()) == expected_keys

    def test_gold_collateral(self) -> None:
        result = comprehensive_approach(
            exposure=1_000_000,
            collateral_value=500_000,
            collateral_type="gold",
        )
        # Hc=0.15, E* = 1M - 500k*(1-0.15) = 1M - 425k = 575k
        assert result["adjusted_exposure"] == pytest.approx(575_000)

    def test_equity_collateral(self) -> None:
        result = comprehensive_approach(
            exposure=1_000_000,
            collateral_value=500_000,
            collateral_type="other_equity",
        )
        # Hc=0.25, E* = 1M - 500k*(1-0.25) = 1M - 375k = 625k
        assert result["adjusted_exposure"] == pytest.approx(625_000)


# ============================================================
# simple_approach
# ============================================================

class TestSimpleApproach:
    """CRE22.35–22.39 simple approach."""

    def test_partial_coverage(self) -> None:
        result = simple_approach(
            exposure=1_000_000,
            collateral_value=400_000,
            exposure_rw=100.0,
            collateral_rw=20.0,
        )
        # Covered 400k at 20%, uncovered 600k at 100%
        assert result["covered_portion"] == pytest.approx(400_000)
        assert result["uncovered_portion"] == pytest.approx(600_000)
        assert result["rwa"] == pytest.approx(600_000 * 1.0 + 400_000 * 0.20)

    def test_full_coverage(self) -> None:
        result = simple_approach(
            exposure=1_000_000,
            collateral_value=1_000_000,
            exposure_rw=100.0,
            collateral_rw=20.0,
        )
        assert result["uncovered_portion"] == pytest.approx(0.0)
        assert result["rwa"] == pytest.approx(1_000_000 * 0.20)

    def test_over_collateralised_caps_at_exposure(self) -> None:
        result = simple_approach(
            exposure=1_000_000,
            collateral_value=2_000_000,
            exposure_rw=100.0,
            collateral_rw=20.0,
        )
        assert result["covered_portion"] == pytest.approx(1_000_000)
        assert result["uncovered_portion"] == pytest.approx(0.0)

    def test_floor_at_20_percent(self) -> None:
        """Collateral RW below 20% is floored at 20%."""
        result = simple_approach(
            exposure=1_000_000,
            collateral_value=1_000_000,
            exposure_rw=100.0,
            collateral_rw=0.0,
        )
        assert result["effective_collateral_rw"] == pytest.approx(20.0)
        assert result["rwa"] == pytest.approx(1_000_000 * 0.20)

    def test_cash_zero_floor(self) -> None:
        """Cash collateral has 0% floor (CRE22.37)."""
        result = simple_approach(
            exposure=1_000_000,
            collateral_value=1_000_000,
            exposure_rw=100.0,
            collateral_rw=0.0,
            is_cash_or_zero_haircut=True,
        )
        assert result["effective_collateral_rw"] == pytest.approx(0.0)
        assert result["rwa"] == pytest.approx(0.0)

    def test_zero_collateral(self) -> None:
        result = simple_approach(
            exposure=1_000_000,
            collateral_value=0,
            exposure_rw=100.0,
            collateral_rw=20.0,
        )
        assert result["covered_portion"] == pytest.approx(0.0)
        assert result["rwa"] == pytest.approx(1_000_000)

    def test_collateral_rw_above_floor(self) -> None:
        """When collateral RW > 20%, use actual RW."""
        result = simple_approach(
            exposure=1_000_000,
            collateral_value=500_000,
            exposure_rw=150.0,
            collateral_rw=50.0,
        )
        assert result["effective_collateral_rw"] == pytest.approx(50.0)
        assert result["rwa"] == pytest.approx(500_000 * 1.50 + 500_000 * 0.50)

    def test_result_keys(self) -> None:
        result = simple_approach(100, 50, 100.0, 20.0)
        expected_keys = {
            "rwa", "covered_portion", "uncovered_portion",
            "effective_collateral_rw", "exposure_rw",
        }
        assert set(result.keys()) == expected_keys


# ============================================================
# guarantee_substitution
# ============================================================

class TestGuaranteeSubstitution:
    """CRE22.78–22.93 substitution approach."""

    def test_full_guarantee(self) -> None:
        result = guarantee_substitution(
            exposure_rw=100.0, guarantor_rw=20.0, coverage_ratio=1.0,
        )
        assert result["effective_rw"] == pytest.approx(20.0)

    def test_no_guarantee(self) -> None:
        result = guarantee_substitution(
            exposure_rw=100.0, guarantor_rw=20.0, coverage_ratio=0.0,
        )
        assert result["effective_rw"] == pytest.approx(100.0)

    def test_partial_guarantee(self) -> None:
        result = guarantee_substitution(
            exposure_rw=100.0, guarantor_rw=20.0, coverage_ratio=0.5,
        )
        # 0.5*20 + 0.5*100 = 60
        assert result["effective_rw"] == pytest.approx(60.0)

    def test_guarantor_rw_higher_than_obligor(self) -> None:
        """Substitution can increase RW if guarantor is worse."""
        result = guarantee_substitution(
            exposure_rw=50.0, guarantor_rw=100.0, coverage_ratio=0.8,
        )
        # 0.8*100 + 0.2*50 = 90
        assert result["effective_rw"] == pytest.approx(90.0)

    def test_both_zero_rw(self) -> None:
        result = guarantee_substitution(
            exposure_rw=0.0, guarantor_rw=0.0, coverage_ratio=0.5,
        )
        assert result["effective_rw"] == pytest.approx(0.0)

    def test_coverage_ratio_out_of_range_high(self) -> None:
        with pytest.raises(ValueError, match="coverage_ratio must be in"):
            guarantee_substitution(100.0, 20.0, 1.5)

    def test_coverage_ratio_out_of_range_negative(self) -> None:
        with pytest.raises(ValueError, match="coverage_ratio must be in"):
            guarantee_substitution(100.0, 20.0, -0.1)

    def test_result_keys(self) -> None:
        result = guarantee_substitution(100.0, 20.0, 0.5)
        expected_keys = {"effective_rw", "exposure_rw", "guarantor_rw", "coverage_ratio"}
        assert set(result.keys()) == expected_keys

    def test_result_passthrough_values(self) -> None:
        result = guarantee_substitution(100.0, 20.0, 0.75)
        assert result["exposure_rw"] == pytest.approx(100.0)
        assert result["guarantor_rw"] == pytest.approx(20.0)
        assert result["coverage_ratio"] == pytest.approx(0.75)


# ============================================================
# maturity_mismatch_adjustment
# ============================================================

class TestMaturityMismatchAdjustment:
    """CRE22.33 maturity mismatch."""

    def test_no_mismatch(self) -> None:
        """t >= T → full recognition."""
        pa = maturity_mismatch_adjustment(1_000_000, 5.0, 3.0)
        assert pa == pytest.approx(1_000_000)

    def test_equal_maturities(self) -> None:
        pa = maturity_mismatch_adjustment(1_000_000, 3.0, 3.0)
        assert pa == pytest.approx(1_000_000)

    def test_basic_mismatch(self) -> None:
        """Pa = P * (t-0.25)/(T-0.25)."""
        # t=2, T=5 → (2-0.25)/(5-0.25) = 1.75/4.75
        pa = maturity_mismatch_adjustment(1_000_000, 2.0, 5.0)
        expected = 1_000_000 * (1.75 / 4.75)
        assert pa == pytest.approx(expected)

    def test_protection_maturity_3_months_or_less(self) -> None:
        """t <= 0.25 → no recognition (Pa = 0)."""
        pa = maturity_mismatch_adjustment(1_000_000, 0.25, 5.0)
        assert pa == pytest.approx(0.0)

    def test_protection_maturity_zero(self) -> None:
        pa = maturity_mismatch_adjustment(1_000_000, 0.0, 5.0)
        assert pa == pytest.approx(0.0)

    def test_protection_just_over_3_months(self) -> None:
        """t just above 0.25 → small but positive."""
        pa = maturity_mismatch_adjustment(1_000_000, 0.30, 5.0)
        expected = 1_000_000 * (0.05 / 4.75)
        assert pa == pytest.approx(expected)

    def test_both_maturities_short(self) -> None:
        """Both <= 0.25 but t >= T → full recognition."""
        pa = maturity_mismatch_adjustment(1_000_000, 0.2, 0.1)
        assert pa == pytest.approx(1_000_000)

    def test_both_maturities_at_025(self) -> None:
        """t = T = 0.25 → no mismatch, full recognition."""
        pa = maturity_mismatch_adjustment(1_000_000, 0.25, 0.25)
        assert pa == pytest.approx(1_000_000)

    def test_exposure_maturity_below_025_protection_below(self) -> None:
        """t < T but T <= 0.25 → full recognition (edge case)."""
        pa = maturity_mismatch_adjustment(1_000_000, 0.1, 0.2)
        # t < T but t <= 0.25 → Pa = 0
        assert pa == pytest.approx(0.0)

    def test_zero_collateral_value(self) -> None:
        pa = maturity_mismatch_adjustment(0, 2.0, 5.0)
        assert pa == pytest.approx(0.0)

    def test_large_mismatch(self) -> None:
        """Very short protection vs long exposure."""
        pa = maturity_mismatch_adjustment(1_000_000, 0.5, 10.0)
        expected = 1_000_000 * (0.25 / 9.75)
        assert pa == pytest.approx(expected)

    def test_exposure_maturity_at_025_protection_below(self) -> None:
        """t < 0.25, T = 0.25: t <= 0.25 so Pa = 0 (but T also <= 0.25,
        however the t <= 0.25 check fires first)."""
        pa = maturity_mismatch_adjustment(1_000_000, 0.1, 0.25)
        assert pa == pytest.approx(0.0)


# ============================================================
# Integration / cross-function tests
# ============================================================

class TestIntegration:
    """End-to-end scenarios combining multiple CRM functions."""

    def test_comprehensive_with_maturity_mismatch(self) -> None:
        """Collateral haircut + maturity mismatch adjustment."""
        raw_collateral = 800_000
        adjusted_collateral = maturity_mismatch_adjustment(
            raw_collateral, collateral_maturity=2.0, exposure_maturity=5.0,
        )
        result = comprehensive_approach(
            exposure=1_000_000,
            collateral_value=adjusted_collateral,
            collateral_type="cash",
        )
        expected_c = 800_000 * (1.75 / 4.75)
        expected_estar = 1_000_000 - expected_c
        assert result["adjusted_exposure"] == pytest.approx(expected_estar)

    def test_guarantee_better_than_obligor(self) -> None:
        """Guarantee from a sovereign (0% RW) on a corporate (100% RW)."""
        result = guarantee_substitution(
            exposure_rw=100.0, guarantor_rw=0.0, coverage_ratio=1.0,
        )
        assert result["effective_rw"] == pytest.approx(0.0)

    def test_haircut_table_completeness(self) -> None:
        """Every entry in HAIRCUT_TABLE yields a valid haircut."""
        for (ctype, cqs), buckets in HAIRCUT_TABLE.items():
            for bucket, value in buckets.items():
                assert isinstance(value, float)
                assert 0.0 <= value <= 1.0

    def test_constants(self) -> None:
        assert CURRENCY_MISMATCH_HAIRCUT == pytest.approx(0.08)
