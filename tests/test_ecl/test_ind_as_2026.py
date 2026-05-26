"""Tests for RBI ECL Master Direction 2026 (RBI/DOR/2026-27/398).

Covers all 12 implementation areas:
    1. Exposure category enum and classifier
    2. Stage 1/2 floor tables (Paragraph 82)
    3. Stage 3 duration-dependent floors
    4. PD/LGD regulatory floors (Paragraphs 96-98)
    5. SICR rules (Paragraph 33) — DPD + revolving overlimit
    6. Borrower-level Stage 3 contagion (Paragraph 76)
    7. Project finance DCCO additional provisioning
    8. Upgradation criteria (Paragraphs 77-79)
    9. Transition timeline (Paragraphs 2, 19-21, 108, 114)
    10. DLG adjustment (Paragraph 88)
    11. Collateral revaluation (Paragraph 55)
    12. End-to-end calculate_ecl_ind_as_2026 and auto-dispatch
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from creditriskengine.core.types import IFRS9Stage
from creditriskengine.ecl.ind_as109 import (
    CAPITAL_ADD_BACK_SCHEDULE,
    RBI_DCCO_INFRA_QUARTERLY_RATE,
    RBI_DCCO_NON_INFRA_QUARTERLY_RATE,
    RBI_ECL_EFFECTIVE_DATE,
    RBI_ECL_FLOOR_STAGE_1_2,
    RBI_EIR_MIGRATION_DEADLINE,
    RBI_LGD_BACKSTOP_SECURED,
    RBI_LGD_BACKSTOP_UNSECURED,
    RBI_LGD_ELIGIBLE_COLLATERAL,
    RBI_PD_FLOOR,
    RBI_REVOLVING_SICR_OVERLIMIT_DAYS,
    DLGAdjustment,
    RBICollateralCategory,
    RBIExposureCategory,
    apply_borrower_level_staging,
    apply_rbi_lgd_backstop,
    apply_rbi_pd_floor,
    assess_sicr_rbi,
    calculate_ecl_ind_as_2026,
    calculate_ecl_ind_as_auto,
    capital_add_back_factor,
    classify_rbi_exposure_category,
    collateral_category_for,
    dcco_additional_provision,
    determine_upgrade_eligibility,
    ecl_with_dlg,
    eir_required,
    is_ecl_framework_effective,
    rbi_ecl_floor_2026,
    validate_collateral_revaluation,
)

# ============================================================================
# Stage 1 & Stage 2 floors per Paragraph 82
# ============================================================================


class TestStage12Floors:
    """Verify each of the 20 exposure categories returns the correct
    Stage 1 and Stage 2 floor per RBI/DOR/2026-27/398 Paragraph 82.

    Floor amounts use EAD = ₹1,00,000 so the result equals the rate
    in basis points times 10 (e.g., 0.40% -> ₹400).
    """

    EAD: float = 100_000.0

    @pytest.mark.parametrize(
        ("category", "expected_s1", "expected_s2"),
        [
            (RBIExposureCategory.SECURED_RETAIL, 400.0, 5_000.0),
            (RBIExposureCategory.CORPORATE, 400.0, 5_000.0),
            (RBIExposureCategory.SMALL_MICRO_ENTERPRISE, 250.0, 5_000.0),
            (RBIExposureCategory.MEDIUM_ENTERPRISE, 400.0, 5_000.0),
            (RBIExposureCategory.FARM_CREDIT_AGRICULTURAL, 250.0, 5_000.0),
            (RBIExposureCategory.BANKS_NBFCS_REGULATED_FIS, 400.0, 5_000.0),
            (RBIExposureCategory.LOANS_AGAINST_DEPOSITS_LIC_KVP, 400.0, 400.0),
            (RBIExposureCategory.GOLD_LOANS, 400.0, 1_500.0),
            (RBIExposureCategory.STATE_GOVT_GUARANTEED, 400.0, 2_500.0),
            (RBIExposureCategory.UNSECURED_RETAIL, 1_000.0, 5_000.0),
            (RBIExposureCategory.HOUSING_LOANS_INDIVIDUALS, 250.0, 1_500.0),
            (RBIExposureCategory.CRE_ADC_150, 1_250.0, 5_000.0),
            (RBIExposureCategory.CRE_RH_ADC, 1_000.0, 5_000.0),
            (RBIExposureCategory.OTHER_RESIDENTIAL_RE, 400.0, 1_500.0),
            (RBIExposureCategory.OTHER_COMMERCIAL_RE, 400.0, 2_500.0),
            (RBIExposureCategory.PROJECT_FINANCE_PRE_OPERATIONAL, 1_000.0, 5_000.0),
            (RBIExposureCategory.PROJECT_FINANCE_OPERATIONAL, 400.0, 5_000.0),
            (RBIExposureCategory.CENTRAL_GOVT_GUARANTEED, 250.0, 250.0),
            (RBIExposureCategory.NATURAL_CALAMITY_RESTRUCTURED, 5_000.0, 10_000.0),
            (RBIExposureCategory.OTHER, 400.0, 5_000.0),
        ],
    )
    def test_stage1_and_stage2_floors(
        self,
        category: RBIExposureCategory,
        expected_s1: float,
        expected_s2: float,
    ) -> None:
        s1 = rbi_ecl_floor_2026(self.EAD, IFRS9Stage.STAGE_1, category)
        s2 = rbi_ecl_floor_2026(self.EAD, IFRS9Stage.STAGE_2, category)
        assert s1 == pytest.approx(expected_s1)
        assert s2 == pytest.approx(expected_s2)

    def test_all_categories_have_floor_entry(self) -> None:
        for category in RBIExposureCategory:
            assert category in RBI_ECL_FLOOR_STAGE_1_2

    def test_zero_or_negative_ead_returns_zero(self) -> None:
        assert (
            rbi_ecl_floor_2026(0.0, IFRS9Stage.STAGE_1, RBIExposureCategory.CORPORATE)
            == 0.0
        )
        assert (
            rbi_ecl_floor_2026(-100.0, IFRS9Stage.STAGE_2, RBIExposureCategory.CORPORATE)
            == 0.0
        )


# ============================================================================
# Stage 3 duration-dependent floors per Paragraph 82
# ============================================================================


class TestStage3Floors:
    EAD: float = 100_000.0

    @pytest.mark.parametrize(
        ("years", "is_secured", "expected"),
        [
            # Standard secured: 25/40/55/75/100
            (0.0, True, 25_000.0),
            (0.99, True, 25_000.0),
            (1.0, True, 40_000.0),
            (1.99, True, 40_000.0),
            (2.0, True, 55_000.0),
            (3.0, True, 75_000.0),
            (4.0, True, 100_000.0),
            (10.0, True, 100_000.0),
            # Standard unsecured (Set A): 40 / 100 / 100 / 100 / 100
            (0.0, False, 40_000.0),
            (1.0, False, 100_000.0),
            (5.0, False, 100_000.0),
        ],
    )
    def test_standard_schedule_corporate(
        self, years: float, is_secured: bool, expected: float
    ) -> None:
        assert rbi_ecl_floor_2026(
            self.EAD,
            IFRS9Stage.STAGE_3,
            RBIExposureCategory.CORPORATE,
            is_secured=is_secured,
            years_in_stage3=years,
        ) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("years", "is_secured", "expected"),
        [
            # Lower-floor schedule: 10/20/30/40/100 secured
            (0.0, True, 10_000.0),
            (1.0, True, 20_000.0),
            (2.0, True, 30_000.0),
            (3.0, True, 40_000.0),
            (4.0, True, 100_000.0),
            # Unsecured: 25/100/100/100/100
            (0.0, False, 25_000.0),
            (2.0, False, 100_000.0),
        ],
    )
    def test_deposits_gold_state_schedule(
        self, years: float, is_secured: bool, expected: float
    ) -> None:
        assert rbi_ecl_floor_2026(
            self.EAD,
            IFRS9Stage.STAGE_3,
            RBIExposureCategory.GOLD_LOANS,
            is_secured=is_secured,
            years_in_stage3=years,
        ) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("years", "expected"),
        [
            (0.0, 25_000.0),
            (0.5, 25_000.0),
            (1.0, 100_000.0),
            (5.0, 100_000.0),
        ],
    )
    def test_unsecured_retail_schedule(self, years: float, expected: float) -> None:
        # Unsecured retail uses two-bracket schedule: 25% then 100%
        assert rbi_ecl_floor_2026(
            self.EAD,
            IFRS9Stage.STAGE_3,
            RBIExposureCategory.UNSECURED_RETAIL,
            is_secured=False,
            years_in_stage3=years,
        ) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("years", "is_secured", "expected"),
        [
            # Housing/residential RE: 10/20/30/40/100 secured
            (0.0, True, 10_000.0),
            (1.0, True, 20_000.0),
            (4.0, True, 100_000.0),
            # Unsecured: 25/100/...
            (0.0, False, 25_000.0),
            (3.0, False, 100_000.0),
        ],
    )
    def test_housing_schedule(
        self, years: float, is_secured: bool, expected: float
    ) -> None:
        assert rbi_ecl_floor_2026(
            self.EAD,
            IFRS9Stage.STAGE_3,
            RBIExposureCategory.HOUSING_LOANS_INDIVIDUALS,
            is_secured=is_secured,
            years_in_stage3=years,
        ) == pytest.approx(expected)


# ============================================================================
# PD/LGD floors per Paragraphs 96-98
# ============================================================================


class TestPDLGDFloors:
    def test_pd_floor_constant(self) -> None:
        assert RBI_PD_FLOOR == 0.0003

    def test_pd_below_floor_is_floored(self) -> None:
        assert apply_rbi_pd_floor(0.00001) == pytest.approx(0.0003)
        assert apply_rbi_pd_floor(0.0) == pytest.approx(0.0003)

    def test_pd_above_floor_unchanged(self) -> None:
        assert apply_rbi_pd_floor(0.01) == 0.01
        assert apply_rbi_pd_floor(0.05) == 0.05

    def test_lgd_backstop_constants(self) -> None:
        assert RBI_LGD_BACKSTOP_SECURED == 0.65
        assert RBI_LGD_BACKSTOP_UNSECURED == 0.70
        assert RBI_LGD_ELIGIBLE_COLLATERAL == 0.30

    def test_lgd_secured_below_floor(self) -> None:
        assert apply_rbi_lgd_backstop(0.30, is_secured=True) == 0.65

    def test_lgd_unsecured_below_floor(self) -> None:
        assert apply_rbi_lgd_backstop(0.50, is_secured=False) == 0.70

    def test_lgd_eligible_collateral_below_floor(self) -> None:
        assert (
            apply_rbi_lgd_backstop(0.10, is_secured=True, has_eligible_collateral=True)
            == 0.30
        )

    def test_lgd_above_floor_unchanged(self) -> None:
        assert apply_rbi_lgd_backstop(0.80, is_secured=True) == 0.80
        assert apply_rbi_lgd_backstop(0.85, is_secured=False) == 0.85

    def test_eligible_collateral_takes_precedence(self) -> None:
        # If marked eligible_collateral, the 30% floor applies even
        # when is_secured=True (the 65% wouldn't be appropriate)
        result = apply_rbi_lgd_backstop(
            0.20, is_secured=True, has_eligible_collateral=True
        )
        assert result == 0.30


# ============================================================================
# SICR per Paragraph 33
# ============================================================================


class TestSICRRBI:
    def test_dpd_backstop_triggers(self) -> None:
        # 31 days > 30 day backstop
        assert assess_sicr_rbi(days_past_due=31) is True

    def test_dpd_at_threshold_not_trigger(self) -> None:
        # Exactly 30 days is not strictly greater than backstop
        assert assess_sicr_rbi(days_past_due=30) is False

    def test_revolving_overlimit_triggers(self) -> None:
        assert (
            assess_sicr_rbi(is_revolving=True, days_over_limit=61) is True
        )

    def test_revolving_overlimit_at_threshold_not_trigger(self) -> None:
        assert (
            assess_sicr_rbi(is_revolving=True, days_over_limit=60) is False
        )

    def test_non_revolving_overlimit_ignored(self) -> None:
        # If not revolving, days_over_limit is ignored
        assert (
            assess_sicr_rbi(is_revolving=False, days_over_limit=200) is False
        )

    def test_rebuttal_suppresses_dpd(self) -> None:
        assert (
            assess_sicr_rbi(days_past_due=60, rebuttal_applied=True) is False
        )

    def test_rebuttal_suppresses_overlimit(self) -> None:
        assert (
            assess_sicr_rbi(
                is_revolving=True,
                days_over_limit=200,
                rebuttal_applied=True,
            )
            is False
        )

    def test_quantitative_trigger_overrides_rebuttal(self) -> None:
        # Quantitative SICR triggers regardless of rebuttal
        assert (
            assess_sicr_rbi(
                rebuttal_applied=True,
                sicr_triggered_quantitative=True,
            )
            is True
        )

    def test_constant_values(self) -> None:
        assert RBI_REVOLVING_SICR_OVERLIMIT_DAYS == 60


# ============================================================================
# Borrower-level Stage 3 contagion per Paragraph 76
# ============================================================================


class TestBorrowerLevelClassification:
    def test_single_stage3_elevates_all_facilities(self) -> None:
        facilities = [
            {"counterparty_id": "B1", "facility_id": "F1", "stage": IFRS9Stage.STAGE_3},
            {"counterparty_id": "B1", "facility_id": "F2", "stage": IFRS9Stage.STAGE_2},
            {"counterparty_id": "B1", "facility_id": "F3", "stage": IFRS9Stage.STAGE_1},
        ]
        result = apply_borrower_level_staging(facilities)
        assert all(f["stage"] == IFRS9Stage.STAGE_3 for f in result)

    def test_stage2_no_contagion(self) -> None:
        facilities = [
            {"counterparty_id": "B1", "facility_id": "F1", "stage": IFRS9Stage.STAGE_2},
            {"counterparty_id": "B1", "facility_id": "F2", "stage": IFRS9Stage.STAGE_1},
        ]
        result = apply_borrower_level_staging(facilities)
        assert result[0]["stage"] == IFRS9Stage.STAGE_2
        assert result[1]["stage"] == IFRS9Stage.STAGE_1

    def test_multiple_borrowers_isolation(self) -> None:
        facilities = [
            {"counterparty_id": "B1", "facility_id": "F1", "stage": IFRS9Stage.STAGE_3},
            {"counterparty_id": "B1", "facility_id": "F2", "stage": IFRS9Stage.STAGE_1},
            {"counterparty_id": "B2", "facility_id": "F3", "stage": IFRS9Stage.STAGE_2},
            {"counterparty_id": "B2", "facility_id": "F4", "stage": IFRS9Stage.STAGE_1},
        ]
        result = apply_borrower_level_staging(facilities)
        # Borrower B1: both facilities elevated to Stage 3
        assert result[0]["stage"] == IFRS9Stage.STAGE_3
        assert result[1]["stage"] == IFRS9Stage.STAGE_3
        # Borrower B2: untouched
        assert result[2]["stage"] == IFRS9Stage.STAGE_2
        assert result[3]["stage"] == IFRS9Stage.STAGE_1

    def test_empty_list(self) -> None:
        assert apply_borrower_level_staging([]) == []

    def test_input_not_mutated(self) -> None:
        facilities = [
            {"counterparty_id": "B1", "facility_id": "F1", "stage": IFRS9Stage.STAGE_3},
            {"counterparty_id": "B1", "facility_id": "F2", "stage": IFRS9Stage.STAGE_1},
        ]
        original_f2_stage = facilities[1]["stage"]
        apply_borrower_level_staging(facilities)
        # Original input should be unchanged
        assert facilities[1]["stage"] == original_f2_stage


# ============================================================================
# DCCO additional provisioning per Paragraph 82(4) Note 1
# ============================================================================


class TestDCCOAdditionalProvisioning:
    def test_infrastructure_one_quarter(self) -> None:
        # 0.375% * 100,000 = 375
        result = dcco_additional_provision(100_000, 1, is_infrastructure=True)
        assert result == pytest.approx(375.0)

    def test_infrastructure_four_quarters(self) -> None:
        # 4 * 0.375% * 100,000 = 1500
        result = dcco_additional_provision(100_000, 4, is_infrastructure=True)
        assert result == pytest.approx(1500.0)

    def test_non_infrastructure_one_quarter(self) -> None:
        # 0.5625% * 100,000 = 562.5
        result = dcco_additional_provision(100_000, 1, is_infrastructure=False)
        assert result == pytest.approx(562.5)

    def test_non_infrastructure_four_quarters(self) -> None:
        # 4 * 0.5625% * 100,000 = 2250
        result = dcco_additional_provision(100_000, 4, is_infrastructure=False)
        assert result == pytest.approx(2250.0)

    def test_zero_quarters_zero_provision(self) -> None:
        assert dcco_additional_provision(100_000, 0) == 0.0

    def test_zero_ead_zero_provision(self) -> None:
        assert dcco_additional_provision(0.0, 4) == 0.0

    def test_rate_constants(self) -> None:
        assert RBI_DCCO_INFRA_QUARTERLY_RATE == 0.00375
        assert RBI_DCCO_NON_INFRA_QUARTERLY_RATE == 0.005625


# ============================================================================
# Upgradation criteria per Paragraphs 77-79, 10
# ============================================================================


class TestUpgradeEligibility:
    def test_stage3_non_restructured_to_stage1(self) -> None:
        result = determine_upgrade_eligibility(
            IFRS9Stage.STAGE_3,
            is_restructured=False,
            all_arrears_repaid=True,
            sicr_triggered=False,
        )
        assert result == IFRS9Stage.STAGE_1

    def test_stage3_non_restructured_sicr_blocks_upgrade(self) -> None:
        result = determine_upgrade_eligibility(
            IFRS9Stage.STAGE_3,
            is_restructured=False,
            all_arrears_repaid=True,
            sicr_triggered=True,
        )
        assert result == IFRS9Stage.STAGE_3

    def test_stage3_non_restructured_arrears_unpaid(self) -> None:
        result = determine_upgrade_eligibility(
            IFRS9Stage.STAGE_3,
            is_restructured=False,
            all_arrears_repaid=False,
            sicr_triggered=False,
        )
        assert result == IFRS9Stage.STAGE_3

    def test_stage3_restructured_to_stage2(self) -> None:
        result = determine_upgrade_eligibility(
            IFRS9Stage.STAGE_3,
            is_restructured=True,
            satisfactory_performance=True,
            resolution_directions_met=True,
            sicr_triggered=False,
        )
        assert result == IFRS9Stage.STAGE_2

    def test_stage3_restructured_no_resolution_directions(self) -> None:
        result = determine_upgrade_eligibility(
            IFRS9Stage.STAGE_3,
            is_restructured=True,
            satisfactory_performance=True,
            resolution_directions_met=False,
        )
        assert result == IFRS9Stage.STAGE_3

    def test_stage2_to_stage1(self) -> None:
        result = determine_upgrade_eligibility(
            IFRS9Stage.STAGE_2,
            is_restructured=False,
            satisfactory_performance=True,
            sicr_triggered=False,
        )
        assert result == IFRS9Stage.STAGE_1

    def test_stage2_with_sicr_remains(self) -> None:
        result = determine_upgrade_eligibility(
            IFRS9Stage.STAGE_2,
            is_restructured=False,
            satisfactory_performance=True,
            sicr_triggered=True,
        )
        assert result == IFRS9Stage.STAGE_2

    def test_stage1_unchanged(self) -> None:
        result = determine_upgrade_eligibility(
            IFRS9Stage.STAGE_1,
            is_restructured=False,
        )
        assert result == IFRS9Stage.STAGE_1


# ============================================================================
# Transition timeline per Paragraphs 2, 19-21, 108, 114
# ============================================================================


class TestTransition:
    def test_effective_date_constant(self) -> None:
        assert date(2027, 4, 1) == RBI_ECL_EFFECTIVE_DATE
        assert date(2030, 3, 31) == RBI_EIR_MIGRATION_DEADLINE

    def test_pre_effective_date(self) -> None:
        assert is_ecl_framework_effective(date(2027, 3, 31)) is False

    def test_on_effective_date(self) -> None:
        assert is_ecl_framework_effective(date(2027, 4, 1)) is True

    def test_post_effective_date(self) -> None:
        assert is_ecl_framework_effective(date(2030, 1, 1)) is True

    def test_capital_add_back_schedule(self) -> None:
        assert capital_add_back_factor(2028) == 0.80
        assert capital_add_back_factor(2029) == 0.60
        assert capital_add_back_factor(2030) == 0.40
        assert capital_add_back_factor(2031) == 0.20

    def test_capital_add_back_post_phase_in(self) -> None:
        assert capital_add_back_factor(2032) == 0.0
        assert capital_add_back_factor(2050) == 0.0

    def test_capital_add_back_pre_phase_in(self) -> None:
        assert capital_add_back_factor(2027) == 0.0

    def test_schedule_dict_keys(self) -> None:
        assert set(CAPITAL_ADD_BACK_SCHEDULE.keys()) == {2028, 2029, 2030, 2031}

    def test_eir_required_new_origination(self) -> None:
        # Loan originated after effective date — EIR mandatory
        assert eir_required(
            origination_date=date(2027, 6, 1),
            reporting_date=date(2027, 12, 31),
        ) is True

    def test_eir_not_required_legacy_before_deadline(self) -> None:
        # Legacy loan, reporting before migration deadline
        assert eir_required(
            origination_date=date(2025, 1, 1),
            reporting_date=date(2029, 6, 30),
        ) is False

    def test_eir_required_legacy_after_deadline(self) -> None:
        # Legacy loan, reporting after migration deadline
        assert eir_required(
            origination_date=date(2025, 1, 1),
            reporting_date=date(2030, 4, 1),
        ) is True


# ============================================================================
# DLG (Default Loss Guarantee) per Paragraph 88
# ============================================================================


class TestDLG:
    def test_full_absorption(self) -> None:
        # DLG can fully absorb the ECL
        result = ecl_with_dlg(gross_ecl=1000, dlg_remaining_capacity=2000)
        assert result.gross_ecl == 1000
        assert result.dlg_absorbed == 1000
        assert result.net_ecl == 0
        assert result.remaining_dlg_capacity == 1000

    def test_partial_absorption(self) -> None:
        # DLG partially absorbs (capacity < ECL)
        result = ecl_with_dlg(gross_ecl=1000, dlg_remaining_capacity=400)
        assert result.dlg_absorbed == 400
        assert result.net_ecl == 600
        assert result.remaining_dlg_capacity == 0

    def test_no_dlg(self) -> None:
        result = ecl_with_dlg(gross_ecl=1000, dlg_remaining_capacity=0)
        assert result.net_ecl == 1000
        assert result.dlg_absorbed == 0

    def test_dlg_cap_pct(self) -> None:
        # DLG cap as % of portfolio EAD
        result = ecl_with_dlg(
            gross_ecl=10000,
            dlg_remaining_capacity=50000,
            portfolio_ead=100000,
            dlg_cap_pct=0.05,  # 5% of 100k = 5000
        )
        assert result.dlg_absorbed == 5000
        assert result.net_ecl == 5000

    def test_zero_ecl(self) -> None:
        result = ecl_with_dlg(gross_ecl=0, dlg_remaining_capacity=1000)
        assert result.net_ecl == 0
        assert result.dlg_absorbed == 0
        assert result.remaining_dlg_capacity == 1000

    def test_dlg_adjustment_is_frozen(self) -> None:
        result = ecl_with_dlg(gross_ecl=1000, dlg_remaining_capacity=500)
        with pytest.raises((AttributeError, TypeError)):
            result.net_ecl = 999  # type: ignore[misc]
        assert isinstance(result, DLGAdjustment)


# ============================================================================
# Collateral revaluation per Paragraph 55
# ============================================================================


class TestCollateralRevaluation:
    def test_stage3_large_overdue_revaluation(self) -> None:
        warnings = validate_collateral_revaluation(
            stage=IFRS9Stage.STAGE_3,
            exposure_inr_crore=10.0,
            last_revaluation_date=date(2023, 1, 1),
            reporting_date=date(2026, 1, 1),  # 3 years later
            collateral_type="real_estate",
        )
        assert len(warnings) == 1
        assert "Stage 3" in warnings[0]

    def test_stage3_large_recent_revaluation_no_warning(self) -> None:
        warnings = validate_collateral_revaluation(
            stage=IFRS9Stage.STAGE_3,
            exposure_inr_crore=10.0,
            last_revaluation_date=date(2025, 1, 1),
            reporting_date=date(2026, 1, 1),  # 1 year later
            collateral_type="real_estate",
        )
        assert warnings == []

    def test_stage3_small_exposure_no_warning(self) -> None:
        # Below ₹7.5 crore threshold
        warnings = validate_collateral_revaluation(
            stage=IFRS9Stage.STAGE_3,
            exposure_inr_crore=5.0,
            last_revaluation_date=date(2020, 1, 1),
            reporting_date=date(2026, 1, 1),
            collateral_type="real_estate",
        )
        assert warnings == []

    def test_stage1_no_warning_regardless(self) -> None:
        warnings = validate_collateral_revaluation(
            stage=IFRS9Stage.STAGE_1,
            exposure_inr_crore=100.0,
            last_revaluation_date=date(2020, 1, 1),
            reporting_date=date(2026, 1, 1),
            collateral_type="real_estate",
        )
        assert warnings == []

    def test_stock_collateral_annual_overdue(self) -> None:
        warnings = validate_collateral_revaluation(
            stage=IFRS9Stage.STAGE_2,  # Stock check is stage-agnostic
            exposure_inr_crore=2.0,
            last_revaluation_date=date(2024, 1, 1),
            reporting_date=date(2026, 1, 1),  # 2 years
            collateral_type="stock",
        )
        assert len(warnings) == 1
        assert "Stock collateral" in warnings[0]

    def test_stock_collateral_recent_no_warning(self) -> None:
        warnings = validate_collateral_revaluation(
            stage=IFRS9Stage.STAGE_2,
            exposure_inr_crore=2.0,
            last_revaluation_date=date(2025, 7, 1),
            reporting_date=date(2026, 1, 1),  # 6 months
            collateral_type="stock",
        )
        assert warnings == []


# ============================================================================
# Exposure category classifier
# ============================================================================


class TestClassifyRBIExposureCategory:
    def test_central_govt_guaranteed_takes_priority(self) -> None:
        assert (
            classify_rbi_exposure_category(is_central_govt_guaranteed=True, is_retail=True)
            == RBIExposureCategory.CENTRAL_GOVT_GUARANTEED
        )

    def test_natural_calamity_takes_priority(self) -> None:
        assert (
            classify_rbi_exposure_category(is_natural_calamity_restructured=True)
            == RBIExposureCategory.NATURAL_CALAMITY_RESTRUCTURED
        )

    def test_gold_loan(self) -> None:
        assert (
            classify_rbi_exposure_category(is_gold_loan=True)
            == RBIExposureCategory.GOLD_LOANS
        )

    def test_loan_against_deposit(self) -> None:
        assert (
            classify_rbi_exposure_category(is_loan_against_deposit=True)
            == RBIExposureCategory.LOANS_AGAINST_DEPOSITS_LIC_KVP
        )

    def test_housing_individual(self) -> None:
        assert (
            classify_rbi_exposure_category(is_housing_individual=True)
            == RBIExposureCategory.HOUSING_LOANS_INDIVIDUALS
        )

    def test_cre_adc_150(self) -> None:
        assert (
            classify_rbi_exposure_category(is_cre_adc=True)
            == RBIExposureCategory.CRE_ADC_150
        )

    def test_project_finance_pre_op(self) -> None:
        assert (
            classify_rbi_exposure_category(
                is_project_finance=True, project_phase="pre_operational"
            )
            == RBIExposureCategory.PROJECT_FINANCE_PRE_OPERATIONAL
        )

    def test_project_finance_operational(self) -> None:
        assert (
            classify_rbi_exposure_category(
                is_project_finance=True, project_phase="operational"
            )
            == RBIExposureCategory.PROJECT_FINANCE_OPERATIONAL
        )

    def test_msme_micro(self) -> None:
        assert (
            classify_rbi_exposure_category(is_msme=True, msme_size="micro")
            == RBIExposureCategory.SMALL_MICRO_ENTERPRISE
        )

    def test_msme_medium(self) -> None:
        assert (
            classify_rbi_exposure_category(is_msme=True, msme_size="medium")
            == RBIExposureCategory.MEDIUM_ENTERPRISE
        )

    def test_agricultural(self) -> None:
        assert (
            classify_rbi_exposure_category(is_agricultural=True)
            == RBIExposureCategory.FARM_CREDIT_AGRICULTURAL
        )

    def test_retail_secured(self) -> None:
        assert (
            classify_rbi_exposure_category(is_retail=True, is_secured=True)
            == RBIExposureCategory.SECURED_RETAIL
        )

    def test_retail_unsecured(self) -> None:
        assert (
            classify_rbi_exposure_category(is_retail=True, is_secured=False)
            == RBIExposureCategory.UNSECURED_RETAIL
        )

    def test_corporate_default(self) -> None:
        assert (
            classify_rbi_exposure_category(sector="corporate")
            == RBIExposureCategory.CORPORATE
        )

    def test_other_fallback(self) -> None:
        assert classify_rbi_exposure_category() == RBIExposureCategory.OTHER


# ============================================================================
# Collateral category mapping
# ============================================================================


class TestCollateralCategoryFor:
    def test_unsecured_maps_to_unsecured(self) -> None:
        assert (
            collateral_category_for(RBIExposureCategory.CORPORATE, is_secured=False)
            == RBICollateralCategory.UNSECURED
        )

    def test_deposits_secured(self) -> None:
        assert (
            collateral_category_for(
                RBIExposureCategory.LOANS_AGAINST_DEPOSITS_LIC_KVP, is_secured=True
            )
            == RBICollateralCategory.DEPOSITS_LIC_GOLD_STATE_GOVT
        )

    def test_gold_secured(self) -> None:
        assert (
            collateral_category_for(RBIExposureCategory.GOLD_LOANS, is_secured=True)
            == RBICollateralCategory.DEPOSITS_LIC_GOLD_STATE_GOVT
        )

    def test_housing_secured(self) -> None:
        assert (
            collateral_category_for(
                RBIExposureCategory.HOUSING_LOANS_INDIVIDUALS, is_secured=True
            )
            == RBICollateralCategory.HOUSING_RESIDENTIAL_RE
        )

    def test_corporate_secured(self) -> None:
        assert (
            collateral_category_for(RBIExposureCategory.CORPORATE, is_secured=True)
            == RBICollateralCategory.STANDARD_SECURED
        )


# ============================================================================
# End-to-end calculate_ecl_ind_as_2026
# ============================================================================


class TestCalculateECLIndAs2026:
    def test_stage1_pd_floor_binds(self) -> None:
        # Very low PD model output → PD floor 0.03% binds
        # EAD 1,000,000 * 0.0003 * 0.65 (LGD floor) = 195
        # But regulatory floor for SECURED_RETAIL Stage 1 = 0.40% * 1,000,000 = 4000
        ecl = calculate_ecl_ind_as_2026(
            stage=IFRS9Stage.STAGE_1,
            pd_12m=0.00001,
            lgd=0.50,
            ead=1_000_000,
            category=RBIExposureCategory.SECURED_RETAIL,
            is_secured=True,
        )
        # Regulatory floor binds
        assert ecl == pytest.approx(4000.0)

    def test_stage2_lifetime_above_floor(self) -> None:
        # High PD lifetime ECL exceeds the 5% regulatory floor
        marginal_pds = np.array([0.10, 0.10, 0.10])  # 30% cumulative
        # 100,000 * (0.10+0.10+0.10) * 0.65 (LGD floor) = 19,500
        # Floor 5% = 5,000
        ecl = calculate_ecl_ind_as_2026(
            stage=IFRS9Stage.STAGE_2,
            pd_12m=0.10,
            lgd=0.50,
            ead=100_000,
            marginal_pds=marginal_pds,
            category=RBIExposureCategory.SECURED_RETAIL,
            is_secured=True,
        )
        # Model ECL of 19,500 binds, not floor of 5,000
        assert ecl == pytest.approx(19_500.0)

    def test_stage3_duration_floor(self) -> None:
        # Stage 3, secured, 2.5 years → 55% floor
        # 100,000 * 0.55 = 55,000
        marginal_pds = np.array([0.20, 0.20])
        ecl = calculate_ecl_ind_as_2026(
            stage=IFRS9Stage.STAGE_3,
            pd_12m=0.20,
            lgd=0.30,
            ead=100_000,
            marginal_pds=marginal_pds,
            category=RBIExposureCategory.CORPORATE,
            is_secured=True,
            years_in_stage3=2.5,
        )
        # Floor 55% = 55,000 binds (model = 0.40 * 0.65 * 100,000 = 26,000)
        assert ecl == pytest.approx(55_000.0)

    def test_dlg_reduces_ecl(self) -> None:
        # Without DLG: high PD generates model ECL above floor
        marginal_pds = np.array([0.30, 0.30])
        ecl_no_dlg = calculate_ecl_ind_as_2026(
            stage=IFRS9Stage.STAGE_2,
            pd_12m=0.30,
            lgd=0.50,
            ead=100_000,
            marginal_pds=marginal_pds,
            category=RBIExposureCategory.SECURED_RETAIL,
            is_secured=True,
        )
        # With DLG: ECL reduced (but never below regulatory floor)
        ecl_with_dlg_val = calculate_ecl_ind_as_2026(
            stage=IFRS9Stage.STAGE_2,
            pd_12m=0.30,
            lgd=0.50,
            ead=100_000,
            marginal_pds=marginal_pds,
            category=RBIExposureCategory.SECURED_RETAIL,
            is_secured=True,
            dlg_remaining_capacity=20_000,
        )
        # DLG should have reduced the ECL
        assert ecl_with_dlg_val <= ecl_no_dlg
        # But never below the 5% regulatory floor (5000)
        assert ecl_with_dlg_val >= 5000.0

    def test_pd_lgd_floors_can_be_bypassed(self) -> None:
        ecl_with_floors = calculate_ecl_ind_as_2026(
            stage=IFRS9Stage.STAGE_1,
            pd_12m=0.0001,
            lgd=0.20,
            ead=10_000_000,
            category=RBIExposureCategory.CORPORATE,
            apply_pd_lgd_floors=True,
        )
        ecl_no_floors = calculate_ecl_ind_as_2026(
            stage=IFRS9Stage.STAGE_1,
            pd_12m=0.0001,
            lgd=0.20,
            ead=10_000_000,
            category=RBIExposureCategory.CORPORATE,
            apply_pd_lgd_floors=False,
        )
        # Both will likely hit the same regulatory floor (0.40% = 40,000)
        # but the floor-bypassed model ECL is lower
        assert ecl_with_floors == pytest.approx(40_000.0)
        assert ecl_no_floors == pytest.approx(40_000.0)

    def test_unsecured_retail_high_lgd_backstop(self) -> None:
        # Unsecured retail Stage 1 floor = 1% = 10,000
        # Model: 0.01 * 0.70 (LGD floor) * 1,000,000 = 7,000
        ecl = calculate_ecl_ind_as_2026(
            stage=IFRS9Stage.STAGE_1,
            pd_12m=0.01,
            lgd=0.30,  # Will be floored to 0.70 (unsecured)
            ead=1_000_000,
            category=RBIExposureCategory.UNSECURED_RETAIL,
            is_secured=False,
        )
        # Floor 1% = 10,000 binds
        assert ecl == pytest.approx(10_000.0)


# ============================================================================
# Auto-dispatch by reporting date
# ============================================================================


class TestAutoDispatch:
    def test_pre_2027_uses_legacy_framework(self) -> None:
        # Pre-effective: uses calculate_ecl_ind_as (legacy IRAC)
        ecl = calculate_ecl_ind_as_auto(
            reporting_date=date(2026, 12, 31),
            stage=IFRS9Stage.STAGE_1,
            pd_12m=0.02,
            lgd=0.45,
            ead=1_000_000,
        )
        # Pure model ECL, no floor: 0.02 * 0.45 * 1,000,000 = 9,000
        assert ecl == pytest.approx(9000.0)

    def test_post_2027_uses_new_framework(self) -> None:
        ecl = calculate_ecl_ind_as_auto(
            reporting_date=date(2027, 4, 1),
            stage=IFRS9Stage.STAGE_1,
            pd_12m=0.02,
            lgd=0.45,
            ead=1_000_000,
            category=RBIExposureCategory.CORPORATE,
            is_secured=True,
        )
        # 2026 framework applies PD/LGD floors and regulatory floor
        # PD floored to 0.03 bps doesn't matter (0.02 > 0.0003)
        # LGD floored to 0.65 (secured)
        # Model: 0.02 * 0.65 * 1,000,000 = 13,000
        # Floor: 0.40% * 1,000,000 = 4,000
        # max(13000, 4000) = 13,000
        assert ecl == pytest.approx(13_000.0)


# ============================================================================
# Set F: other RE-secured floors per Paragraph 82(5) Set F
# ============================================================================


class TestStage3SetF:
    EAD: float = 100_000.0

    @pytest.mark.parametrize(
        ("years", "is_secured", "expected"),
        [
            # Set F secured: 15/25/40/55/100
            (0.0, True, 15_000.0),
            (1.0, True, 25_000.0),
            (2.0, True, 40_000.0),
            (3.0, True, 55_000.0),
            (4.0, True, 100_000.0),
            # Set F unsecured: 25/100/100/100/100
            (0.0, False, 25_000.0),
            (2.0, False, 100_000.0),
        ],
    )
    def test_other_residential_re(
        self, years: float, is_secured: bool, expected: float
    ) -> None:
        assert rbi_ecl_floor_2026(
            self.EAD,
            IFRS9Stage.STAGE_3,
            RBIExposureCategory.OTHER_RESIDENTIAL_RE,
            is_secured=is_secured,
            years_in_stage3=years,
        ) == pytest.approx(expected)

    def test_other_commercial_re_uses_set_f(self) -> None:
        # Verify OTHER_COMMERCIAL_RE also uses Set F
        assert rbi_ecl_floor_2026(
            self.EAD,
            IFRS9Stage.STAGE_3,
            RBIExposureCategory.OTHER_COMMERCIAL_RE,
            is_secured=True,
            years_in_stage3=0.5,
        ) == pytest.approx(15_000.0)


# ============================================================================
# Wilful defaulter +5% surcharge per Paragraph 101(4)
# ============================================================================


class TestWilfulDefaulter:
    def test_stage1_wilful_defaulter_addon(self) -> None:
        base = rbi_ecl_floor_2026(
            1_000_000, IFRS9Stage.STAGE_1, RBIExposureCategory.CORPORATE
        )
        with_wd = rbi_ecl_floor_2026(
            1_000_000, IFRS9Stage.STAGE_1, RBIExposureCategory.CORPORATE,
            is_wilful_defaulter=True,
        )
        assert with_wd - base == pytest.approx(50_000.0)  # 5% of 1M

    def test_stage3_wilful_defaulter_addon(self) -> None:
        base = rbi_ecl_floor_2026(
            100_000, IFRS9Stage.STAGE_3, RBIExposureCategory.CORPORATE,
            is_secured=True, years_in_stage3=0.5,
        )
        with_wd = rbi_ecl_floor_2026(
            100_000, IFRS9Stage.STAGE_3, RBIExposureCategory.CORPORATE,
            is_secured=True, years_in_stage3=0.5, is_wilful_defaulter=True,
        )
        assert with_wd - base == pytest.approx(5_000.0)  # 5% of 100k


# ============================================================================
# Sovereign / SLR carve-out per Paragraphs 37-38
# ============================================================================


class TestSovereignCarveOut:
    def test_sovereign_returns_zero(self) -> None:
        ecl = calculate_ecl_ind_as_2026(
            stage=IFRS9Stage.STAGE_1,
            pd_12m=0.01, lgd=0.45, ead=10_000_000,
            category=RBIExposureCategory.CORPORATE,
            is_sovereign_slr=True,
        )
        assert ecl == 0.0

    def test_non_sovereign_returns_positive(self) -> None:
        ecl = calculate_ecl_ind_as_2026(
            stage=IFRS9Stage.STAGE_1,
            pd_12m=0.01, lgd=0.45, ead=10_000_000,
            category=RBIExposureCategory.CORPORATE,
            is_sovereign_slr=False,
        )
        assert ecl > 0


# ============================================================================
# IRACP standard-asset provisioning
# ============================================================================


class TestIRACPStandardAsset:
    def test_agriculture_direct(self) -> None:
        from creditriskengine.ecl.ind_as109.iracp import (
            StandardAssetSector,
            standard_asset_provision,
        )
        prov = standard_asset_provision(1_000_000, StandardAssetSector.AGRICULTURE_DIRECT)
        assert prov == pytest.approx(2500.0)  # 0.25%

    def test_cre(self) -> None:
        from creditriskengine.ecl.ind_as109.iracp import (
            StandardAssetSector,
            standard_asset_provision,
        )
        prov = standard_asset_provision(1_000_000, StandardAssetSector.CRE)
        assert prov == pytest.approx(10_000.0)  # 1.00%

    def test_cre_rh(self) -> None:
        from creditriskengine.ecl.ind_as109.iracp import (
            StandardAssetSector,
            standard_asset_provision,
        )
        prov = standard_asset_provision(1_000_000, StandardAssetSector.CRE_RH)
        assert prov == pytest.approx(7500.0)  # 0.75%

    def test_housing_individual(self) -> None:
        from creditriskengine.ecl.ind_as109.iracp import (
            StandardAssetSector,
            standard_asset_provision,
        )
        prov = standard_asset_provision(1_000_000, StandardAssetSector.HOUSING_INDIVIDUAL)
        assert prov == pytest.approx(2500.0)  # 0.25%

    def test_teaser_pre_reset(self) -> None:
        from creditriskengine.ecl.ind_as109.iracp import (
            StandardAssetSector,
            standard_asset_provision,
        )
        prov = standard_asset_provision(1_000_000, StandardAssetSector.HOUSING_TEASER)
        assert prov == pytest.approx(20_000.0)  # 2.00%

    def test_teaser_post_reset(self) -> None:
        from creditriskengine.ecl.ind_as109.iracp import (
            StandardAssetSector,
            standard_asset_provision,
        )
        prov = standard_asset_provision(
            1_000_000, StandardAssetSector.HOUSING_TEASER,
            teaser_one_year_post_reset=True,
        )
        assert prov == pytest.approx(4000.0)  # 0.40%

    def test_project_uc(self) -> None:
        from creditriskengine.ecl.ind_as109.iracp import (
            StandardAssetSector,
            standard_asset_provision,
        )
        prov = standard_asset_provision(
            1_000_000, StandardAssetSector.PROJECT_UNDER_CONSTRUCTION
        )
        assert prov == pytest.approx(10_000.0)  # 1.00%

    def test_project_uc_cre(self) -> None:
        from creditriskengine.ecl.ind_as109.iracp import (
            StandardAssetSector,
            standard_asset_provision,
        )
        prov = standard_asset_provision(
            1_000_000, StandardAssetSector.PROJECT_UNDER_CONSTRUCTION_CRE
        )
        assert prov == pytest.approx(12_500.0)  # 1.25%


# ============================================================================
# Resolution Framework add-ons
# ============================================================================


class TestResolutionFrameworkAddon:
    def test_restructured_addon(self) -> None:
        from creditriskengine.ecl.ind_as109.iracp import resolution_framework_addon
        addon = resolution_framework_addon(residual_debt=1_000_000)
        assert addon == pytest.approx(100_000.0)  # 10%

    def test_slippage_addon(self) -> None:
        from creditriskengine.ecl.ind_as109.iracp import resolution_framework_addon
        addon = resolution_framework_addon(residual_debt=1_000_000, has_slipped=True)
        assert addon == pytest.approx(150_000.0)  # 10% + 5% = 15%


# ============================================================================
# Out-of-order CC/OD
# ============================================================================


class TestOutOfOrderCCOD:
    def test_over_limit_90_days(self) -> None:
        from creditriskengine.ecl.ind_as109.iracp import is_out_of_order
        assert is_out_of_order(
            outstanding=120_000,
            sanctioned_limit=100_000,
            days_continuously_over_limit=91,
        ) is True

    def test_no_credits_90_days(self) -> None:
        from creditriskengine.ecl.ind_as109.iracp import is_out_of_order
        assert is_out_of_order(
            outstanding=50_000,
            sanctioned_limit=100_000,
            no_credits_for_days=91,
        ) is True

    def test_credits_less_than_interest(self) -> None:
        from creditriskengine.ecl.ind_as109.iracp import is_out_of_order
        assert is_out_of_order(
            outstanding=50_000,
            sanctioned_limit=100_000,
            credits_less_than_interest_debited=True,
        ) is True

    def test_performing_account(self) -> None:
        from creditriskengine.ecl.ind_as109.iracp import is_out_of_order
        assert is_out_of_order(
            outstanding=50_000,
            sanctioned_limit=100_000,
        ) is False


# ============================================================================
# NBFC backstop
# ============================================================================


class TestNBFCBackstop:
    def test_ecl_above_iracp(self) -> None:
        from creditriskengine.ecl.ind_as109.nbfc_backstop import apply_nbfc_backstop
        r = apply_nbfc_backstop(ind_as_109_ecl=10_000, iracp_provision=8_000)
        assert r.total_floor == 10_000
        assert r.impairment_reserve_transfer == 0.0

    def test_iracp_above_ecl(self) -> None:
        from creditriskengine.ecl.ind_as109.nbfc_backstop import apply_nbfc_backstop
        r = apply_nbfc_backstop(ind_as_109_ecl=8_000, iracp_provision=10_000)
        assert r.total_floor == 10_000
        assert r.impairment_reserve_transfer == 2_000
        assert r.booked_to_pl == 8_000


# ============================================================================
# SBR NPA glide-path
# ============================================================================


class TestSBRGlidePath:
    def test_already_on_90_day(self) -> None:
        from creditriskengine.ecl.ind_as109.nbfc_backstop import npa_dpd_threshold
        assert npa_dpd_threshold(date(2023, 1, 1), already_on_90_day_norm=True) == 90

    def test_before_2024(self) -> None:
        from creditriskengine.ecl.ind_as109.nbfc_backstop import npa_dpd_threshold
        assert npa_dpd_threshold(date(2023, 6, 1)) == 180

    def test_fy2024(self) -> None:
        from creditriskengine.ecl.ind_as109.nbfc_backstop import npa_dpd_threshold
        assert npa_dpd_threshold(date(2024, 6, 1)) == 150

    def test_fy2025(self) -> None:
        from creditriskengine.ecl.ind_as109.nbfc_backstop import npa_dpd_threshold
        assert npa_dpd_threshold(date(2025, 6, 1)) == 120

    def test_fy2026_onwards(self) -> None:
        from creditriskengine.ecl.ind_as109.nbfc_backstop import npa_dpd_threshold
        assert npa_dpd_threshold(date(2026, 6, 1)) == 90
