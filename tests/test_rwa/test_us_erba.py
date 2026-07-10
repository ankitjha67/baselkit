"""Tests for the US Basel III Endgame ERBA module (PROPOSED framework).

All expected values are from the 2023 NPR operative text (88 FR 64028).
"""

from datetime import date

import pytest

from creditriskengine.rwa.standardized.us_erba import (
    ERBABankGrade,
    ERBAEquityCategory,
    ERBAOffBalanceItem,
    ERBARetailCategory,
    aoci_included_pct_reproposal,
    aoci_optout_remaining_pct,
    dual_stack_binding_rwa,
    erba_bank_rw,
    erba_ccf,
    erba_corporate_rw,
    erba_equity_rw,
    erba_residential_mortgage_rw,
    erba_retail_rw,
    expanded_rwa_transition_pct,
    single_stack_rwa,
)


class TestResidentialMortgage:
    """Tables 5 & 6, 88 FR 64190-91."""

    def test_not_dependent_grid(self) -> None:
        assert erba_residential_mortgage_rw(0.45) == pytest.approx(40.0)
        assert erba_residential_mortgage_rw(0.55) == pytest.approx(45.0)
        assert erba_residential_mortgage_rw(0.75) == pytest.approx(50.0)
        assert erba_residential_mortgage_rw(0.85) == pytest.approx(60.0)
        assert erba_residential_mortgage_rw(0.95) == pytest.approx(70.0)
        assert erba_residential_mortgage_rw(1.10) == pytest.approx(90.0)

    def test_dependent_grid(self) -> None:
        dep = {0.45: 50.0, 0.75: 65.0, 0.95: 95.0, 1.10: 125.0}
        for ltv, expected in dep.items():
            rw = erba_residential_mortgage_rw(ltv, is_cashflow_dependent=True)
            assert rw == pytest.approx(expected)

    def test_band_boundaries_inclusive(self) -> None:
        assert erba_residential_mortgage_rw(0.50) == pytest.approx(40.0)
        assert erba_residential_mortgage_rw(0.80) == pytest.approx(50.0)
        assert erba_residential_mortgage_rw(1.00) == pytest.approx(70.0)

    def test_non_regulatory_other(self) -> None:
        # s.__.111(f)(7): 100% not-dependent / 150% dependent.
        assert erba_residential_mortgage_rw(
            0.75, is_regulatory_residential=False
        ) == pytest.approx(100.0)
        assert erba_residential_mortgage_rw(
            0.75, is_regulatory_residential=False, is_cashflow_dependent=True
        ) == pytest.approx(150.0)

    def test_defaulted(self) -> None:
        # 150%, except non-cashflow-dependent residential at 100%.
        assert erba_residential_mortgage_rw(0.75, is_defaulted=True) == pytest.approx(100.0)
        assert erba_residential_mortgage_rw(
            0.75, is_defaulted=True, is_cashflow_dependent=True
        ) == pytest.approx(150.0)

    def test_currency_mismatch(self) -> None:
        # 50% * 1.5 = 75%.
        assert erba_residential_mortgage_rw(
            0.75, is_currency_mismatched=True
        ) == pytest.approx(75.0)
        # 125% * 1.5 capped at 150%.
        assert erba_residential_mortgage_rw(
            1.10, is_cashflow_dependent=True, is_currency_mismatched=True
        ) == pytest.approx(150.0)

    def test_negative_ltv_raises(self) -> None:
        with pytest.raises(ValueError, match="ltv"):
            erba_residential_mortgage_rw(-0.1)


class TestRetail:
    def test_categories(self) -> None:
        assert erba_retail_rw(ERBARetailCategory.TRANSACTOR) == pytest.approx(55.0)
        assert erba_retail_rw(ERBARetailCategory.REGULATORY_RETAIL) == pytest.approx(85.0)
        assert erba_retail_rw(ERBARetailCategory.OTHER_RETAIL) == pytest.approx(110.0)

    def test_currency_mismatch_capped(self) -> None:
        rw = erba_retail_rw(ERBARetailCategory.OTHER_RETAIL, is_currency_mismatched=True)
        assert rw == pytest.approx(150.0)  # min(110*1.5, 150)


class TestCorporate:
    def test_ig_with_public_security_65(self) -> None:
        assert erba_corporate_rw(
            is_investment_grade=True, has_public_security=True
        ) == pytest.approx(65.0)

    def test_ig_without_public_security_100(self) -> None:
        # BOTH conditions required in the 2023 NPR.
        assert erba_corporate_rw(is_investment_grade=True) == pytest.approx(100.0)

    def test_base_100(self) -> None:
        assert erba_corporate_rw() == pytest.approx(100.0)

    def test_subordinated_150(self) -> None:
        assert erba_corporate_rw(is_subordinated_debt=True) == pytest.approx(150.0)

    def test_project_finance_preoperational_130(self) -> None:
        assert erba_corporate_rw(
            is_project_finance_preoperational=True
        ) == pytest.approx(130.0)


class TestBank:
    def test_grades(self) -> None:
        assert erba_bank_rw(ERBABankGrade.GRADE_A) == pytest.approx(40.0)
        assert erba_bank_rw(ERBABankGrade.GRADE_B) == pytest.approx(75.0)
        assert erba_bank_rw(ERBABankGrade.GRADE_C) == pytest.approx(150.0)

    def test_short_term(self) -> None:
        assert erba_bank_rw(ERBABankGrade.GRADE_A, is_short_term=True) == pytest.approx(20.0)
        assert erba_bank_rw(ERBABankGrade.GRADE_B, is_short_term=True) == pytest.approx(50.0)
        assert erba_bank_rw(ERBABankGrade.GRADE_C, is_short_term=True) == pytest.approx(150.0)


class TestCCF:
    def test_all_ccfs(self) -> None:
        assert erba_ccf(ERBAOffBalanceItem.UNCONDITIONALLY_CANCELLABLE) == pytest.approx(0.10)
        assert erba_ccf(ERBAOffBalanceItem.TRADE_SHORT_TERM) == pytest.approx(0.20)
        assert erba_ccf(ERBAOffBalanceItem.COMMITMENT) == pytest.approx(0.40)
        assert erba_ccf(ERBAOffBalanceItem.TRANSACTION_RELATED) == pytest.approx(0.50)
        assert erba_ccf(ERBAOffBalanceItem.DIRECT_CREDIT_SUBSTITUTE) == pytest.approx(1.00)


class TestEquity:
    def test_ladder(self) -> None:
        assert erba_equity_rw(ERBAEquityCategory.SOVEREIGN) == pytest.approx(0.0)
        assert erba_equity_rw(ERBAEquityCategory.PSE_FHLB_FARMER_MAC) == pytest.approx(20.0)
        assert erba_equity_rw(ERBAEquityCategory.COMMUNITY_DEVELOPMENT) == pytest.approx(100.0)
        assert erba_equity_rw(ERBAEquityCategory.PUBLICLY_TRADED) == pytest.approx(250.0)
        assert erba_equity_rw(ERBAEquityCategory.NON_PUBLICLY_TRADED) == pytest.approx(400.0)
        assert erba_equity_rw(ERBAEquityCategory.INVESTMENT_FIRM) == pytest.approx(1250.0)


class TestFrameworkStructure:
    def test_dual_stack_higher_of(self) -> None:
        r = dual_stack_binding_rwa(1000.0, 1200.0)
        assert r.binding_rwa == pytest.approx(1200.0)
        assert r.binding_stack == "expanded"
        r2 = dual_stack_binding_rwa(1300.0, 1200.0)
        assert r2.binding_stack == "standardized"

    def test_dual_stack_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            dual_stack_binding_rwa(-1.0, 100.0)

    def test_single_stack_cat_i_ii_uses_erba(self) -> None:
        # >= $700bn: ERBA only, even when it is LOWER (no higher-of).
        r = single_stack_rwa(1300.0, 1200.0, total_assets_usd=800e9)
        assert r.binding_rwa == pytest.approx(1200.0)
        assert r.binding_stack == "expanded"

    def test_single_stack_smaller_bank_uses_sa(self) -> None:
        r = single_stack_rwa(1300.0, 1200.0, total_assets_usd=200e9)
        assert r.binding_stack == "standardized"
        assert r.binding_rwa == pytest.approx(1300.0)

    def test_single_stack_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            single_stack_rwa(100.0, 100.0, total_assets_usd=-1.0)


class TestTransitions:
    def test_expanded_rwa_schedule_no_95_step(self) -> None:
        # Table 9: 80/85/90/100 — explicitly NO 95% step.
        assert expanded_rwa_transition_pct(date(2025, 1, 1)) == pytest.approx(0.0)
        assert expanded_rwa_transition_pct(date(2025, 7, 1)) == pytest.approx(0.80)
        assert expanded_rwa_transition_pct(date(2026, 7, 1)) == pytest.approx(0.85)
        assert expanded_rwa_transition_pct(date(2027, 7, 1)) == pytest.approx(0.90)
        assert expanded_rwa_transition_pct(date(2028, 7, 1)) == pytest.approx(1.00)
        assert expanded_rwa_transition_pct(date(2030, 1, 1)) == pytest.approx(1.00)

    def test_aoci_optout_2023_schedule(self) -> None:
        # Table 10: 75/50/25/0.
        assert aoci_optout_remaining_pct(date(2025, 1, 1)) == pytest.approx(1.00)
        assert aoci_optout_remaining_pct(date(2025, 7, 1)) == pytest.approx(0.75)
        assert aoci_optout_remaining_pct(date(2026, 7, 1)) == pytest.approx(0.50)
        assert aoci_optout_remaining_pct(date(2027, 7, 1)) == pytest.approx(0.25)
        assert aoci_optout_remaining_pct(date(2028, 7, 1)) == pytest.approx(0.0)

    def test_aoci_reproposal_20pct_per_year(self) -> None:
        # 2026 reproposal: 20%/year from 1 Jan 2027, full from 2031.
        assert aoci_included_pct_reproposal(date(2026, 12, 31)) == pytest.approx(0.0)
        assert aoci_included_pct_reproposal(date(2027, 6, 1)) == pytest.approx(0.20)
        assert aoci_included_pct_reproposal(date(2029, 1, 1)) == pytest.approx(0.60)
        assert aoci_included_pct_reproposal(date(2031, 1, 1)) == pytest.approx(1.00)
        assert aoci_included_pct_reproposal(date(2035, 1, 1)) == pytest.approx(1.00)
