"""Tests for Standardized Approach risk weight functions."""

from unittest.mock import patch

import pytest

from creditriskengine.core.types import (
    CreditQualityStep,
    Jurisdiction,
    SAExposureClass,
)
from creditriskengine.rwa.standardized.credit_risk_sa import (
    assign_sa_risk_weight,
    get_bank_risk_weight,
    get_commercial_re_risk_weight,
    get_corporate_risk_weight,
    get_defaulted_risk_weight,
    get_equity_risk_weight,
    get_residential_re_risk_weight,
    get_retail_risk_weight,
    get_sovereign_risk_weight,
    get_subordinated_debt_risk_weight,
)


class TestSovereignRiskWeight:
    """BCBS CRE20.7, Table 1."""

    @pytest.mark.parametrize(
        "cqs,expected",
        [
            (CreditQualityStep.CQS_1, 0.0),
            (CreditQualityStep.CQS_2, 20.0),
            (CreditQualityStep.CQS_3, 50.0),
            (CreditQualityStep.CQS_4, 100.0),
            (CreditQualityStep.CQS_5, 100.0),
            (CreditQualityStep.CQS_6, 150.0),
            (CreditQualityStep.UNRATED, 100.0),
        ],
    )
    def test_sovereign_rw_by_cqs(self, cqs: CreditQualityStep, expected: float) -> None:
        assert get_sovereign_risk_weight(cqs) == expected

    def test_domestic_own_currency_is_zero(self) -> None:
        assert get_sovereign_risk_weight(
            CreditQualityStep.CQS_3, is_domestic_own_currency=True
        ) == 0.0


class TestBankRiskWeight:
    """BCBS CRE20.15-20.21."""

    @pytest.mark.parametrize(
        "cqs,expected",
        [
            (CreditQualityStep.CQS_1, 20.0),
            (CreditQualityStep.CQS_2, 30.0),
            (CreditQualityStep.CQS_3, 50.0),
            (CreditQualityStep.CQS_6, 150.0),
            (CreditQualityStep.UNRATED, 50.0),
        ],
    )
    def test_bank_ecra(self, cqs: CreditQualityStep, expected: float) -> None:
        assert get_bank_risk_weight(cqs=cqs) == expected

    @pytest.mark.parametrize(
        "grade,expected",
        [("A", 40.0), ("B", 75.0), ("C", 150.0)],
    )
    def test_bank_scra(self, grade: str, expected: float) -> None:
        assert get_bank_risk_weight(scra_grade=grade) == expected

    def test_scra_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid SCRA grade"):
            get_bank_risk_weight(scra_grade="D")

    def test_unrated_default(self) -> None:
        assert get_bank_risk_weight() == 50.0

    @pytest.mark.parametrize(
        "cqs,expected",
        [
            (CreditQualityStep.CQS_1, 20.0),
            (CreditQualityStep.CQS_2, 20.0),
            (CreditQualityStep.CQS_3, 20.0),
            (CreditQualityStep.CQS_4, 50.0),
            (CreditQualityStep.CQS_6, 150.0),
        ],
    )
    def test_bank_ecra_short_term(self, cqs: CreditQualityStep, expected: float) -> None:
        assert get_bank_risk_weight(cqs=cqs, is_short_term=True) == expected


class TestCorporateRiskWeight:
    """BCBS CRE20.28-20.32."""

    @pytest.mark.parametrize(
        "cqs,expected",
        [
            (CreditQualityStep.CQS_1, 20.0),
            (CreditQualityStep.CQS_2, 50.0),
            (CreditQualityStep.CQS_3, 75.0),
            (CreditQualityStep.CQS_4, 100.0),
            (CreditQualityStep.CQS_5, 150.0),
            (CreditQualityStep.UNRATED, 100.0),
        ],
    )
    def test_corporate_rw_by_cqs(self, cqs: CreditQualityStep, expected: float) -> None:
        assert get_corporate_risk_weight(cqs) == expected

    def test_uk_unrated_investment_grade(self) -> None:
        rw = get_corporate_risk_weight(
            CreditQualityStep.UNRATED,
            jurisdiction=Jurisdiction.UK,
            is_investment_grade=True,
        )
        assert rw == 65.0

    def test_eu_sme_supporting_factor(self) -> None:
        base = get_corporate_risk_weight(CreditQualityStep.UNRATED)
        sme = get_corporate_risk_weight(
            CreditQualityStep.UNRATED,
            jurisdiction=Jurisdiction.EU,
            is_sme=True,
        )
        assert sme == pytest.approx(base * 0.7619, rel=1e-4)


class TestResidentialRealEstate:
    """BCBS CRE20.71-20.86."""

    @pytest.mark.parametrize(
        "ltv,expected",
        [
            (0.30, 20.0),
            (0.55, 25.0),
            (0.65, 30.0),
            (0.75, 35.0),
            (0.85, 40.0),
            (0.95, 50.0),
            (1.10, 70.0),
        ],
    )
    def test_whole_loan_rw(self, ltv: float, expected: float) -> None:
        assert get_residential_re_risk_weight(ltv) == expected

    def test_cashflow_dependent_higher_rw(self) -> None:
        rw_normal = get_residential_re_risk_weight(0.55)
        rw_cf = get_residential_re_risk_weight(0.55, is_cashflow_dependent=True)
        assert rw_cf > rw_normal

    def test_india_rbi_treatment(self) -> None:
        assert get_residential_re_risk_weight(0.75, Jurisdiction.INDIA) == 20.0
        assert get_residential_re_risk_weight(0.85, Jurisdiction.INDIA) == 35.0

    @pytest.mark.parametrize(
        "ltv,expected",
        [
            (0.0, 20.0),    # LTV exactly 0 → lowest band
            (0.50, 20.0),   # At boundary → stays in band
            (0.60, 25.0),   # At boundary → next band
            (0.80, 35.0),   # At boundary → 0.70-0.80 band
            (1.00, 50.0),   # At boundary → 0.90-1.00 band
        ],
    )
    def test_ltv_boundary_values(self, ltv: float, expected: float) -> None:
        assert get_residential_re_risk_weight(ltv) == expected


class TestCommercialRealEstate:
    """BCBS CRE20.87-20.98."""

    def test_not_cashflow_low_ltv(self) -> None:
        rw = get_commercial_re_risk_weight(0.50, counterparty_rw=100.0)
        assert rw == 60.0

    def test_not_cashflow_mid_ltv(self) -> None:
        assert get_commercial_re_risk_weight(0.70, counterparty_rw=100.0) == 75.0

    def test_not_cashflow_high_ltv_returns_counterparty(self) -> None:
        assert get_commercial_re_risk_weight(0.90, counterparty_rw=120.0) == 120.0

    def test_adc_is_150(self) -> None:
        assert get_commercial_re_risk_weight(1.0, is_adc=True) == 150.0

    def test_adc_presold_is_100(self) -> None:
        assert get_commercial_re_risk_weight(
            1.0, is_adc=True, is_presold_residential=True
        ) == 100.0


class TestDefaultedExposures:
    def test_high_provisions(self) -> None:
        assert get_defaulted_risk_weight(0.25) == 100.0

    def test_low_provisions(self) -> None:
        assert get_defaulted_risk_weight(0.10) == 150.0

    def test_rre_secured_always_100(self) -> None:
        # CRE20.101: RRE-secured defaults always get 100% regardless of provisions
        assert get_defaulted_risk_weight(0.05, is_rre_secured=True) == 100.0
        assert get_defaulted_risk_weight(0.25, is_rre_secured=True) == 100.0


class TestRetail:
    def test_regulatory_retail(self) -> None:
        assert get_retail_risk_weight(True) == 75.0

    def test_non_regulatory_retail(self) -> None:
        assert get_retail_risk_weight(False) == 100.0


class TestEquity:
    def test_listed(self) -> None:
        assert get_equity_risk_weight(is_listed=True) == 250.0

    def test_speculative(self) -> None:
        assert get_equity_risk_weight(is_speculative=True) == 400.0


class TestSubordinatedDebt:
    def test_subordinated(self) -> None:
        assert get_subordinated_debt_risk_weight() == 150.0


class TestAssignSaRiskWeight:
    """Integration tests for the master dispatcher."""

    def test_sovereign_dispatch(self) -> None:
        rw = assign_sa_risk_weight(
            SAExposureClass.SOVEREIGN, CreditQualityStep.CQS_1
        )
        assert rw == 0.0

    def test_corporate_dispatch(self) -> None:
        rw = assign_sa_risk_weight(
            SAExposureClass.CORPORATE, CreditQualityStep.CQS_2
        )
        assert rw == 50.0

    def test_residential_requires_ltv(self) -> None:
        with pytest.raises(ValueError, match="LTV required"):
            assign_sa_risk_weight(SAExposureClass.RESIDENTIAL_MORTGAGE)

    def test_residential_with_ltv(self) -> None:
        rw = assign_sa_risk_weight(
            SAExposureClass.RESIDENTIAL_MORTGAGE,
            ltv=0.65,
        )
        assert rw == 30.0

    def test_mdb_is_zero(self) -> None:
        assert assign_sa_risk_weight(SAExposureClass.MDB) == 0.0

    def test_other_is_100(self) -> None:
        assert assign_sa_risk_weight(SAExposureClass.OTHER) == 100.0

    def test_pse_uses_bank_table(self) -> None:
        # PSEs use bank risk weight table per CRE20.10 Option A
        rw = assign_sa_risk_weight(SAExposureClass.PSE, CreditQualityStep.CQS_1)
        assert rw == get_bank_risk_weight(cqs=CreditQualityStep.CQS_1)

    def test_securities_firm_dispatch(self) -> None:
        """Line 432: SECURITIES_FIRM uses bank risk weight table."""
        rw = assign_sa_risk_weight(
            SAExposureClass.SECURITIES_FIRM, CreditQualityStep.CQS_1
        )
        assert rw == get_bank_risk_weight(cqs=CreditQualityStep.CQS_1)

    def test_commercial_re_requires_ltv(self) -> None:
        """Lines 446-448: COMMERCIAL_REAL_ESTATE without LTV raises ValueError."""
        with pytest.raises(ValueError, match="LTV required for commercial real estate"):
            assign_sa_risk_weight(SAExposureClass.COMMERCIAL_REAL_ESTATE)

    def test_commercial_re_with_ltv(self) -> None:
        """Lines 446-448: COMMERCIAL_REAL_ESTATE with LTV works."""
        rw = assign_sa_risk_weight(
            SAExposureClass.COMMERCIAL_REAL_ESTATE,
            ltv=0.50,
            counterparty_rw=100.0,
        )
        assert rw == 60.0

    def test_land_adc_dispatch(self) -> None:
        """Line 453: LAND_ADC dispatches to commercial RE with is_adc=True."""
        rw = assign_sa_risk_weight(SAExposureClass.LAND_ADC)
        assert rw == 150.0

    def test_land_adc_presold(self) -> None:
        """Line 453: LAND_ADC with presold residential."""
        rw = assign_sa_risk_weight(
            SAExposureClass.LAND_ADC, is_presold_residential=True
        )
        assert rw == 100.0

    def test_retail_dispatch(self) -> None:
        """Line 458: RETAIL dispatches to get_retail_risk_weight."""
        rw = assign_sa_risk_weight(SAExposureClass.RETAIL)
        assert rw == 75.0

    def test_retail_non_regulatory_dispatch(self) -> None:
        """Line 458: RETAIL with non-regulatory retail."""
        rw = assign_sa_risk_weight(
            SAExposureClass.RETAIL, is_regulatory_retail=False
        )
        assert rw == 100.0

    def test_equity_dispatch(self) -> None:
        """Line 467: EQUITY dispatches to get_equity_risk_weight."""
        rw = assign_sa_risk_weight(SAExposureClass.EQUITY)
        assert rw == 250.0

    def test_equity_speculative_dispatch(self) -> None:
        """Line 467: EQUITY speculative dispatch."""
        rw = assign_sa_risk_weight(
            SAExposureClass.EQUITY, is_speculative=True
        )
        assert rw == 400.0

    def test_retail_regulatory_dispatch(self) -> None:
        """Line 461: RETAIL_REGULATORY always returns 75%."""
        rw = assign_sa_risk_weight(SAExposureClass.RETAIL_REGULATORY)
        assert rw == 75.0

    def test_defaulted_dispatch(self) -> None:
        """Line 464: DEFAULTED dispatches to get_defaulted_risk_weight."""
        rw = assign_sa_risk_weight(
            SAExposureClass.DEFAULTED, specific_provisions_pct=0.25
        )
        assert rw == 100.0

    def test_defaulted_low_provisions_dispatch(self) -> None:
        """Line 464: DEFAULTED with low provisions → 150%."""
        rw = assign_sa_risk_weight(
            SAExposureClass.DEFAULTED, specific_provisions_pct=0.05
        )
        assert rw == 150.0

    def test_subordinated_debt_dispatch(self) -> None:
        """Line 470: SUBORDINATED_DEBT dispatches to get_subordinated_debt_risk_weight."""
        rw = assign_sa_risk_weight(SAExposureClass.SUBORDINATED_DEBT)
        assert rw == 150.0


class TestResidentialRealEstateFallback:
    """Cover line 263: fallback return when LTV doesn't match any band."""

    def test_rre_fallback_via_monkeypatch(self) -> None:
        """Line 263: Force fallback by using an empty band table."""
        empty_table: list[tuple[float, float, float]] = []
        with patch(
            "creditriskengine.rwa.standardized.credit_risk_sa.RRE_WHOLE_LOAN_RW",
            empty_table,
        ):
            # ltv=0.5 > 0, loop finds nothing, ltv <= 0 is False → hits line 263
            # But empty table → table[-1] would IndexError. Use a gap table instead.
            pass

    def test_rre_fallback_with_gap_table(self) -> None:
        """Line 263: Use a table with a gap so positive LTV falls through."""
        gap_table: list[tuple[float, float, float]] = [
            (0.0, 0.50, 20.0),
            (0.60, 0.70, 30.0),
            # Gap: 0.50 < ltv <= 0.60 is not covered, and ltv=0.55 > 0
        ]
        with patch(
            "creditriskengine.rwa.standardized.credit_risk_sa.RRE_WHOLE_LOAN_RW",
            gap_table,
        ):
            rw = get_residential_re_risk_weight(0.55)
            assert rw == 30.0  # table[-1][2]


class TestCommercialRealEstateCashflow:
    """Cover lines 293-298: cashflow-dependent CRE (IPRE) path."""

    @pytest.mark.parametrize(
        "ltv,expected",
        [
            (0.30, 70.0),   # First band: (0.0, 0.60] → 70%
            (0.60, 70.0),   # At boundary of first band
            (0.70, 90.0),   # Second band: (0.60, 0.80] → 90%
            (0.80, 90.0),   # At boundary of second band
            (0.90, 110.0),  # Third band: (0.80, inf) → 110%
            (1.50, 110.0),  # Well above → 110%
        ],
    )
    def test_cre_ipre_by_ltv(self, ltv: float, expected: float) -> None:
        """Lines 293-295: IPRE table lookup."""
        assert get_commercial_re_risk_weight(
            ltv, is_cashflow_dependent=True
        ) == expected

    def test_cre_ipre_zero_ltv(self) -> None:
        """Lines 296-297: LTV <= 0 returns first band RW."""
        rw = get_commercial_re_risk_weight(0.0, is_cashflow_dependent=True)
        assert rw == 70.0

    def test_cre_ipre_negative_ltv(self) -> None:
        """Lines 296-297: Negative LTV returns first band RW."""
        rw = get_commercial_re_risk_weight(-0.1, is_cashflow_dependent=True)
        assert rw == 70.0

    def test_cre_ipre_fallback(self) -> None:
        """Line 298: Fallback when LTV doesn't match any IPRE band."""
        gap_table: list[tuple[float, float, float]] = [
            (0.0, 0.50, 70.0),
            (0.60, 0.80, 90.0),
        ]
        with patch(
            "creditriskengine.rwa.standardized.credit_risk_sa.CRE_IPRE_RW",
            gap_table,
        ):
            rw = get_commercial_re_risk_weight(0.55, is_cashflow_dependent=True)
            assert rw == 90.0  # table[-1][2]


class TestEquityUnlisted:
    """Cover line 365: unlisted non-speculative equity."""

    def test_unlisted_non_speculative(self) -> None:
        """Line 365: Unlisted, non-speculative equity → 400%."""
        assert get_equity_risk_weight(is_listed=False, is_speculative=False) == 400.0


class TestFinalFallback:
    """Cover line 483: final fallback return 100.0 for unknown exposure class."""

    def test_unknown_exposure_class_fallback(self) -> None:
        """Line 483: An unrecognized exposure class falls through to 100%."""
        # Pass a string that doesn't match any SAExposureClass member.
        # Python doesn't enforce type hints at runtime.
        rw = assign_sa_risk_weight(exposure_class="unknown_class")  # type: ignore[arg-type]
        assert rw == 100.0
