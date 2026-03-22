"""Tests for Standardized Approach risk weight functions."""

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
    def test_sovereign_rw_by_cqs(self, cqs, expected):
        assert get_sovereign_risk_weight(cqs) == expected

    def test_domestic_own_currency_is_zero(self):
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
    def test_bank_ecra(self, cqs, expected):
        assert get_bank_risk_weight(cqs=cqs) == expected

    @pytest.mark.parametrize(
        "grade,expected",
        [("A", 40.0), ("B", 75.0), ("C", 150.0)],
    )
    def test_bank_scra(self, grade, expected):
        assert get_bank_risk_weight(scra_grade=grade) == expected

    def test_scra_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid SCRA grade"):
            get_bank_risk_weight(scra_grade="D")

    def test_unrated_default(self):
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
    def test_bank_ecra_short_term(self, cqs, expected):
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
    def test_corporate_rw_by_cqs(self, cqs, expected):
        assert get_corporate_risk_weight(cqs) == expected

    def test_uk_unrated_investment_grade(self):
        rw = get_corporate_risk_weight(
            CreditQualityStep.UNRATED,
            jurisdiction=Jurisdiction.UK,
            is_investment_grade=True,
        )
        assert rw == 65.0

    def test_eu_sme_supporting_factor(self):
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
    def test_whole_loan_rw(self, ltv, expected):
        assert get_residential_re_risk_weight(ltv) == expected

    def test_cashflow_dependent_higher_rw(self):
        rw_normal = get_residential_re_risk_weight(0.55)
        rw_cf = get_residential_re_risk_weight(0.55, is_cashflow_dependent=True)
        assert rw_cf > rw_normal

    def test_india_rbi_treatment(self):
        assert get_residential_re_risk_weight(0.75, Jurisdiction.INDIA) == 20.0
        assert get_residential_re_risk_weight(0.85, Jurisdiction.INDIA) == 35.0


class TestCommercialRealEstate:
    """BCBS CRE20.87-20.98."""

    def test_not_cashflow_low_ltv(self):
        rw = get_commercial_re_risk_weight(0.50, counterparty_rw=100.0)
        assert rw == 60.0

    def test_not_cashflow_mid_ltv(self):
        assert get_commercial_re_risk_weight(0.70, counterparty_rw=100.0) == 75.0

    def test_not_cashflow_high_ltv_returns_counterparty(self):
        assert get_commercial_re_risk_weight(0.90, counterparty_rw=120.0) == 120.0

    def test_adc_is_150(self):
        assert get_commercial_re_risk_weight(1.0, is_adc=True) == 150.0

    def test_adc_presold_is_100(self):
        assert get_commercial_re_risk_weight(
            1.0, is_adc=True, is_presold_residential=True
        ) == 100.0


class TestDefaultedExposures:
    def test_high_provisions(self):
        assert get_defaulted_risk_weight(0.25) == 100.0

    def test_low_provisions(self):
        assert get_defaulted_risk_weight(0.10) == 150.0

    def test_rre_secured_always_100(self):
        # CRE20.101: RRE-secured defaults always get 100% regardless of provisions
        assert get_defaulted_risk_weight(0.05, is_rre_secured=True) == 100.0
        assert get_defaulted_risk_weight(0.25, is_rre_secured=True) == 100.0


class TestRetail:
    def test_regulatory_retail(self):
        assert get_retail_risk_weight(True) == 75.0

    def test_non_regulatory_retail(self):
        assert get_retail_risk_weight(False) == 100.0


class TestEquity:
    def test_listed(self):
        assert get_equity_risk_weight(is_listed=True) == 250.0

    def test_speculative(self):
        assert get_equity_risk_weight(is_speculative=True) == 400.0


class TestSubordinatedDebt:
    def test_subordinated(self):
        assert get_subordinated_debt_risk_weight() == 150.0


class TestAssignSaRiskWeight:
    """Integration tests for the master dispatcher."""

    def test_sovereign_dispatch(self):
        rw = assign_sa_risk_weight(
            SAExposureClass.SOVEREIGN, CreditQualityStep.CQS_1
        )
        assert rw == 0.0

    def test_corporate_dispatch(self):
        rw = assign_sa_risk_weight(
            SAExposureClass.CORPORATE, CreditQualityStep.CQS_2
        )
        assert rw == 50.0

    def test_residential_requires_ltv(self):
        with pytest.raises(ValueError, match="LTV required"):
            assign_sa_risk_weight(SAExposureClass.RESIDENTIAL_MORTGAGE)

    def test_residential_with_ltv(self):
        rw = assign_sa_risk_weight(
            SAExposureClass.RESIDENTIAL_MORTGAGE,
            ltv=0.65,
        )
        assert rw == 30.0

    def test_mdb_is_zero(self):
        assert assign_sa_risk_weight(SAExposureClass.MDB) == 0.0

    def test_other_is_100(self):
        assert assign_sa_risk_weight(SAExposureClass.OTHER) == 100.0

    def test_pse_uses_bank_table(self):
        # PSEs use bank risk weight table per CRE20.10 Option A
        rw = assign_sa_risk_weight(SAExposureClass.PSE, CreditQualityStep.CQS_1)
        assert rw == get_bank_risk_weight(cqs=CreditQualityStep.CQS_1)
