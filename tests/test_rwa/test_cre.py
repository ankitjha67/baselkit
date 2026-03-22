"""Tests for Commercial Real Estate risk weight logic — BCBS d424, CRE20.87-20.98."""

import pytest

from creditriskengine.core.types import Jurisdiction
from creditriskengine.rwa.standardized.cre import (
    CRE_IPRE_RW,
    CRE_NOT_CASHFLOW_RW,
    LAND_ADC_PRESOLD_RESIDENTIAL_RW,
    LAND_ADC_RW,
    _lookup_ltv_table,
    get_cre_risk_weight,
    get_cre_risk_weight_eu,
)


class TestConstants:
    """Verify regulatory constants."""

    def test_land_adc_rw(self) -> None:
        assert LAND_ADC_RW == 150.0

    def test_land_adc_presold_residential_rw(self) -> None:
        assert LAND_ADC_PRESOLD_RESIDENTIAL_RW == 100.0

    def test_cre_not_cashflow_table_structure(self) -> None:
        assert len(CRE_NOT_CASHFLOW_RW) == 3
        # First bucket 0-60%, second 60-80%, third 80+
        assert CRE_NOT_CASHFLOW_RW[0] == (0.0, 0.60, 60.0)

    def test_cre_ipre_table_structure(self) -> None:
        assert len(CRE_IPRE_RW) == 3


class TestLookupLtvTable:
    """Test _lookup_ltv_table helper."""

    def test_first_bucket(self) -> None:
        # LTV=0.30 falls in first bucket (0, 0.60] -> rw=60, min(60, counterparty_rw)
        rw = _lookup_ltv_table(CRE_NOT_CASHFLOW_RW, 0.30, 100.0)
        assert rw == pytest.approx(60.0)

    def test_first_bucket_lower_counterparty_rw(self) -> None:
        # min(60%, counterparty_rw) when counterparty_rw < 60
        rw = _lookup_ltv_table(CRE_NOT_CASHFLOW_RW, 0.30, 40.0)
        assert rw == pytest.approx(40.0)

    def test_second_bucket(self) -> None:
        # LTV=0.70 falls in (0.60, 0.80] -> rw=80
        rw = _lookup_ltv_table(CRE_NOT_CASHFLOW_RW, 0.70, 100.0)
        assert rw == pytest.approx(80.0)

    def test_third_bucket_sentinel(self) -> None:
        # LTV=0.90 falls in (0.80, inf] -> sentinel -1, returns counterparty_rw
        rw = _lookup_ltv_table(CRE_NOT_CASHFLOW_RW, 0.90, 120.0)
        assert rw == pytest.approx(120.0)

    def test_ltv_zero(self) -> None:
        # LTV exactly 0 hits the special case
        rw = _lookup_ltv_table(CRE_NOT_CASHFLOW_RW, 0.0, 100.0)
        assert rw == pytest.approx(60.0)

    def test_ltv_zero_lower_counterparty_rw(self) -> None:
        rw = _lookup_ltv_table(CRE_NOT_CASHFLOW_RW, 0.0, 30.0)
        assert rw == pytest.approx(30.0)

    def test_ltv_negative(self) -> None:
        # Negative LTV treated as below first bucket
        rw = _lookup_ltv_table(CRE_NOT_CASHFLOW_RW, -0.1, 50.0)
        assert rw == pytest.approx(50.0)

    def test_ipre_first_bucket(self) -> None:
        rw = _lookup_ltv_table(CRE_IPRE_RW, 0.50, 100.0)
        assert rw == pytest.approx(70.0)

    def test_ipre_second_bucket(self) -> None:
        rw = _lookup_ltv_table(CRE_IPRE_RW, 0.70, 100.0)
        assert rw == pytest.approx(90.0)

    def test_ipre_third_bucket(self) -> None:
        rw = _lookup_ltv_table(CRE_IPRE_RW, 0.90, 100.0)
        assert rw == pytest.approx(110.0)

    def test_ltv_exactly_at_bucket_boundary(self) -> None:
        # LTV=0.60 hits first bucket boundary
        rw = _lookup_ltv_table(CRE_NOT_CASHFLOW_RW, 0.60, 100.0)
        assert rw == pytest.approx(60.0)

    def test_ltv_exactly_at_second_boundary(self) -> None:
        rw = _lookup_ltv_table(CRE_NOT_CASHFLOW_RW, 0.80, 100.0)
        assert rw == pytest.approx(80.0)


class TestGetCRERiskWeight:
    """Test main CRE risk weight function."""

    def test_adc_returns_150(self) -> None:
        rw = get_cre_risk_weight(ltv=0.50, counterparty_rw=100.0, is_adc=True)
        assert rw == pytest.approx(150.0)

    def test_adc_presold_returns_100(self) -> None:
        rw = get_cre_risk_weight(
            ltv=0.50, counterparty_rw=100.0, is_adc=True, is_presold=True
        )
        assert rw == pytest.approx(100.0)

    def test_non_income_producing_low_ltv(self) -> None:
        rw = get_cre_risk_weight(ltv=0.50, counterparty_rw=100.0)
        assert rw == pytest.approx(60.0)

    def test_non_income_producing_high_ltv(self) -> None:
        rw = get_cre_risk_weight(ltv=0.85, counterparty_rw=100.0)
        assert rw == pytest.approx(100.0)  # counterparty RW

    def test_income_producing_low_ltv(self) -> None:
        rw = get_cre_risk_weight(
            ltv=0.50, counterparty_rw=100.0, is_income_producing=True
        )
        assert rw == pytest.approx(70.0)

    def test_income_producing_mid_ltv(self) -> None:
        rw = get_cre_risk_weight(
            ltv=0.70, counterparty_rw=100.0, is_income_producing=True
        )
        assert rw == pytest.approx(90.0)

    def test_income_producing_high_ltv(self) -> None:
        rw = get_cre_risk_weight(
            ltv=0.90, counterparty_rw=100.0, is_income_producing=True
        )
        assert rw == pytest.approx(110.0)

    def test_eu_jurisdiction_delegates(self) -> None:
        rw = get_cre_risk_weight(
            ltv=0.50, counterparty_rw=100.0, jurisdiction=Jurisdiction.EU
        )
        # EU non-income-producing, LTV<=0.55 -> 60%
        assert rw == pytest.approx(60.0)

    def test_eu_jurisdiction_ipre(self) -> None:
        rw = get_cre_risk_weight(
            ltv=0.50,
            counterparty_rw=100.0,
            is_income_producing=True,
            jurisdiction=Jurisdiction.EU,
        )
        assert rw == pytest.approx(70.0)

    def test_bcbs_jurisdiction_explicit(self) -> None:
        rw = get_cre_risk_weight(
            ltv=0.70, counterparty_rw=100.0, jurisdiction=Jurisdiction.BCBS
        )
        assert rw == pytest.approx(80.0)


class TestGetCRERiskWeightEU:
    """Test EU CRR3 Art. 126 CRE risk weights."""

    def test_non_cashflow_low_ltv(self) -> None:
        rw = get_cre_risk_weight_eu(0.50, 100.0)
        assert rw == pytest.approx(60.0)

    def test_non_cashflow_55_60_sub_bucket(self) -> None:
        # EU specific 55-60% sub-bucket at 60%
        rw = get_cre_risk_weight_eu(0.57, 100.0)
        assert rw == pytest.approx(60.0)

    def test_non_cashflow_60_80(self) -> None:
        rw = get_cre_risk_weight_eu(0.70, 100.0)
        assert rw == pytest.approx(80.0)

    def test_non_cashflow_high_ltv_counterparty_rw(self) -> None:
        rw = get_cre_risk_weight_eu(0.90, 120.0)
        assert rw == pytest.approx(120.0)

    def test_ipre_low_ltv(self) -> None:
        rw = get_cre_risk_weight_eu(0.50, 100.0, is_income_producing=True)
        assert rw == pytest.approx(70.0)

    def test_ipre_mid_ltv(self) -> None:
        rw = get_cre_risk_weight_eu(0.70, 100.0, is_income_producing=True)
        assert rw == pytest.approx(90.0)

    def test_ipre_high_ltv(self) -> None:
        rw = get_cre_risk_weight_eu(0.90, 100.0, is_income_producing=True)
        assert rw == pytest.approx(110.0)

    def test_min_counterparty_rw_for_low_ltv(self) -> None:
        # min(60%, 40%) = 40%
        rw = get_cre_risk_weight_eu(0.40, 40.0)
        assert rw == pytest.approx(40.0)
