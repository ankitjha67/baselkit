"""Tests for RBI Project Finance Directions 2025 and Gold Loan Directions 2025."""

import pytest

from creditriskengine.ecl.ind_as109.gold_loans import (
    assess_gold_loan_ltv,
    gold_loan_max_ltv,
)
from creditriskengine.ecl.ind_as109.project_finance import (
    dcco_deferment_provision,
    dcco_max_deferral_years,
)


class TestDCCOMaxDeferral:
    def test_infrastructure_three_years(self) -> None:
        assert dcco_max_deferral_years(True) == pytest.approx(3.0)

    def test_non_infrastructure_two_years(self) -> None:
        assert dcco_max_deferral_years(False) == pytest.approx(2.0)


class TestDCCODefermentProvision:
    def test_infra_two_quarters(self) -> None:
        # Infra, 0.5y = 2 quarters: add-on = 2 * 0.375% = 0.75%;
        # total = 1.00% base + 0.75% = 1.75% on 100 crore.
        r = dcco_deferment_provision(
            1_000_000_000.0, deferral_years=0.5,
            base_provision_rate=0.0100, is_infrastructure=True,
        )
        assert r.quarters_deferred == 2
        assert r.within_permitted_window is True
        assert r.additional_provision_rate == pytest.approx(0.0075)
        assert r.total_provision_rate == pytest.approx(0.0175)
        assert r.total_provision == pytest.approx(0.0175 * 1_000_000_000.0)

    def test_non_infra_higher_quarterly_rate(self) -> None:
        # Non-infra (CRE), 1 quarter: add-on = 0.5625%.
        r = dcco_deferment_provision(
            100.0, deferral_years=0.25,
            base_provision_rate=0.0125, is_infrastructure=False,
        )
        assert r.quarters_deferred == 1
        assert r.additional_provision_rate == pytest.approx(0.005625)
        assert r.total_provision_rate == pytest.approx(0.018125)

    def test_partial_quarter_rounds_up(self) -> None:
        r = dcco_deferment_provision(
            100.0, deferral_years=0.3,  # 1.2 quarters -> 2
            base_provision_rate=0.0100, is_infrastructure=True,
        )
        assert r.quarters_deferred == 2

    def test_beyond_window_loses_standard(self) -> None:
        # Non-infra deferred 2.5y > 2y limit -> not standard, zero fields.
        r = dcco_deferment_provision(
            100.0, deferral_years=2.5,
            base_provision_rate=0.0125, is_infrastructure=False,
        )
        assert r.within_permitted_window is False
        assert r.total_provision == pytest.approx(0.0)

    def test_infra_full_window_ok(self) -> None:
        # Infra at exactly 3y (12 quarters) stays standard.
        r = dcco_deferment_provision(
            100.0, deferral_years=3.0,
            base_provision_rate=0.0100, is_infrastructure=True,
        )
        assert r.within_permitted_window is True
        assert r.quarters_deferred == 12
        assert r.additional_provision_rate == pytest.approx(12 * 0.00375)

    def test_zero_deferral(self) -> None:
        r = dcco_deferment_provision(
            100.0, deferral_years=0.0,
            base_provision_rate=0.0100, is_infrastructure=True,
        )
        assert r.quarters_deferred == 0
        assert r.total_provision_rate == pytest.approx(0.0100)

    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError, match="funded_outstanding"):
            dcco_deferment_provision(-1.0, 0.5, 0.01, True)
        with pytest.raises(ValueError, match="deferral_years"):
            dcco_deferment_provision(100.0, -0.5, 0.01, True)
        with pytest.raises(ValueError, match="base_provision_rate"):
            dcco_deferment_provision(100.0, 0.5, -0.01, True)


class TestGoldLoanLTV:
    def test_small_loan_85(self) -> None:
        assert gold_loan_max_ltv(200_000.0) == pytest.approx(0.85)

    def test_tier_boundary_2_5_lakh(self) -> None:
        assert gold_loan_max_ltv(250_000.0) == pytest.approx(0.85)

    def test_mid_tier_80(self) -> None:
        assert gold_loan_max_ltv(400_000.0) == pytest.approx(0.80)

    def test_tier_boundary_5_lakh(self) -> None:
        assert gold_loan_max_ltv(500_000.0) == pytest.approx(0.80)

    def test_large_loan_75(self) -> None:
        assert gold_loan_max_ltv(1_000_000.0) == pytest.approx(0.75)

    def test_invalid_amount(self) -> None:
        with pytest.raises(ValueError, match="loan_amount_inr"):
            gold_loan_max_ltv(0.0)


class TestAssessGoldLoanLTV:
    def test_compliant(self) -> None:
        # 2 lakh loan on 2.5 lakh collateral: LTV 80% <= 85% ceiling.
        r = assess_gold_loan_ltv(200_000.0, 250_000.0)
        assert r.ltv == pytest.approx(0.80)
        assert r.max_ltv == pytest.approx(0.85)
        assert r.is_compliant is True
        assert r.max_permissible_loan == pytest.approx(212_500.0)

    def test_breach(self) -> None:
        # 9 lakh on 10 lakh collateral: LTV 90% > 75% ceiling.
        r = assess_gold_loan_ltv(900_000.0, 1_000_000.0)
        assert r.max_ltv == pytest.approx(0.75)
        assert r.is_compliant is False

    def test_invalid_collateral(self) -> None:
        with pytest.raises(ValueError, match="collateral_value_inr"):
            assess_gold_loan_ltv(100_000.0, 0.0)
