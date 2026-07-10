"""Tests for EU CRR3 SA credit-risk drivers (Arts. 465(3), 501, 501a, 123a)."""

from datetime import date

import pytest

from creditriskengine.core.types import CreditQualityStep, Jurisdiction
from creditriskengine.rwa.standardized.credit_risk_sa import (
    EU_INFRASTRUCTURE_FACTOR,
    EU_SME_FACTOR_HIGH,
    EU_SME_FACTOR_LOW,
    currency_mismatch_multiplier,
    eu_sme_supporting_factor,
    get_corporate_risk_weight,
    get_residential_re_risk_weight,
    get_retail_risk_weight,
)


class TestUnratedCorporateTransitional:
    """CRR3 Art. 465(3): 65% for unrated corporates with PD <= 0.5%, to 2032."""

    def test_low_pd_gets_65(self) -> None:
        rw = get_corporate_risk_weight(
            CreditQualityStep.UNRATED, Jurisdiction.EU, pd=0.004
        )
        assert rw == pytest.approx(65.0)

    def test_pd_at_ceiling_gets_65(self) -> None:
        rw = get_corporate_risk_weight(
            CreditQualityStep.UNRATED, Jurisdiction.EU, pd=0.005
        )
        assert rw == pytest.approx(65.0)

    def test_high_pd_gets_100(self) -> None:
        rw = get_corporate_risk_weight(
            CreditQualityStep.UNRATED, Jurisdiction.EU, pd=0.006
        )
        assert rw == pytest.approx(100.0)

    def test_no_pd_gets_100(self) -> None:
        rw = get_corporate_risk_weight(CreditQualityStep.UNRATED, Jurisdiction.EU)
        assert rw == pytest.approx(100.0)

    def test_expires_after_2032(self) -> None:
        rw = get_corporate_risk_weight(
            CreditQualityStep.UNRATED,
            Jurisdiction.EU,
            pd=0.004,
            reporting_date=date(2033, 1, 1),
        )
        assert rw == pytest.approx(100.0)

    def test_within_window_with_date(self) -> None:
        rw = get_corporate_risk_weight(
            CreditQualityStep.UNRATED,
            Jurisdiction.EU,
            pd=0.004,
            reporting_date=date(2030, 6, 30),
        )
        assert rw == pytest.approx(65.0)

    def test_not_applied_outside_eu(self) -> None:
        rw = get_corporate_risk_weight(
            CreditQualityStep.UNRATED, Jurisdiction.BCBS, pd=0.004
        )
        assert rw == pytest.approx(100.0)


class TestSMESupportingFactor:
    """CRR3 Art. 501: 0.7619 up to EUR 2.5m, 0.85 on the excess."""

    def test_below_threshold_flat_low_factor(self) -> None:
        assert eu_sme_supporting_factor(1_000_000.0) == pytest.approx(EU_SME_FACTOR_LOW)

    def test_at_threshold(self) -> None:
        assert eu_sme_supporting_factor(2_500_000.0) == pytest.approx(EU_SME_FACTOR_LOW)

    def test_blended_above_threshold(self) -> None:
        # EUR 5m: (0.7619*2.5 + 0.85*2.5) / 5 = 0.80595
        assert eu_sme_supporting_factor(5_000_000.0) == pytest.approx(0.80595, abs=1e-5)

    def test_converges_to_high_factor(self) -> None:
        f = eu_sme_supporting_factor(1_000_000_000.0)
        assert f == pytest.approx(EU_SME_FACTOR_HIGH, abs=1e-3)
        assert f < EU_SME_FACTOR_HIGH

    def test_invalid_exposure(self) -> None:
        with pytest.raises(ValueError, match="total_exposure_eur"):
            eu_sme_supporting_factor(0.0)

    def test_corporate_rw_uses_tiered_factor(self) -> None:
        # Unrated EU SME, total exposure EUR 5m: 100% * 0.80595 = 80.595%.
        rw = get_corporate_risk_weight(
            CreditQualityStep.UNRATED,
            Jurisdiction.EU,
            is_sme=True,
            total_sme_exposure_eur=5_000_000.0,
        )
        assert rw == pytest.approx(80.595, abs=1e-3)

    def test_corporate_rw_flat_factor_without_total(self) -> None:
        rw = get_corporate_risk_weight(
            CreditQualityStep.UNRATED, Jurisdiction.EU, is_sme=True
        )
        assert rw == pytest.approx(76.19, abs=1e-2)


class TestInfrastructureFactor:
    """CRR3 Art. 501a: qualifying infrastructure exposures x 0.75."""

    def test_applied_for_eu(self) -> None:
        rw = get_corporate_risk_weight(
            CreditQualityStep.UNRATED,
            Jurisdiction.EU,
            is_qualifying_infrastructure=True,
        )
        assert rw == pytest.approx(75.0)

    def test_not_applied_for_uk(self) -> None:
        # UK PRA did not adopt the infrastructure supporting factor.
        rw = get_corporate_risk_weight(
            CreditQualityStep.UNRATED,
            Jurisdiction.UK,
            is_qualifying_infrastructure=True,
        )
        assert rw == pytest.approx(100.0)

    def test_stacks_with_sme_factor(self) -> None:
        rw = get_corporate_risk_weight(
            CreditQualityStep.UNRATED,
            Jurisdiction.EU,
            is_sme=True,
            is_qualifying_infrastructure=True,
        )
        assert rw == pytest.approx(100.0 * 0.7619 * EU_INFRASTRUCTURE_FACTOR, abs=1e-3)


class TestCurrencyMismatch:
    """CRR3 Art. 123a / BCBS CRE20.92: RW x 1.5 capped at 150%."""

    def test_multiplier_basic(self) -> None:
        assert currency_mismatch_multiplier(75.0) == pytest.approx(112.5)

    def test_multiplier_capped(self) -> None:
        assert currency_mismatch_multiplier(120.0) == pytest.approx(150.0)

    def test_retail_mismatch(self) -> None:
        rw = get_retail_risk_weight(is_currency_mismatched=True)
        assert rw == pytest.approx(112.5)  # 75 * 1.5

    def test_other_retail_mismatch_capped(self) -> None:
        rw = get_retail_risk_weight(
            is_regulatory_retail=False, is_currency_mismatched=True
        )
        assert rw == pytest.approx(150.0)  # min(100*1.5, 150)

    def test_rre_mismatch(self) -> None:
        base = get_residential_re_risk_weight(0.75)
        stressed = get_residential_re_risk_weight(0.75, is_currency_mismatched=True)
        assert stressed == pytest.approx(min(base * 1.5, 150.0))

    def test_rre_india_mismatch(self) -> None:
        rw = get_residential_re_risk_weight(
            0.70, jurisdiction=Jurisdiction.INDIA, is_currency_mismatched=True
        )
        assert rw == pytest.approx(30.0)  # 20 * 1.5

    def test_rre_no_mismatch_unchanged(self) -> None:
        assert get_residential_re_risk_weight(0.75) == pytest.approx(
            get_residential_re_risk_weight(0.75, is_currency_mismatched=False)
        )
