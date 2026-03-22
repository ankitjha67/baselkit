"""Multi-jurisdiction tests for SA risk weight divergences.

Tests jurisdiction-specific overrides and variations from the base
BCBS framework, verifying that each national implementation produces
the correct risk weights per its own regulatory text.
"""

from datetime import date

import pytest

from creditriskengine.core.types import (
    CreditQualityStep,
    Jurisdiction,
    SAExposureClass,
)
from creditriskengine.rwa.output_floor import get_output_floor_pct
from creditriskengine.rwa.standardized.credit_risk_sa import (
    assign_sa_risk_weight,
    get_corporate_risk_weight,
    get_residential_re_risk_weight,
    get_sovereign_risk_weight,
)


class TestSovereignRiskWeights:
    """Sovereign risk weights should be 0% for domestic currency across jurisdictions."""

    @pytest.mark.parametrize("jurisdiction", [
        Jurisdiction.BCBS, Jurisdiction.EU, Jurisdiction.UK,
        Jurisdiction.INDIA, Jurisdiction.SINGAPORE,
    ])
    def test_domestic_sovereign_zero(self, jurisdiction: Jurisdiction) -> None:
        """All jurisdictions assign 0% to own sovereign in domestic currency."""
        rw = get_sovereign_risk_weight(
            CreditQualityStep.UNRATED,
            jurisdiction=jurisdiction,
            is_domestic_own_currency=True,
        )
        assert rw == 0.0

    def test_bcbs_sovereign_table(self) -> None:
        """BCBS CRE20 Table 1 sovereign risk weights."""
        expected = {
            CreditQualityStep.CQS_1: 0.0,
            CreditQualityStep.CQS_2: 20.0,
            CreditQualityStep.CQS_3: 50.0,
            CreditQualityStep.CQS_4: 100.0,
            CreditQualityStep.CQS_5: 100.0,
            CreditQualityStep.CQS_6: 150.0,
            CreditQualityStep.UNRATED: 100.0,
        }
        for cqs, expected_rw in expected.items():
            rw = get_sovereign_risk_weight(cqs, Jurisdiction.BCBS)
            assert rw == expected_rw, f"CQS {cqs}: expected {expected_rw}, got {rw}"


class TestCorporateJurisdictionDivergences:
    """Test jurisdiction-specific corporate risk weight treatments."""

    def test_uk_unrated_investment_grade_65pct(self) -> None:
        """PRA PS9/24 para 3.17: unrated IG corporates get 65% RW."""
        rw = get_corporate_risk_weight(
            CreditQualityStep.UNRATED,
            jurisdiction=Jurisdiction.UK,
            is_investment_grade=True,
        )
        assert rw == 65.0

    def test_uk_unrated_non_ig_100pct(self) -> None:
        """UK: non-IG unrated corporates remain at 100%."""
        rw = get_corporate_risk_weight(
            CreditQualityStep.UNRATED,
            jurisdiction=Jurisdiction.UK,
            is_investment_grade=False,
        )
        assert rw == 100.0

    def test_eu_unrated_100pct(self) -> None:
        """EU CRR3: unrated corporates always 100% (no IG exception)."""
        rw = get_corporate_risk_weight(
            CreditQualityStep.UNRATED,
            jurisdiction=Jurisdiction.EU,
            is_investment_grade=True,
        )
        assert rw == 100.0

    def test_eu_sme_supporting_factor(self) -> None:
        """EU CRR3 Art. 501: SME supporting factor 0.7619."""
        rw = get_corporate_risk_weight(
            CreditQualityStep.UNRATED,
            jurisdiction=Jurisdiction.EU,
            is_sme=True,
        )
        assert rw == pytest.approx(100.0 * 0.7619, rel=1e-4)

    def test_bcbs_no_sme_factor(self) -> None:
        """BCBS: no SME supporting factor in standardized approach."""
        rw_plain = get_corporate_risk_weight(
            CreditQualityStep.UNRATED, jurisdiction=Jurisdiction.BCBS,
        )
        rw_sme = get_corporate_risk_weight(
            CreditQualityStep.UNRATED, jurisdiction=Jurisdiction.BCBS, is_sme=True,
        )
        assert rw_plain == rw_sme  # No factor applied


class TestResidentialREJurisdictions:
    """Test jurisdiction-specific RRE risk weight treatments."""

    @pytest.mark.parametrize("ltv,expected", [
        (0.50, 20.0), (0.60, 25.0), (0.70, 30.0),
        (0.80, 35.0), (0.90, 40.0), (1.00, 50.0), (1.10, 70.0),
    ])
    def test_bcbs_whole_loan_table(self, ltv: float, expected: float) -> None:
        """BCBS CRE20 Table 12 whole-loan RRE risk weights."""
        rw = get_residential_re_risk_weight(ltv=ltv, jurisdiction=Jurisdiction.BCBS)
        assert rw == expected

    def test_india_rbi_low_ltv(self) -> None:
        """RBI: LTV <= 80% → 20% RW for residential housing."""
        rw = get_residential_re_risk_weight(ltv=0.70, jurisdiction=Jurisdiction.INDIA)
        assert rw == 20.0

    def test_india_rbi_high_ltv(self) -> None:
        """RBI: LTV > 80% → 35% RW."""
        rw = get_residential_re_risk_weight(ltv=0.85, jurisdiction=Jurisdiction.INDIA)
        assert rw == 35.0


class TestOutputFloorJurisdictions:
    """Test output floor phase-in schedules by jurisdiction."""

    def test_eu_2025_50pct(self) -> None:
        """EU CRR3 Art. 92a: 50% from 1 Jan 2025."""
        floor = get_output_floor_pct(Jurisdiction.EU, date(2025, 6, 1))
        assert floor == pytest.approx(0.50)

    def test_eu_2030_fully_phased(self) -> None:
        """EU CRR3: 72.5% from 1 Jan 2030."""
        floor = get_output_floor_pct(Jurisdiction.EU, date(2030, 6, 1))
        assert floor == pytest.approx(0.725)

    def test_uk_delayed_start(self) -> None:
        """UK PRA PS1/26: phase-in starts 1 Jan 2027 (later than EU)."""
        floor = get_output_floor_pct(Jurisdiction.UK, date(2027, 6, 1))
        assert floor == pytest.approx(0.50)

    def test_india_rbi_80pct_floor(self) -> None:
        """RBI: 80% output floor (more conservative than BCBS 72.5%)."""
        floor = get_output_floor_pct(Jurisdiction.INDIA, date(2030, 6, 1))
        assert floor == pytest.approx(0.80)

    def test_australia_apra_immediate_72_5(self) -> None:
        """APRA: 72.5% from 1 Jan 2023 (immediate, no phase-in)."""
        floor = get_output_floor_pct(Jurisdiction.AUSTRALIA, date(2023, 6, 1))
        assert floor == pytest.approx(0.725)

    def test_canada_osfi_immediate_72_5(self) -> None:
        """OSFI: 72.5% from Q2 2024 (immediate, no phase-in)."""
        floor = get_output_floor_pct(Jurisdiction.CANADA, date(2024, 7, 1))
        assert floor == pytest.approx(0.725)


class TestSADispatcher:
    """Test the master SA risk weight dispatcher across jurisdictions."""

    def test_mdb_zero_pct(self) -> None:
        """Qualifying MDBs get 0% risk weight."""
        rw = assign_sa_risk_weight(SAExposureClass.MDB)
        assert rw == 0.0

    def test_retail_regulatory_75pct(self) -> None:
        """Regulatory retail always 75% across jurisdictions."""
        for j in [Jurisdiction.BCBS, Jurisdiction.EU, Jurisdiction.UK]:
            rw = assign_sa_risk_weight(
                SAExposureClass.RETAIL_REGULATORY, jurisdiction=j,
            )
            assert rw == 75.0

    def test_subordinated_debt_150pct(self) -> None:
        """Subordinated debt always 150% per CRE20.49."""
        rw = assign_sa_risk_weight(SAExposureClass.SUBORDINATED_DEBT)
        assert rw == 150.0

    def test_defaulted_provisions_threshold(self) -> None:
        """Defaulted: 150% if provisions < 20%, 100% if >= 20%."""
        rw_low = assign_sa_risk_weight(
            SAExposureClass.DEFAULTED, specific_provisions_pct=0.10,
        )
        rw_high = assign_sa_risk_weight(
            SAExposureClass.DEFAULTED, specific_provisions_pct=0.25,
        )
        assert rw_low == 150.0
        assert rw_high == 100.0
