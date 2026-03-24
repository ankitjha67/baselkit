"""Regulatory back-testing against Basel III worked examples (CRE31-32)."""

from datetime import date

import pytest

from creditriskengine.core.types import CreditQualityStep, Jurisdiction
from creditriskengine.rwa.irb.formulas import (
    PD_FLOOR,
    asset_correlation_corporate,
    asset_correlation_other_retail,
    asset_correlation_qrre,
    asset_correlation_residential_mortgage,
    irb_risk_weight,
    maturity_adjustment,
    sme_firm_size_adjustment,
)
from creditriskengine.rwa.output_floor import OutputFloorCalculator, get_output_floor_pct
from creditriskengine.rwa.standardized.credit_risk_sa import (
    get_bank_risk_weight,
    get_commercial_re_risk_weight,
    get_corporate_risk_weight,
    get_residential_re_risk_weight,
    get_sovereign_risk_weight,
)

# ============================================================
# IRB Corporate Risk Weights — CRE31 reference ranges
# ============================================================

class TestCorporateIRBRiskWeights:
    """Verify IRB corporate risk weights against known Basel regulatory ranges.

    LGD = 45% (F-IRB supervisory LGD for senior unsecured),
    M = 2.5 years (F-IRB fixed maturity).
    """

    LGD = 0.45
    MATURITY = 2.5

    def test_pd_at_floor(self):
        """PD = 0.03% (floor). Very low PD -> high correlation -> moderate RW."""
        rw = irb_risk_weight(
            pd=PD_FLOOR, lgd=self.LGD, asset_class="corporate", maturity=self.MATURITY
        )
        # At PD floor, corporate RW is approximately 14-18%
        assert 10.0 < rw < 25.0, f"RW at PD floor: {rw:.2f}%"

    def test_pd_one_percent(self):
        """PD = 1%. Typical investment-grade/sub-investment-grade boundary."""
        rw = irb_risk_weight(
            pd=0.01, lgd=self.LGD, asset_class="corporate", maturity=self.MATURITY
        )
        # Basel III: corporate PD=1%, LGD=45%, M=2.5 -> RW approx 70-90%
        assert 60.0 < rw < 100.0, f"RW at PD=1%: {rw:.2f}%"

    def test_pd_five_percent(self):
        """PD = 5%. Sub-investment-grade exposure."""
        rw = irb_risk_weight(
            pd=0.05, lgd=self.LGD, asset_class="corporate", maturity=self.MATURITY
        )
        # PD=5%, LGD=45%, M=2.5 -> RW approx 115-160%
        assert 100.0 < rw < 180.0, f"RW at PD=5%: {rw:.2f}%"

    def test_pd_twenty_percent(self):
        """PD = 20%. Highly distressed exposure approaching default."""
        rw = irb_risk_weight(
            pd=0.20, lgd=self.LGD, asset_class="corporate", maturity=self.MATURITY
        )
        # PD=20%, LGD=45%, M=2.5 -> RW approx 160-250%
        assert 140.0 < rw < 280.0, f"RW at PD=20%: {rw:.2f}%"

    def test_rw_monotonically_increases_with_pd(self):
        """Risk weight must increase with PD (all else equal)."""
        pds = [PD_FLOOR, 0.005, 0.01, 0.05, 0.10, 0.20]
        rws = [
            irb_risk_weight(
                pd=pd, lgd=self.LGD, asset_class="corporate", maturity=self.MATURITY
            )
            for pd in pds
        ]
        for i in range(1, len(rws)):
            assert rws[i] > rws[i - 1], (
                f"RW not monotonic: RW({pds[i]})={rws[i]:.2f} "
                f"<= RW({pds[i-1]})={rws[i-1]:.2f}"
            )

    def test_defaulted_exposure_zero_rw(self):
        """PD = 100% (defaulted) -> RW = 0 (capital handled via EL provisions)."""
        rw = irb_risk_weight(
            pd=1.0, lgd=self.LGD, asset_class="corporate", maturity=self.MATURITY
        )
        assert rw == 0.0


# ============================================================
# Correlation bounds — CRE31.5-31.10
# ============================================================

class TestCorrelationBounds:
    """Asset correlation functions must stay within their regulatory bounds."""

    def test_corporate_correlation_high_pd(self):
        """High PD -> correlation approaches lower bound 0.12."""
        r = asset_correlation_corporate(0.50)
        assert 0.12 <= r <= 0.13

    def test_corporate_correlation_low_pd(self):
        """Low PD -> correlation approaches upper bound 0.24."""
        r = asset_correlation_corporate(PD_FLOOR)
        assert 0.23 <= r <= 0.24

    def test_corporate_correlation_range(self):
        """Corporate correlation is always in [0.12, 0.24]."""
        for pd in [0.0003, 0.001, 0.01, 0.05, 0.10, 0.30, 0.99]:
            r = asset_correlation_corporate(pd)
            assert 0.12 <= r <= 0.24, f"R({pd})={r} outside [0.12, 0.24]"

    def test_residential_mortgage_fixed(self):
        """Residential mortgage correlation is fixed at 0.15."""
        assert asset_correlation_residential_mortgage(0.01) == 0.15
        assert asset_correlation_residential_mortgage(0.20) == 0.15

    def test_qrre_fixed(self):
        """QRRE correlation is fixed at 0.04."""
        assert asset_correlation_qrre(0.01) == 0.04
        assert asset_correlation_qrre(0.20) == 0.04

    def test_other_retail_range(self):
        """Other retail correlation in [0.03, 0.16]."""
        for pd in [0.0003, 0.01, 0.05, 0.10, 0.50]:
            r = asset_correlation_other_retail(pd)
            assert 0.03 <= r <= 0.16, f"Other retail R({pd})={r} outside [0.03, 0.16]"

    def test_sme_adjustment_small_firm(self):
        """SME adjustment at EUR 5M turnover = -0.04."""
        adj = sme_firm_size_adjustment(5.0)
        assert adj == pytest.approx(-0.04, abs=1e-9)

    def test_sme_adjustment_large_firm(self):
        """SME adjustment at EUR 50M turnover = 0.0 (no adjustment)."""
        adj = sme_firm_size_adjustment(50.0)
        assert adj == pytest.approx(0.0, abs=1e-9)

    def test_sme_adjustment_mid_firm(self):
        """SME adjustment at EUR 27.5M (midpoint) = -0.02."""
        adj = sme_firm_size_adjustment(27.5)
        assert adj == pytest.approx(-0.02, abs=1e-9)

    def test_sme_floor_at_5m(self):
        """Turnover below EUR 5M is floored to EUR 5M."""
        assert sme_firm_size_adjustment(1.0) == sme_firm_size_adjustment(5.0)


# ============================================================
# Maturity adjustment — CRE31.7
# ============================================================

class TestMaturityAdjustment:
    """Maturity adjustment factor tests."""

    def test_at_2_5_years(self):
        """At M=2.5, the (M-2.5)*b numerator term is zero."""
        # MA = 1 / (1 - 1.5*b), which is > 1
        ma = maturity_adjustment(0.01, 2.5)
        assert ma > 1.0

    def test_shorter_maturity_reduces(self):
        """Shorter maturity (M<2.5) should give a lower MA than M=2.5."""
        ma_short = maturity_adjustment(0.01, 1.0)
        ma_base = maturity_adjustment(0.01, 2.5)
        assert ma_short < ma_base

    def test_longer_maturity_increases(self):
        """Longer maturity (M>2.5) should give a higher MA than M=2.5."""
        ma_long = maturity_adjustment(0.01, 5.0)
        ma_base = maturity_adjustment(0.01, 2.5)
        assert ma_long > ma_base


# ============================================================
# SA risk weight tables — CRE20 exact values
# ============================================================

class TestSATablesExact:
    """SA risk weight tables must exactly match BCBS CRE20."""

    def test_sovereign_table(self):
        """BCBS CRE20.7 Table 1 — sovereign risk weights."""
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
            rw = get_sovereign_risk_weight(cqs)
            assert rw == expected_rw, f"Sovereign CQS={cqs}: got {rw}, expected {expected_rw}"

    def test_bank_ecra_table(self):
        """BCBS CRE20.15-18 Table 4 — bank ECRA risk weights."""
        expected = {
            CreditQualityStep.CQS_1: 20.0,
            CreditQualityStep.CQS_2: 30.0,
            CreditQualityStep.CQS_3: 50.0,
            CreditQualityStep.CQS_4: 100.0,
            CreditQualityStep.CQS_5: 100.0,
            CreditQualityStep.CQS_6: 150.0,
            CreditQualityStep.UNRATED: 50.0,
        }
        for cqs, expected_rw in expected.items():
            rw = get_bank_risk_weight(cqs=cqs)
            assert rw == expected_rw, f"Bank ECRA CQS={cqs}: got {rw}, expected {expected_rw}"

    def test_bank_scra_table(self):
        """BCBS CRE20.19-21 — bank SCRA risk weights."""
        assert get_bank_risk_weight(scra_grade="A") == 40.0
        assert get_bank_risk_weight(scra_grade="B") == 75.0
        assert get_bank_risk_weight(scra_grade="C") == 150.0

    def test_corporate_table(self):
        """BCBS CRE20.28-32 Table 7 — corporate risk weights."""
        expected = {
            CreditQualityStep.CQS_1: 20.0,
            CreditQualityStep.CQS_2: 50.0,
            CreditQualityStep.CQS_3: 75.0,
            CreditQualityStep.CQS_4: 100.0,
            CreditQualityStep.CQS_5: 150.0,
            CreditQualityStep.CQS_6: 150.0,
            CreditQualityStep.UNRATED: 100.0,
        }
        for cqs, expected_rw in expected.items():
            rw = get_corporate_risk_weight(cqs)
            assert rw == expected_rw, f"Corp CQS={cqs}: got {rw}, expected {expected_rw}"

    def test_rre_whole_loan_table(self):
        """BCBS CRE20.73 Table 12 — RRE whole-loan risk weights."""
        # Test representative LTV from each bucket
        test_cases = [
            (0.40, 20.0),   # LTV <= 50%
            (0.55, 25.0),   # 50% < LTV <= 60%
            (0.65, 30.0),   # 60% < LTV <= 70%
            (0.75, 35.0),   # 70% < LTV <= 80%
            (0.85, 40.0),   # 80% < LTV <= 90%
            (0.95, 50.0),   # 90% < LTV <= 100%
            (1.10, 70.0),   # LTV > 100%
        ]
        for ltv, expected_rw in test_cases:
            rw = get_residential_re_risk_weight(ltv=ltv)
            assert rw == expected_rw, f"RRE LTV={ltv}: got {rw}, expected {expected_rw}"

    def test_rre_cashflow_dependent_table(self):
        """BCBS CRE20.78 Table 13 — RRE cashflow-dependent risk weights."""
        test_cases = [
            (0.40, 30.0),
            (0.55, 35.0),
            (0.65, 45.0),
            (0.75, 50.0),
            (0.85, 60.0),
            (0.95, 75.0),
            (1.10, 105.0),
        ]
        for ltv, expected_rw in test_cases:
            rw = get_residential_re_risk_weight(ltv=ltv, is_cashflow_dependent=True)
            assert rw == expected_rw, f"RRE CF LTV={ltv}: got {rw}, expected {expected_rw}"

    def test_cre_not_cashflow_dependent(self):
        """BCBS CRE20.89 Table 14 — CRE not dependent on cashflows."""
        # LTV <= 60%: min(60%, counterparty_rw)
        assert get_commercial_re_risk_weight(ltv=0.50, counterparty_rw=100.0) == 60.0
        # LTV <= 80%: 75%
        assert get_commercial_re_risk_weight(ltv=0.70, counterparty_rw=100.0) == 75.0
        # LTV > 80%: counterparty RW
        assert get_commercial_re_risk_weight(ltv=0.90, counterparty_rw=100.0) == 100.0

    def test_cre_adc(self):
        """BCBS CRE20.97 — ADC = 150%, presold = 100%."""
        assert get_commercial_re_risk_weight(ltv=1.0, is_adc=True) == 150.0
        assert get_commercial_re_risk_weight(
            ltv=1.0, is_adc=True, is_presold_residential=True
        ) == 100.0


# ============================================================
# Output floor dates — RBC25
# ============================================================

class TestOutputFloorDates:
    """Output floor phase-in schedule tests."""

    def test_bcbs_2023_50pct(self):
        """BCBS: 50% floor from 2023-01-01."""
        pct = get_output_floor_pct(Jurisdiction.BCBS, date(2023, 6, 30))
        assert pct == pytest.approx(0.50)

    def test_bcbs_2028_fully_phased_in(self):
        """BCBS: 72.5% floor from 2028-01-01."""
        pct = get_output_floor_pct(Jurisdiction.BCBS, date(2028, 6, 30))
        assert pct == pytest.approx(0.725)

    def test_eu_delayed_start(self):
        """EU CRR3: starts 2025, not 2023. Before start -> 0%."""
        pct = get_output_floor_pct(Jurisdiction.EU, date(2024, 12, 31))
        assert pct == pytest.approx(0.0)

    def test_eu_2025_50pct(self):
        """EU CRR3: 50% from 2025-01-01."""
        pct = get_output_floor_pct(Jurisdiction.EU, date(2025, 6, 30))
        assert pct == pytest.approx(0.50)

    def test_uk_delayed_start(self):
        """UK PRA: starts 2027. Before start -> 0%."""
        pct = get_output_floor_pct(Jurisdiction.UK, date(2026, 12, 31))
        assert pct == pytest.approx(0.0)

    def test_india_80pct_flat(self):
        """India RBI: 80% floor, no phase-in."""
        pct = get_output_floor_pct(Jurisdiction.INDIA, date(2025, 1, 1))
        assert pct == pytest.approx(0.80)

    def test_australia_725_immediate(self):
        """Australia APRA: 72.5% from 2023."""
        pct = get_output_floor_pct(Jurisdiction.AUSTRALIA, date(2023, 7, 1))
        assert pct == pytest.approx(0.725)

    def test_output_floor_calculator_binding(self):
        """When floor is binding, floored_rwa = floor_pct * sa_rwa."""
        calc = OutputFloorCalculator(Jurisdiction.BCBS, date(2028, 6, 30))
        result = calc.calculate(irb_rwa=500.0, sa_rwa=1000.0)
        # 72.5% * 1000 = 725 > 500 -> binding
        assert result["is_binding"] is True
        assert result["floored_rwa"] == pytest.approx(725.0)

    def test_output_floor_calculator_not_binding(self):
        """When IRB RWA exceeds floor, floored_rwa = irb_rwa."""
        calc = OutputFloorCalculator(Jurisdiction.BCBS, date(2028, 6, 30))
        result = calc.calculate(irb_rwa=800.0, sa_rwa=1000.0)
        # 72.5% * 1000 = 725 < 800 -> not binding
        assert result["is_binding"] is False
        assert result["floored_rwa"] == pytest.approx(800.0)
