"""Tests for IRB risk weight formulas — BCBS d424 CRE31."""


import pytest

from creditriskengine.rwa.irb.formulas import (
    PD_FLOOR,
    asset_correlation_corporate,
    asset_correlation_other_retail,
    asset_correlation_qrre,
    asset_correlation_residential_mortgage,
    irb_capital_requirement_k,
    irb_risk_weight,
    maturity_adjustment,
    sme_firm_size_adjustment,
)


class TestAssetCorrelation:
    """BCBS CRE31.5-31.10."""

    def test_corporate_boundaries(self) -> None:
        # R -> 0.24 as PD -> 0
        r_low_pd = asset_correlation_corporate(0.0003)
        assert r_low_pd == pytest.approx(0.24, abs=0.005)

        # R -> 0.12 as PD -> 1
        r_high_pd = asset_correlation_corporate(1.0)
        assert r_high_pd == pytest.approx(0.12, abs=0.001)

    def test_corporate_monotone_decreasing(self) -> None:
        pds = [0.001, 0.01, 0.05, 0.10, 0.20]
        rs = [asset_correlation_corporate(pd) for pd in pds]
        for i in range(len(rs) - 1):
            assert rs[i] > rs[i + 1], f"R not decreasing: {rs[i]} <= {rs[i+1]}"

    def test_residential_mortgage_fixed(self) -> None:
        assert asset_correlation_residential_mortgage(0.01) == 0.15
        assert asset_correlation_residential_mortgage(0.10) == 0.15

    def test_qrre_fixed(self) -> None:
        assert asset_correlation_qrre(0.05) == 0.04

    def test_other_retail_range(self) -> None:
        r_low = asset_correlation_other_retail(0.0003)
        r_high = asset_correlation_other_retail(1.0)
        assert r_low == pytest.approx(0.16, abs=0.005)
        assert r_high == pytest.approx(0.03, abs=0.001)


class TestSMEFirmSizeAdjustment:
    """BCBS CRE31.6."""

    def test_max_adjustment_at_5m(self) -> None:
        adj = sme_firm_size_adjustment(5.0)
        assert adj == pytest.approx(-0.04, abs=1e-6)

    def test_no_adjustment_at_50m(self) -> None:
        adj = sme_firm_size_adjustment(50.0)
        assert adj == pytest.approx(0.0, abs=1e-6)

    def test_floor_below_5m(self) -> None:
        assert sme_firm_size_adjustment(3.0) == sme_firm_size_adjustment(5.0)

    def test_cap_above_50m(self) -> None:
        assert sme_firm_size_adjustment(100.0) == sme_firm_size_adjustment(50.0)

    def test_midpoint(self) -> None:
        adj = sme_firm_size_adjustment(27.5)
        assert adj == pytest.approx(-0.02, abs=1e-6)


class TestMaturityAdjustment:
    """BCBS CRE31.7."""

    def test_at_2_5_years(self) -> None:
        # At M=2.5: numerator = 1, MA = 1 / (1 - 1.5*b) > 1
        ma = maturity_adjustment(0.01, 2.5)
        assert ma > 1.0

    def test_longer_maturity_higher(self) -> None:
        ma_short = maturity_adjustment(0.01, 1.0)
        ma_long = maturity_adjustment(0.01, 5.0)
        assert ma_long > ma_short

    def test_pd_floor_applied(self) -> None:
        ma = maturity_adjustment(0.0001, 2.5)
        ma_floor = maturity_adjustment(PD_FLOOR, 2.5)
        assert ma == pytest.approx(ma_floor, rel=1e-6)


class TestIRBCapitalRequirement:
    """BCBS CRE31.4."""

    def test_k_non_negative(self) -> None:
        k = irb_capital_requirement_k(0.01, 0.45, 0.20)
        assert k >= 0.0

    def test_k_increases_with_pd(self) -> None:
        k_low = irb_capital_requirement_k(0.001, 0.45, 0.20)
        k_high = irb_capital_requirement_k(0.05, 0.45, 0.20)
        assert k_high > k_low

    def test_k_increases_with_lgd(self) -> None:
        k_low = irb_capital_requirement_k(0.01, 0.25, 0.20)
        k_high = irb_capital_requirement_k(0.01, 0.45, 0.20)
        assert k_high > k_low

    def test_known_value_corporate(self) -> None:
        # PD=1%, LGD=45%, R=corporate correlation
        r = asset_correlation_corporate(0.01)
        k = irb_capital_requirement_k(0.01, 0.45, r)
        # K should be roughly in the range 3-8% for typical corporate
        assert 0.03 < k < 0.10

    def test_invalid_correlation_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="Correlation must be in \\(0, 1\\)"):
            irb_capital_requirement_k(0.01, 0.45, 0.0)

    def test_invalid_correlation_one_raises(self) -> None:
        with pytest.raises(ValueError, match="Correlation must be in \\(0, 1\\)"):
            irb_capital_requirement_k(0.01, 0.45, 1.0)

    def test_invalid_correlation_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="Correlation must be in \\(0, 1\\)"):
            irb_capital_requirement_k(0.01, 0.45, -0.1)


class TestIRBRiskWeight:
    """Full IRB risk weight calculation."""

    def test_corporate_typical(self) -> None:
        rw = irb_risk_weight(pd=0.01, lgd=0.45, asset_class="corporate")
        # Typical corporate RW for PD=1%, LGD=45% is roughly 60-100%
        assert 50.0 < rw < 120.0

    def test_pd_floor_applied(self) -> None:
        rw = irb_risk_weight(pd=0.0001, lgd=0.45, asset_class="corporate")
        rw_floor = irb_risk_weight(pd=PD_FLOOR, lgd=0.45, asset_class="corporate")
        assert rw == pytest.approx(rw_floor, rel=1e-6)

    def test_sme_lower_than_general_corporate(self) -> None:
        rw_general = irb_risk_weight(pd=0.01, lgd=0.45, asset_class="corporate")
        rw_sme = irb_risk_weight(
            pd=0.01, lgd=0.45, asset_class="corporate",
            turnover_eur_millions=10.0,
        )
        assert rw_sme < rw_general

    def test_residential_mortgage(self) -> None:
        rw = irb_risk_weight(pd=0.01, lgd=0.20, asset_class="residential_mortgage")
        assert rw > 0.0

    def test_qrre_transactor_no_scalar(self) -> None:
        # QRRE transactor scalar was removed in Basel III d424
        rw = irb_risk_weight(pd=0.02, lgd=0.80, asset_class="qrre")
        rw_trans = irb_risk_weight(
            pd=0.02, lgd=0.80, asset_class="qrre", is_qrre_transactor=True
        )
        assert rw_trans == pytest.approx(rw, rel=1e-6)

    def test_other_retail(self) -> None:
        rw = irb_risk_weight(pd=0.02, lgd=0.50, asset_class="other_retail")
        assert rw > 0.0

    def test_defaulted_returns_zero(self) -> None:
        assert irb_risk_weight(pd=1.0, lgd=0.45, asset_class="corporate") == 0.0

    def test_unknown_asset_class_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown asset class"):
            irb_risk_weight(pd=0.01, lgd=0.45, asset_class="invalid")

    def test_maturity_effect(self) -> None:
        rw_short = irb_risk_weight(
            pd=0.01, lgd=0.45, asset_class="corporate", maturity=1.0
        )
        rw_long = irb_risk_weight(
            pd=0.01, lgd=0.45, asset_class="corporate", maturity=5.0
        )
        assert rw_long > rw_short
