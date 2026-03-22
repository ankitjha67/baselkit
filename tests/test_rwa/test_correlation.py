"""Tests for IRB asset correlation routing — BCBS d424, CRE31.5-31.10."""

import pytest

from creditriskengine.rwa.irb.correlation import (
    asset_correlation_corporate,
    asset_correlation_other_retail,
    asset_correlation_qrre,
    asset_correlation_residential_mortgage,
    get_asset_correlation,
    sme_firm_size_adjustment,
)


class TestGetAssetCorrelation:
    """Test the unified routing function."""

    def test_corporate(self) -> None:
        r = get_asset_correlation("corporate", 0.01)
        assert 0.12 <= r <= 0.24

    def test_sovereign(self) -> None:
        r = get_asset_correlation("sovereign", 0.01)
        # Same formula as corporate
        expected = asset_correlation_corporate(0.01)
        assert r == pytest.approx(expected)

    def test_bank(self) -> None:
        r = get_asset_correlation("bank", 0.05)
        expected = asset_correlation_corporate(0.05)
        assert r == pytest.approx(expected)

    def test_residential_mortgage(self) -> None:
        r = get_asset_correlation("residential_mortgage", 0.02)
        expected = asset_correlation_residential_mortgage(0.02)
        assert r == pytest.approx(expected)

    def test_qrre(self) -> None:
        r = get_asset_correlation("qrre", 0.03)
        expected = asset_correlation_qrre(0.03)
        assert r == pytest.approx(expected)

    def test_other_retail(self) -> None:
        r = get_asset_correlation("other_retail", 0.05)
        expected = asset_correlation_other_retail(0.05)
        assert r == pytest.approx(expected)

    def test_unknown_asset_class_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown asset class"):
            get_asset_correlation("unknown_class", 0.01)

    def test_corporate_with_sme_adjustment(self) -> None:
        r_no_sme = get_asset_correlation("corporate", 0.01)
        r_sme = get_asset_correlation("corporate", 0.01, turnover_eur_millions=10.0)
        # SME adjustment is negative, so r_sme < r_no_sme
        assert r_sme < r_no_sme

    def test_corporate_sme_at_50m_no_adjustment(self) -> None:
        r_no_sme = get_asset_correlation("corporate", 0.01)
        r_sme_50 = get_asset_correlation("corporate", 0.01, turnover_eur_millions=50.0)
        # At 50M, adjustment is 0
        assert r_sme_50 == pytest.approx(r_no_sme)

    def test_corporate_sme_at_5m_max_adjustment(self) -> None:
        r_no_sme = get_asset_correlation("corporate", 0.01)
        r_sme_5 = get_asset_correlation("corporate", 0.01, turnover_eur_millions=5.0)
        # Max reduction is 0.04
        assert r_sme_5 == pytest.approx(r_no_sme - 0.04)

    def test_sovereign_ignores_turnover(self) -> None:
        # turnover_eur_millions should be ignored for sovereign
        r1 = get_asset_correlation("sovereign", 0.01)
        r2 = get_asset_correlation("sovereign", 0.01, turnover_eur_millions=10.0)
        assert r1 == pytest.approx(r2)

    def test_bank_ignores_turnover(self) -> None:
        r1 = get_asset_correlation("bank", 0.01)
        r2 = get_asset_correlation("bank", 0.01, turnover_eur_millions=10.0)
        assert r1 == pytest.approx(r2)

    def test_corporate_sme_below_5m_floors_at_5m(self) -> None:
        r_5 = get_asset_correlation("corporate", 0.01, turnover_eur_millions=5.0)
        r_1 = get_asset_correlation("corporate", 0.01, turnover_eur_millions=1.0)
        # Below 5M is floored at 5M, so same adjustment
        assert r_1 == pytest.approx(r_5)

    def test_correlation_non_negative(self) -> None:
        # Even with max SME adjustment, correlation should be >= 0
        r = get_asset_correlation("corporate", 0.99, turnover_eur_millions=5.0)
        assert r >= 0.0


class TestAssetCorrelationCorporate:
    """Test corporate/sovereign/bank correlation formula."""

    def test_low_pd_high_correlation(self) -> None:
        r = asset_correlation_corporate(0.0003)
        assert r > 0.20

    def test_high_pd_low_correlation(self) -> None:
        r = asset_correlation_corporate(0.50)
        assert r < 0.15

    def test_range(self) -> None:
        for pd in [0.0003, 0.01, 0.05, 0.10, 0.50, 1.0]:
            r = asset_correlation_corporate(pd)
            assert 0.12 <= r <= 0.24


class TestSMEFirmSizeAdjustment:
    """Test SME firm-size adjustment."""

    def test_at_5m(self) -> None:
        adj = sme_firm_size_adjustment(5.0)
        assert adj == pytest.approx(-0.04)

    def test_at_50m(self) -> None:
        adj = sme_firm_size_adjustment(50.0)
        assert adj == pytest.approx(0.0)

    def test_at_27_5m(self) -> None:
        adj = sme_firm_size_adjustment(27.5)
        assert adj == pytest.approx(-0.02)

    def test_below_5m_floors(self) -> None:
        adj = sme_firm_size_adjustment(1.0)
        assert adj == pytest.approx(-0.04)

    def test_above_50m_caps(self) -> None:
        adj = sme_firm_size_adjustment(100.0)
        assert adj == pytest.approx(0.0)
