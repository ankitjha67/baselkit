"""Tests for equity investments in funds (BCBS CRE60)."""

import pytest

from creditriskengine.rwa.equity_in_funds import (
    FundApproach,
    fall_back_rwa,
    fund_average_risk_weight,
    fund_leverage,
    look_through_rwa,
    mandate_based_rwa,
)


class TestFundAverageRiskWeight:
    def test_basic(self) -> None:
        # 750 RWA on 1000 assets -> 75% average risk weight.
        assert fund_average_risk_weight(750.0, 1000.0) == pytest.approx(0.75)

    def test_invalid_assets(self) -> None:
        with pytest.raises(ValueError, match="fund_total_assets must be positive"):
            fund_average_risk_weight(750.0, 0.0)

    def test_negative_rwa(self) -> None:
        with pytest.raises(ValueError, match="underlying_rwa must be non-negative"):
            fund_average_risk_weight(-1.0, 1000.0)


class TestFundLeverage:
    def test_basic(self) -> None:
        assert fund_leverage(1500.0, 1000.0) == pytest.approx(1.5)

    def test_invalid_equity(self) -> None:
        with pytest.raises(ValueError, match="total_equity must be positive"):
            fund_leverage(1500.0, 0.0)

    def test_negative_assets(self) -> None:
        with pytest.raises(ValueError, match="total_assets must be non-negative"):
            fund_leverage(-1.0, 1000.0)


class TestLookThrough:
    def test_basic(self) -> None:
        # avg_RW 75% x leverage 1.5 = 112.5% -> RWA = 1.125 * 200 = 225.
        r = look_through_rwa(200.0, average_risk_weight=0.75, leverage=1.5)
        assert r.approach == FundApproach.LOOK_THROUGH
        assert r.effective_risk_weight == pytest.approx(1.125)
        assert r.rwa == pytest.approx(225.0)
        assert r.capped is False

    def test_cap_binds(self) -> None:
        # avg_RW 200% x leverage 8 = 1600% -> capped at 1250%.
        r = look_through_rwa(100.0, average_risk_weight=2.0, leverage=8.0)
        assert r.effective_risk_weight == pytest.approx(12.50)
        assert r.rwa == pytest.approx(1250.0)
        assert r.capped is True

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="inputs must be non-negative"):
            look_through_rwa(-1.0, 0.75, 1.5)


class TestMandateBased:
    def test_basic(self) -> None:
        r = mandate_based_rwa(100.0, average_risk_weight=1.0, leverage=2.0)
        assert r.approach == FundApproach.MANDATE_BASED
        assert r.rwa == pytest.approx(200.0)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="inputs must be non-negative"):
            mandate_based_rwa(100.0, -0.1, 2.0)


class TestFallBack:
    def test_1250_percent(self) -> None:
        r = fall_back_rwa(100.0)
        assert r.approach == FundApproach.FALL_BACK
        assert r.effective_risk_weight == pytest.approx(12.50)
        assert r.rwa == pytest.approx(1250.0)
        assert r.capped is True

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="investment must be non-negative"):
            fall_back_rwa(-1.0)
