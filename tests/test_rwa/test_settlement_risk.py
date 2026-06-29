"""Tests for settlement / failed-trade risk capital (BCBS CRE70)."""

import pytest

from creditriskengine.rwa.settlement_risk import (
    dvp_settlement_capital,
    dvp_settlement_multiplier,
    non_dvp_risk_weight,
)


class TestDvPMultiplier:
    def test_below_5_days_zero(self) -> None:
        assert dvp_settlement_multiplier(4) == pytest.approx(0.0)

    def test_5_to_15_days(self) -> None:
        assert dvp_settlement_multiplier(5) == pytest.approx(0.08)
        assert dvp_settlement_multiplier(15) == pytest.approx(0.08)

    def test_16_to_30_days(self) -> None:
        assert dvp_settlement_multiplier(16) == pytest.approx(0.50)
        assert dvp_settlement_multiplier(30) == pytest.approx(0.50)

    def test_31_to_45_days(self) -> None:
        assert dvp_settlement_multiplier(31) == pytest.approx(0.75)
        assert dvp_settlement_multiplier(45) == pytest.approx(0.75)

    def test_46_plus_days(self) -> None:
        assert dvp_settlement_multiplier(46) == pytest.approx(1.00)
        assert dvp_settlement_multiplier(200) == pytest.approx(1.00)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            dvp_settlement_multiplier(-1)


class TestDvPSettlementCapital:
    def test_charge_and_rwa(self) -> None:
        # 20 days late -> 50% of 1000 = 500 charge; RWA = 6250.
        r = dvp_settlement_capital(1000.0, business_days_late=20)
        assert r.multiplier == pytest.approx(0.50)
        assert r.capital_charge == pytest.approx(500.0)
        assert r.rwa_equivalent == pytest.approx(6250.0)

    def test_below_threshold_zero_charge(self) -> None:
        r = dvp_settlement_capital(1000.0, business_days_late=3)
        assert r.capital_charge == pytest.approx(0.0)
        assert r.rwa_equivalent == pytest.approx(0.0)

    def test_full_multiplier(self) -> None:
        r = dvp_settlement_capital(800.0, business_days_late=60)
        assert r.multiplier == pytest.approx(1.0)
        assert r.capital_charge == pytest.approx(800.0)

    def test_negative_exposure_raises(self) -> None:
        with pytest.raises(ValueError, match="positive_current_exposure"):
            dvp_settlement_capital(-1.0, business_days_late=10)


class TestNonDvP:
    def test_within_4_days_counterparty_rw(self) -> None:
        assert non_dvp_risk_weight(2, counterparty_risk_weight=1.0) == pytest.approx(1.0)

    def test_before_second_leg_counterparty_rw(self) -> None:
        assert non_dvp_risk_weight(-3, counterparty_risk_weight=0.5) == pytest.approx(0.5)

    def test_5_plus_days_full_deduction(self) -> None:
        assert non_dvp_risk_weight(5, counterparty_risk_weight=0.2) == pytest.approx(12.50)

    def test_negative_rw_raises(self) -> None:
        with pytest.raises(ValueError, match="counterparty_risk_weight"):
            non_dvp_risk_weight(2, counterparty_risk_weight=-0.1)
