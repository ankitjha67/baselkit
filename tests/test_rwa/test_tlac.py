"""Tests for the TLAC framework (FSB / BCBS)."""

import pytest

from creditriskengine.rwa.tlac import (
    TLAC_LEVERAGE_MINIMUM,
    TLAC_RWA_MINIMUM,
    available_tlac,
    tlac_ratios,
)


class TestConstants:
    def test_steady_state_minimums(self) -> None:
        assert pytest.approx(0.18) == TLAC_RWA_MINIMUM
        assert pytest.approx(0.0675) == TLAC_LEVERAGE_MINIMUM


class TestAvailableTLAC:
    def test_excludes_buffer_cet1(self) -> None:
        # 100+20+30+50 = 200; buffer 3.5% of 1000 = 35 -> 165.
        a = available_tlac(100.0, 20.0, 30.0, 50.0, buffer_requirement_pct=0.035, rwa=1000.0)
        assert a == pytest.approx(165.0)

    def test_floored_at_zero(self) -> None:
        a = available_tlac(10.0, 0.0, 0.0, 0.0, buffer_requirement_pct=0.50, rwa=1000.0)
        assert a == pytest.approx(0.0)

    def test_negative_capital_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            available_tlac(-1.0, 0.0, 0.0, 0.0, 0.035, 1000.0)

    def test_invalid_buffer_pct(self) -> None:
        with pytest.raises(ValueError, match="buffer_requirement_pct"):
            available_tlac(100.0, 0.0, 0.0, 0.0, 1.5, 1000.0)


class TestTLACRatios:
    def test_compliant(self) -> None:
        r = tlac_ratios(700.0, rwa=1000.0, leverage_exposure=10000.0)
        assert r.rwa_ratio == pytest.approx(0.70)
        assert r.leverage_ratio == pytest.approx(0.07)
        assert r.is_compliant is True
        assert r.rwa_shortfall == pytest.approx(0.0)
        assert r.leverage_shortfall == pytest.approx(0.0)

    def test_leverage_binding_and_breached(self) -> None:
        # Meets 18% RWA but fails 6.75% leverage.
        r = tlac_ratios(180.0, rwa=1000.0, leverage_exposure=10000.0)
        assert r.rwa_ratio == pytest.approx(0.18)
        assert r.is_compliant is False
        assert r.binding_constraint == "leverage"
        # lev_required 675 - 180 = 495.
        assert r.leverage_shortfall == pytest.approx(495.0)
        assert r.rwa_shortfall == pytest.approx(0.0)

    def test_rwa_binding(self) -> None:
        # rwa=1000, leverage=2000 -> rwa_required 180 > lev_required 135.
        r = tlac_ratios(180.0, rwa=1000.0, leverage_exposure=2000.0)
        assert r.binding_constraint == "rwa"
        assert r.is_compliant is True  # 18% and 9% both met

    def test_rwa_breached(self) -> None:
        r = tlac_ratios(100.0, rwa=1000.0, leverage_exposure=2000.0)
        assert r.is_compliant is False
        assert r.rwa_shortfall == pytest.approx(80.0)  # 180 - 100

    def test_conformance_period_minimums(self) -> None:
        r = tlac_ratios(160.0, rwa=1000.0, leverage_exposure=2000.0, conformance_period=True)
        assert r.rwa_minimum == pytest.approx(0.16)
        assert r.leverage_minimum == pytest.approx(0.06)
        assert r.is_compliant is True  # 16% and 8% both met

    def test_invalid_rwa(self) -> None:
        with pytest.raises(ValueError, match="rwa must be positive"):
            tlac_ratios(100.0, rwa=0.0, leverage_exposure=2000.0)

    def test_invalid_leverage(self) -> None:
        with pytest.raises(ValueError, match="leverage_exposure must be positive"):
            tlac_ratios(100.0, rwa=1000.0, leverage_exposure=0.0)

    def test_negative_tlac(self) -> None:
        with pytest.raises(ValueError, match="tlac_available must be non-negative"):
            tlac_ratios(-1.0, rwa=1000.0, leverage_exposure=2000.0)
