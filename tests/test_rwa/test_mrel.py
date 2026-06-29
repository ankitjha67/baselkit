"""Tests for the MREL framework (BRRD2 / SRMR2)."""

import pytest

from creditriskengine.rwa.mrel import (
    assess_mrel,
    mrel_tem_requirement,
    mrel_trea_requirement,
)


class TestMRELTREARequirement:
    def test_laa_plus_rca(self) -> None:
        # LAA = 8% + 2% = 10%; RCA = 8% + 2% + 2.5% MCC = 12.5%; MREL = 22.5%.
        assert mrel_trea_requirement(0.02, market_confidence_charge=0.025) == pytest.approx(0.225)

    def test_no_mcc(self) -> None:
        assert mrel_trea_requirement(0.02) == pytest.approx(0.20)

    def test_gsii_floor_binds(self) -> None:
        # 2*(8%+0) = 16% -> floored to 18%.
        assert mrel_trea_requirement(0.0, is_gsii=True) == pytest.approx(0.18)

    def test_gsii_floor_not_binding(self) -> None:
        # 2*(8%+3%) = 22% > 18%.
        assert mrel_trea_requirement(0.03, is_gsii=True) == pytest.approx(0.22)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            mrel_trea_requirement(-0.01)


class TestMRELTEMRequirement:
    def test_default(self) -> None:
        assert mrel_tem_requirement() == pytest.approx(0.06)

    def test_with_leverage_p2r(self) -> None:
        assert mrel_tem_requirement(leverage_p2r=0.01) == pytest.approx(0.08)

    def test_gsii_floor_binds(self) -> None:
        assert mrel_tem_requirement(is_gsii=True) == pytest.approx(0.0675)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            mrel_tem_requirement(-0.01)


class TestAssessMREL:
    def test_tem_binding_and_breached(self) -> None:
        # req_trea=22.5% -> 225; req_tem=6% -> 300. TEM binds.
        r = assess_mrel(250.0, trea=1000.0, tem=5000.0, p2r=0.02, market_confidence_charge=0.025)
        assert r.binding_constraint == "tem"
        assert r.is_compliant is False
        assert r.tem_shortfall == pytest.approx(50.0)  # 300 - 250
        assert r.trea_shortfall == pytest.approx(0.0)

    def test_compliant(self) -> None:
        r = assess_mrel(400.0, trea=1000.0, tem=5000.0, p2r=0.02, market_confidence_charge=0.025)
        # 400/1000=40% >= 22.5%; 400/5000=8% >= 6%.
        assert r.is_compliant is True

    def test_trea_binding(self) -> None:
        # High TREA requirement, small TEM -> TREA binds.
        r = assess_mrel(300.0, trea=1000.0, tem=1000.0, p2r=0.05, market_confidence_charge=0.03)
        # req_trea=2*(8%+5%)+3% = 29%; req_tem=6%. TREA amt 290 > TEM amt 60.
        assert r.binding_constraint == "trea"
        assert r.required_trea_pct == pytest.approx(0.29)

    def test_gsii_uses_floors(self) -> None:
        r = assess_mrel(
            175.0, trea=1000.0, tem=5000.0, p2r=0.0, is_gsii=True
        )
        assert r.required_trea_pct == pytest.approx(0.18)
        assert r.required_tem_pct == pytest.approx(0.0675)

    def test_invalid_trea(self) -> None:
        with pytest.raises(ValueError, match="trea must be positive"):
            assess_mrel(100.0, trea=0.0, tem=5000.0, p2r=0.02)

    def test_invalid_tem(self) -> None:
        with pytest.raises(ValueError, match="tem must be positive"):
            assess_mrel(100.0, trea=1000.0, tem=0.0, p2r=0.02)

    def test_negative_eligible(self) -> None:
        with pytest.raises(ValueError, match="eligible_mrel must be non-negative"):
            assess_mrel(-1.0, trea=1000.0, tem=5000.0, p2r=0.02)
