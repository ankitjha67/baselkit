"""Tests for SFT minimum haircut floors (BCBS CRE56)."""

import pytest

from creditriskengine.rwa.sft_haircut_floors import (
    SFTCollateralType,
    SFTLeg,
    assess_sft_floor,
    minimum_haircut_floor,
    portfolio_floor_compliant,
    sft_haircut,
)


class TestMinimumHaircutFloor:
    def test_cash_and_sovereign_exempt(self) -> None:
        assert minimum_haircut_floor(SFTCollateralType.CASH) == pytest.approx(0.0)
        assert minimum_haircut_floor(SFTCollateralType.SOVEREIGN_DEBT, 7.0) == pytest.approx(0.0)

    def test_corporate_debt_buckets(self) -> None:
        assert minimum_haircut_floor(SFTCollateralType.CORPORATE_DEBT, 0.5) == pytest.approx(0.005)
        assert minimum_haircut_floor(SFTCollateralType.CORPORATE_DEBT, 3.0) == pytest.approx(0.015)
        assert minimum_haircut_floor(SFTCollateralType.CORPORATE_DEBT, 8.0) == pytest.approx(0.03)
        assert minimum_haircut_floor(SFTCollateralType.CORPORATE_DEBT, 20.0) == pytest.approx(0.04)

    def test_securitisation_buckets(self) -> None:
        assert minimum_haircut_floor(SFTCollateralType.SECURITISATION, 1.0) == pytest.approx(0.01)
        assert minimum_haircut_floor(SFTCollateralType.SECURITISATION, 5.0) == pytest.approx(0.04)
        assert minimum_haircut_floor(SFTCollateralType.SECURITISATION, 10.0) == pytest.approx(0.06)
        assert minimum_haircut_floor(SFTCollateralType.SECURITISATION, 11.0) == pytest.approx(0.07)

    def test_equity_and_other(self) -> None:
        assert minimum_haircut_floor(SFTCollateralType.MAIN_INDEX_EQUITY) == pytest.approx(0.06)
        assert minimum_haircut_floor(SFTCollateralType.OTHER_ASSET) == pytest.approx(0.10)

    def test_negative_maturity_raises(self) -> None:
        with pytest.raises(ValueError, match="residual_maturity_years"):
            minimum_haircut_floor(SFTCollateralType.CORPORATE_DEBT, -1.0)


class TestSFTHaircut:
    def test_over_collateralisation(self) -> None:
        # 102 collateral on 100 exposure -> 2% haircut.
        assert sft_haircut(100.0, 102.0) == pytest.approx(0.02)

    def test_invalid_exposure(self) -> None:
        with pytest.raises(ValueError, match="exposure must be positive"):
            sft_haircut(0.0, 100.0)

    def test_negative_collateral(self) -> None:
        with pytest.raises(ValueError, match="collateral_value must be non-negative"):
            sft_haircut(100.0, -1.0)


class TestAssessSFTFloor:
    def test_meets_floor(self) -> None:
        # 2% haircut vs 1.5% floor (corporate, 3y) -> recognised.
        r = assess_sft_floor(100.0, 102.0, SFTCollateralType.CORPORATE_DEBT, 3.0)
        assert r.floor == pytest.approx(0.015)
        assert r.meets_floor is True
        assert r.collateral_recognised is True

    def test_below_floor_not_recognised(self) -> None:
        # 1% haircut vs 4% floor (securitisation, 5y) -> treated as unsecured.
        r = assess_sft_floor(100.0, 101.0, SFTCollateralType.SECURITISATION, 5.0)
        assert r.meets_floor is False
        assert r.collateral_recognised is False


class TestPortfolioFloor:
    def test_compliant(self) -> None:
        legs = [
            SFTLeg(100.0, 103.0, SFTCollateralType.CORPORATE_DEBT, 3.0),  # floor 1.5%
            SFTLeg(100.0, 108.0, SFTCollateralType.MAIN_INDEX_EQUITY),  # floor 6%
        ]
        # portfolio haircut = (211-200)/200 = 5.5%; weighted floor = (1.5+6)/2 = 3.75%.
        assert portfolio_floor_compliant(legs) is True

    def test_non_compliant(self) -> None:
        legs = [
            SFTLeg(100.0, 101.0, SFTCollateralType.OTHER_ASSET),  # floor 10%
        ]
        # haircut 1% < floor 10%.
        assert portfolio_floor_compliant(legs) is False

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one SFT leg"):
            portfolio_floor_compliant([])

    def test_zero_exposure_raises(self) -> None:
        with pytest.raises(ValueError, match="total exposure must be positive"):
            portfolio_floor_compliant([SFTLeg(0.0, 0.0, SFTCollateralType.CASH)])
