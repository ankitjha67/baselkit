"""Tests for IRB asset class classification -- BCBS d424, CRE30.4-30.15."""

import pytest

from creditriskengine.core.exposure import Exposure
from creditriskengine.core.types import (
    CreditRiskApproach,
    IRBAssetClass,
    IRBCorporateSubClass,
    IRBRetailSubClass,
    Jurisdiction,
    SAExposureClass,
)
from creditriskengine.rwa.irb.asset_classes import (
    classify_irb_asset_class,
    get_retail_subclass,
    is_specialised_lending,
    requires_slotting,
)


def _make_exposure(**overrides: object) -> Exposure:
    """Helper to create a minimal exposure."""
    defaults = dict(
        exposure_id="EXP-AC01",
        counterparty_id="CP-001",
        ead=1_000_000,
        drawn_amount=1_000_000,
        jurisdiction=Jurisdiction.BCBS,
        approach=CreditRiskApproach.FIRB,
    )
    defaults.update(overrides)
    return Exposure(**defaults)


class TestClassifyIRBAssetClass:
    """CRE30.4: IRB asset class classification."""

    def test_explicit_corporate(self) -> None:
        exp = _make_exposure(irb_asset_class=IRBAssetClass.CORPORATE)
        assert classify_irb_asset_class(exp) == IRBAssetClass.CORPORATE

    def test_explicit_sovereign(self) -> None:
        exp = _make_exposure(irb_asset_class=IRBAssetClass.SOVEREIGN)
        assert classify_irb_asset_class(exp) == IRBAssetClass.SOVEREIGN

    def test_explicit_bank(self) -> None:
        exp = _make_exposure(irb_asset_class=IRBAssetClass.BANK)
        assert classify_irb_asset_class(exp) == IRBAssetClass.BANK

    def test_explicit_retail(self) -> None:
        exp = _make_exposure(irb_asset_class=IRBAssetClass.RETAIL)
        assert classify_irb_asset_class(exp) == IRBAssetClass.RETAIL

    def test_explicit_equity(self) -> None:
        exp = _make_exposure(irb_asset_class=IRBAssetClass.EQUITY)
        assert classify_irb_asset_class(exp) == IRBAssetClass.EQUITY

    def test_infer_from_sa_sovereign(self) -> None:
        exp = _make_exposure(sa_exposure_class=SAExposureClass.SOVEREIGN)
        assert classify_irb_asset_class(exp) == IRBAssetClass.SOVEREIGN

    def test_infer_from_sa_bank(self) -> None:
        exp = _make_exposure(sa_exposure_class=SAExposureClass.BANK)
        assert classify_irb_asset_class(exp) == IRBAssetClass.BANK

    def test_infer_from_sa_corporate(self) -> None:
        exp = _make_exposure(sa_exposure_class=SAExposureClass.CORPORATE)
        assert classify_irb_asset_class(exp) == IRBAssetClass.CORPORATE

    def test_infer_from_sa_residential_mortgage(self) -> None:
        exp = _make_exposure(sa_exposure_class=SAExposureClass.RESIDENTIAL_MORTGAGE)
        assert classify_irb_asset_class(exp) == IRBAssetClass.RETAIL

    def test_infer_from_sa_equity(self) -> None:
        exp = _make_exposure(sa_exposure_class=SAExposureClass.EQUITY)
        assert classify_irb_asset_class(exp) == IRBAssetClass.EQUITY

    def test_unmapped_sa_class_raises(self) -> None:
        exp = _make_exposure(sa_exposure_class=SAExposureClass.DEFAULTED)
        with pytest.raises(ValueError, match="Cannot determine IRB asset class"):
            classify_irb_asset_class(exp)

    def test_no_class_info_raises(self) -> None:
        exp = _make_exposure()
        with pytest.raises(ValueError, match="Cannot determine IRB asset class"):
            classify_irb_asset_class(exp)


class TestIsSpecialisedLending:
    """CRE30.7: specialised lending classification."""

    def test_explicit_sl_subclass(self) -> None:
        exp = _make_exposure(
            irb_asset_class=IRBAssetClass.CORPORATE,
            irb_corporate_subclass=IRBCorporateSubClass.SPECIALISED_LENDING,
        )
        assert is_specialised_lending(exp) is True

    def test_general_corporate_not_sl(self) -> None:
        exp = _make_exposure(
            irb_asset_class=IRBAssetClass.CORPORATE,
            irb_corporate_subclass=IRBCorporateSubClass.GENERAL_CORPORATE,
        )
        assert is_specialised_lending(exp) is False

    def test_ipre_inference(self) -> None:
        exp = _make_exposure(
            irb_asset_class=IRBAssetClass.CORPORATE,
            is_income_producing=True,
            is_materially_dependent_on_cashflows=True,
        )
        assert is_specialised_lending(exp) is True

    def test_income_producing_but_not_dependent(self) -> None:
        exp = _make_exposure(
            irb_asset_class=IRBAssetClass.CORPORATE,
            is_income_producing=True,
            is_materially_dependent_on_cashflows=False,
        )
        assert is_specialised_lending(exp) is False


class TestGetRetailSubclass:
    """CRE30.11-30.15: retail sub-class determination."""

    def test_explicit_mortgage(self) -> None:
        exp = _make_exposure(
            irb_asset_class=IRBAssetClass.RETAIL,
            irb_retail_subclass=IRBRetailSubClass.RESIDENTIAL_MORTGAGE,
        )
        assert get_retail_subclass(exp) == IRBRetailSubClass.RESIDENTIAL_MORTGAGE

    def test_explicit_qrre(self) -> None:
        exp = _make_exposure(
            irb_asset_class=IRBAssetClass.RETAIL,
            irb_retail_subclass=IRBRetailSubClass.QRRE,
        )
        assert get_retail_subclass(exp) == IRBRetailSubClass.QRRE

    def test_infer_from_sa_residential_mortgage(self) -> None:
        exp = _make_exposure(sa_exposure_class=SAExposureClass.RESIDENTIAL_MORTGAGE)
        assert get_retail_subclass(exp) == IRBRetailSubClass.RESIDENTIAL_MORTGAGE

    def test_sme_retail_from_sa(self) -> None:
        exp = _make_exposure(
            sa_exposure_class=SAExposureClass.CORPORATE_SME,
            ead=500_000,
        )
        assert get_retail_subclass(exp) == IRBRetailSubClass.SME_RETAIL

    def test_default_other_retail(self) -> None:
        exp = _make_exposure(sa_exposure_class=SAExposureClass.RETAIL)
        assert get_retail_subclass(exp) == IRBRetailSubClass.OTHER_RETAIL


class TestRequiresSlotting:
    """CRE30.7, CRE34: slotting requirement check."""

    def test_sl_without_pd_requires_slotting(self) -> None:
        exp = _make_exposure(
            irb_asset_class=IRBAssetClass.CORPORATE,
            irb_corporate_subclass=IRBCorporateSubClass.SPECIALISED_LENDING,
            pd=None,
        )
        assert requires_slotting(exp) is True

    def test_sl_with_pd_no_slotting(self) -> None:
        exp = _make_exposure(
            irb_asset_class=IRBAssetClass.CORPORATE,
            irb_corporate_subclass=IRBCorporateSubClass.SPECIALISED_LENDING,
            pd=0.02,
        )
        assert requires_slotting(exp) is False

    def test_non_sl_no_slotting(self) -> None:
        exp = _make_exposure(
            irb_asset_class=IRBAssetClass.CORPORATE,
            irb_corporate_subclass=IRBCorporateSubClass.GENERAL_CORPORATE,
            pd=None,
        )
        assert requires_slotting(exp) is False
