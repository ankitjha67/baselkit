"""Tests for supervisory slotting approach -- BCBS d424, CRE34.1-34.7."""

import pytest

from creditriskengine.core.types import IRBSpecialisedLendingType
from creditriskengine.rwa.irb.slotting import (
    SlottingCategory,
    assign_slotting_category,
    slotting_risk_weight,
)


class TestSlottingRiskWeight:
    """CRE34.2 Table 1: non-HVCRE standard risk weights."""

    def test_strong_70(self) -> None:
        assert slotting_risk_weight(SlottingCategory.STRONG) == 70.0

    def test_good_90(self) -> None:
        assert slotting_risk_weight(SlottingCategory.GOOD) == 90.0

    def test_satisfactory_115(self) -> None:
        assert slotting_risk_weight(SlottingCategory.SATISFACTORY) == 115.0

    def test_weak_250(self) -> None:
        assert slotting_risk_weight(SlottingCategory.WEAK) == 250.0

    def test_default_0(self) -> None:
        assert slotting_risk_weight(SlottingCategory.DEFAULT) == 0.0


class TestSlottingRiskWeightHVCRE:
    """CRE34.3 Table 2: HVCRE risk weights."""

    def test_hvcre_strong_95(self) -> None:
        rw = slotting_risk_weight(
            SlottingCategory.STRONG,
            sl_type=IRBSpecialisedLendingType.HIGH_VOLATILITY_CRE,
        )
        assert rw == 95.0

    def test_hvcre_good_120(self) -> None:
        rw = slotting_risk_weight(
            SlottingCategory.GOOD,
            sl_type=IRBSpecialisedLendingType.HIGH_VOLATILITY_CRE,
        )
        assert rw == 120.0

    def test_hvcre_satisfactory_140(self) -> None:
        rw = slotting_risk_weight(
            SlottingCategory.SATISFACTORY,
            sl_type=IRBSpecialisedLendingType.HIGH_VOLATILITY_CRE,
        )
        assert rw == 140.0

    def test_hvcre_weak_250(self) -> None:
        rw = slotting_risk_weight(
            SlottingCategory.WEAK,
            sl_type=IRBSpecialisedLendingType.HIGH_VOLATILITY_CRE,
        )
        assert rw == 250.0

    def test_hvcre_default_0(self) -> None:
        rw = slotting_risk_weight(
            SlottingCategory.DEFAULT,
            sl_type=IRBSpecialisedLendingType.HIGH_VOLATILITY_CRE,
        )
        assert rw == 0.0


class TestSlottingPreferential:
    """CRE34.4: national discretion preferential weights."""

    def test_preferential_strong_50(self) -> None:
        rw = slotting_risk_weight(SlottingCategory.STRONG, use_preferential=True)
        assert rw == 50.0

    def test_preferential_good_70(self) -> None:
        rw = slotting_risk_weight(SlottingCategory.GOOD, use_preferential=True)
        assert rw == 70.0

    def test_preferential_satisfactory_unchanged_115(self) -> None:
        rw = slotting_risk_weight(SlottingCategory.SATISFACTORY, use_preferential=True)
        assert rw == 115.0

    def test_preferential_hvcre_strong_70(self) -> None:
        rw = slotting_risk_weight(
            SlottingCategory.STRONG,
            sl_type=IRBSpecialisedLendingType.HIGH_VOLATILITY_CRE,
            use_preferential=True,
        )
        assert rw == 70.0

    def test_preferential_hvcre_good_95(self) -> None:
        rw = slotting_risk_weight(
            SlottingCategory.GOOD,
            sl_type=IRBSpecialisedLendingType.HIGH_VOLATILITY_CRE,
            use_preferential=True,
        )
        assert rw == 95.0


class TestAssignSlottingCategory:
    """Conservative aggregation: worst dimension wins."""

    def test_all_strong(self) -> None:
        result = assign_slotting_category(
            financial_strength="strong",
            political_and_legal="strong",
            transaction_characteristics="strong",
            asset_characteristics="strong",
            sponsor_strength="strong",
        )
        assert result == SlottingCategory.STRONG

    def test_one_weak_all_others_strong(self) -> None:
        result = assign_slotting_category(
            financial_strength="strong",
            political_and_legal="strong",
            transaction_characteristics="strong",
            asset_characteristics="weak",
            sponsor_strength="strong",
        )
        assert result == SlottingCategory.WEAK

    def test_mixed_good_satisfactory(self) -> None:
        result = assign_slotting_category(
            financial_strength="good",
            political_and_legal="satisfactory",
            transaction_characteristics="good",
            asset_characteristics="good",
            sponsor_strength="good",
        )
        assert result == SlottingCategory.SATISFACTORY

    def test_all_default(self) -> None:
        result = assign_slotting_category(
            financial_strength="default",
            political_and_legal="default",
            transaction_characteristics="default",
            asset_characteristics="default",
            sponsor_strength="default",
        )
        assert result == SlottingCategory.DEFAULT

    def test_case_insensitive(self) -> None:
        result = assign_slotting_category(
            financial_strength="Strong",
            political_and_legal="STRONG",
            transaction_characteristics="  strong  ",
            asset_characteristics="strong",
            sponsor_strength="strong",
        )
        assert result == SlottingCategory.STRONG

    def test_invalid_rating_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid slotting rating"):
            assign_slotting_category(
                financial_strength="excellent",
                political_and_legal="strong",
                transaction_characteristics="strong",
                asset_characteristics="strong",
                sponsor_strength="strong",
            )

    def test_non_hvcre_type_uses_standard_table(self) -> None:
        rw = slotting_risk_weight(
            SlottingCategory.STRONG,
            sl_type=IRBSpecialisedLendingType.PROJECT_FINANCE,
        )
        assert rw == 70.0
