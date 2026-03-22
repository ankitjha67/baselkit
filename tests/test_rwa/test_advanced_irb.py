"""Tests for Advanced IRB (A-IRB) calculator -- BCBS d424, CRE31-32."""

import pytest

from creditriskengine.core.exposure import Collateral, Exposure
from creditriskengine.core.types import (
    CollateralType,
    CreditRiskApproach,
    IRBAssetClass,
    IRBRetailSubClass,
    Jurisdiction,
)
from creditriskengine.rwa.irb.advanced import (
    AdvancedIRBCalculator,
    apply_ccf_floor,
    apply_lgd_floor,
)


def _make_exposure(**overrides) -> Exposure:
    """Helper to create a minimal A-IRB exposure."""
    defaults = dict(
        exposure_id="EXP-A001",
        counterparty_id="CP-001",
        ead=1_000_000,
        drawn_amount=800_000,
        undrawn_commitment=200_000,
        jurisdiction=Jurisdiction.EU,
        approach=CreditRiskApproach.AIRB,
        irb_asset_class=IRBAssetClass.CORPORATE,
        pd=0.02,
        lgd=0.35,
    )
    defaults.update(overrides)
    return Exposure(**defaults)


class TestApplyLGDFloor:
    """CRE32.23-32.25: LGD floors by asset class and collateral type."""

    def test_corporate_unsecured_floor_25pct(self):
        result = apply_lgd_floor(0.10, "corporate")
        assert result == pytest.approx(0.25, abs=1e-6)

    def test_corporate_unsecured_no_floor_needed(self):
        result = apply_lgd_floor(0.40, "corporate")
        assert result == pytest.approx(0.40, abs=1e-6)

    def test_corporate_secured_financial_0pct(self):
        result = apply_lgd_floor(0.0, "corporate", collateral_type=CollateralType.CASH)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_corporate_secured_receivables_10pct(self):
        result = apply_lgd_floor(0.05, "corporate", collateral_type=CollateralType.RECEIVABLES)
        assert result == pytest.approx(0.10, abs=1e-6)

    def test_corporate_secured_rre_10pct(self):
        result = apply_lgd_floor(
            0.05, "corporate", collateral_type=CollateralType.RESIDENTIAL_REAL_ESTATE
        )
        assert result == pytest.approx(0.10, abs=1e-6)

    def test_corporate_secured_other_physical_15pct(self):
        result = apply_lgd_floor(0.10, "corporate", collateral_type=CollateralType.OTHER_PHYSICAL)
        assert result == pytest.approx(0.15, abs=1e-6)

    def test_sovereign_unsecured_floor(self):
        result = apply_lgd_floor(0.10, "sovereign")
        assert result == pytest.approx(0.25, abs=1e-6)

    def test_bank_unsecured_floor(self):
        result = apply_lgd_floor(0.10, "bank")
        assert result == pytest.approx(0.25, abs=1e-6)

    def test_retail_mortgage_floor_5pct(self):
        result = apply_lgd_floor(
            0.03,
            "residential_mortgage",
            retail_subclass=IRBRetailSubClass.RESIDENTIAL_MORTGAGE,
        )
        assert result == pytest.approx(0.05, abs=1e-6)

    def test_qrre_floor_50pct(self):
        result = apply_lgd_floor(0.30, "qrre", retail_subclass=IRBRetailSubClass.QRRE)
        assert result == pytest.approx(0.50, abs=1e-6)

    def test_other_retail_floor_25pct(self):
        result = apply_lgd_floor(0.10, "other_retail", retail_subclass=IRBRetailSubClass.OTHER_RETAIL)
        assert result == pytest.approx(0.25, abs=1e-6)

    def test_sme_retail_floor_25pct(self):
        result = apply_lgd_floor(0.10, "other_retail", retail_subclass=IRBRetailSubClass.SME_RETAIL)
        assert result == pytest.approx(0.25, abs=1e-6)

    def test_unknown_asset_class_conservative_fallback(self):
        result = apply_lgd_floor(0.10, "unknown_class")
        assert result == pytest.approx(0.25, abs=1e-6)


class TestApplyCCFFloor:
    """CRE32.33: 50% CCF floor."""

    def test_ccf_below_floor(self):
        assert apply_ccf_floor(0.30) == pytest.approx(0.50, abs=1e-6)

    def test_ccf_at_floor(self):
        assert apply_ccf_floor(0.50) == pytest.approx(0.50, abs=1e-6)

    def test_ccf_above_floor(self):
        assert apply_ccf_floor(0.80) == pytest.approx(0.80, abs=1e-6)

    def test_ccf_zero(self):
        assert apply_ccf_floor(0.0) == pytest.approx(0.50, abs=1e-6)

    def test_ccf_one(self):
        assert apply_ccf_floor(1.0) == pytest.approx(1.0, abs=1e-6)


class TestAdvancedIRBCalculator:
    """A-IRB end-to-end calculation."""

    def test_basic_corporate(self):
        calc = AdvancedIRBCalculator()
        exp = _make_exposure(pd=0.02, lgd=0.35, drawn_amount=1_000_000, undrawn_commitment=0)
        result = calc.calculate(exp)

        assert result.exposure_id == "EXP-A001"
        assert result.approach == CreditRiskApproach.AIRB
        assert result.risk_weight > 0
        assert result.rwa > 0
        assert result.capital_requirement == pytest.approx(result.rwa * 0.08, rel=1e-6)

    def test_pd_floor_applied(self):
        calc = AdvancedIRBCalculator()
        exp = _make_exposure(pd=0.0001, lgd=0.35, drawn_amount=1_000_000, undrawn_commitment=0)
        result = calc.calculate(exp)
        assert result.details["pd"] == pytest.approx(0.0003, abs=1e-6)

    def test_lgd_floor_applied(self):
        calc = AdvancedIRBCalculator()
        exp = _make_exposure(pd=0.02, lgd=0.10, drawn_amount=1_000_000, undrawn_commitment=0)
        result = calc.calculate(exp)
        # Corporate unsecured floor is 25%
        assert result.details["lgd"] == pytest.approx(0.25, abs=1e-6)

    def test_defaulted_exposure_zero_rw(self):
        calc = AdvancedIRBCalculator()
        exp = _make_exposure(is_defaulted=True, pd=1.0, lgd=0.45)
        result = calc.calculate(exp)
        assert result.risk_weight == 0.0
        assert result.rwa == 0.0

    def test_missing_pd_raises(self):
        calc = AdvancedIRBCalculator()
        exp = _make_exposure(pd=None, lgd=0.35)
        with pytest.raises(ValueError, match="A-IRB requires"):
            calc.calculate(exp)

    def test_missing_lgd_raises(self):
        calc = AdvancedIRBCalculator()
        exp = _make_exposure(pd=0.02, lgd=None)
        with pytest.raises(ValueError, match="A-IRB requires"):
            calc.calculate(exp)

    def test_ead_model_used_when_provided(self):
        calc = AdvancedIRBCalculator()
        exp = _make_exposure(
            pd=0.02, lgd=0.35, ead_model=500_000, drawn_amount=800_000, undrawn_commitment=200_000
        )
        result = calc.calculate(exp)
        assert result.ead == pytest.approx(500_000, rel=1e-6)

    def test_retail_mortgage_asset_class(self):
        calc = AdvancedIRBCalculator()
        exp = _make_exposure(
            irb_asset_class=IRBAssetClass.RETAIL,
            irb_retail_subclass=IRBRetailSubClass.RESIDENTIAL_MORTGAGE,
            pd=0.01,
            lgd=0.15,
            drawn_amount=500_000,
            undrawn_commitment=0,
        )
        result = calc.calculate(exp)
        assert result.asset_class == "residential_mortgage"
        assert result.rwa > 0
