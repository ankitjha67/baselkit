"""Tests for Foundation IRB (F-IRB) calculator -- BCBS d424, CRE31-32."""

import pytest

from creditriskengine.core.exposure import Collateral, Exposure
from creditriskengine.core.types import (
    CollateralType,
    CreditRiskApproach,
    IRBAssetClass,
    Jurisdiction,
)
from creditriskengine.rwa.irb.foundation import (
    FoundationIRBCalculator,
    get_supervisory_ccf,
    get_supervisory_lgd,
)


def _make_exposure(**overrides) -> Exposure:
    """Helper to create a minimal F-IRB exposure."""
    defaults = dict(
        exposure_id="EXP-001",
        counterparty_id="CP-001",
        ead=1_000_000,
        drawn_amount=800_000,
        undrawn_commitment=200_000,
        jurisdiction=Jurisdiction.EU,
        approach=CreditRiskApproach.FIRB,
        irb_asset_class=IRBAssetClass.CORPORATE,
        pd=0.02,
    )
    defaults.update(overrides)
    return Exposure(**defaults)


class TestGetSupervisoryLGD:
    """CRE32.17-32.22: supervisory LGD values."""

    def test_senior_unsecured_no_collateral(self):
        exp = _make_exposure()
        assert get_supervisory_lgd(exp) == 0.45

    def test_subordinated(self):
        exp = _make_exposure()
        assert get_supervisory_lgd(exp, is_subordinated=True) == 0.75

    def test_fully_secured_by_cash(self):
        exp = _make_exposure(
            collaterals=[Collateral(collateral_type=CollateralType.CASH, value=1_000_000)],
        )
        lgd = get_supervisory_lgd(exp)
        assert lgd == pytest.approx(0.0, abs=1e-6)

    def test_fully_secured_by_receivables(self):
        exp = _make_exposure(
            collaterals=[Collateral(collateral_type=CollateralType.RECEIVABLES, value=1_000_000)],
        )
        lgd = get_supervisory_lgd(exp)
        assert lgd == pytest.approx(0.35, abs=1e-6)

    def test_fully_secured_by_rre(self):
        exp = _make_exposure(
            collaterals=[
                Collateral(collateral_type=CollateralType.RESIDENTIAL_REAL_ESTATE, value=1_000_000)
            ],
        )
        lgd = get_supervisory_lgd(exp)
        assert lgd == pytest.approx(0.35, abs=1e-6)

    def test_fully_secured_by_other_physical(self):
        exp = _make_exposure(
            collaterals=[
                Collateral(collateral_type=CollateralType.OTHER_PHYSICAL, value=1_000_000)
            ],
        )
        lgd = get_supervisory_lgd(exp)
        assert lgd == pytest.approx(0.40, abs=1e-6)

    def test_partially_secured_50pct_cash(self):
        exp = _make_exposure(
            collaterals=[Collateral(collateral_type=CollateralType.CASH, value=500_000)],
        )
        lgd = get_supervisory_lgd(exp)
        # coverage=50%, secured portion LGD=0.0, unsecured=0.45
        expected = 0.5 * 0.0 + 0.5 * 0.45
        assert lgd == pytest.approx(expected, abs=1e-6)

    def test_zero_ead_returns_senior_unsecured(self):
        exp = _make_exposure(
            ead=0,
            collaterals=[Collateral(collateral_type=CollateralType.CASH, value=500_000)],
        )
        assert get_supervisory_lgd(exp) == 0.45

    def test_zero_collateral_value_returns_senior_unsecured(self):
        exp = _make_exposure(
            collaterals=[Collateral(collateral_type=CollateralType.CASH, value=0)],
        )
        assert get_supervisory_lgd(exp) == 0.45

    def test_subordinated_ignores_collateral(self):
        exp = _make_exposure(
            collaterals=[Collateral(collateral_type=CollateralType.CASH, value=1_000_000)],
        )
        assert get_supervisory_lgd(exp, is_subordinated=True) == 0.75


class TestGetSupervisoryCCF:
    """CRE32.28-32.32: supervisory CCF values."""

    def test_zero_undrawn_returns_zero(self):
        exp = _make_exposure(undrawn_commitment=0)
        assert get_supervisory_ccf(exp) == 0.0

    def test_default_committed_facility(self):
        exp = _make_exposure()
        assert get_supervisory_ccf(exp) == 0.75

    def test_direct_credit_substitute(self):
        exp = _make_exposure()
        assert get_supervisory_ccf(exp, is_direct_credit_substitute=True) == 1.00

    def test_unconditionally_cancellable(self):
        exp = _make_exposure()
        assert get_supervisory_ccf(exp, is_unconditionally_cancellable=True) == 0.40

    def test_trade_related(self):
        exp = _make_exposure()
        assert get_supervisory_ccf(exp, is_trade_related=True) == 0.20

    def test_transaction_related(self):
        exp = _make_exposure()
        assert get_supervisory_ccf(exp, is_transaction_related=True) == 0.50

    def test_priority_direct_substitute_over_cancellable(self):
        exp = _make_exposure()
        ccf = get_supervisory_ccf(
            exp,
            is_direct_credit_substitute=True,
            is_unconditionally_cancellable=True,
        )
        assert ccf == 1.00


class TestFoundationIRBCalculator:
    """F-IRB end-to-end calculation."""

    def test_basic_corporate_calculation(self):
        calc = FoundationIRBCalculator()
        exp = _make_exposure(pd=0.02, ead=1_000_000, drawn_amount=1_000_000, undrawn_commitment=0)
        result = calc.calculate(exp)

        assert result.exposure_id == "EXP-001"
        assert result.approach == CreditRiskApproach.FIRB
        assert result.risk_weight > 0
        assert result.rwa > 0
        assert result.capital_requirement == pytest.approx(result.rwa * 0.08, rel=1e-6)
        assert result.details["pd"] == 0.02
        assert result.details["lgd"] == 0.45

    def test_pd_floor_applied(self):
        calc = FoundationIRBCalculator()
        exp = _make_exposure(pd=0.0001, drawn_amount=1_000_000, undrawn_commitment=0)
        result = calc.calculate(exp)
        assert result.details["pd"] == pytest.approx(0.0003, abs=1e-6)

    def test_defaulted_exposure_zero_rw(self):
        calc = FoundationIRBCalculator()
        exp = _make_exposure(is_defaulted=True, pd=1.0, drawn_amount=1_000_000, undrawn_commitment=0)
        result = calc.calculate(exp)
        assert result.risk_weight == 0.0
        assert result.rwa == 0.0

    def test_missing_pd_raises(self):
        calc = FoundationIRBCalculator()
        exp = _make_exposure(pd=None)
        with pytest.raises(ValueError, match="no PD estimate"):
            calc.calculate(exp)

    def test_ead_includes_ccf_times_undrawn(self):
        calc = FoundationIRBCalculator()
        exp = _make_exposure(pd=0.02, drawn_amount=800_000, undrawn_commitment=200_000)
        result = calc.calculate(exp)
        # CCF=0.75 for committed facility -> EAD = 800k + 0.75*200k = 950k
        assert result.ead == pytest.approx(950_000, rel=1e-6)

    def test_maturity_fixed_at_2_5_years(self):
        calc = FoundationIRBCalculator()
        exp = _make_exposure(pd=0.02, drawn_amount=1_000_000, undrawn_commitment=0)
        result = calc.calculate(exp)
        assert result.details["maturity"] == 2.5

    def test_sovereign_asset_class(self):
        calc = FoundationIRBCalculator()
        exp = _make_exposure(
            irb_asset_class=IRBAssetClass.SOVEREIGN,
            pd=0.01,
            drawn_amount=1_000_000,
            undrawn_commitment=0,
        )
        result = calc.calculate(exp)
        assert result.asset_class == "sovereign"
        assert result.rwa > 0

    def test_unknown_asset_class_raises(self):
        calc = FoundationIRBCalculator()
        exp = _make_exposure(irb_asset_class=IRBAssetClass.EQUITY, pd=0.02)
        with pytest.raises(ValueError, match="Cannot determine IRB asset class"):
            calc.calculate(exp)
