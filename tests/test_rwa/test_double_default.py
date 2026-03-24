"""Tests for double default and equity IRB functions — BCBS d424 CRE32-33."""

import pytest

from creditriskengine.rwa.irb.formulas import (
    PD_FLOOR,
    double_default_rw,
    equity_irb_rw,
    irb_risk_weight,
)


class TestDoubleDefaultRW:
    """BCBS CRE32.38-41 — Substitution approach."""

    def test_guarantor_pd_lower_reduces_rw(self) -> None:
        """A better-rated guarantor should produce a lower RW than the obligor alone."""
        rw_obligor = irb_risk_weight(pd=0.05, lgd=0.45, asset_class="corporate")
        rw_dd = double_default_rw(
            pd_obligor=0.05, pd_guarantor=0.005, lgd=0.45, asset_class="corporate"
        )
        assert rw_dd < rw_obligor

    def test_guarantor_pd_floored(self) -> None:
        """Guarantor PD below PD_FLOOR should be floored at 0.03%."""
        rw_tiny = double_default_rw(
            pd_obligor=0.05, pd_guarantor=0.0001, lgd=0.45
        )
        rw_floor = double_default_rw(
            pd_obligor=0.05, pd_guarantor=PD_FLOOR, lgd=0.45
        )
        assert rw_tiny == pytest.approx(rw_floor, rel=1e-6)

    def test_obligor_correlation_used(self) -> None:
        """Correlation should come from obligor PD, not guarantor PD.

        Two calls with different obligor PDs but same guarantor PD should
        produce different RWs (because correlation differs).
        """
        rw_a = double_default_rw(
            pd_obligor=0.001, pd_guarantor=0.005, lgd=0.45
        )
        rw_b = double_default_rw(
            pd_obligor=0.10, pd_guarantor=0.005, lgd=0.45
        )
        assert rw_a != pytest.approx(rw_b, rel=1e-3)

    def test_defaulted_guarantor_returns_zero(self) -> None:
        """If guarantor PD >= 1.0, function returns 0.0 (no benefit)."""
        rw = double_default_rw(
            pd_obligor=0.05, pd_guarantor=1.0, lgd=0.45
        )
        assert rw == 0.0

    def test_corporate_typical_range(self) -> None:
        """Typical corporate double default RW should be in a reasonable range."""
        rw = double_default_rw(
            pd_obligor=0.02, pd_guarantor=0.003, lgd=0.45,
            maturity=2.5, asset_class="corporate",
        )
        assert 20.0 < rw < 100.0

    def test_maturity_effect(self) -> None:
        """Longer maturity should produce higher RW for non-retail."""
        rw_short = double_default_rw(
            pd_obligor=0.02, pd_guarantor=0.005, lgd=0.45, maturity=1.0
        )
        rw_long = double_default_rw(
            pd_obligor=0.02, pd_guarantor=0.005, lgd=0.45, maturity=5.0
        )
        assert rw_long > rw_short

    def test_retail_no_maturity_adjustment(self) -> None:
        """Retail asset classes should not apply maturity adjustment.

        Changing maturity should not affect the result for retail.
        """
        rw_a = double_default_rw(
            pd_obligor=0.02, pd_guarantor=0.005, lgd=0.50,
            maturity=1.0, asset_class="residential_mortgage",
        )
        rw_b = double_default_rw(
            pd_obligor=0.02, pd_guarantor=0.005, lgd=0.50,
            maturity=5.0, asset_class="residential_mortgage",
        )
        assert rw_a == pytest.approx(rw_b, rel=1e-6)

    def test_sovereign_asset_class(self) -> None:
        """Sovereign asset class should use corporate correlation formula."""
        rw = double_default_rw(
            pd_obligor=0.01, pd_guarantor=0.003, lgd=0.45,
            asset_class="sovereign",
        )
        assert rw > 0.0

    def test_bank_asset_class(self) -> None:
        """Bank asset class should use corporate correlation formula."""
        rw = double_default_rw(
            pd_obligor=0.01, pd_guarantor=0.003, lgd=0.45,
            asset_class="bank",
        )
        assert rw > 0.0

    def test_qrre_asset_class(self) -> None:
        """QRRE asset class should use fixed 0.04 correlation."""
        rw = double_default_rw(
            pd_obligor=0.02, pd_guarantor=0.005, lgd=0.80,
            asset_class="qrre",
        )
        assert rw > 0.0

    def test_other_retail_asset_class(self) -> None:
        """Other retail asset class should use the other retail correlation."""
        rw = double_default_rw(
            pd_obligor=0.03, pd_guarantor=0.005, lgd=0.50,
            asset_class="other_retail",
        )
        assert rw > 0.0

    def test_unknown_asset_class_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown asset class"):
            double_default_rw(
                pd_obligor=0.01, pd_guarantor=0.005, lgd=0.45,
                asset_class="invalid",
            )

    def test_lgd_effect(self) -> None:
        """Higher LGD should produce higher RW."""
        rw_low = double_default_rw(
            pd_obligor=0.02, pd_guarantor=0.005, lgd=0.25
        )
        rw_high = double_default_rw(
            pd_obligor=0.02, pd_guarantor=0.005, lgd=0.45
        )
        assert rw_high > rw_low

    def test_same_pd_equals_standard_irb(self) -> None:
        """When guarantor PD equals obligor PD, result should match standard IRB."""
        rw_dd = double_default_rw(
            pd_obligor=0.01, pd_guarantor=0.01, lgd=0.45,
            maturity=2.5, asset_class="corporate",
        )
        rw_irb = irb_risk_weight(
            pd=0.01, lgd=0.45, asset_class="corporate", maturity=2.5
        )
        assert rw_dd == pytest.approx(rw_irb, rel=1e-6)


class TestEquityIRBRW:
    """BCBS CRE33 — Simple risk weight method for equity."""

    def test_listed_floor_200(self) -> None:
        """Listed equity must have at least 200% RW."""
        rw = equity_irb_rw(pd=0.0003, equity_type="listed")
        assert rw >= 200.0

    def test_private_floor_300(self) -> None:
        """Private equity must have at least 300% RW."""
        rw = equity_irb_rw(pd=0.0003, equity_type="private")
        assert rw >= 300.0

    def test_listed_is_2_5x_corporate(self) -> None:
        """Listed equity RW should be max(200%, 2.5 * corporate RW)."""
        corporate_rw = irb_risk_weight(
            pd=0.05, lgd=0.90, asset_class="corporate", maturity=5.0
        )
        expected = max(200.0, 2.5 * corporate_rw)
        actual = equity_irb_rw(pd=0.05, equity_type="listed")
        assert actual == pytest.approx(expected, rel=1e-6)

    def test_private_is_2_5x_corporate(self) -> None:
        """Private equity RW should be max(300%, 2.5 * corporate RW)."""
        corporate_rw = irb_risk_weight(
            pd=0.05, lgd=0.90, asset_class="corporate", maturity=5.0
        )
        expected = max(300.0, 2.5 * corporate_rw)
        actual = equity_irb_rw(pd=0.05, equity_type="private")
        assert actual == pytest.approx(expected, rel=1e-6)

    def test_listed_default_type(self) -> None:
        """Default equity type should be 'listed'."""
        rw_default = equity_irb_rw(pd=0.01)
        rw_listed = equity_irb_rw(pd=0.01, equity_type="listed")
        assert rw_default == rw_listed

    def test_high_pd_listed(self) -> None:
        """At high PD, 2.5x corporate RW exceeds the 200% floor."""
        rw = equity_irb_rw(pd=0.10, equity_type="listed")
        corporate_rw = irb_risk_weight(
            pd=0.10, lgd=0.90, asset_class="corporate", maturity=5.0
        )
        assert rw == pytest.approx(2.5 * corporate_rw, rel=1e-6)
        assert rw > 200.0

    def test_high_pd_private(self) -> None:
        """At high PD, 2.5x corporate RW exceeds the 300% floor."""
        rw = equity_irb_rw(pd=0.10, equity_type="private")
        corporate_rw = irb_risk_weight(
            pd=0.10, lgd=0.90, asset_class="corporate", maturity=5.0
        )
        assert rw == pytest.approx(2.5 * corporate_rw, rel=1e-6)
        assert rw > 300.0

    def test_private_gte_listed(self) -> None:
        """Private equity RW should always be >= listed for the same PD."""
        for pd in [0.001, 0.01, 0.05, 0.10]:
            rw_listed = equity_irb_rw(pd=pd, equity_type="listed")
            rw_private = equity_irb_rw(pd=pd, equity_type="private")
            assert rw_private >= rw_listed, f"Failed at PD={pd}"

    def test_invalid_equity_type_raises(self) -> None:
        with pytest.raises(ValueError, match="equity_type must be"):
            equity_irb_rw(pd=0.01, equity_type="preferred")

    def test_pd_floor_applied(self) -> None:
        """Very low PD should be floored at 0.03%."""
        rw_tiny = equity_irb_rw(pd=0.0001, equity_type="listed")
        rw_floor = equity_irb_rw(pd=PD_FLOOR, equity_type="listed")
        assert rw_tiny == pytest.approx(rw_floor, rel=1e-6)

    def test_uses_lgd_90_and_m_5(self) -> None:
        """Verify that LGD=90% and M=5 are used internally."""
        rw = equity_irb_rw(pd=0.03, equity_type="listed")
        corporate_rw = irb_risk_weight(
            pd=0.03, lgd=0.90, asset_class="corporate", maturity=5.0
        )
        assert rw >= 2.5 * corporate_rw - 0.01  # Allow tiny float rounding
