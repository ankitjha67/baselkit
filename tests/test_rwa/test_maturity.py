"""Tests for IRB maturity module — CRE31.7, CRE32.44-32.47."""

import pytest

from creditriskengine.rwa.irb.maturity import (
    effective_maturity_airb,
    effective_maturity_firb,
    needs_maturity_adjustment,
)


class TestEffectiveMaturityFIRB:
    def test_returns_fixed_2_5(self) -> None:
        assert effective_maturity_firb() == 2.5


class TestEffectiveMaturityAIRB:
    def test_within_range(self) -> None:
        assert effective_maturity_airb(3.0) == 3.0

    def test_floor_at_1_year(self) -> None:
        assert effective_maturity_airb(0.5) == 1.0
        assert effective_maturity_airb(0.0) == 1.0
        assert effective_maturity_airb(-1.0) == 1.0

    def test_cap_at_5_years(self) -> None:
        assert effective_maturity_airb(7.0) == 5.0
        assert effective_maturity_airb(100.0) == 5.0

    def test_boundary_values(self) -> None:
        assert effective_maturity_airb(1.0) == 1.0
        assert effective_maturity_airb(5.0) == 5.0

    def test_fractional_maturity(self) -> None:
        assert effective_maturity_airb(2.7) == 2.7


class TestNeedsMaturityAdjustment:
    def test_corporate(self) -> None:
        assert needs_maturity_adjustment("corporate") is True

    def test_sovereign(self) -> None:
        assert needs_maturity_adjustment("sovereign") is True

    def test_bank(self) -> None:
        assert needs_maturity_adjustment("bank") is True

    def test_residential_mortgage(self) -> None:
        assert needs_maturity_adjustment("residential_mortgage") is False

    def test_qrre(self) -> None:
        assert needs_maturity_adjustment("qrre") is False

    def test_other_retail(self) -> None:
        assert needs_maturity_adjustment("other_retail") is False

    def test_unknown_class(self) -> None:
        assert needs_maturity_adjustment("something_else") is False
