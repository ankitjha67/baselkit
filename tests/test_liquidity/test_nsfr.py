"""Tests for the Net Stable Funding Ratio (BCBS NSF)."""

import pytest

from creditriskengine.liquidity.nsfr import (
    ASFCategory,
    RSFCategory,
    available_stable_funding,
    net_stable_funding_ratio,
    required_stable_funding,
)


class TestAvailableStableFunding:
    def test_weighted_sum(self) -> None:
        asf = available_stable_funding(
            {
                ASFCategory.CAPITAL: 100.0,  # x1.00 = 100
                ASFCategory.STABLE_DEPOSITS: 200.0,  # x0.95 = 190
                ASFCategory.OTHER_LT1Y: 500.0,  # x0.00 = 0
            }
        )
        assert asf == pytest.approx(290.0)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="ASF amounts"):
            available_stable_funding({ASFCategory.CAPITAL: -1.0})


class TestRequiredStableFunding:
    def test_weighted_sum(self) -> None:
        rsf = required_stable_funding(
            {
                RSFCategory.CASH_AND_L1: 50.0,  # x0.05 = 2.5
                RSFCategory.OTHER_LOANS_GE1Y: 100.0,  # x0.85 = 85
                RSFCategory.RESIDENTIAL_MORTGAGE: 100.0,  # x0.65 = 65
            }
        )
        assert rsf == pytest.approx(152.5)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="RSF amounts"):
            required_stable_funding({RSFCategory.OTHER_ASSETS: -1.0})


class TestNSFR:
    def test_compliant(self) -> None:
        r = net_stable_funding_ratio(
            {ASFCategory.CAPITAL: 100.0, ASFCategory.STABLE_DEPOSITS: 200.0},
            {RSFCategory.CASH_AND_L1: 50.0, RSFCategory.OTHER_LOANS_GE1Y: 100.0},
        )
        # ASF=290, RSF=2.5+85=87.5 -> NSFR=3.314
        assert r.asf == pytest.approx(290.0)
        assert r.rsf == pytest.approx(87.5)
        assert r.nsfr == pytest.approx(290.0 / 87.5, abs=1e-4)
        assert r.is_compliant is True

    def test_non_compliant(self) -> None:
        r = net_stable_funding_ratio(
            {ASFCategory.OTHER_LT1Y: 1000.0},  # x0 -> ASF=0
            {RSFCategory.OTHER_ASSETS: 100.0},  # x1 -> RSF=100
        )
        assert r.nsfr == pytest.approx(0.0)
        assert r.is_compliant is False

    def test_zero_rsf_is_infinite(self) -> None:
        r = net_stable_funding_ratio({ASFCategory.CAPITAL: 100.0}, {})
        assert r.rsf == 0.0
        assert r.nsfr == float("inf")
        assert r.is_compliant is True
