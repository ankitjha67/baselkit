"""Tests for the Liquidity Coverage Ratio (BCBS LCR)."""

import pytest

from creditriskengine.liquidity.lcr import (
    liquidity_coverage_ratio,
    net_cash_outflows,
    stock_of_hqla,
)


class TestStockOfHQLA:
    def test_level1_only(self) -> None:
        h = stock_of_hqla(100.0, 0.0, 0.0)
        assert h.total_hqla == pytest.approx(100.0)
        assert h.cap_adjustment == pytest.approx(0.0)

    def test_level2a_haircut(self) -> None:
        # L1=100, L2A_pre=20 -> l2a=17; total within the 40% cap.
        h = stock_of_hqla(100.0, 20.0, 0.0)
        assert h.level2a == pytest.approx(17.0)
        assert h.total_hqla == pytest.approx(117.0)
        assert h.cap_adjustment == pytest.approx(0.0)

    def test_level2b_15pct_cap(self) -> None:
        # L1=50, L2B_pre=100 -> l2b post-haircut 50, capped at 15% of HQLA.
        h = stock_of_hqla(50.0, 0.0, 100.0)
        # l2b_cap = (15/85)*50 = 8.8235
        assert h.level2b == pytest.approx(8.8235, abs=1e-3)
        assert h.total_hqla == pytest.approx(58.8235, abs=1e-3)
        # Level 2B is exactly 15% of HQLA after the cap.
        assert h.level2b / h.total_hqla == pytest.approx(0.15, abs=1e-4)

    def test_level2_40pct_cap(self) -> None:
        # L1=50, L2A_pre=100 -> l2a=85, total L2 capped at 40% of HQLA.
        h = stock_of_hqla(50.0, 100.0, 0.0)
        # l2_cap = (40/60)*50 = 33.33; total = 83.33; L2 share = 40%.
        assert h.level2a == pytest.approx(33.3333, abs=1e-3)
        assert h.total_hqla == pytest.approx(83.3333, abs=1e-3)
        assert h.cap_adjustment == pytest.approx(85.0 - 33.3333, abs=1e-3)

    def test_40pct_cap_reduces_2b_first(self) -> None:
        # Large L2B that survives the 15% cap but the 40% cap bites,
        # reducing Level 2B before Level 2A.
        h = stock_of_hqla(10.0, 20.0, 60.0)
        # l2a=17, l2b=30; l2b_cap=(15/85)*(10+17)=4.76 -> l2b=4.76
        # l2_total=21.76; l2_cap=(40/60)*10=6.667 -> excess=15.09
        # reduce_2b=min(4.76,15.09)=4.76 -> l2b=0; excess=10.33 -> l2a=17-10.33=6.667
        assert h.level2b == pytest.approx(0.0, abs=1e-6)
        assert h.level2a == pytest.approx(6.6667, abs=1e-3)
        assert h.total_hqla == pytest.approx(16.6667, abs=1e-3)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            stock_of_hqla(-1.0, 0.0, 0.0)


class TestNetCashOutflows:
    def test_inflows_below_cap(self) -> None:
        # inflows 50 < 75% of 100 -> net = 100 - 50 = 50.
        assert net_cash_outflows(100.0, 50.0) == pytest.approx(50.0)

    def test_inflows_capped(self) -> None:
        # inflows 90 capped to 75 -> net = 25.
        assert net_cash_outflows(100.0, 90.0) == pytest.approx(25.0)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            net_cash_outflows(-1.0, 0.0)


class TestLCR:
    def test_compliant(self) -> None:
        r = liquidity_coverage_ratio(120.0, 0.0, 0.0, total_outflows=100.0, total_inflows=0.0)
        assert r.lcr == pytest.approx(1.2)
        assert r.lcr_pct == pytest.approx(120.0)
        assert r.is_compliant is True

    def test_non_compliant(self) -> None:
        r = liquidity_coverage_ratio(80.0, 0.0, 0.0, total_outflows=100.0, total_inflows=0.0)
        assert r.lcr == pytest.approx(0.8)
        assert r.is_compliant is False

    def test_zero_net_outflows_is_infinite(self) -> None:
        r = liquidity_coverage_ratio(100.0, 0.0, 0.0, total_outflows=0.0, total_inflows=0.0)
        assert r.net_cash_outflows == 0.0
        assert r.lcr == float("inf")
        assert r.is_compliant is True
