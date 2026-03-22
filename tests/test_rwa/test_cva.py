"""Tests for CVA risk charge — BCBS d424, CVA25-27."""

import math

import pytest

from creditriskengine.rwa.cva import (
    BETA,
    RHO,
    SECTOR_RW,
    CVACounterparty,
    CVAHedge,
    _supervisory_discount_factor,
    ba_cva_capital,
    sa_cva_capital,
    scva_standalone,
)


class TestCVAParameters:
    """Verify regulatory constants."""

    def test_rho_value(self) -> None:
        assert RHO == 0.50

    def test_beta_value(self) -> None:
        assert BETA == 0.25


class TestStandaloneCVA:
    """Test standalone CVA capital for a single counterparty."""

    def test_positive_capital(self) -> None:
        cp = CVACounterparty(
            counterparty_id="CP1", ead=10_000_000,
            credit_spread=100.0, maturity_years=3.0,
            sector="corporate",
        )
        scva = scva_standalone(cp)
        assert scva > 0.0

    def test_exempt_counterparty_zero(self) -> None:
        cp = CVACounterparty(
            counterparty_id="SOV1", ead=50_000_000,
            credit_spread=10.0, maturity_years=5.0,
            sector="sovereign", is_exempt=True,
        )
        scva = scva_standalone(cp)
        assert scva == 0.0

    def test_higher_ead_higher_scva(self) -> None:
        cp_small = CVACounterparty(
            counterparty_id="CP1", ead=1_000_000,
            credit_spread=100.0, maturity_years=2.5,
        )
        cp_large = CVACounterparty(
            counterparty_id="CP2", ead=100_000_000,
            credit_spread=100.0, maturity_years=2.5,
        )
        assert scva_standalone(cp_large) > scva_standalone(cp_small)


class TestBACVA:
    """Test BA-CVA capital calculation per CVA25."""

    def test_single_counterparty(self) -> None:
        counterparties = [
            CVACounterparty(
                counterparty_id="CP1", ead=10_000_000,
                credit_spread=100.0, maturity_years=3.0,
            ),
        ]
        capital = ba_cva_capital(counterparties)
        assert capital > 0.0

    def test_multiple_counterparties(self) -> None:
        counterparties = [
            CVACounterparty(
                counterparty_id=f"CP{i}", ead=5_000_000,
                credit_spread=80.0 + i * 20, maturity_years=2.5,
            )
            for i in range(5)
        ]
        capital = ba_cva_capital(counterparties)
        assert capital > 0.0

    def test_hedges_reduce_capital(self) -> None:
        counterparties = [
            CVACounterparty(
                counterparty_id="CP1", ead=10_000_000,
                credit_spread=100.0, maturity_years=3.0,
            ),
        ]
        hedges = [
            CVAHedge(counterparty_id="CP1", notional=5_000_000, maturity_years=3.0),
        ]
        capital_unhedged = ba_cva_capital(counterparties)
        capital_hedged = ba_cva_capital(counterparties, hedges)
        assert capital_hedged <= capital_unhedged

    def test_exempt_excluded(self) -> None:
        counterparties = [
            CVACounterparty(
                counterparty_id="CP1", ead=10_000_000,
                credit_spread=100.0, maturity_years=3.0, is_exempt=True,
            ),
        ]
        capital = ba_cva_capital(counterparties)
        assert capital == 0.0

    def test_empty_counterparties(self) -> None:
        assert ba_cva_capital([]) == 0.0

    def test_all_exempt(self) -> None:
        counterparties = [
            CVACounterparty(
                counterparty_id=f"CP{i}", ead=1_000_000,
                credit_spread=50.0, maturity_years=2.0, is_exempt=True,
            )
            for i in range(3)
        ]
        assert ba_cva_capital(counterparties) == 0.0

    def test_multiple_hedges_same_counterparty(self) -> None:
        counterparties = [
            CVACounterparty(
                counterparty_id="CP1", ead=10_000_000,
                credit_spread=100.0, maturity_years=3.0,
            ),
        ]
        hedges = [
            CVAHedge(counterparty_id="CP1", notional=2_000_000, maturity_years=3.0),
            CVAHedge(counterparty_id="CP1", notional=3_000_000, maturity_years=2.0),
        ]
        capital = ba_cva_capital(counterparties, hedges)
        assert capital > 0.0


class TestSupervisoryDiscountFactor:
    """Test the supervisory discount factor calculation."""

    def test_zero_maturity(self) -> None:
        assert _supervisory_discount_factor(0.0) == pytest.approx(1.0)

    def test_negative_maturity(self) -> None:
        assert _supervisory_discount_factor(-1.0) == pytest.approx(1.0)

    def test_positive_maturity(self) -> None:
        expected = (1.0 - math.exp(-0.25)) / 0.25
        assert _supervisory_discount_factor(5.0) == pytest.approx(expected)

    def test_very_small_maturity(self) -> None:
        df = _supervisory_discount_factor(1e-12)
        assert df == pytest.approx(1.0)

    def test_decreasing_with_maturity(self) -> None:
        df1 = _supervisory_discount_factor(1.0)
        df5 = _supervisory_discount_factor(5.0)
        assert df1 > df5


class TestSCVAStandaloneExtended:
    """Additional standalone CVA tests."""

    def test_unknown_sector_uses_fallback(self) -> None:
        cp = CVACounterparty(
            counterparty_id="CP1", ead=10_000_000,
            credit_spread=100.0, maturity_years=3.0,
            sector="nonexistent_sector",
        )
        scva = scva_standalone(cp)
        assert scva > 0.0

    def test_longer_maturity_higher_scva(self) -> None:
        cp_short = CVACounterparty(
            counterparty_id="CP1", ead=10_000_000,
            credit_spread=100.0, maturity_years=1.0,
        )
        cp_long = CVACounterparty(
            counterparty_id="CP2", ead=10_000_000,
            credit_spread=100.0, maturity_years=5.0,
        )
        assert scva_standalone(cp_long) > scva_standalone(cp_short)

    def test_formula_values(self) -> None:
        cp = CVACounterparty(
            counterparty_id="CP1", ead=10_000_000,
            credit_spread=100.0, maturity_years=2.5,
            sector="corporate_ig",
        )
        rw = SECTOR_RW["corporate_ig"]
        df = _supervisory_discount_factor(2.5)
        expected = (2.0 / 3.0) * rw * 2.5 * 10_000_000 * df
        assert scva_standalone(cp) == pytest.approx(expected)


class TestSACVA:
    """Test SA-CVA capital calculation per CVA26."""

    def test_empty_counterparties(self) -> None:
        assert sa_cva_capital([]) == 0.0

    def test_single_counterparty(self) -> None:
        counterparties = [
            CVACounterparty(
                counterparty_id="CP1", ead=10_000_000,
                credit_spread=100.0, maturity_years=3.0,
                sector="corporate_ig",
            ),
        ]
        capital = sa_cva_capital(counterparties)
        assert capital > 0.0

    def test_multiple_same_sector(self) -> None:
        counterparties = [
            CVACounterparty(
                counterparty_id=f"CP{i}", ead=5_000_000,
                credit_spread=80.0, maturity_years=2.5,
                sector="corporate_ig",
            )
            for i in range(3)
        ]
        capital = sa_cva_capital(counterparties)
        assert capital > 0.0

    def test_multiple_different_sectors(self) -> None:
        counterparties = [
            CVACounterparty(
                counterparty_id="CP1", ead=5_000_000,
                credit_spread=80.0, maturity_years=2.5,
                sector="corporate_ig",
            ),
            CVACounterparty(
                counterparty_id="CP2", ead=5_000_000,
                credit_spread=80.0, maturity_years=2.5,
                sector="financial_ig",
            ),
        ]
        capital = sa_cva_capital(counterparties)
        assert capital > 0.0

    def test_exempt_excluded(self) -> None:
        counterparties = [
            CVACounterparty(
                counterparty_id="CP1", ead=10_000_000,
                credit_spread=100.0, maturity_years=3.0, is_exempt=True,
            ),
        ]
        assert sa_cva_capital(counterparties) == 0.0

    def test_hedges_reduce_capital(self) -> None:
        counterparties = [
            CVACounterparty(
                counterparty_id="CP1", ead=10_000_000,
                credit_spread=100.0, maturity_years=3.0,
                sector="corporate_ig",
            ),
        ]
        hedges = [
            CVAHedge(counterparty_id="CP1", notional=5_000_000, maturity_years=3.0),
        ]
        k_unhedged = sa_cva_capital(counterparties)
        k_hedged = sa_cva_capital(counterparties, hedges)
        assert k_hedged <= k_unhedged

    def test_unknown_sector_uses_other(self) -> None:
        counterparties = [
            CVACounterparty(
                counterparty_id="CP1", ead=10_000_000,
                credit_spread=100.0, maturity_years=3.0,
                sector="exotic_sector",
            ),
        ]
        capital = sa_cva_capital(counterparties)
        assert capital > 0.0
