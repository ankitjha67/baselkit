"""Tests for CVA risk charge — BCBS d424, CVA25-27."""

import pytest

from creditriskengine.rwa.cva import (
    BETA,
    RHO,
    CVACounterparty,
    CVAHedge,
    ba_cva_capital,
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
