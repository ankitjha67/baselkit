"""Tests for securitisation risk weight approaches — BCBS d424, CRE40-44."""

import pytest

from creditriskengine.rwa.securitisation import (
    SecuritisationPool,
    SecuritisationTranche,
    assign_securitisation_approach,
    sec_erba_risk_weight,
    sec_irba_risk_weight,
    sec_sa_risk_weight,
)


class TestSecIRBA:
    """SEC-IRBA risk weight tests per CRE41."""

    def test_senior_tranche_low_kirb(self) -> None:
        pool = SecuritisationPool(kirb=0.02, ksa=0.08, pool_ead=1_000_000, n_effective=100)
        tranche = SecuritisationTranche(
            tranche_id="T1", attachment_point=0.10, detachment_point=1.00,
            notional=900_000, is_senior=True,
        )
        rw = sec_irba_risk_weight(tranche, pool)
        assert rw >= 0.15  # Floor is 15%
        assert rw <= 12.50  # Cap is 1250%

    def test_mezzanine_tranche_higher_or_equal_rw(self) -> None:
        pool = SecuritisationPool(kirb=0.05, ksa=0.10, pool_ead=1_000_000, n_effective=50)
        senior = SecuritisationTranche(
            tranche_id="T1", attachment_point=0.15, detachment_point=1.00,
            notional=850_000, is_senior=True,
        )
        mezzanine = SecuritisationTranche(
            tranche_id="T2", attachment_point=0.05, detachment_point=0.15,
            notional=100_000, is_senior=False,
        )
        rw_senior = sec_irba_risk_weight(senior, pool)
        rw_mezz = sec_irba_risk_weight(mezzanine, pool)
        # Mezzanine should have >= senior RW; both may hit the 15% floor
        assert rw_mezz >= rw_senior

    def test_tranche_below_kirb_gets_1250(self) -> None:
        pool = SecuritisationPool(kirb=0.10, ksa=0.15, pool_ead=1_000_000, n_effective=50)
        tranche = SecuritisationTranche(
            tranche_id="T3", attachment_point=0.0, detachment_point=0.05,
            notional=50_000, is_senior=False,
        )
        rw = sec_irba_risk_weight(tranche, pool)
        assert rw == pytest.approx(12.50, abs=0.01)


class TestSecERBA:
    """SEC-ERBA risk weight tests per CRE43."""

    def test_senior_cqs1(self) -> None:
        tranche = SecuritisationTranche(
            tranche_id="T1", attachment_point=0.10, detachment_point=1.00,
            notional=900_000, external_rating=1, is_senior=True,
            maturity_years=1.0,  # Use 1y maturity to get base RW without adjustment
        )
        rw = sec_erba_risk_weight(tranche)
        assert rw == pytest.approx(0.15)

    def test_non_senior_higher_rw(self) -> None:
        senior = SecuritisationTranche(
            tranche_id="T1", attachment_point=0.10, detachment_point=1.00,
            notional=900_000, external_rating=3, is_senior=True,
            maturity_years=1.0,
        )
        non_senior = SecuritisationTranche(
            tranche_id="T2", attachment_point=0.05, detachment_point=0.10,
            notional=50_000, external_rating=3, is_senior=False,
            maturity_years=1.0,
        )
        assert sec_erba_risk_weight(non_senior) >= sec_erba_risk_weight(senior)


class TestSecSA:
    """SEC-SA risk weight tests per CRE42."""

    def test_senior_tranche(self) -> None:
        pool = SecuritisationPool(kirb=0.05, ksa=0.08, pool_ead=1_000_000, n_effective=100)
        tranche = SecuritisationTranche(
            tranche_id="T1", attachment_point=0.10, detachment_point=1.00,
            notional=900_000, is_senior=True,
        )
        rw = sec_sa_risk_weight(tranche, pool)
        assert rw >= 0.15
        assert rw <= 12.50


class TestApproachAssignment:
    """Test approach hierarchy per CRE40.4."""

    def test_irb_preferred_when_available(self) -> None:
        pool = SecuritisationPool(kirb=0.05, ksa=0.08, pool_ead=1_000_000, n_effective=50)
        tranche = SecuritisationTranche(
            tranche_id="T1", attachment_point=0.10, detachment_point=1.00,
            notional=900_000, external_rating=1, is_senior=True,
        )
        approach = assign_securitisation_approach(tranche, pool, has_irb_approval=True)
        assert approach == "SEC-IRBA"

    def test_erba_when_rated_no_irb(self) -> None:
        pool = SecuritisationPool(kirb=0.0, ksa=0.08, pool_ead=1_000_000, n_effective=50)
        tranche = SecuritisationTranche(
            tranche_id="T1", attachment_point=0.10, detachment_point=1.00,
            notional=900_000, external_rating=2, is_senior=True,
        )
        approach = assign_securitisation_approach(tranche, pool, has_irb_approval=False)
        assert approach == "SEC-ERBA"

    def test_sa_fallback(self) -> None:
        pool = SecuritisationPool(kirb=0.0, ksa=0.08, pool_ead=1_000_000, n_effective=50)
        tranche = SecuritisationTranche(
            tranche_id="T1", attachment_point=0.10, detachment_point=1.00,
            notional=900_000, is_senior=True,
        )
        approach = assign_securitisation_approach(tranche, pool, has_irb_approval=False)
        assert approach == "SEC-SA"
