"""Tests for securitisation risk weight approaches — BCBS d424, CRE40-44."""

import pytest

from creditriskengine.rwa.securitisation import (
    SecuritisationPool,
    SecuritisationTranche,
    _compute_a_parameter,
    _kssfa,
    assign_securitisation_approach,
    sec_erba_risk_weight,
    sec_irba_risk_weight,
    sec_risk_weight_cap,
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


class TestKSSFA:
    """Tests for the KSSFA supervisory formula internals."""

    def test_a_near_zero_returns_thickness(self) -> None:
        """Line 154: abs(a) < 1e-10 branch."""
        result = _kssfa(0.0, 0.05, 0.20)
        assert result == pytest.approx(0.15)

    def test_denominator_near_zero_returns_thickness(self) -> None:
        """Line 159: abs(denominator) < 1e-15 branch.

        We need abs(a) >= 1e-10 but a*(exp(a)-1) ~ 0.
        For very small a (but >= 1e-10), exp(a) ~ 1 + a, so
        denominator ~ a * a which is ~1e-20 < 1e-15.
        """
        result = _kssfa(1e-10, 0.05, 0.20)
        assert result == pytest.approx(0.15)


class TestComputeAParameter:
    """Tests for _compute_a_parameter."""

    def test_zero_capital_req_returns_zero(self) -> None:
        """Line 231: capital_req <= 0 branch."""
        assert _compute_a_parameter(0.5, 0.0, 0.05, 0.20) == 0.0

    def test_negative_capital_req_returns_zero(self) -> None:
        """Line 231: capital_req <= 0 branch (negative)."""
        assert _compute_a_parameter(0.5, -0.01, 0.05, 0.20) == 0.0

    def test_zero_p_returns_zero(self) -> None:
        """Line 231: p <= 0 branch."""
        assert _compute_a_parameter(0.0, 0.05, 0.05, 0.20) == 0.0


class TestSecSATrancheBelowKSA:
    """SEC-SA tranche below KSA tests."""

    def test_tranche_below_ksa_gets_1250(self) -> None:
        """Lines 363-369: d_point <= ka returns RW_CAP."""
        pool = SecuritisationPool(kirb=0.05, ksa=0.15, pool_ead=1_000_000, n_effective=50)
        tranche = SecuritisationTranche(
            tranche_id="T_SA_below", attachment_point=0.0, detachment_point=0.10,
            notional=100_000, is_senior=False,
        )
        rw = sec_sa_risk_weight(tranche, pool)
        assert rw == pytest.approx(12.50, abs=0.01)


class TestSecERBAEdgeCases:
    """SEC-ERBA edge case tests."""

    def test_no_external_rating_raises(self) -> None:
        """Line 439: external_rating is None raises ValueError."""
        tranche = SecuritisationTranche(
            tranche_id="T_no_rating", attachment_point=0.05, detachment_point=0.20,
            notional=100_000, external_rating=None, is_senior=False,
        )
        with pytest.raises(ValueError, match="no external rating"):
            sec_erba_risk_weight(tranche)

    def test_thin_tranche_uses_thin_table(self) -> None:
        """Line 451: non-senior thin tranche (thickness < 0.03)."""
        tranche = SecuritisationTranche(
            tranche_id="T_thin", attachment_point=0.05, detachment_point=0.07,
            notional=20_000, external_rating=1, is_senior=False,
            maturity_years=1.0,
        )
        rw = sec_erba_risk_weight(tranche)
        # Thin table CQS 1 = 0.25
        assert rw == pytest.approx(0.25)

    def test_unknown_cqs_returns_1250(self) -> None:
        """Lines 457-463: CQS not in table returns RW_CAP."""
        tranche = SecuritisationTranche(
            tranche_id="T_bad_cqs", attachment_point=0.05, detachment_point=0.20,
            notional=100_000, external_rating=99, is_senior=True,
            maturity_years=1.0,
        )
        rw = sec_erba_risk_weight(tranche)
        assert rw == pytest.approx(12.50)

    def test_resecuritisation_doubles_rw(self) -> None:
        """Line 472: is_resecuritisation multiplies RW by 2."""
        tranche = SecuritisationTranche(
            tranche_id="T_resec", attachment_point=0.10, detachment_point=1.00,
            notional=900_000, external_rating=1, is_senior=True,
            maturity_years=1.0, is_resecuritisation=True,
        )
        rw = sec_erba_risk_weight(tranche)
        # Base senior CQS 1 = 0.15, doubled = 0.30
        assert rw == pytest.approx(0.30)


class TestSecRiskWeightCap:
    """Tests for sec_risk_weight_cap — lines 517-526."""

    def test_zero_notional_returns_cap(self) -> None:
        """Line 517-518: notional <= 0 returns RW_CAP."""
        pool = SecuritisationPool(kirb=0.05, ksa=0.08, pool_ead=1_000_000, n_effective=50)
        tranche = SecuritisationTranche(
            tranche_id="T_zero", attachment_point=0.05, detachment_point=0.20,
            notional=0.0, is_senior=False,
        )
        rw = sec_risk_weight_cap(tranche, pool)
        assert rw == pytest.approx(12.50)

    def test_positive_notional_returns_prorata_cap(self) -> None:
        """Lines 521-526: normal pro-rata calculation."""
        pool = SecuritisationPool(kirb=0.05, ksa=0.08, pool_ead=1_000_000, n_effective=50)
        tranche = SecuritisationTranche(
            tranche_id="T_cap", attachment_point=0.05, detachment_point=0.20,
            notional=150_000, is_senior=False,
        )
        rw = sec_risk_weight_cap(tranche, pool)
        # tranche_share = 150_000 / 1_000_000 = 0.15
        # pool_capital = 0.05 * 1_000_000 = 50_000
        # max_capital = 50_000 * 0.15 = 7_500
        # rw_max = 7_500 / 150_000 * 12.5 = 0.625
        assert rw == pytest.approx(0.625)
        assert rw <= 12.50
