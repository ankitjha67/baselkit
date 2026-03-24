"""Tests for covered bonds and MDB tiering — BCBS d424 CRE20."""

import pytest

from creditriskengine.core.types import (
    CreditQualityStep,
    SAExposureClass,
)
from creditriskengine.rwa.standardized.credit_risk_sa import (
    BANK_ECRA_RW,
    CORPORATE_RW,
    COVERED_BOND_RW,
    assign_sa_risk_weight,
    get_bank_risk_weight,
    get_covered_bond_risk_weight,
    get_mdb_risk_weight,
)


class TestCoveredBondRiskWeight:
    """BCBS CRE20.60-67, Table 10."""

    @pytest.mark.parametrize(
        "cqs,expected",
        [
            (CreditQualityStep.CQS_1, 10.0),
            (CreditQualityStep.CQS_2, 20.0),
            (CreditQualityStep.CQS_3, 20.0),
            (CreditQualityStep.CQS_4, 50.0),
            (CreditQualityStep.CQS_5, 50.0),
            (CreditQualityStep.CQS_6, 100.0),
        ],
    )
    def test_qualifying_rated_by_cqs(
        self, cqs: CreditQualityStep, expected: float
    ) -> None:
        """Qualifying covered bonds with a rating use Table 10."""
        assert get_covered_bond_risk_weight(cqs, is_qualifying=True) == expected

    def test_qualifying_unrated_falls_back_to_issuer(self) -> None:
        """Unrated qualifying covered bonds use issuer bank RW."""
        rw = get_covered_bond_risk_weight(
            CreditQualityStep.UNRATED,
            is_qualifying=True,
            issuer_cqs=CreditQualityStep.CQS_2,
        )
        expected = get_bank_risk_weight(cqs=CreditQualityStep.CQS_2)
        assert rw == expected

    def test_qualifying_unrated_no_issuer_defaults(self) -> None:
        """Unrated qualifying with no issuer CQS uses unrated bank default."""
        rw = get_covered_bond_risk_weight(
            CreditQualityStep.UNRATED, is_qualifying=True
        )
        assert rw == get_bank_risk_weight()  # 50.0

    def test_non_qualifying_uses_issuer_bank_rw(self) -> None:
        """Non-qualifying covered bonds use issuer bank ECRA table."""
        rw = get_covered_bond_risk_weight(
            CreditQualityStep.CQS_1,
            is_qualifying=False,
            issuer_cqs=CreditQualityStep.CQS_3,
        )
        expected = get_bank_risk_weight(cqs=CreditQualityStep.CQS_3)
        assert rw == expected

    def test_non_qualifying_no_issuer_defaults(self) -> None:
        """Non-qualifying with no issuer CQS uses unrated bank default."""
        rw = get_covered_bond_risk_weight(
            CreditQualityStep.CQS_1, is_qualifying=False
        )
        assert rw == get_bank_risk_weight()  # 50.0

    def test_qualifying_lower_than_bank(self) -> None:
        """Qualifying covered bond RW should be <= issuer bank RW for same CQS."""
        for cqs in [
            CreditQualityStep.CQS_1,
            CreditQualityStep.CQS_2,
            CreditQualityStep.CQS_3,
        ]:
            cb_rw = get_covered_bond_risk_weight(cqs, is_qualifying=True)
            bank_rw = get_bank_risk_weight(cqs=cqs)
            assert cb_rw <= bank_rw, (
                f"Covered bond RW ({cb_rw}) > bank RW ({bank_rw}) for {cqs}"
            )

    def test_table_completeness(self) -> None:
        """All rated CQS values should be in the COVERED_BOND_RW table."""
        for cqs in [
            CreditQualityStep.CQS_1,
            CreditQualityStep.CQS_2,
            CreditQualityStep.CQS_3,
            CreditQualityStep.CQS_4,
            CreditQualityStep.CQS_5,
            CreditQualityStep.CQS_6,
        ]:
            assert cqs.value in COVERED_BOND_RW


class TestCoveredBondDispatcher:
    """Covered bond routing through assign_sa_risk_weight."""

    def test_covered_bond_dispatch_qualifying(self) -> None:
        rw = assign_sa_risk_weight(
            SAExposureClass.COVERED_BOND,
            cqs=CreditQualityStep.CQS_1,
            is_qualifying=True,
        )
        assert rw == 10.0

    def test_covered_bond_dispatch_non_qualifying(self) -> None:
        rw = assign_sa_risk_weight(
            SAExposureClass.COVERED_BOND,
            cqs=CreditQualityStep.CQS_1,
            is_qualifying=False,
            issuer_cqs=CreditQualityStep.CQS_1,
        )
        assert rw == get_bank_risk_weight(cqs=CreditQualityStep.CQS_1)

    def test_covered_bond_dispatch_unrated(self) -> None:
        rw = assign_sa_risk_weight(
            SAExposureClass.COVERED_BOND,
            cqs=CreditQualityStep.UNRATED,
            issuer_cqs=CreditQualityStep.CQS_2,
        )
        assert rw == get_bank_risk_weight(cqs=CreditQualityStep.CQS_2)


class TestMDBRiskWeight:
    """BCBS CRE20.8-9 — Multilateral development banks."""

    def test_category_1_qualifying_is_zero(self) -> None:
        """Qualifying MDBs (Category 1) receive 0% RW."""
        assert get_mdb_risk_weight(mdb_category=1) == 0.0

    def test_category_1_ignores_cqs(self) -> None:
        """Category 1 MDB should be 0% regardless of CQS."""
        for cqs in CreditQualityStep:
            assert get_mdb_risk_weight(mdb_category=1, cqs=cqs) == 0.0

    @pytest.mark.parametrize(
        "cqs,expected",
        [
            (CreditQualityStep.CQS_1, 20.0),
            (CreditQualityStep.CQS_2, 30.0),
            (CreditQualityStep.CQS_3, 50.0),
            (CreditQualityStep.CQS_4, 100.0),
            (CreditQualityStep.CQS_5, 100.0),
            (CreditQualityStep.CQS_6, 150.0),
            (CreditQualityStep.UNRATED, 50.0),
        ],
    )
    def test_category_2_uses_bank_ecra(
        self, cqs: CreditQualityStep, expected: float
    ) -> None:
        """Category 2 MDBs use the bank ECRA table."""
        assert get_mdb_risk_weight(mdb_category=2, cqs=cqs) == expected

    def test_category_2_matches_bank_table(self) -> None:
        """Category 2 result should match BANK_ECRA_RW directly."""
        for cqs in CreditQualityStep:
            mdb_rw = get_mdb_risk_weight(mdb_category=2, cqs=cqs)
            bank_rw = BANK_ECRA_RW.get(cqs.value, 50.0)
            assert mdb_rw == bank_rw

    @pytest.mark.parametrize(
        "cqs,expected",
        [
            (CreditQualityStep.CQS_1, 20.0),
            (CreditQualityStep.CQS_2, 50.0),
            (CreditQualityStep.CQS_3, 75.0),
            (CreditQualityStep.CQS_4, 100.0),
            (CreditQualityStep.CQS_5, 150.0),
            (CreditQualityStep.CQS_6, 150.0),
            (CreditQualityStep.UNRATED, 100.0),
        ],
    )
    def test_non_qualifying_uses_corporate_table(
        self, cqs: CreditQualityStep, expected: float
    ) -> None:
        """Non-qualifying MDBs (category 3+) use corporate table."""
        assert get_mdb_risk_weight(mdb_category=3, cqs=cqs) == expected

    def test_non_qualifying_matches_corporate_table(self) -> None:
        """Category 3+ result should match CORPORATE_RW directly."""
        for cqs in CreditQualityStep:
            mdb_rw = get_mdb_risk_weight(mdb_category=3, cqs=cqs)
            corp_rw = CORPORATE_RW.get(cqs.value, 100.0)
            assert mdb_rw == corp_rw

    def test_high_category_still_corporate(self) -> None:
        """Category 4, 5, etc. should also use corporate table."""
        rw = get_mdb_risk_weight(mdb_category=5, cqs=CreditQualityStep.CQS_1)
        assert rw == CORPORATE_RW[CreditQualityStep.CQS_1.value]


class TestMDBDispatcher:
    """MDB routing through assign_sa_risk_weight."""

    def test_mdb_dispatch_category_1(self) -> None:
        rw = assign_sa_risk_weight(
            SAExposureClass.MDB, mdb_category=1
        )
        assert rw == 0.0

    def test_mdb_dispatch_category_2(self) -> None:
        rw = assign_sa_risk_weight(
            SAExposureClass.MDB,
            mdb_category=2,
            cqs=CreditQualityStep.CQS_1,
        )
        assert rw == 20.0

    def test_mdb_dispatch_category_3(self) -> None:
        rw = assign_sa_risk_weight(
            SAExposureClass.MDB,
            mdb_category=3,
            cqs=CreditQualityStep.CQS_2,
        )
        assert rw == 50.0

    def test_mdb_dispatch_default_is_qualifying(self) -> None:
        """Default mdb_category=1, so dispatching MDB without args gives 0%."""
        rw = assign_sa_risk_weight(SAExposureClass.MDB)
        assert rw == 0.0
