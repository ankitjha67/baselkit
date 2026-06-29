"""Tests for EAD modeling: CCF, regulatory EAD, term structure."""

import numpy as np
import pytest

from creditriskengine.models.ead.ead_model import (
    CCF_FLOOR_AIRB,
    EADModel,
    amortising_balance_schedule,
    apply_ccf_floor,
    calculate_ead,
    ead_term_structure,
    ead_term_structure_from_schedule,
    estimate_ccf,
    get_sa_ccf,
    get_supervisory_ccf,
)


class TestCalculateEAD:
    def test_fully_drawn(self) -> None:
        ead = calculate_ead(drawn_amount=1000.0, undrawn_commitment=0.0, ccf=0.75)
        assert ead == pytest.approx(1000.0)

    def test_with_undrawn(self) -> None:
        ead = calculate_ead(drawn_amount=800.0, undrawn_commitment=200.0, ccf=0.75)
        assert ead == pytest.approx(950.0)

    def test_zero_ccf(self) -> None:
        ead = calculate_ead(drawn_amount=500.0, undrawn_commitment=500.0, ccf=0.0)
        assert ead == pytest.approx(500.0)

    def test_full_ccf(self) -> None:
        ead = calculate_ead(drawn_amount=500.0, undrawn_commitment=500.0, ccf=1.0)
        assert ead == pytest.approx(1000.0)


class TestEstimateCCF:
    def test_basic(self) -> None:
        # Drawn at ref = 700, Limit = 1000, EAD at default = 850
        # CCF = (850 - 700) / (1000 - 700) = 150/300 = 0.5
        ccf = estimate_ccf(ead_at_default=850.0, drawn_at_reference=700.0, limit=1000.0)
        assert ccf == pytest.approx(0.5)

    def test_fully_drawn_at_reference(self) -> None:
        ccf = estimate_ccf(ead_at_default=1000.0, drawn_at_reference=1000.0, limit=1000.0)
        assert ccf == pytest.approx(1.0)

    def test_clipped(self) -> None:
        ccf = estimate_ccf(ead_at_default=500.0, drawn_at_reference=700.0, limit=1000.0)
        assert ccf == pytest.approx(0.0)  # negative → clipped to 0


class TestSupervisoryCCF:
    def test_committed_other(self) -> None:
        assert get_supervisory_ccf("committed_other") == pytest.approx(0.75)

    def test_direct_credit_sub(self) -> None:
        assert get_supervisory_ccf("direct_credit_substitutes") == pytest.approx(1.0)

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown facility type"):
            get_supervisory_ccf("unknown_type")


class TestApplyCCFFloor:
    def test_airb_floor(self) -> None:
        assert apply_ccf_floor(0.30, approach="airb") == CCF_FLOOR_AIRB

    def test_airb_above_floor(self) -> None:
        assert apply_ccf_floor(0.80, approach="airb") == pytest.approx(0.80)

    def test_firb_no_floor(self) -> None:
        assert apply_ccf_floor(0.30, approach="firb") == pytest.approx(0.30)


class TestEADTermStructure:
    def test_no_amortization(self) -> None:
        ts = ead_term_structure(800.0, 200.0, ccf=0.75, n_periods=3)
        expected = 800.0 + 0.75 * 200.0
        np.testing.assert_allclose(ts, [expected, expected, expected])

    def test_with_amortization(self) -> None:
        ts = ead_term_structure(1000.0, 0.0, ccf=0.0, n_periods=3, amortization_rate=0.10)
        np.testing.assert_allclose(ts, [1000.0, 900.0, 810.0])

    def test_shape(self) -> None:
        ts = ead_term_structure(500.0, 100.0, ccf=0.5, n_periods=5)
        assert len(ts) == 5

    def test_ead_non_negative(self) -> None:
        # Fully amortized with large rate
        ts = ead_term_structure(100.0, 0.0, ccf=0.0, n_periods=5, amortization_rate=0.99)
        assert all(t >= 0.0 for t in ts)


class TestAmortisingBalanceSchedule:
    def test_zero_rate_straight_line(self) -> None:
        # 0% rate, fully amortising over 4 periods → 75, 50, 25, 0
        sched = amortising_balance_schedule(100.0, 0.0, 4)
        np.testing.assert_allclose(sched, [75.0, 50.0, 25.0, 0.0])

    def test_fully_amortising_ends_at_zero(self) -> None:
        sched = amortising_balance_schedule(1000.0, 0.06, 10)
        assert sched[-1] == pytest.approx(0.0, abs=1e-6)

    def test_monotonic_non_increasing(self) -> None:
        sched = amortising_balance_schedule(1000.0, 0.08, 12, periods_per_year=12)
        assert np.all(np.diff(sched) <= 1e-9)

    def test_annuity_balance_known_value(self) -> None:
        # Annual annuity: P=1000, i=10%, n=3.
        # Instalment = 1000 * 0.1 / (1 - 1.1^-3) = 402.1148...
        # End-of-period 1 balance = 1000*1.1 - 402.1148 = 697.885
        sched = amortising_balance_schedule(1000.0, 0.10, 3)
        assert sched[0] == pytest.approx(697.8852, abs=1e-3)
        assert sched[-1] == pytest.approx(0.0, abs=1e-6)

    def test_bullet_holds_then_repays(self) -> None:
        # Pure bullet: balance held until maturity, repaid at the end.
        sched = amortising_balance_schedule(1000.0, 0.05, 5, balloon_fraction=1.0)
        np.testing.assert_allclose(sched[:-1], [1000.0, 1000.0, 1000.0, 1000.0])
        assert sched[-1] == pytest.approx(0.0)

    def test_balloon_partial(self) -> None:
        # 50% balloon: amortises down toward the balloon, never below it
        # until the final maturity repayment.
        sched = amortising_balance_schedule(1000.0, 0.06, 5, balloon_fraction=0.5)
        assert all(sched[:-1] >= 500.0 - 1e-6)
        assert sched[-1] == pytest.approx(0.0)

    def test_zero_principal(self) -> None:
        sched = amortising_balance_schedule(0.0, 0.05, 4)
        np.testing.assert_allclose(sched, np.zeros(4))

    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError, match="principal must be"):
            amortising_balance_schedule(-1.0, 0.05, 4)
        with pytest.raises(ValueError, match="n_periods must be"):
            amortising_balance_schedule(100.0, 0.05, 0)
        with pytest.raises(ValueError, match="balloon_fraction"):
            amortising_balance_schedule(100.0, 0.05, 4, balloon_fraction=1.5)
        with pytest.raises(ValueError, match="periods_per_year must be"):
            amortising_balance_schedule(100.0, 0.05, 4, periods_per_year=0)


class TestGetSACCF:
    def test_baseline_lookup(self) -> None:
        assert get_sa_ccf("unconditionally_cancellable") == pytest.approx(0.10)

    def test_unknown_facility_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown SA facility type"):
            get_sa_ccf("not_a_real_facility")


class TestEADTermStructureFromSchedule:
    def test_drawn_only_matches_balance_schedule(self) -> None:
        sched = amortising_balance_schedule(1000.0, 0.06, 5)
        ead = ead_term_structure_from_schedule(1000.0, 0.06, 5)
        np.testing.assert_allclose(ead, sched)

    def test_adds_undrawn_ccf(self) -> None:
        ead = ead_term_structure_from_schedule(
            1000.0, 0.06, 5, undrawn_commitment=400.0, ccf=0.5
        )
        sched = amortising_balance_schedule(1000.0, 0.06, 5)
        np.testing.assert_allclose(ead, sched + 0.5 * 400.0)

    def test_feeds_ecl_lifetime(self) -> None:
        # End-to-end: a Stage 2 amortising loan whose EAD declines over life.
        from creditriskengine.ecl.ifrs9.ecl_calc import ecl_lifetime

        ead_curve = ead_term_structure_from_schedule(100_000.0, 0.07, 5)
        marginal_pds = np.full(5, 0.02)
        ecl_amort = ecl_lifetime(marginal_pds, lgds=0.45, eads=ead_curve, eir=0.07)
        ecl_flat = ecl_lifetime(marginal_pds, lgds=0.45, eads=100_000.0, eir=0.07)
        # Amortising EAD must give a strictly smaller lifetime ECL than
        # holding today's balance flat.
        assert ecl_amort < ecl_flat
        assert ecl_amort > 0


class TestEADModel:
    """Test sklearn-compatible EAD model."""

    def test_fit_predict_supervisory(self) -> None:
        model = EADModel(ccf_method="supervisory", facility_type="committed_other")
        X = np.array([[800.0, 200.0], [600.0, 400.0], [900.0, 100.0]])  # noqa: N806
        y = np.array([950.0, 900.0, 975.0])
        model.fit(X, y)
        assert model.is_fitted_ is True
        preds = model.predict(X)
        assert len(preds) == 3
        # supervisory CCF for committed_other = 0.75
        expected_0 = 800.0 + 0.75 * 200.0
        assert preds[0] == pytest.approx(expected_0)

    def test_fit_predict_estimated(self) -> None:
        model = EADModel(ccf_method="estimated")
        X = np.array([[800.0, 200.0], [600.0, 400.0]])  # noqa: N806
        y = np.array([900.0, 800.0])
        model.fit(X, y)
        assert model.mean_ccf_ is not None
        preds = model.predict(X)
        assert len(preds) == 2

    def test_predict_before_fit_raises(self) -> None:
        model = EADModel()
        with pytest.raises(AssertionError):
            model.predict(np.array([[100.0, 50.0]]))

    def test_fit_single_column(self) -> None:
        model = EADModel(ccf_method="estimated")
        X = np.array([[800.0]])  # noqa: N806
        y = np.array([900.0])
        model.fit(X, y)
        assert model.mean_ccf_ == 0.75  # default when < 2 columns

    def test_fit_zero_undrawn(self) -> None:
        model = EADModel(ccf_method="estimated")
        X = np.array([[1000.0, 0.0], [500.0, 0.0]])  # noqa: N806
        y = np.array([1000.0, 500.0])
        model.fit(X, y)
        assert model.mean_ccf_ == 0.75  # default when no undrawn > 0

    def test_mean_ccf_clipped(self) -> None:
        model = EADModel(ccf_method="estimated")
        # EAD much larger than drawn + undrawn -> ccf > 1 -> should clip
        X = np.array([[100.0, 100.0]])  # noqa: N806
        y = np.array([500.0])
        model.fit(X, y)
        assert model.mean_ccf_ is not None
        assert model.mean_ccf_ <= 1.0
