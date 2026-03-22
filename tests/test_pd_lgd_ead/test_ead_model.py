"""Tests for EAD modeling: CCF, regulatory EAD, term structure."""

import numpy as np
import pytest

from creditriskengine.models.ead.ead_model import (
    CCF_FLOOR_AIRB,
    EADModel,
    apply_ccf_floor,
    calculate_ead,
    ead_term_structure,
    estimate_ccf,
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
