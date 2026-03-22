"""Tests for LGD modeling: workout, downturn, term structure, floors."""

import numpy as np
import pytest

from creditriskengine.models.lgd.lgd_model import (
    LGD_FLOOR_SECURED_FINANCIAL,
    LGD_FLOOR_SECURED_OTHER,
    LGD_FLOOR_SECURED_RECEIVABLES,
    LGD_FLOOR_UNSECURED,
    SUPERVISORY_LGD_SENIOR_UNSECURED,
    SUPERVISORY_LGD_SUBORDINATED,
    LGDModel,
    apply_lgd_floor,
    downturn_lgd,
    lgd_term_structure,
    workout_lgd,
)


class TestWorkoutLGD:
    def test_full_recovery(self) -> None:
        lgd = workout_lgd(ead_at_default=1000.0, total_recoveries=1000.0, total_costs=0.0)
        assert lgd == pytest.approx(0.0)

    def test_zero_recovery(self) -> None:
        lgd = workout_lgd(ead_at_default=1000.0, total_recoveries=0.0, total_costs=0.0)
        assert lgd == pytest.approx(1.0)

    def test_partial_recovery(self) -> None:
        lgd = workout_lgd(ead_at_default=1000.0, total_recoveries=600.0, total_costs=100.0)
        assert lgd == pytest.approx(0.5)

    def test_with_discounting(self) -> None:
        lgd_no_disc = workout_lgd(1000.0, 500.0, 0.0, discount_rate=0.0)
        lgd_disc = workout_lgd(1000.0, 500.0, 0.0, discount_rate=0.05, time_to_recovery_years=2.0)
        assert lgd_disc > lgd_no_disc  # discounting reduces PV of recoveries

    def test_zero_ead(self) -> None:
        assert workout_lgd(0.0, 500.0, 0.0) == pytest.approx(1.0)

    def test_clipped_to_unit(self) -> None:
        # Costs exceed EAD → LGD capped at 1.0
        lgd = workout_lgd(100.0, 0.0, 200.0)
        assert lgd == pytest.approx(1.0)


class TestDownturnLGD:
    def test_additive(self) -> None:
        dt = downturn_lgd(0.40, downturn_add_on=0.10, method="additive")
        assert dt == pytest.approx(0.50)

    def test_haircut(self) -> None:
        dt = downturn_lgd(0.40, downturn_add_on=0.10, method="haircut")
        # 1 - (1-0.40)*(1-0.10) = 1 - 0.54 = 0.46
        assert dt == pytest.approx(0.46)

    def test_regulatory(self) -> None:
        dt = downturn_lgd(0.30, method="regulatory")
        assert dt == pytest.approx(max(0.30, 0.10 + 0.40 * 0.30))

    def test_capped_at_one(self) -> None:
        dt = downturn_lgd(0.95, downturn_add_on=0.20, method="additive")
        assert dt == pytest.approx(1.0)


class TestLGDTermStructure:
    def test_flat(self) -> None:
        ts = lgd_term_structure(0.45, n_periods=5)
        np.testing.assert_allclose(ts, [0.45] * 5)

    def test_with_recovery_curve(self) -> None:
        recovery = np.array([0.0, 0.20, 0.40, 0.55, 0.55])
        ts = lgd_term_structure(0.45, n_periods=5, recovery_curve=recovery)
        expected = 1.0 - recovery
        np.testing.assert_allclose(ts, expected)

    def test_padding(self) -> None:
        recovery = np.array([0.0, 0.20])
        ts = lgd_term_structure(0.45, n_periods=4, recovery_curve=recovery)
        assert len(ts) == 4
        assert ts[2] == pytest.approx(ts[1])  # padded with last value


class TestApplyLGDFloor:
    def test_unsecured_floor(self) -> None:
        assert apply_lgd_floor(0.10) == LGD_FLOOR_UNSECURED

    def test_above_floor(self) -> None:
        assert apply_lgd_floor(0.50) == pytest.approx(0.50)

    def test_secured_cre(self) -> None:
        assert apply_lgd_floor(0.05, is_secured=True, collateral_type="cre") == pytest.approx(0.10)


class TestApplyLGDFloorExtended:
    """Additional floor tests for all collateral types."""

    def test_secured_financial(self) -> None:
        result = apply_lgd_floor(0.0, is_secured=True, collateral_type="financial")
        assert result == LGD_FLOOR_SECURED_FINANCIAL

    def test_secured_receivables(self) -> None:
        result = apply_lgd_floor(0.05, is_secured=True, collateral_type="receivables")
        assert result == LGD_FLOOR_SECURED_RECEIVABLES

    def test_secured_rre(self) -> None:
        assert apply_lgd_floor(0.05, is_secured=True, collateral_type="rre") == pytest.approx(0.10)

    def test_secured_other(self) -> None:
        result = apply_lgd_floor(0.05, is_secured=True, collateral_type="other")
        assert result == LGD_FLOOR_SECURED_OTHER

    def test_above_floor_secured(self) -> None:
        result = apply_lgd_floor(0.50, is_secured=True, collateral_type="other")
        assert result == pytest.approx(0.50)


class TestDownturnLGDExtended:
    """Additional downturn LGD tests."""

    def test_additive_default_addon(self) -> None:
        dt = downturn_lgd(0.40, method="additive")
        assert dt == pytest.approx(0.48)  # 0.40 + 0.08

    def test_haircut_default_addon(self) -> None:
        dt = downturn_lgd(0.40, method="haircut")
        # 1 - (1-0.40)*(1-0.08) = 1 - 0.552 = 0.448
        assert dt == pytest.approx(0.448)

    def test_regulatory_high_lgd(self) -> None:
        dt = downturn_lgd(0.60, method="regulatory")
        # max(0.60, 0.10 + 0.24) = max(0.60, 0.34) = 0.60
        assert dt == pytest.approx(0.60)

    def test_regulatory_low_lgd(self) -> None:
        dt = downturn_lgd(0.10, method="regulatory")
        # max(0.10, 0.10 + 0.04) = 0.14
        assert dt == pytest.approx(0.14)

    def test_clipped_to_zero(self) -> None:
        # Negative base_lgd shouldn't happen but clip handles it
        dt = downturn_lgd(-0.10, downturn_add_on=-0.20, method="additive")
        assert dt == pytest.approx(0.0)


class TestLGDTermStructureExtended:
    """Additional term structure tests."""

    def test_empty_recovery_curve_padding(self) -> None:
        recovery = np.array([0.0])
        ts = lgd_term_structure(0.45, n_periods=3, recovery_curve=recovery)
        assert len(ts) == 3
        assert ts[0] == pytest.approx(1.0)  # 1 - 0.0
        assert ts[1] == pytest.approx(1.0)  # padded
        assert ts[2] == pytest.approx(1.0)

    def test_recovery_exceeds_periods(self) -> None:
        recovery = np.array([0.0, 0.20, 0.40, 0.55, 0.55, 0.60])
        ts = lgd_term_structure(0.45, n_periods=3, recovery_curve=recovery)
        assert len(ts) == 3
        np.testing.assert_allclose(ts, [1.0, 0.80, 0.60])

    def test_clipped_to_unit_range(self) -> None:
        # Recovery > 1.0 should be clipped
        recovery = np.array([0.0, 1.5])
        ts = lgd_term_structure(0.45, n_periods=2, recovery_curve=recovery)
        assert all(0.0 <= t <= 1.0 for t in ts)


class TestSupervisoryConstants:
    def test_senior_unsecured(self) -> None:
        assert SUPERVISORY_LGD_SENIOR_UNSECURED == 0.45

    def test_subordinated(self) -> None:
        assert SUPERVISORY_LGD_SUBORDINATED == 0.75


class TestLGDModel:
    """Test sklearn-compatible LGD model."""

    def test_fit_predict_workout(self) -> None:
        model = LGDModel(method="workout")
        X = np.array([[1.0], [2.0], [3.0], [4.0]])  # noqa: N806
        y = np.array([0.40, 0.45, 0.50, 0.55])
        model.fit(X, y)
        assert model.is_fitted_ is True
        preds = model.predict(X)
        assert len(preds) == 4
        # For workout, returns mean LGD with floor
        assert all(p >= LGD_FLOOR_UNSECURED for p in preds)

    def test_fit_predict_downturn(self) -> None:
        model = LGDModel(method="downturn", downturn_method="additive", downturn_add_on=0.10)
        X = np.array([[1.0], [2.0], [3.0]])  # noqa: N806
        y = np.array([0.30, 0.35, 0.40])
        model.fit(X, y)
        preds = model.predict(X)
        # Mean LGD = 0.35, downturn = 0.35 + 0.10 = 0.45, floor = 0.25 -> 0.45
        assert all(p >= 0.25 for p in preds)

    def test_predict_before_fit_raises(self) -> None:
        model = LGDModel()
        with pytest.raises(AssertionError):
            model.predict(np.array([[1.0]]))

    def test_secured_floor(self) -> None:
        model = LGDModel(method="workout", collateral_type="financial")
        X = np.array([[1.0], [2.0]])  # noqa: N806
        y = np.array([0.01, 0.02])
        model.fit(X, y)
        preds = model.predict(X)
        assert all(p >= LGD_FLOOR_SECURED_FINANCIAL for p in preds)

    def test_mean_lgd_stored(self) -> None:
        model = LGDModel()
        y = np.array([0.30, 0.40, 0.50])
        model.fit(np.array([[1], [2], [3]]), y)
        assert model.mean_lgd_ == pytest.approx(0.40)
