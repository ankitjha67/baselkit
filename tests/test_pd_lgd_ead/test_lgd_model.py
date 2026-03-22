"""Tests for LGD modeling: workout, downturn, term structure, floors."""

import numpy as np
import pytest

from creditriskengine.models.lgd.lgd_model import (
    LGD_FLOOR_UNSECURED,
    SUPERVISORY_LGD_SENIOR_UNSECURED,
    SUPERVISORY_LGD_SUBORDINATED,
    apply_lgd_floor,
    downturn_lgd,
    lgd_term_structure,
    workout_lgd,
)


class TestWorkoutLGD:
    def test_full_recovery(self):
        lgd = workout_lgd(ead_at_default=1000.0, total_recoveries=1000.0, total_costs=0.0)
        assert lgd == pytest.approx(0.0)

    def test_zero_recovery(self):
        lgd = workout_lgd(ead_at_default=1000.0, total_recoveries=0.0, total_costs=0.0)
        assert lgd == pytest.approx(1.0)

    def test_partial_recovery(self):
        lgd = workout_lgd(ead_at_default=1000.0, total_recoveries=600.0, total_costs=100.0)
        assert lgd == pytest.approx(0.5)

    def test_with_discounting(self):
        lgd_no_disc = workout_lgd(1000.0, 500.0, 0.0, discount_rate=0.0)
        lgd_disc = workout_lgd(1000.0, 500.0, 0.0, discount_rate=0.05, time_to_recovery_years=2.0)
        assert lgd_disc > lgd_no_disc  # discounting reduces PV of recoveries

    def test_zero_ead(self):
        assert workout_lgd(0.0, 500.0, 0.0) == pytest.approx(1.0)

    def test_clipped_to_unit(self):
        # Costs exceed EAD → LGD capped at 1.0
        lgd = workout_lgd(100.0, 0.0, 200.0)
        assert lgd == pytest.approx(1.0)


class TestDownturnLGD:
    def test_additive(self):
        dt = downturn_lgd(0.40, downturn_add_on=0.10, method="additive")
        assert dt == pytest.approx(0.50)

    def test_haircut(self):
        dt = downturn_lgd(0.40, downturn_add_on=0.10, method="haircut")
        # 1 - (1-0.40)*(1-0.10) = 1 - 0.54 = 0.46
        assert dt == pytest.approx(0.46)

    def test_regulatory(self):
        dt = downturn_lgd(0.30, method="regulatory")
        assert dt == pytest.approx(max(0.30, 0.10 + 0.40 * 0.30))

    def test_capped_at_one(self):
        dt = downturn_lgd(0.95, downturn_add_on=0.20, method="additive")
        assert dt == pytest.approx(1.0)


class TestLGDTermStructure:
    def test_flat(self):
        ts = lgd_term_structure(0.45, n_periods=5)
        np.testing.assert_allclose(ts, [0.45] * 5)

    def test_with_recovery_curve(self):
        recovery = np.array([0.0, 0.20, 0.40, 0.55, 0.55])
        ts = lgd_term_structure(0.45, n_periods=5, recovery_curve=recovery)
        expected = 1.0 - recovery
        np.testing.assert_allclose(ts, expected)

    def test_padding(self):
        recovery = np.array([0.0, 0.20])
        ts = lgd_term_structure(0.45, n_periods=4, recovery_curve=recovery)
        assert len(ts) == 4
        assert ts[2] == pytest.approx(ts[1])  # padded with last value


class TestApplyLGDFloor:
    def test_unsecured_floor(self):
        assert apply_lgd_floor(0.10) == LGD_FLOOR_UNSECURED

    def test_above_floor(self):
        assert apply_lgd_floor(0.50) == pytest.approx(0.50)

    def test_secured_cre(self):
        assert apply_lgd_floor(0.05, is_secured=True, collateral_type="cre") == pytest.approx(0.10)


class TestSupervisoryConstants:
    def test_senior_unsecured(self):
        assert SUPERVISORY_LGD_SENIOR_UNSECURED == 0.45

    def test_subordinated(self):
        assert SUPERVISORY_LGD_SUBORDINATED == 0.75
