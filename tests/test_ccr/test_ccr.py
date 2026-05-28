"""Tests for counterparty credit risk exposure and wrong-way risk."""

from __future__ import annotations

import numpy as np
import pytest

from creditriskengine.ccr import (
    ExposureProfile,
    alpha_wrong_way_multiplier,
    conditional_epe_wwr,
    effective_epe,
    effective_expected_exposure,
    expected_positive_exposure,
    netting_set_exposure,
    potential_future_exposure,
    specific_wwr_flag,
)
from creditriskengine.ccr.exposure import (
    IMM_ALPHA,
    expected_exposure,
    summarise_exposure,
)

# ============================================================================
# Exposure profiles
# ============================================================================


class TestExposureProfiles:
    def _paths(self) -> np.ndarray:
        # 4 paths × 3 timesteps
        return np.array([
            [10.0, 20.0, 5.0],
            [-5.0, 15.0, 25.0],
            [0.0, 30.0, -10.0],
            [8.0, 12.0, 18.0],
        ])

    def test_expected_exposure_positive_only(self) -> None:
        ee = expected_exposure(self._paths())
        # t=0: mean(max(10,0), max(-5,0), max(0,0), max(8,0)) = (10+0+0+8)/4 = 4.5
        assert ee[0] == pytest.approx(4.5)

    def test_epe_is_time_average(self) -> None:
        epe = expected_positive_exposure(self._paths())
        ee = expected_exposure(self._paths())
        assert epe == pytest.approx(np.mean(ee))

    def test_eee_non_decreasing(self) -> None:
        eee = effective_expected_exposure(self._paths())
        assert np.all(np.diff(eee) >= -1e-12)

    def test_eepe_ge_epe(self) -> None:
        # Effective EPE >= EPE always (running max)
        paths = self._paths()
        assert effective_epe(paths) >= expected_positive_exposure(paths) - 1e-9

    def test_pfe_higher_than_ee(self) -> None:
        paths = self._paths()
        ee = expected_exposure(paths)
        pfe = potential_future_exposure(paths, 0.95)
        assert np.all(pfe >= ee - 1e-9)

    def test_pfe_invalid_confidence(self) -> None:
        with pytest.raises(ValueError, match="confidence_level"):
            potential_future_exposure(self._paths(), 1.5)

    def test_summarise(self) -> None:
        profile = summarise_exposure(self._paths())
        assert isinstance(profile, ExposureProfile)
        assert profile.regulatory_ead == pytest.approx(IMM_ALPHA * profile.eepe)
        assert profile.peak_pfe > 0

    def test_time_weighted_epe(self) -> None:
        paths = self._paths()
        weights = np.array([1.0, 1.0, 1.0])
        epe_w = expected_positive_exposure(paths, weights)
        epe_u = expected_positive_exposure(paths)
        assert epe_w == pytest.approx(epe_u)


class TestNettingSet:
    def test_netting_reduces_exposure(self) -> None:
        # Two offsetting trades net to a smaller exposure
        trades = np.array([100.0, -60.0])
        netted = netting_set_exposure(trades)
        assert netted == pytest.approx(40.0)

    def test_collateral_reduces(self) -> None:
        trades = np.array([100.0, 50.0])
        netted = netting_set_exposure(trades, collateral=30.0)
        assert netted == pytest.approx(120.0)

    def test_floored_at_zero(self) -> None:
        trades = np.array([20.0, -50.0])
        assert netting_set_exposure(trades) == 0.0

    def test_2d_per_row(self) -> None:
        trades = np.array([[100.0, -60.0], [50.0, 50.0]])
        netted = netting_set_exposure(trades)
        np.testing.assert_allclose(netted, [40.0, 100.0])


# ============================================================================
# Wrong-way risk
# ============================================================================


class TestWrongWayRisk:
    def test_swwr_own_issuance(self) -> None:
        assert specific_wwr_flag(collateral_is_own_issuance=True)

    def test_swwr_fx_linked(self) -> None:
        assert specific_wwr_flag(
            exposure_fx_linked_to_counterparty_sovereign=True
        )

    def test_no_swwr(self) -> None:
        assert not specific_wwr_flag()

    def test_alpha_no_adjustment_for_negative_correlation(self) -> None:
        assert alpha_wrong_way_multiplier(1.4, correlation=-0.3) == 1.4

    def test_alpha_increases_with_correlation(self) -> None:
        a_low = alpha_wrong_way_multiplier(1.4, correlation=0.2, stress_factor=2.0)
        a_high = alpha_wrong_way_multiplier(1.4, correlation=0.8, stress_factor=2.0)
        assert a_high > a_low >= 1.4

    def test_alpha_capped(self) -> None:
        a = alpha_wrong_way_multiplier(1.4, correlation=1.0, stress_factor=10.0)
        assert a == 2.5

    def test_conditional_epe_increases(self) -> None:
        base = 100.0
        cond = conditional_epe_wwr(base, exposure_credit_correlation=0.5, counterparty_pd=0.02)
        assert cond > base

    def test_conditional_epe_no_wwr(self) -> None:
        base = 100.0
        cond = conditional_epe_wwr(base, exposure_credit_correlation=0.0, counterparty_pd=0.02)
        assert cond == base

    def test_conditional_epe_negative_correlation(self) -> None:
        base = 100.0
        cond = conditional_epe_wwr(base, exposure_credit_correlation=-0.5, counterparty_pd=0.02)
        assert cond == base
