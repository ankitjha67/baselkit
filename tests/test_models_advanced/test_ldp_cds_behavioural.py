"""Tests for LDP (Pluto-Tasche), CDS-implied PD, and behavioural scoring."""

from __future__ import annotations

import pytest

from creditriskengine.models.pd import (
    BehaviouralAttributes,
    behavioural_pd,
    behavioural_score,
    cds_implied_hazard_rate,
    cds_implied_pd,
    cds_pd_term_structure,
    early_warning_flag,
    pluto_tasche_multi_grade,
    pluto_tasche_single,
    risk_neutral_to_real_world,
)

# ============================================================================
# Pluto-Tasche LDP
# ============================================================================


class TestPlutoTascheSingle:
    def test_zero_defaults_nonzero_pd(self) -> None:
        # 100 obligors, 0 defaults, 90% confidence
        # PD = 1 - 0.1^(1/100) ≈ 0.0228
        pd = pluto_tasche_single(100, 0, 0.90)
        assert pd > 0
        assert pd == pytest.approx(1.0 - 0.1 ** (1 / 100), abs=1e-6)

    def test_more_obligors_lower_pd(self) -> None:
        pd_100 = pluto_tasche_single(100, 0, 0.90)
        pd_1000 = pluto_tasche_single(1000, 0, 0.90)
        assert pd_1000 < pd_100

    def test_higher_confidence_higher_pd(self) -> None:
        pd_90 = pluto_tasche_single(100, 0, 0.90)
        pd_99 = pluto_tasche_single(100, 0, 0.99)
        assert pd_99 > pd_90

    def test_with_defaults(self) -> None:
        pd = pluto_tasche_single(100, 5, 0.90)
        # Should be above the naive 5% MLE
        assert pd > 0.05

    def test_invalid_obligors(self) -> None:
        with pytest.raises(ValueError, match="n_obligors must be positive"):
            pluto_tasche_single(0, 0)

    def test_invalid_defaults(self) -> None:
        with pytest.raises(ValueError, match="n_defaults must be"):
            pluto_tasche_single(100, 200)

    def test_invalid_confidence(self) -> None:
        with pytest.raises(ValueError, match="confidence_level"):
            pluto_tasche_single(100, 0, 1.5)


class TestPlutoTascheMultiGrade:
    def test_monotonic_non_decreasing(self) -> None:
        obligors = [1000, 500, 200, 50]
        defaults = [0, 1, 3, 5]
        pds = pluto_tasche_multi_grade(obligors, defaults, 0.90)
        assert len(pds) == 4
        # Worst grade (last) should have highest PD
        for i in range(len(pds) - 1):
            assert pds[i] <= pds[i + 1] + 1e-9

    def test_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            pluto_tasche_multi_grade([100, 50], [0])


# ============================================================================
# CDS-implied PD
# ============================================================================


class TestCDSImplied:
    def test_hazard_rate_credit_triangle(self) -> None:
        # 100bps spread, 40% recovery → hazard = 0.01 / 0.6 = 0.01667
        h = cds_implied_hazard_rate(100, 0.40)
        assert h == pytest.approx(0.01 / 0.6)

    def test_pd_increases_with_tenor(self) -> None:
        pd_1y = cds_implied_pd(100, 1, 0.40)
        pd_5y = cds_implied_pd(100, 5, 0.40)
        assert pd_5y > pd_1y

    def test_pd_increases_with_spread(self) -> None:
        pd_low = cds_implied_pd(50, 5, 0.40)
        pd_high = cds_implied_pd(500, 5, 0.40)
        assert pd_high > pd_low

    def test_pd_in_unit_interval(self) -> None:
        pd = cds_implied_pd(1000, 10, 0.40)
        assert 0.0 <= pd <= 1.0

    def test_invalid_recovery(self) -> None:
        with pytest.raises(ValueError, match="recovery_rate"):
            cds_implied_hazard_rate(100, 1.0)

    def test_term_structure(self) -> None:
        spreads = {1.0: 50, 3.0: 80, 5.0: 100}
        ts = cds_pd_term_structure(spreads, 0.40)
        assert len(ts) == 3
        # Cumulative PD increases with tenor
        assert ts[1.0] < ts[3.0] < ts[5.0]

    def test_q_to_p_lowers_pd(self) -> None:
        pd_q = 0.05
        pd_p = risk_neutral_to_real_world(pd_q, sharpe_ratio=0.40)
        # Real-world PD should be lower than risk-neutral
        assert pd_p < pd_q

    def test_q_to_p_zero_sharpe_unchanged(self) -> None:
        pd_q = 0.05
        pd_p = risk_neutral_to_real_world(pd_q, sharpe_ratio=0.0)
        assert pd_p == pytest.approx(pd_q)


# ============================================================================
# Behavioural scoring
# ============================================================================


class TestBehaviouralScoring:
    def _good_account(self) -> BehaviouralAttributes:
        return BehaviouralAttributes(
            utilisation=0.20,
            payment_ratio=2.0,
            months_on_book=48,
            max_dpd_12m=0,
            balance_velocity=-0.05,
            n_times_overlimit_12m=0,
        )

    def _bad_account(self) -> BehaviouralAttributes:
        return BehaviouralAttributes(
            utilisation=0.98,
            payment_ratio=0.5,
            months_on_book=6,
            max_dpd_12m=60,
            balance_velocity=0.30,
            n_times_overlimit_12m=4,
        )

    def test_bad_account_higher_pd(self) -> None:
        good_pd = behavioural_pd(self._good_account())
        bad_pd = behavioural_pd(self._bad_account())
        assert bad_pd > good_pd

    def test_pd_in_unit_interval(self) -> None:
        assert 0.0 < behavioural_pd(self._good_account()) < 1.0
        assert 0.0 < behavioural_pd(self._bad_account()) < 1.0

    def test_score_monotonic_in_utilisation(self) -> None:
        low = BehaviouralAttributes(0.10, 1.0, 24, 0, 0.0, 0)
        high = BehaviouralAttributes(0.90, 1.0, 24, 0, 0.0, 0)
        assert behavioural_score(high) > behavioural_score(low)

    def test_early_warning_high_utilisation(self) -> None:
        attrs = BehaviouralAttributes(0.95, 2.0, 36, 0, 0.0, 0)
        assert early_warning_flag(attrs)

    def test_early_warning_dpd(self) -> None:
        attrs = BehaviouralAttributes(0.30, 2.0, 36, 45, 0.0, 0)
        assert early_warning_flag(attrs)

    def test_no_early_warning_good_account(self) -> None:
        assert not early_warning_flag(self._good_account())
