"""Tests for RAROC, loan pricing, and capital allocation."""

from __future__ import annotations

import numpy as np
import pytest

from creditriskengine.pricing import (
    break_even_spread,
    economic_value_added,
    euler_var_contributions,
    expected_shortfall_contributions,
    marginal_contributions,
    raroc,
    raroc_hurdle_check,
    risk_based_loan_rate,
)

# ============================================================================
# RAROC / EVA
# ============================================================================


class TestRAROC:
    def test_basic_raroc(self) -> None:
        # Revenue 50, EL = 0.02*0.45*1000 = 9, EC = 80
        # RAROC = (50 - 9) / 80 = 0.5125
        result = raroc(
            revenue=50.0, pd=0.02, lgd=0.45, ead=1000.0,
            economic_capital=80.0,
        )
        assert result.expected_loss == pytest.approx(9.0)
        assert result.raroc == pytest.approx((50.0 - 9.0) / 80.0)

    def test_capital_benefit_raises_raroc(self) -> None:
        base = raroc(50.0, 0.02, 0.45, 1000.0, 80.0)
        with_benefit = raroc(
            50.0, 0.02, 0.45, 1000.0, 80.0, capital_benefit_rate=0.03
        )
        assert with_benefit.raroc > base.raroc

    def test_operating_cost_lowers_raroc(self) -> None:
        base = raroc(50.0, 0.02, 0.45, 1000.0, 80.0)
        with_opex = raroc(50.0, 0.02, 0.45, 1000.0, 80.0, operating_cost=10.0)
        assert with_opex.raroc < base.raroc

    def test_zero_capital_raises(self) -> None:
        with pytest.raises(ValueError, match="economic_capital must be positive"):
            raroc(50.0, 0.02, 0.45, 1000.0, 0.0)

    def test_eva_positive_when_above_hurdle(self) -> None:
        result = raroc(50.0, 0.02, 0.45, 1000.0, 80.0)
        eva = economic_value_added(result, hurdle_rate=0.12)
        # RAROC ≈ 0.51 >> 0.12 → positive EVA
        assert eva > 0

    def test_eva_negative_when_below_hurdle(self) -> None:
        result = raroc(15.0, 0.05, 0.60, 1000.0, 100.0)
        eva = economic_value_added(result, hurdle_rate=0.15)
        # Revenue 15, EL = 30 → negative return → negative EVA
        assert eva < 0

    def test_hurdle_check(self) -> None:
        result = raroc(50.0, 0.02, 0.45, 1000.0, 80.0)
        assert raroc_hurdle_check(result, 0.12)
        assert not raroc_hurdle_check(result, 0.80)


# ============================================================================
# Loan pricing
# ============================================================================


class TestLoanPricing:
    def test_break_even_covers_el_and_capital(self) -> None:
        # EL = 0.02*0.45*1000 = 9; capital cost = 0.12*80 = 9.6
        # spread = (9 + 9.6) / 1000 = 0.0186
        spread = break_even_spread(
            pd=0.02, lgd=0.45, ead=1000.0,
            economic_capital=80.0, hurdle_rate=0.12,
        )
        assert spread == pytest.approx((9.0 + 9.6) / 1000.0)

    def test_break_even_raroc_equals_hurdle(self) -> None:
        # Pricing at break-even spread should yield RAROC = hurdle
        pd, lgd, ead, ec, hurdle = 0.02, 0.45, 1000.0, 80.0, 0.12
        spread = break_even_spread(pd, lgd, ead, ec, hurdle)
        revenue = spread * ead
        result = raroc(revenue, pd, lgd, ead, ec)
        assert result.raroc == pytest.approx(hurdle)

    def test_zero_ead_raises(self) -> None:
        with pytest.raises(ValueError, match="ead must be positive"):
            break_even_spread(0.02, 0.45, 0.0, 80.0, 0.12)

    def test_risk_based_rate_adds_funding(self) -> None:
        spread = break_even_spread(0.02, 0.45, 1000.0, 80.0, 0.12)
        rate = risk_based_loan_rate(
            0.02, 0.45, 1000.0, 80.0, cost_of_funds=0.03, hurdle_rate=0.12
        )
        assert rate == pytest.approx(0.03 + spread)


# ============================================================================
# Capital allocation
# ============================================================================


class TestCapitalAllocation:
    def _scenarios(self) -> np.ndarray:
        rng = np.random.default_rng(42)
        # 3 correlated exposures, 10000 scenarios
        z = rng.normal(0, 1, (10000, 1))
        idio = rng.normal(0, 1, (10000, 3))
        losses = np.maximum(0.6 * z + 0.4 * idio, 0) * np.array([100, 200, 150])
        return losses

    def test_marginal_contributions_shape(self) -> None:
        contribs = marginal_contributions(self._scenarios(), 0.99)
        assert len(contribs) == 3

    def test_es_contributions_sum_to_es(self) -> None:
        scenarios = self._scenarios()
        contribs = expected_shortfall_contributions(scenarios, 0.975)
        total = scenarios.sum(axis=1)
        var = np.quantile(total, 0.975)
        es = total[total >= var].mean()
        # ES contributions sum exactly to portfolio ES
        assert contribs.sum() == pytest.approx(es, rel=1e-6)

    def test_euler_contributions_shape(self) -> None:
        contribs = euler_var_contributions(self._scenarios(), 0.99)
        assert len(contribs) == 3

    def test_larger_exposure_larger_contribution(self) -> None:
        # Exposure 1 (200) should contribute more than exposure 0 (100)
        contribs = expected_shortfall_contributions(self._scenarios(), 0.975)
        assert contribs[1] > contribs[0]

    def test_invalid_confidence_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence_level"):
            marginal_contributions(self._scenarios(), 1.5)
        with pytest.raises(ValueError, match="confidence_level"):
            expected_shortfall_contributions(self._scenarios(), 0.0)
