"""Tests for lifetime PD term structure construction."""

import numpy as np
import pytest

from creditriskengine.ecl.ifrs9.lifetime_pd import (
    cumulative_pd_from_annual,
    flat_pd_term_structure,
    lifetime_pd_from_rating_transitions,
    marginal_pd_from_cumulative,
    survival_probabilities,
)


class TestCumulativePD:
    def test_single_period(self):
        cum = cumulative_pd_from_annual([0.02])
        assert cum[0] == pytest.approx(0.02)

    def test_multi_period(self):
        cum = cumulative_pd_from_annual([0.02, 0.02, 0.02])
        # Year 1: 0.02, Year 2: 1-(0.98*0.98)=0.0396, Year 3: 1-(0.98^3)
        assert cum[0] == pytest.approx(0.02)
        assert cum[1] == pytest.approx(1.0 - 0.98**2)
        assert cum[2] == pytest.approx(1.0 - 0.98**3)

    def test_monotonically_increasing(self):
        cum = cumulative_pd_from_annual([0.01, 0.02, 0.03, 0.04])
        for i in range(len(cum) - 1):
            assert cum[i + 1] >= cum[i]


class TestMarginalPD:
    def test_roundtrip(self):
        annual = [0.02, 0.02, 0.02]
        cum = cumulative_pd_from_annual(annual)
        marginal = marginal_pd_from_cumulative(cum)
        # Marginal should recover approximately the annual PDs
        # Marginal(1) = Cum(1) = 0.02, Marginal(2) = Cum(2)-Cum(1)
        assert marginal[0] == pytest.approx(0.02)
        assert marginal[1] == pytest.approx(0.98 * 0.02)
        assert all(m >= 0 for m in marginal)

    def test_sum_equals_cumulative(self):
        cum = cumulative_pd_from_annual([0.01, 0.02, 0.03])
        marginal = marginal_pd_from_cumulative(cum)
        assert np.sum(marginal) == pytest.approx(cum[-1])


class TestSurvivalProbabilities:
    def test_basic(self):
        cum = np.array([0.02, 0.04, 0.06])
        surv = survival_probabilities(cum)
        np.testing.assert_allclose(surv, [0.98, 0.96, 0.94])


class TestFlatPDTermStructure:
    def test_shape(self):
        cum, marg = flat_pd_term_structure(0.02, 5)
        assert len(cum) == 5
        assert len(marg) == 5

    def test_terminal_cumulative(self):
        cum, _ = flat_pd_term_structure(0.02, 5)
        assert cum[-1] == pytest.approx(1.0 - 0.98**5)


class TestTransitionMatrixPD:
    def test_absorbing_state(self):
        # Simple 2x2: good -> default with 5% annual
        tm = np.array([[0.95, 0.05], [0.0, 1.0]])
        cum = lifetime_pd_from_rating_transitions(tm, initial_rating=0, default_state=1, horizon_years=3)
        assert cum[0] == pytest.approx(0.05)
        assert cum[1] == pytest.approx(1.0 - 0.95**2)
        assert cum[2] == pytest.approx(1.0 - 0.95**3)
