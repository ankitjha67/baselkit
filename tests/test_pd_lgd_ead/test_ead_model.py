"""Tests for EAD modeling: CCF, regulatory EAD, term structure."""

import numpy as np
import pytest

from creditriskengine.models.ead.ead_model import (
    CCF_FLOOR_AIRB,
    apply_ccf_floor,
    calculate_ead,
    ead_term_structure,
    estimate_ccf,
    get_supervisory_ccf,
)


class TestCalculateEAD:
    def test_fully_drawn(self):
        ead = calculate_ead(drawn_amount=1000.0, undrawn_commitment=0.0, ccf=0.75)
        assert ead == pytest.approx(1000.0)

    def test_with_undrawn(self):
        ead = calculate_ead(drawn_amount=800.0, undrawn_commitment=200.0, ccf=0.75)
        assert ead == pytest.approx(950.0)

    def test_zero_ccf(self):
        ead = calculate_ead(drawn_amount=500.0, undrawn_commitment=500.0, ccf=0.0)
        assert ead == pytest.approx(500.0)

    def test_full_ccf(self):
        ead = calculate_ead(drawn_amount=500.0, undrawn_commitment=500.0, ccf=1.0)
        assert ead == pytest.approx(1000.0)


class TestEstimateCCF:
    def test_basic(self):
        # Drawn at ref = 700, Limit = 1000, EAD at default = 850
        # CCF = (850 - 700) / (1000 - 700) = 150/300 = 0.5
        ccf = estimate_ccf(ead_at_default=850.0, drawn_at_reference=700.0, limit=1000.0)
        assert ccf == pytest.approx(0.5)

    def test_fully_drawn_at_reference(self):
        ccf = estimate_ccf(ead_at_default=1000.0, drawn_at_reference=1000.0, limit=1000.0)
        assert ccf == pytest.approx(1.0)

    def test_clipped(self):
        ccf = estimate_ccf(ead_at_default=500.0, drawn_at_reference=700.0, limit=1000.0)
        assert ccf == pytest.approx(0.0)  # negative → clipped to 0


class TestSupervisoryCCF:
    def test_committed_other(self):
        assert get_supervisory_ccf("committed_other") == pytest.approx(0.75)

    def test_direct_credit_sub(self):
        assert get_supervisory_ccf("direct_credit_substitutes") == pytest.approx(1.0)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown facility type"):
            get_supervisory_ccf("unknown_type")


class TestApplyCCFFloor:
    def test_airb_floor(self):
        assert apply_ccf_floor(0.30, approach="airb") == CCF_FLOOR_AIRB

    def test_airb_above_floor(self):
        assert apply_ccf_floor(0.80, approach="airb") == pytest.approx(0.80)

    def test_firb_no_floor(self):
        assert apply_ccf_floor(0.30, approach="firb") == pytest.approx(0.30)


class TestEADTermStructure:
    def test_no_amortization(self):
        ts = ead_term_structure(800.0, 200.0, ccf=0.75, n_periods=3)
        expected = 800.0 + 0.75 * 200.0
        np.testing.assert_allclose(ts, [expected, expected, expected])

    def test_with_amortization(self):
        ts = ead_term_structure(1000.0, 0.0, ccf=0.0, n_periods=3, amortization_rate=0.10)
        np.testing.assert_allclose(ts, [1000.0, 900.0, 810.0])

    def test_shape(self):
        ts = ead_term_structure(500.0, 100.0, ccf=0.5, n_periods=5)
        assert len(ts) == 5
