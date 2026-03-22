"""Tests for CECL (ASC 326) calculations."""

import numpy as np
import pytest

from creditriskengine.ecl.cecl.cecl_calc import cecl_loss_rate, cecl_pd_lgd
from creditriskengine.ecl.cecl.methods import dcf_method, vintage_analysis, warm_method
from creditriskengine.ecl.cecl.qualitative import (
    QualitativeFactor,
    apply_q_factors,
    total_q_factor_adjustment,
)


class TestCECLPDLGD:
    def test_basic(self):
        pds = np.array([0.02, 0.02, 0.02])
        ecl = cecl_pd_lgd(pds, lgds=0.45, eads=1000.0)
        # 3 * 0.02 * 0.45 * 1000 = 27.0 (no discounting)
        assert ecl == pytest.approx(27.0)

    def test_with_discounting(self):
        pds = np.array([0.02, 0.02])
        ecl_no_disc = cecl_pd_lgd(pds, 0.45, 1000.0, discount_rate=0.0)
        ecl_disc = cecl_pd_lgd(pds, 0.45, 1000.0, discount_rate=0.05)
        assert ecl_disc < ecl_no_disc


class TestCECLLossRate:
    def test_basic(self):
        ecl = cecl_loss_rate(ead=1000.0, historical_loss_rate=0.01, remaining_life_years=3.0)
        assert ecl == pytest.approx(30.0)

    def test_with_adjustments(self):
        ecl = cecl_loss_rate(
            ead=1000.0,
            historical_loss_rate=0.01,
            qualitative_adjustment=0.005,
            forecast_adjustment=0.002,
            remaining_life_years=2.0,
        )
        assert ecl == pytest.approx(1000.0 * 0.017 * 2.0)

    def test_floor_at_zero(self):
        ecl = cecl_loss_rate(ead=1000.0, historical_loss_rate=0.01, qualitative_adjustment=-0.05)
        assert ecl == pytest.approx(0.0)


class TestWARMMethod:
    def test_basic(self):
        ecl = warm_method(ead=1000.0, historical_loss_rate=0.01, remaining_life_years=3.0)
        assert ecl == pytest.approx(30.0)

    def test_with_q_factor(self):
        ecl = warm_method(ead=1000.0, historical_loss_rate=0.01, remaining_life_years=3.0, qualitative_factor=0.5)
        assert ecl == pytest.approx(1000.0 * 0.015 * 3.0)


class TestVintageAnalysis:
    def test_basic(self):
        # 2 vintages, 3 ages; vintage 0 is at age 1, vintage 1 at age 0
        matrix = np.array([
            [0.01, 0.03, 0.0],
            [0.02, 0.0, 0.0],
        ])
        balances = np.array([100.0, 200.0])
        ecl = vintage_analysis(matrix, balances)
        # Ultimate loss (max of last column) = max(0.0, 0.0) = 0 → fallback to max of all last col
        # Actually: np.max(matrix[:, -1]) = 0.0
        assert ecl == pytest.approx(0.0)

    def test_with_ultimate_losses(self):
        matrix = np.array([
            [0.01, 0.03, 0.05],
            [0.02, 0.0, 0.0],
        ])
        balances = np.array([100.0, 200.0])
        ecl = vintage_analysis(matrix, balances)
        # Ultimate = max of col[-1] = 0.05
        # Vintage 0: at age 2, cum=0.05, remaining=0.0
        # Vintage 1: at age 0, cum=0.02, remaining=0.03
        # Total = 0 + 200*0.03 = 6.0
        assert ecl == pytest.approx(6.0)


class TestDCFMethod:
    def test_basic(self):
        contractual = np.array([100.0, 100.0, 100.0])
        expected = np.array([95.0, 90.0, 85.0])
        ecl = dcf_method(contractual, expected, discount_rate=0.05)
        # PV diff should be positive
        assert ecl > 0
        # Manual: PV(contractual) - PV(expected)
        dfs = 1.0 / (1.05 ** np.array([1, 2, 3]))
        expected_ecl = float(np.sum((contractual - expected) * dfs))
        assert ecl == pytest.approx(expected_ecl)

    def test_no_losses(self):
        cf = np.array([100.0, 100.0])
        assert dcf_method(cf, cf, discount_rate=0.05) == pytest.approx(0.0)


class TestQualitativeFactors:
    def test_total_adjustment(self):
        factors = [
            QualitativeFactor(name="econ", adjustment_bps=25),
            QualitativeFactor(name="portfolio", adjustment_bps=-10),
        ]
        assert total_q_factor_adjustment(factors) == pytest.approx(15 / 10_000)

    def test_apply_q_factors(self):
        factors = [QualitativeFactor(name="econ", adjustment_bps=50)]
        adjusted = apply_q_factors(0.01, factors)
        assert adjusted == pytest.approx(0.01 + 50 / 10_000)

    def test_apply_q_factors_floor(self):
        factors = [QualitativeFactor(name="negative", adjustment_bps=-200)]
        adjusted = apply_q_factors(0.01, factors, floor=0.005)
        assert adjusted == pytest.approx(0.005)
