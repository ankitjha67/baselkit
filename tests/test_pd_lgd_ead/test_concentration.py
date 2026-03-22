"""Tests for concentration risk analytics."""

import numpy as np
import pytest

from creditriskengine.models.concentration.concentration import (
    granularity_adjustment,
    sector_concentration,
    single_name_concentration,
)


class TestSingleNameConcentration:
    def test_equal_exposures(self):
        eads = np.full(100, 10.0)
        result = single_name_concentration(eads)
        assert result["hhi"] == pytest.approx(0.01)
        assert result["top_1_share"] == pytest.approx(0.01)
        assert result["n_obligors"] == 100

    def test_single_exposure(self):
        result = single_name_concentration(np.array([1000.0]))
        assert result["hhi"] == pytest.approx(1.0)
        assert result["top_1_share"] == pytest.approx(1.0)

    def test_concentrated(self):
        eads = np.array([900.0, 50.0, 50.0])
        result = single_name_concentration(eads)
        assert result["hhi"] > 0.80
        assert result["top_1_share"] == pytest.approx(0.9)

    def test_empty(self):
        result = single_name_concentration(np.array([]))
        assert result["hhi"] == 0.0
        assert result["n_obligors"] == 0


class TestSectorConcentration:
    def test_single_sector(self):
        eads = np.array([100.0, 200.0, 300.0])
        sectors = np.array(["A", "A", "A"])
        result = sector_concentration(eads, sectors)
        assert result["sector_hhi"] == pytest.approx(1.0)

    def test_equal_sectors(self):
        eads = np.array([100.0, 100.0, 100.0, 100.0])
        sectors = np.array(["A", "B", "C", "D"])
        result = sector_concentration(eads, sectors)
        assert result["sector_hhi"] == pytest.approx(0.25)
        assert result["sector_shares"]["A"] == pytest.approx(0.25)

    def test_empty(self):
        result = sector_concentration(np.array([]), np.array([]))
        assert result["sector_hhi"] == 0.0


class TestGranularityAdjustment:
    def test_diversified_portfolio(self):
        n = 1000
        eads = np.full(n, 1.0)
        pds = np.full(n, 0.02)
        lgds = np.full(n, 0.45)
        ga = granularity_adjustment(eads, pds, lgds, rho=0.15)
        assert ga >= 0.0
        assert ga < 0.01  # small for diversified portfolio

    def test_concentrated_portfolio(self):
        eads = np.array([990.0, 10.0])
        pds = np.array([0.02, 0.02])
        lgds = np.array([0.45, 0.45])
        ga_conc = granularity_adjustment(eads, pds, lgds, rho=0.15)

        eads_div = np.full(100, 10.0)
        pds_div = np.full(100, 0.02)
        lgds_div = np.full(100, 0.45)
        ga_div = granularity_adjustment(eads_div, pds_div, lgds_div, rho=0.15)

        assert ga_conc > ga_div

    def test_zero_ead(self):
        ga = granularity_adjustment(np.array([]), np.array([]), np.array([]), rho=0.15)
        assert ga == 0.0
