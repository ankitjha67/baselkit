"""Tests for IFRS 9 ECL calculation."""

import numpy as np
import pytest

from creditriskengine.core.types import IFRS9Stage
from creditriskengine.ecl.ifrs9.ecl_calc import (
    calculate_ecl,
    discount_factors,
    ecl_12_month,
    ecl_lifetime,
)


class TestDiscountFactors:
    def test_zero_eir(self):
        dfs = discount_factors(0.0, 5)
        np.testing.assert_allclose(dfs, np.ones(5))

    def test_positive_eir(self):
        dfs = discount_factors(0.05, 3)
        expected = np.array([1 / 1.05, 1 / 1.05**2, 1 / 1.05**3])
        np.testing.assert_allclose(dfs, expected, rtol=1e-10)

    def test_decreasing(self):
        dfs = discount_factors(0.10, 10)
        assert all(dfs[i] > dfs[i + 1] for i in range(len(dfs) - 1))


class TestECL12Month:
    def test_basic(self):
        ecl = ecl_12_month(pd_12m=0.02, lgd=0.45, ead=1000.0)
        assert ecl == pytest.approx(9.0, rel=1e-6)

    def test_with_discounting(self):
        ecl_no_disc = ecl_12_month(pd_12m=0.02, lgd=0.45, ead=1000.0, eir=0.0)
        ecl_disc = ecl_12_month(pd_12m=0.02, lgd=0.45, ead=1000.0, eir=0.05)
        assert ecl_disc < ecl_no_disc

    def test_zero_pd(self):
        assert ecl_12_month(0.0, 0.45, 1000.0) == 0.0

    def test_full_default(self):
        ecl = ecl_12_month(1.0, 0.45, 1000.0)
        assert ecl == pytest.approx(450.0)


class TestECLLifetime:
    def test_basic(self):
        marginal_pds = np.array([0.02, 0.03, 0.04])
        ecl = ecl_lifetime(marginal_pds, lgds=0.45, eads=1000.0)
        expected = sum(pd * 0.45 * 1000.0 for pd in marginal_pds)
        assert ecl == pytest.approx(expected, rel=1e-6)

    def test_with_term_structure(self):
        marginal_pds = np.array([0.02, 0.03])
        lgds = np.array([0.45, 0.40])
        eads = np.array([1000.0, 950.0])
        ecl = ecl_lifetime(marginal_pds, lgds, eads, eir=0.05)
        # Manual: 0.02*0.45*1000/1.05 + 0.03*0.40*950/1.05^2
        expected = 0.02 * 0.45 * 1000.0 / 1.05 + 0.03 * 0.40 * 950.0 / 1.05**2
        assert ecl == pytest.approx(expected, rel=1e-6)

    def test_discounting_reduces_ecl(self):
        marginal_pds = np.array([0.02, 0.03, 0.04, 0.05])
        ecl_no_disc = ecl_lifetime(marginal_pds, 0.45, 1000.0, eir=0.0)
        ecl_disc = ecl_lifetime(marginal_pds, 0.45, 1000.0, eir=0.05)
        assert ecl_disc < ecl_no_disc


class TestCalculateECL:
    def test_stage1_uses_12m(self):
        ecl = calculate_ecl(
            stage=IFRS9Stage.STAGE_1,
            pd_12m=0.02,
            lgd=0.45,
            ead=1000.0,
        )
        assert ecl == pytest.approx(9.0, rel=1e-6)

    def test_stage2_requires_marginal_pds(self):
        with pytest.raises(ValueError, match="marginal_pds required"):
            calculate_ecl(
                stage=IFRS9Stage.STAGE_2,
                pd_12m=0.02,
                lgd=0.45,
                ead=1000.0,
            )

    def test_stage2_lifetime(self):
        marginal_pds = np.array([0.02, 0.03, 0.04])
        ecl = calculate_ecl(
            stage=IFRS9Stage.STAGE_2,
            pd_12m=0.02,
            lgd=0.45,
            ead=1000.0,
            marginal_pds=marginal_pds,
        )
        expected = sum(pd * 0.45 * 1000.0 for pd in marginal_pds)
        assert ecl == pytest.approx(expected, rel=1e-6)

    def test_stage3_lifetime(self):
        marginal_pds = np.array([0.50, 0.30, 0.20])
        ecl = calculate_ecl(
            stage=IFRS9Stage.STAGE_3,
            pd_12m=0.50,
            lgd=0.60,
            ead=500.0,
            marginal_pds=marginal_pds,
        )
        expected = sum(pd * 0.60 * 500.0 for pd in marginal_pds)
        assert ecl == pytest.approx(expected, rel=1e-6)
