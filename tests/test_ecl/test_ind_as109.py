"""Tests for Ind AS 109 ECL calculation."""

import numpy as np
import pytest

from creditriskengine.core.types import IFRS9Stage
from creditriskengine.ecl.ind_as109.ind_as_ecl import (
    RBI_DEFAULT_DPD_THRESHOLD,
    assign_stage_ind_as,
    calculate_ecl_ind_as,
)


class TestAssignStageIndAS:
    def test_performing(self):
        assert assign_stage_ind_as(days_past_due=0) == IFRS9Stage.STAGE_1

    def test_npa_stage3(self):
        assert assign_stage_ind_as(days_past_due=0, is_npa=True) == IFRS9Stage.STAGE_3

    def test_90dpd_stage3(self):
        assert assign_stage_ind_as(days_past_due=90) == IFRS9Stage.STAGE_3

    def test_89dpd_not_stage3(self):
        # 89 DPD → not NPA, but > 30 DPD backstop → Stage 2
        assert assign_stage_ind_as(days_past_due=89) == IFRS9Stage.STAGE_2

    def test_31dpd_stage2(self):
        assert assign_stage_ind_as(days_past_due=31) == IFRS9Stage.STAGE_2

    def test_poci(self):
        assert assign_stage_ind_as(days_past_due=0, is_poci=True) == IFRS9Stage.POCI

    def test_rbi_threshold_value(self):
        assert RBI_DEFAULT_DPD_THRESHOLD == 90


class TestCalculateECLIndAS:
    def test_stage1(self):
        ecl = calculate_ecl_ind_as(IFRS9Stage.STAGE_1, pd_12m=0.02, lgd=0.45, ead=1000.0)
        assert ecl == pytest.approx(9.0)

    def test_stage2_requires_marginal_pds(self):
        with pytest.raises(ValueError, match="marginal_pds required"):
            calculate_ecl_ind_as(IFRS9Stage.STAGE_2, pd_12m=0.02, lgd=0.45, ead=1000.0)

    def test_stage2_lifetime(self):
        marginal_pds = np.array([0.02, 0.03])
        ecl = calculate_ecl_ind_as(
            IFRS9Stage.STAGE_2, pd_12m=0.02, lgd=0.45, ead=1000.0,
            marginal_pds=marginal_pds,
        )
        expected = (0.02 + 0.03) * 0.45 * 1000.0
        assert ecl == pytest.approx(expected)
