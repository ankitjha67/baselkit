"""Tests for IFRS 9 staging and SICR assessment."""

import pytest

from creditriskengine.core.types import IFRS9Stage
from creditriskengine.ecl.ifrs9.sicr import assess_sicr
from creditriskengine.ecl.ifrs9.staging import assign_stage, stage_allocation_summary


class TestAssignStage:
    def test_performing_stage1(self) -> None:
        assert assign_stage(days_past_due=0) == IFRS9Stage.STAGE_1

    def test_poci_takes_priority(self) -> None:
        assert assign_stage(days_past_due=60, is_poci=True, is_defaulted=True) == IFRS9Stage.POCI

    def test_defaulted_stage3(self) -> None:
        assert assign_stage(days_past_due=0, is_defaulted=True) == IFRS9Stage.STAGE_3

    def test_credit_impaired_stage3(self) -> None:
        assert assign_stage(days_past_due=0, is_credit_impaired=True) == IFRS9Stage.STAGE_3

    def test_sicr_triggered_stage2(self) -> None:
        assert assign_stage(days_past_due=0, sicr_triggered=True) == IFRS9Stage.STAGE_2

    def test_dpd_backstop_stage2(self) -> None:
        assert assign_stage(days_past_due=31) == IFRS9Stage.STAGE_2

    def test_dpd_exactly_at_backstop_stays_stage1(self) -> None:
        assert assign_stage(days_past_due=30) == IFRS9Stage.STAGE_1

    def test_custom_dpd_backstop(self) -> None:
        assert assign_stage(days_past_due=60, dpd_backstop=60) == IFRS9Stage.STAGE_1
        assert assign_stage(days_past_due=61, dpd_backstop=60) == IFRS9Stage.STAGE_2


class TestAssessSICR:
    def test_dpd_backstop_triggers(self) -> None:
        assert assess_sicr(0.01, 0.01, days_past_due=31) is True

    def test_dpd_backstop_disabled(self) -> None:
        assert assess_sicr(0.01, 0.01, days_past_due=31, use_dpd_backstop=False) is False

    def test_relative_change_triggers(self) -> None:
        # 3x increase > default 2.0 threshold
        assert assess_sicr(0.03, 0.01) is True

    def test_relative_change_below_threshold(self) -> None:
        assert assess_sicr(0.015, 0.01) is False

    def test_absolute_change_triggers(self) -> None:
        # 0.06 - 0.05 = 0.01 > 0.005 threshold, but relative = 1.2x < 2.0
        assert assess_sicr(0.06, 0.05) is True

    def test_no_sicr(self) -> None:
        assert assess_sicr(0.01, 0.01) is False

    def test_zero_origination_pd(self) -> None:
        # Should use absolute threshold
        assert assess_sicr(0.01, 0.0) is True
        assert assess_sicr(0.004, 0.0) is False


class TestStageAllocationSummary:
    def test_basic_summary(self) -> None:
        stages = [IFRS9Stage.STAGE_1, IFRS9Stage.STAGE_1, IFRS9Stage.STAGE_2, IFRS9Stage.STAGE_3]
        eads = [100.0, 200.0, 150.0, 50.0]
        summary = stage_allocation_summary(stages, eads)

        assert summary["STAGE_1"]["count"] == 2
        assert summary["STAGE_1"]["ead"] == pytest.approx(300.0)
        assert summary["STAGE_2"]["count"] == 1
        assert summary["STAGE_3"]["ead"] == pytest.approx(50.0)
