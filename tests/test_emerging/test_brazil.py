"""Tests for Brazil CMN 4.966 three-stage ECL classification."""

import pytest

from creditriskengine.core.types import IFRS9Stage
from creditriskengine.ecl.emerging.brazil import (
    classify_cmn_4966_stage,
    uses_simplified_model,
)


class TestCMN4966Staging:
    def test_current_is_stage_1(self) -> None:
        assert classify_cmn_4966_stage(0) == IFRS9Stage.STAGE_1

    def test_30_dpd_boundary_still_stage_1(self) -> None:
        # Backstop is > 30 DPD.
        assert classify_cmn_4966_stage(30) == IFRS9Stage.STAGE_1

    def test_31_dpd_stage_2(self) -> None:
        assert classify_cmn_4966_stage(31) == IFRS9Stage.STAGE_2

    def test_sicr_flag_stage_2_regardless_of_dpd(self) -> None:
        assert classify_cmn_4966_stage(0, has_sicr=True) == IFRS9Stage.STAGE_2

    def test_90_dpd_boundary_still_stage_2(self) -> None:
        # Default is > 90 DPD.
        assert classify_cmn_4966_stage(90) == IFRS9Stage.STAGE_2

    def test_91_dpd_stage_3(self) -> None:
        assert classify_cmn_4966_stage(91) == IFRS9Stage.STAGE_3

    def test_objective_evidence_stage_3_regardless_of_dpd(self) -> None:
        assert (
            classify_cmn_4966_stage(0, has_objective_loss_evidence=True)
            == IFRS9Stage.STAGE_3
        )

    def test_negative_dpd_raises(self) -> None:
        with pytest.raises(ValueError, match="days_past_due"):
            classify_cmn_4966_stage(-1)


class TestSimplifiedModelSegments:
    def test_full_model_s1_s2(self) -> None:
        assert uses_simplified_model("S1") is False
        assert uses_simplified_model("S2") is False

    def test_simplified_s3_to_s5(self) -> None:
        assert uses_simplified_model("S3") is True
        assert uses_simplified_model("s4") is True  # case-insensitive
        assert uses_simplified_model("S5") is True

    def test_unknown_segment_raises(self) -> None:
        with pytest.raises(ValueError, match="prudential segment"):
            uses_simplified_model("S9")


class TestConfigConsistency:
    def test_bcb_yml_marks_2682_as_legacy(self) -> None:
        from creditriskengine.core.types import Jurisdiction
        from creditriskengine.regulatory.loader import load_config

        cfg = load_config(Jurisdiction.BRAZIL)
        # The new ECL framework block exists with the 4.966 backstops...
        ecl = cfg["ecl_framework"]
        assert ecl["stage_2_sicr_dpd_backstop"] == 30
        assert ecl["stage_3_default_dpd"] == 90
        assert ecl["effective_date"] == "2025-01-01"
        # ...and the repealed 2.682 table is quarantined under legacy.
        assert "legacy_res_2682" in cfg
        assert "provisioning_rules" not in cfg.get("default_definition", {})
