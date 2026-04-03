"""Tests for Ind AS 109 ECL calculation with full RBI IRAC norms."""

import numpy as np
import pytest

from creditriskengine.core.types import IFRS9Stage
from creditriskengine.ecl.ind_as109.ind_as_ecl import (
    RBI_AGRI_SHORT_CROP_DPD,
    RBI_DEFAULT_DPD_THRESHOLD,
    IRACAssetClass,
    assign_stage_ind_as,
    calculate_ecl_ind_as,
    classify_irac,
    irac_to_ifrs9_stage,
    rbi_minimum_provision,
    restructured_account_stage,
)


class TestAssignStageIndAS:
    def test_performing(self) -> None:
        assert assign_stage_ind_as(days_past_due=0) == IFRS9Stage.STAGE_1

    def test_npa_stage3(self) -> None:
        assert assign_stage_ind_as(days_past_due=0, is_npa=True) == IFRS9Stage.STAGE_3

    def test_90dpd_stage3(self) -> None:
        assert assign_stage_ind_as(days_past_due=90) == IFRS9Stage.STAGE_3

    def test_89dpd_not_stage3(self) -> None:
        assert assign_stage_ind_as(days_past_due=89) == IFRS9Stage.STAGE_2

    def test_31dpd_stage2(self) -> None:
        assert assign_stage_ind_as(days_past_due=31) == IFRS9Stage.STAGE_2

    def test_poci(self) -> None:
        assert assign_stage_ind_as(days_past_due=0, is_poci=True) == IFRS9Stage.POCI

    def test_rbi_threshold_value(self) -> None:
        assert RBI_DEFAULT_DPD_THRESHOLD == 90


class TestClassifyIRAC:
    def test_standard(self) -> None:
        assert classify_irac(days_past_due=0) == IRACAssetClass.STANDARD

    def test_sma_0(self) -> None:
        assert classify_irac(days_past_due=15) == IRACAssetClass.SMA_0
        assert classify_irac(days_past_due=30) == IRACAssetClass.SMA_0

    def test_sma_1(self) -> None:
        assert classify_irac(days_past_due=45) == IRACAssetClass.SMA_1
        assert classify_irac(days_past_due=60) == IRACAssetClass.SMA_1

    def test_sma_2(self) -> None:
        assert classify_irac(days_past_due=75) == IRACAssetClass.SMA_2
        assert classify_irac(days_past_due=89) == IRACAssetClass.SMA_2

    def test_substandard(self) -> None:
        assert classify_irac(days_past_due=90, months_as_npa=0) == IRACAssetClass.SUBSTANDARD
        assert classify_irac(days_past_due=120, months_as_npa=6) == IRACAssetClass.SUBSTANDARD
        assert classify_irac(days_past_due=90, months_as_npa=12) == IRACAssetClass.SUBSTANDARD

    def test_doubtful_1(self) -> None:
        assert classify_irac(days_past_due=90, months_as_npa=13) == IRACAssetClass.DOUBTFUL_1
        assert classify_irac(days_past_due=90, months_as_npa=24) == IRACAssetClass.DOUBTFUL_1

    def test_doubtful_2(self) -> None:
        assert classify_irac(days_past_due=90, months_as_npa=25) == IRACAssetClass.DOUBTFUL_2
        assert classify_irac(days_past_due=90, months_as_npa=36) == IRACAssetClass.DOUBTFUL_2

    def test_doubtful_3(self) -> None:
        assert classify_irac(days_past_due=90, months_as_npa=37) == IRACAssetClass.DOUBTFUL_3

    def test_loss_overrides_dpd(self) -> None:
        assert classify_irac(days_past_due=0, is_loss=True) == IRACAssetClass.LOSS

    def test_agri_short_crop_threshold(self) -> None:
        # Short-duration crop: NPA at 60 DPD
        assert classify_irac(
            days_past_due=60,
            is_agricultural=True,
            is_short_duration_crop=True,
        ) == IRACAssetClass.SUBSTANDARD
        # 59 DPD is still SMA for short crop
        assert classify_irac(
            days_past_due=59,
            is_agricultural=True,
            is_short_duration_crop=True,
        ) == IRACAssetClass.SMA_1

    def test_agri_long_crop_uses_90dpd(self) -> None:
        assert classify_irac(
            days_past_due=89,
            is_agricultural=True,
            is_short_duration_crop=False,
        ) == IRACAssetClass.SMA_2

    def test_agri_short_crop_dpd_constant(self) -> None:
        assert RBI_AGRI_SHORT_CROP_DPD == 60


class TestIRACToIFRS9Stage:
    def test_standard_to_stage1(self) -> None:
        assert irac_to_ifrs9_stage(IRACAssetClass.STANDARD) == IFRS9Stage.STAGE_1

    def test_sma0_to_stage1(self) -> None:
        assert irac_to_ifrs9_stage(IRACAssetClass.SMA_0) == IFRS9Stage.STAGE_1

    def test_sma1_to_stage2(self) -> None:
        assert irac_to_ifrs9_stage(IRACAssetClass.SMA_1) == IFRS9Stage.STAGE_2

    def test_sma2_to_stage2(self) -> None:
        assert irac_to_ifrs9_stage(IRACAssetClass.SMA_2) == IFRS9Stage.STAGE_2

    def test_npa_classes_to_stage3(self) -> None:
        for cls in (
            IRACAssetClass.SUBSTANDARD,
            IRACAssetClass.DOUBTFUL_1,
            IRACAssetClass.DOUBTFUL_2,
            IRACAssetClass.DOUBTFUL_3,
            IRACAssetClass.LOSS,
        ):
            assert irac_to_ifrs9_stage(cls) == IFRS9Stage.STAGE_3


class TestRBIMinimumProvision:
    def test_standard_commercial(self) -> None:
        prov = rbi_minimum_provision(100_000.0, IRACAssetClass.STANDARD, sector="commercial")
        assert prov == pytest.approx(400.0)  # 0.40%

    def test_standard_cre(self) -> None:
        prov = rbi_minimum_provision(100_000.0, IRACAssetClass.STANDARD, sector="cre")
        assert prov == pytest.approx(1000.0)  # 1.00%

    def test_standard_sme(self) -> None:
        prov = rbi_minimum_provision(100_000.0, IRACAssetClass.STANDARD, sector="sme")
        assert prov == pytest.approx(250.0)  # 0.25%

    def test_sma_uses_standard_rate(self) -> None:
        # SMA accounts use standard provisioning rates
        prov = rbi_minimum_provision(100_000.0, IRACAssetClass.SMA_1, sector="commercial")
        assert prov == pytest.approx(400.0)

    def test_substandard_secured(self) -> None:
        prov = rbi_minimum_provision(100_000.0, IRACAssetClass.SUBSTANDARD, is_secured=True)
        assert prov == pytest.approx(15_000.0)  # 15%

    def test_substandard_unsecured(self) -> None:
        prov = rbi_minimum_provision(100_000.0, IRACAssetClass.SUBSTANDARD, is_secured=False)
        assert prov == pytest.approx(25_000.0)  # 25%

    def test_doubtful_1_secured(self) -> None:
        prov = rbi_minimum_provision(100_000.0, IRACAssetClass.DOUBTFUL_1, is_secured=True)
        assert prov == pytest.approx(25_000.0)  # 25%

    def test_doubtful_1_unsecured(self) -> None:
        prov = rbi_minimum_provision(100_000.0, IRACAssetClass.DOUBTFUL_1, is_secured=False)
        assert prov == pytest.approx(100_000.0)  # 100%

    def test_doubtful_3(self) -> None:
        prov = rbi_minimum_provision(100_000.0, IRACAssetClass.DOUBTFUL_3)
        assert prov == pytest.approx(100_000.0)  # 100%

    def test_loss(self) -> None:
        prov = rbi_minimum_provision(100_000.0, IRACAssetClass.LOSS)
        assert prov == pytest.approx(100_000.0)  # 100%


class TestRestructuredAccountStage:
    def test_overdue_post_restructure(self) -> None:
        stage = restructured_account_stage(
            days_past_due_post_restructure=91,
            months_since_restructure=24,
        )
        assert stage == IFRS9Stage.STAGE_3

    def test_at_point_of_restructuring_is_stage3(self) -> None:
        # Per RBI norms: NPA at the point of restructuring
        stage = restructured_account_stage(
            days_past_due_post_restructure=0,
            months_since_restructure=0,
        )
        assert stage == IFRS9Stage.STAGE_3

    def test_within_probation_period(self) -> None:
        stage = restructured_account_stage(
            days_past_due_post_restructure=0,
            months_since_restructure=6,
        )
        assert stage == IFRS9Stage.STAGE_2

    def test_satisfactory_performance_upgrade(self) -> None:
        stage = restructured_account_stage(
            days_past_due_post_restructure=0,
            months_since_restructure=12,
        )
        assert stage == IFRS9Stage.STAGE_1

    def test_performing_but_mild_dpd(self) -> None:
        stage = restructured_account_stage(
            days_past_due_post_restructure=35,
            months_since_restructure=18,
        )
        assert stage == IFRS9Stage.STAGE_2


class TestCalculateECLIndAS:
    def test_stage1(self) -> None:
        ecl = calculate_ecl_ind_as(IFRS9Stage.STAGE_1, pd_12m=0.02, lgd=0.45, ead=1000.0)
        assert ecl == pytest.approx(9.0)

    def test_stage2_requires_marginal_pds(self) -> None:
        with pytest.raises(ValueError, match="marginal_pds required"):
            calculate_ecl_ind_as(IFRS9Stage.STAGE_2, pd_12m=0.02, lgd=0.45, ead=1000.0)

    def test_stage2_lifetime(self) -> None:
        marginal_pds = np.array([0.02, 0.03])
        ecl = calculate_ecl_ind_as(
            IFRS9Stage.STAGE_2, pd_12m=0.02, lgd=0.45, ead=1000.0,
            marginal_pds=marginal_pds,
        )
        expected = (0.02 + 0.03) * 0.45 * 1000.0
        assert ecl == pytest.approx(expected)

    def test_irac_floor_applied(self) -> None:
        # Model ECL = 9.0 (0.02 * 0.45 * 1000), but RBI sub-standard = 15% = 150
        ecl = calculate_ecl_ind_as(
            IFRS9Stage.STAGE_3,
            pd_12m=0.02,
            lgd=0.45,
            ead=1000.0,
            marginal_pds=np.array([0.02]),
            irac_class=IRACAssetClass.SUBSTANDARD,
            is_secured=True,
        )
        assert ecl == pytest.approx(150.0)  # RBI floor binds

    def test_model_ecl_above_floor(self) -> None:
        # Model ECL with high PD should exceed RBI standard floor
        ecl = calculate_ecl_ind_as(
            IFRS9Stage.STAGE_1,
            pd_12m=0.10,
            lgd=0.60,
            ead=1000.0,
            irac_class=IRACAssetClass.STANDARD,
            sector="commercial",
        )
        # Model ECL = 60.0, RBI floor = 4.0 → model binds
        assert ecl == pytest.approx(60.0)
