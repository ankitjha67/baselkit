"""Tests for emerging-market asset classification (China NFRA, Indonesia OJK)."""

from __future__ import annotations

import pytest

from creditriskengine.core.types import IFRS9Stage
from creditriskengine.ecl.emerging import (
    NFRAFiveTier,
    OJKCollectability,
    classify_nfra_five_tier,
    classify_ojk_collectability,
    nfra_tier_to_ifrs9_stage,
    ojk_minimum_provision,
    ojk_to_ifrs9_stage,
)

# ============================================================================
# China NFRA five-tier
# ============================================================================


class TestNFRAClassification:
    def test_normal(self) -> None:
        assert classify_nfra_five_tier(0) == NFRAFiveTier.NORMAL

    def test_special_mention(self) -> None:
        assert classify_nfra_five_tier(45) == NFRAFiveTier.SPECIAL_MENTION

    def test_substandard_at_90dpd(self) -> None:
        assert classify_nfra_five_tier(90) == NFRAFiveTier.SUBSTANDARD
        assert classify_nfra_five_tier(200) == NFRAFiveTier.SUBSTANDARD

    def test_doubtful_above_270(self) -> None:
        assert classify_nfra_five_tier(300) == NFRAFiveTier.DOUBTFUL

    def test_loss_above_360(self) -> None:
        assert classify_nfra_five_tier(400) == NFRAFiveTier.LOSS

    def test_bankruptcy_override(self) -> None:
        assert classify_nfra_five_tier(0, is_bankrupt=True) == NFRAFiveTier.LOSS

    def test_ecl_drives_doubtful(self) -> None:
        # Low DPD but ECL 60% → Doubtful (worst of the two)
        assert classify_nfra_five_tier(10, ecl_ratio=0.60) == NFRAFiveTier.DOUBTFUL

    def test_ecl_drives_loss(self) -> None:
        assert classify_nfra_five_tier(10, ecl_ratio=0.95) == NFRAFiveTier.LOSS

    def test_worst_of_dpd_and_ecl(self) -> None:
        # 300 DPD (Doubtful) + 95% ECL (Loss) → Loss
        assert classify_nfra_five_tier(300, ecl_ratio=0.95) == NFRAFiveTier.LOSS

    def test_stage_mapping(self) -> None:
        assert nfra_tier_to_ifrs9_stage(NFRAFiveTier.NORMAL) == IFRS9Stage.STAGE_1
        assert nfra_tier_to_ifrs9_stage(NFRAFiveTier.SPECIAL_MENTION) == IFRS9Stage.STAGE_2
        assert nfra_tier_to_ifrs9_stage(NFRAFiveTier.SUBSTANDARD) == IFRS9Stage.STAGE_3
        assert nfra_tier_to_ifrs9_stage(NFRAFiveTier.DOUBTFUL) == IFRS9Stage.STAGE_3
        assert nfra_tier_to_ifrs9_stage(NFRAFiveTier.LOSS) == IFRS9Stage.STAGE_3


# ============================================================================
# Indonesia OJK collectability
# ============================================================================


class TestOJKClassification:
    def test_current(self) -> None:
        assert classify_ojk_collectability(0) == OJKCollectability.CURRENT

    def test_special_mention(self) -> None:
        assert classify_ojk_collectability(30) == OJKCollectability.SPECIAL_MENTION
        assert classify_ojk_collectability(90) == OJKCollectability.SPECIAL_MENTION

    def test_substandard(self) -> None:
        assert classify_ojk_collectability(91) == OJKCollectability.SUBSTANDARD
        assert classify_ojk_collectability(120) == OJKCollectability.SUBSTANDARD

    def test_doubtful(self) -> None:
        assert classify_ojk_collectability(121) == OJKCollectability.DOUBTFUL
        assert classify_ojk_collectability(180) == OJKCollectability.DOUBTFUL

    def test_loss(self) -> None:
        assert classify_ojk_collectability(181) == OJKCollectability.LOSS

    def test_loss_override(self) -> None:
        assert classify_ojk_collectability(0, is_loss=True) == OJKCollectability.LOSS

    def test_provision_current(self) -> None:
        # 1% of gross
        assert ojk_minimum_provision(1_000_000, OJKCollectability.CURRENT) == pytest.approx(10_000)

    def test_provision_special_mention(self) -> None:
        prov = ojk_minimum_provision(1_000_000, OJKCollectability.SPECIAL_MENTION)
        assert prov == pytest.approx(50_000)

    def test_provision_substandard_net_of_collateral(self) -> None:
        # 15% of (1M - 400k) = 90k
        prov = ojk_minimum_provision(
            1_000_000, OJKCollectability.SUBSTANDARD, eligible_collateral_value=400_000
        )
        assert prov == pytest.approx(90_000)

    def test_provision_loss_full(self) -> None:
        # 100% of uncovered
        prov = ojk_minimum_provision(
            1_000_000, OJKCollectability.LOSS, eligible_collateral_value=200_000
        )
        assert prov == pytest.approx(800_000)

    def test_provision_zero_ead(self) -> None:
        assert ojk_minimum_provision(0, OJKCollectability.LOSS) == 0.0

    def test_stage_mapping(self) -> None:
        assert ojk_to_ifrs9_stage(OJKCollectability.CURRENT) == IFRS9Stage.STAGE_1
        assert ojk_to_ifrs9_stage(OJKCollectability.SPECIAL_MENTION) == IFRS9Stage.STAGE_2
        assert ojk_to_ifrs9_stage(OJKCollectability.LOSS) == IFRS9Stage.STAGE_3
