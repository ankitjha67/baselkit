"""
Indonesia OJK five-tier collectability and provisioning.

Reference:
    - OJK POJK 40/POJK.03/2019 (Quality Assessment of Asset for
      Commercial Banks), effective January 1, 2020.
    - PSAK 71 (Indonesian IFRS 9 equivalent).

Five collectability tiers with minimum regulatory provisioning (CKPN):
    Current (Lancar)                       : 1%
    Special Mention (Dalam Perhatian Khusus): 5%
    Sub-Standard (Kurang Lancar)           : 15%
    Doubtful (Diragukan)                   : 50%
    Loss (Macet)                           : 100%

Regulatory minimum provisions for the NPA tiers are computed net of
eligible collateral value; the headline rates apply to the uncovered
portion.
"""

from __future__ import annotations

import logging
from enum import StrEnum

from creditriskengine.core.types import IFRS9Stage

logger = logging.getLogger(__name__)


class OJKCollectability(StrEnum):
    """OJK five-tier collectability classification."""

    CURRENT = "current"
    SPECIAL_MENTION = "special_mention"
    SUBSTANDARD = "substandard"
    DOUBTFUL = "doubtful"
    LOSS = "loss"


# Minimum provisioning rates per POJK 40/2019 (on uncovered portion for
# NPA tiers; Current/Special Mention apply to the gross balance).
OJK_PROVISION_RATES: dict[OJKCollectability, float] = {
    OJKCollectability.CURRENT: 0.01,
    OJKCollectability.SPECIAL_MENTION: 0.05,
    OJKCollectability.SUBSTANDARD: 0.15,
    OJKCollectability.DOUBTFUL: 0.50,
    OJKCollectability.LOSS: 1.00,
}
"""OJK minimum provisioning rates (POJK 40/2019)."""

# DPD boundaries (commercial loan convention).
_SPECIAL_MENTION_DPD: int = 1
_SUBSTANDARD_DPD: int = 91
_DOUBTFUL_DPD: int = 121
_LOSS_DPD: int = 181


def classify_ojk_collectability(
    days_past_due: int,
    is_loss: bool = False,
) -> OJKCollectability:
    """Classify a loan under the OJK five-tier collectability framework.

    DPD convention (commercial loans, POJK 40/2019):
        0 DPD          : Current
        1-90 DPD       : Special Mention
        91-120 DPD     : Sub-Standard
        121-180 DPD    : Doubtful
        > 180 DPD      : Loss

    Args:
        days_past_due: Days past due.
        is_loss: Whether the asset is identified as a loss (override).

    Returns:
        :class:`OJKCollectability` classification.
    """
    if is_loss or days_past_due >= _LOSS_DPD:
        return OJKCollectability.LOSS
    if days_past_due >= _DOUBTFUL_DPD:
        return OJKCollectability.DOUBTFUL
    if days_past_due >= _SUBSTANDARD_DPD:
        return OJKCollectability.SUBSTANDARD
    if days_past_due >= _SPECIAL_MENTION_DPD:
        return OJKCollectability.SPECIAL_MENTION
    return OJKCollectability.CURRENT


def ojk_minimum_provision(
    ead: float,
    collectability: OJKCollectability,
    eligible_collateral_value: float = 0.0,
) -> float:
    """Calculate the OJK minimum provision (CKPN).

    For the NPA tiers (Sub-Standard, Doubtful, Loss) the rate applies to
    the exposure net of eligible collateral value. For Current and
    Special Mention the rate applies to the gross exposure.

    Reference: POJK 40/2019.

    Args:
        ead: Exposure at default / outstanding amount.
        collectability: OJK collectability tier.
        eligible_collateral_value: Value of eligible collateral
            deductible for NPA tiers.

    Returns:
        Minimum provision amount.
    """
    if ead <= 0:
        return 0.0

    rate = OJK_PROVISION_RATES[collectability]

    if collectability in (
        OJKCollectability.CURRENT,
        OJKCollectability.SPECIAL_MENTION,
    ):
        return ead * rate

    # NPA tiers: net of eligible collateral
    uncovered = max(ead - eligible_collateral_value, 0.0)
    return uncovered * rate


def ojk_to_ifrs9_stage(collectability: OJKCollectability) -> IFRS9Stage:
    """Map OJK collectability to a PSAK 71 / IFRS 9 stage.

    Current → Stage 1; Special Mention → Stage 2; Sub-Standard /
    Doubtful / Loss → Stage 3.

    Args:
        collectability: OJK collectability tier.

    Returns:
        Corresponding IFRS 9 stage.
    """
    mapping: dict[OJKCollectability, IFRS9Stage] = {
        OJKCollectability.CURRENT: IFRS9Stage.STAGE_1,
        OJKCollectability.SPECIAL_MENTION: IFRS9Stage.STAGE_2,
        OJKCollectability.SUBSTANDARD: IFRS9Stage.STAGE_3,
        OJKCollectability.DOUBTFUL: IFRS9Stage.STAGE_3,
        OJKCollectability.LOSS: IFRS9Stage.STAGE_3,
    }
    return mapping[collectability]
