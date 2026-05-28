"""
China NFRA five-tier risk classification.

Reference:
    - NFRA/PBOC "Measures for the Risk Classification of Financial
      Assets of Commercial Banks" (February 2023).
    - Effective July 1, 2023; full reclassification by December 31, 2025.
    - CAS 22 (Chinese Accounting Standard 22, aligned with IFRS 9).

Five-tier classification driven by days-past-due and ECL ratio:
    Normal          : performing, no adverse indicators.
    Special Mention : overdue or adverse factors, < 90 DPD.
    Substandard     : NPA — 90-270 DPD, or ECL 40-50%.
    Doubtful        : 270-360 DPD, or ECL 50-90%.
    Loss            : > 360 DPD, or ECL > 90%, or bankruptcy.
"""

from __future__ import annotations

import logging
from enum import StrEnum

from creditriskengine.core.types import IFRS9Stage

logger = logging.getLogger(__name__)


NFRA_SUBSTANDARD_DPD: int = 90
"""NPA threshold: 90 DPD (Substandard)."""

NFRA_DOUBTFUL_DPD: int = 270
"""Doubtful threshold: > 270 DPD."""

NFRA_LOSS_DPD: int = 360
"""Loss threshold: > 360 DPD."""

NFRA_DOUBTFUL_ECL: float = 0.50
"""ECL ratio threshold for Doubtful: > 50%."""

NFRA_LOSS_ECL: float = 0.90
"""ECL ratio threshold for Loss: > 90%."""


class NFRAFiveTier(StrEnum):
    """NFRA five-tier asset risk classification."""

    NORMAL = "normal"
    SPECIAL_MENTION = "special_mention"
    SUBSTANDARD = "substandard"
    DOUBTFUL = "doubtful"
    LOSS = "loss"


def classify_nfra_five_tier(
    days_past_due: int,
    ecl_ratio: float = 0.0,
    is_bankrupt: bool = False,
) -> NFRAFiveTier:
    """Classify a financial asset under the NFRA five-tier framework.

    The classification takes the worst (most severe) of the DPD-based
    and ECL-ratio-based assessments, plus a bankruptcy override to Loss.

    Reference: NFRA Measures (February 2023), Articles 9-11.

    Args:
        days_past_due: Days past due of principal or interest.
        ecl_ratio: Expected credit loss as a fraction of exposure
            (used as a quantitative classification trigger).
        is_bankrupt: Whether the borrower is bankrupt / liquidated.

    Returns:
        :class:`NFRAFiveTier` classification.
    """
    if is_bankrupt:
        return NFRAFiveTier.LOSS

    # DPD-based tier
    if days_past_due > NFRA_LOSS_DPD:
        dpd_tier = NFRAFiveTier.LOSS
    elif days_past_due > NFRA_DOUBTFUL_DPD:
        dpd_tier = NFRAFiveTier.DOUBTFUL
    elif days_past_due >= NFRA_SUBSTANDARD_DPD:
        dpd_tier = NFRAFiveTier.SUBSTANDARD
    elif days_past_due > 0:
        dpd_tier = NFRAFiveTier.SPECIAL_MENTION
    else:
        dpd_tier = NFRAFiveTier.NORMAL

    # ECL-based tier
    if ecl_ratio > NFRA_LOSS_ECL:
        ecl_tier = NFRAFiveTier.LOSS
    elif ecl_ratio > NFRA_DOUBTFUL_ECL:
        ecl_tier = NFRAFiveTier.DOUBTFUL
    elif ecl_ratio >= 0.40:
        ecl_tier = NFRAFiveTier.SUBSTANDARD
    else:
        ecl_tier = NFRAFiveTier.NORMAL

    # Take the more severe tier (higher ordinal)
    order = list(NFRAFiveTier)
    return max(dpd_tier, ecl_tier, key=order.index)


def nfra_tier_to_ifrs9_stage(tier: NFRAFiveTier) -> IFRS9Stage:
    """Map an NFRA five-tier classification to an IFRS 9 / CAS 22 stage.

    Normal → Stage 1; Special Mention → Stage 2; Substandard / Doubtful /
    Loss → Stage 3 (credit-impaired).

    Reference: CAS 22 staging alignment.

    Args:
        tier: NFRA five-tier classification.

    Returns:
        Corresponding IFRS 9 stage.
    """
    mapping: dict[NFRAFiveTier, IFRS9Stage] = {
        NFRAFiveTier.NORMAL: IFRS9Stage.STAGE_1,
        NFRAFiveTier.SPECIAL_MENTION: IFRS9Stage.STAGE_2,
        NFRAFiveTier.SUBSTANDARD: IFRS9Stage.STAGE_3,
        NFRAFiveTier.DOUBTFUL: IFRS9Stage.STAGE_3,
        NFRAFiveTier.LOSS: IFRS9Stage.STAGE_3,
    }
    return mapping[tier]
