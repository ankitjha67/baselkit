"""BCBS SCO60 — Prudential treatment of crypto-asset exposures.

Reference: BCBS d545 (December 2022), amended July 17, 2024.
Effective: January 1, 2026.

Classification:
    Group 1a: Tokenised traditional assets — same RW as underlying.
    Group 1b: Stablecoins meeting redemption/reserve quality tests —
              modified SA treatment + infrastructure risk add-on.
    Group 2a: Other crypto with hedge recognition.
    Group 2b: Unbacked crypto (e.g., Bitcoin) — 1250% risk weight.

Exposure limits (SCO60.75-80):
    Group 2 total must not exceed 1% of Tier 1 capital (soft limit).
    Breach above 2% of Tier 1 → entire Group 2 reclassified as 2b.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

logger = logging.getLogger(__name__)


class CryptoAssetGroup(StrEnum):
    """BCBS SCO60 crypto-asset classification groups."""

    GROUP_1A = "group_1a"
    """Tokenised traditional assets (same treatment as underlying)."""

    GROUP_1B = "group_1b"
    """Stablecoins meeting stabilisation mechanism requirements."""

    GROUP_2A = "group_2a"
    """Other crypto with hedge recognition (limited capital relief)."""

    GROUP_2B = "group_2b"
    """Unbacked crypto — 1250% risk weight. Per SCO60.60."""


GROUP_2B_RISK_WEIGHT: float = 12.50
"""1250% RW expressed as a multiplier (12.50 × 100% = 1250%).

Reference: SCO60.60.
"""

GROUP_1B_INFRASTRUCTURE_ADDON: float = 0.025
"""2.5% infrastructure risk add-on for Group 1b stablecoins.

Reference: SCO60.40.
"""

TIER1_SOFT_LIMIT_PCT: float = 0.01
"""Group 2 exposure soft limit: 1% of Tier 1 capital.

Reference: SCO60.75.
"""

TIER1_HARD_LIMIT_PCT: float = 0.02
"""Group 2 exposure hard limit: 2% of Tier 1 capital.

Breach triggers reclassification of all Group 2 to Group 2b.

Reference: SCO60.80.
"""


@dataclass(frozen=True)
class CryptoCapitalResult:
    """Result of crypto-asset capital calculation.

    Attributes:
        group: Assigned BCBS SCO60 group.
        exposure: Exposure amount.
        risk_weight_pct: Applied risk weight (%).
        rwa: Risk-weighted assets.
        infrastructure_addon: Group 1b infrastructure add-on (if applicable).
        limit_breach: Whether Group 2 Tier 1 limit is breached.
        limit_breach_level: ``"none"``, ``"soft"`` (1%), or ``"hard"`` (2%).
    """

    group: CryptoAssetGroup
    exposure: float
    risk_weight_pct: float
    rwa: float
    infrastructure_addon: float
    limit_breach: bool
    limit_breach_level: str


def crypto_asset_rwa(
    exposure: float,
    group: CryptoAssetGroup,
    underlying_rw_pct: float = 100.0,
    tier1_capital: float | None = None,
    total_group2_exposure: float | None = None,
) -> CryptoCapitalResult:
    """Calculate RWA for a crypto-asset exposure per BCBS SCO60.

    Args:
        exposure: Exposure amount.
        group: BCBS SCO60 classification group.
        underlying_rw_pct: Risk weight of the underlying traditional
            asset (only for Group 1a). Default 100%.
        tier1_capital: Bank's Tier 1 capital (for limit checks on
            Group 2 exposures).
        total_group2_exposure: Total Group 2 exposure across all
            crypto-asset holdings (for aggregate limit check).

    Returns:
        :class:`CryptoCapitalResult`.

    Reference:
        BCBS SCO60 (July 2024), effective January 1, 2026.
    """
    infra_addon = 0.0
    limit_breach = False
    limit_level = "none"

    if group == CryptoAssetGroup.GROUP_1A:
        rw = underlying_rw_pct
    elif group == CryptoAssetGroup.GROUP_1B:
        rw = underlying_rw_pct
        infra_addon = exposure * GROUP_1B_INFRASTRUCTURE_ADDON
    elif group == CryptoAssetGroup.GROUP_2A:
        rw = 1250.0
    else:
        rw = 1250.0

    rwa = exposure * (rw / 100.0) + infra_addon

    # Group 2 limit check
    if (
        group in (CryptoAssetGroup.GROUP_2A, CryptoAssetGroup.GROUP_2B)
        and tier1_capital is not None
        and total_group2_exposure is not None
    ):
        ratio = total_group2_exposure / tier1_capital if tier1_capital > 0 else 0.0
        if ratio > TIER1_HARD_LIMIT_PCT:
            limit_breach = True
            limit_level = "hard"
            rw = 1250.0
            rwa = exposure * (rw / 100.0)
        elif ratio > TIER1_SOFT_LIMIT_PCT:
            limit_breach = True
            limit_level = "soft"

    return CryptoCapitalResult(
        group=group,
        exposure=exposure,
        risk_weight_pct=rw,
        rwa=rwa,
        infrastructure_addon=infra_addon,
        limit_breach=limit_breach,
        limit_breach_level=limit_level,
    )
