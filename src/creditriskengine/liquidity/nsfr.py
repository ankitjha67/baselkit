"""Net Stable Funding Ratio (NSFR) per BCBS NSF.

Reference:
    - BCBS "Basel III: the net stable funding ratio" (d295, October 2014;
      NSF in the consolidated framework).
    - EU CRR2 Part Six Title IV (Arts. 428a-428az).

The NSFR requires that long-term and illiquid assets be funded by stable
sources over a one-year horizon:

    NSFR = Available Stable Funding (ASF) / Required Stable Funding (RSF) >= 100%

ASF is the carrying value of capital and liabilities weighted by an ASF
factor (more stable funding -> higher factor). RSF is the carrying value
of assets and off-balance-sheet items weighted by an RSF factor (more
illiquid / longer-dated assets -> higher factor).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

logger = logging.getLogger(__name__)

NSFR_MINIMUM: float = 1.00  # 100%


class ASFCategory(StrEnum):
    """Available Stable Funding categories with their factors (NSF.10)."""

    CAPITAL = "capital"  # regulatory capital + other >=1yr liabilities
    STABLE_DEPOSITS = "stable_deposits"  # retail/SME, stable
    LESS_STABLE_DEPOSITS = "less_stable_deposits"  # retail/SME, less stable
    WHOLESALE_NONFIN_LT1Y = "wholesale_nonfin_lt1y"  # non-fin corp < 1yr
    OTHER_LT1Y = "other_lt1y"  # other funding < 1yr (incl. financial)


class RSFCategory(StrEnum):
    """Required Stable Funding categories with their factors (NSF.30)."""

    CASH_AND_L1 = "cash_and_l1"  # coins, central bank reserves, Level 1
    LEVEL_2A = "level_2a"
    LEVEL_2B = "level_2b"
    LOANS_LT1Y = "loans_lt1y"  # performing loans < 1yr to non-fin
    RESIDENTIAL_MORTGAGE = "residential_mortgage"  # >=1yr, <=35% RW
    OTHER_LOANS_GE1Y = "other_loans_ge1y"  # other performing loans >=1yr
    OTHER_ASSETS = "other_assets"  # NSF residual (fixed assets, defaulted)


# NSF.10 — ASF factors.
ASF_FACTORS: dict[ASFCategory, float] = {
    ASFCategory.CAPITAL: 1.00,
    ASFCategory.STABLE_DEPOSITS: 0.95,
    ASFCategory.LESS_STABLE_DEPOSITS: 0.90,
    ASFCategory.WHOLESALE_NONFIN_LT1Y: 0.50,
    ASFCategory.OTHER_LT1Y: 0.00,
}

# NSF.30-40 — RSF factors.
RSF_FACTORS: dict[RSFCategory, float] = {
    RSFCategory.CASH_AND_L1: 0.05,
    RSFCategory.LEVEL_2A: 0.15,
    RSFCategory.LEVEL_2B: 0.50,
    RSFCategory.LOANS_LT1Y: 0.50,
    RSFCategory.RESIDENTIAL_MORTGAGE: 0.65,
    RSFCategory.OTHER_LOANS_GE1Y: 0.85,
    RSFCategory.OTHER_ASSETS: 1.00,
}


def available_stable_funding(amounts: Mapping[ASFCategory, float]) -> float:
    """Weighted Available Stable Funding (NSF.10).

    Args:
        amounts: Carrying amount per ASF category.

    Returns:
        Total ASF.

    Raises:
        ValueError: If any amount is negative.
    """
    if any(v < 0.0 for v in amounts.values()):
        raise ValueError("ASF amounts must be non-negative")
    return float(sum(ASF_FACTORS[cat] * amt for cat, amt in amounts.items()))


def required_stable_funding(amounts: Mapping[RSFCategory, float]) -> float:
    """Weighted Required Stable Funding (NSF.30-40).

    Args:
        amounts: Carrying amount per RSF category.

    Returns:
        Total RSF.

    Raises:
        ValueError: If any amount is negative.
    """
    if any(v < 0.0 for v in amounts.values()):
        raise ValueError("RSF amounts must be non-negative")
    return float(sum(RSF_FACTORS[cat] * amt for cat, amt in amounts.items()))


@dataclass(frozen=True)
class NSFRResult:
    """Net Stable Funding Ratio result.

    Attributes:
        asf: Total available stable funding.
        rsf: Total required stable funding.
        nsfr: ASF / RSF (1.0 = 100%).
        nsfr_pct: NSFR as a percentage.
        is_compliant: True if NSFR >= 100%.
    """

    asf: float
    rsf: float
    nsfr: float
    nsfr_pct: float
    is_compliant: bool


def net_stable_funding_ratio(
    asf_amounts: Mapping[ASFCategory, float],
    rsf_amounts: Mapping[RSFCategory, float],
) -> NSFRResult:
    """Compute the Net Stable Funding Ratio.

    Args:
        asf_amounts: Carrying amounts per ASF category.
        rsf_amounts: Carrying amounts per RSF category.

    Returns:
        An :class:`NSFRResult`. When RSF is zero the NSFR is reported as
        ``inf`` and treated as compliant.
    """
    asf = available_stable_funding(asf_amounts)
    rsf = required_stable_funding(rsf_amounts)

    if rsf <= 0.0:
        return NSFRResult(
            asf=round(asf, 6), rsf=0.0,
            nsfr=float("inf"), nsfr_pct=float("inf"), is_compliant=True,
        )

    nsfr = asf / rsf
    return NSFRResult(
        asf=round(asf, 6),
        rsf=round(rsf, 6),
        nsfr=round(nsfr, 6),
        nsfr_pct=round(nsfr * 100.0, 4),
        is_compliant=nsfr >= NSFR_MINIMUM,
    )
