"""Liquidity Coverage Ratio (LCR) per BCBS LCR.

Reference:
    - BCBS "Basel III: The Liquidity Coverage Ratio and liquidity risk
      monitoring tools" (d238, January 2013; LCR in the consolidated
      framework).
    - EU CRR / Commission Delegated Regulation (EU) 2015/61.

The LCR requires a bank to hold enough unencumbered high-quality liquid
assets (HQLA) to survive a 30-calendar-day stress scenario:

    LCR = stock of HQLA / total net cash outflows over 30 days >= 100%

HQLA is split into tiers with prescribed haircuts and caps:
    - Level 1   : no haircut, no cap.
    - Level 2A  : 15% haircut.
    - Level 2B  : 50% haircut.
    - Level 2 (2A + 2B) is capped at 40% of total HQLA.
    - Level 2B is capped at 15% of total HQLA.

Net cash outflows = total expected outflows - min(total expected inflows,
75% of total expected outflows).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

L2A_HAIRCUT: float = 0.15
L2B_HAIRCUT: float = 0.50
INFLOW_CAP_FRACTION: float = 0.75  # inflows capped at 75% of outflows
LCR_MINIMUM: float = 1.00  # 100%


@dataclass(frozen=True)
class HQLAResult:
    """High-quality liquid asset stock after haircuts and caps.

    Attributes:
        level1: Level 1 assets (no haircut).
        level2a: Level 2A assets after the 15% haircut.
        level2b: Level 2B assets after the 50% haircut and the 15% cap.
        total_hqla: Total HQLA after applying the 40% Level 2 cap.
        cap_adjustment: Amount removed by the Level 2 / 2B caps.
    """

    level1: float
    level2a: float
    level2b: float
    total_hqla: float
    cap_adjustment: float


def stock_of_hqla(
    level1: float,
    level2a_pre_haircut: float,
    level2b_pre_haircut: float,
) -> HQLAResult:
    """Compute the stock of HQLA with haircuts and the Level 2 caps.

    Applies the 15%/50% haircuts to Level 2A/2B, then the 15%-of-HQLA cap
    on Level 2B and the 40%-of-HQLA cap on total Level 2, using the
    closed-form cap ratios (LCR40.13-40.15):

        Level 2B post-haircut <= (15/85) * (Level 1 + Level 2A)
        Level 2  post-haircut <= (40/60) * Level 1

    Args:
        level1: Level 1 assets (market value, no haircut).
        level2a_pre_haircut: Level 2A assets before the 15% haircut.
        level2b_pre_haircut: Level 2B assets before the 50% haircut.

    Returns:
        An :class:`HQLAResult`.

    Raises:
        ValueError: If any input is negative.
    """
    if min(level1, level2a_pre_haircut, level2b_pre_haircut) < 0.0:
        raise ValueError("HQLA amounts must be non-negative")

    l1 = level1
    l2a = level2a_pre_haircut * (1.0 - L2A_HAIRCUT)
    l2b = level2b_pre_haircut * (1.0 - L2B_HAIRCUT)
    uncapped_total = l1 + l2a + l2b

    # Level 2B cap: 15% of HQLA  <=>  L2B <= (15/85) * (L1 + L2A).
    l2b_cap = (15.0 / 85.0) * (l1 + l2a)
    l2b_capped = min(l2b, l2b_cap)

    # Total Level 2 cap: 40% of HQLA  <=>  L2 <= (40/60) * L1.
    l2_total = l2a + l2b_capped
    l2_cap = (40.0 / 60.0) * l1
    if l2_total > l2_cap:
        # Reduce Level 2B first, then Level 2A, to meet the 40% cap.
        excess = l2_total - l2_cap
        reduce_2b = min(l2b_capped, excess)
        l2b_capped -= reduce_2b
        excess -= reduce_2b
        l2a -= excess

    total_hqla = l1 + l2a + l2b_capped
    return HQLAResult(
        level1=round(l1, 6),
        level2a=round(l2a, 6),
        level2b=round(l2b_capped, 6),
        total_hqla=round(total_hqla, 6),
        cap_adjustment=round(uncapped_total - total_hqla, 6),
    )


def net_cash_outflows(total_outflows: float, total_inflows: float) -> float:
    """Net cash outflows over the 30-day horizon (LCR40.2).

    Inflows are capped at 75% of outflows::

        net = total_outflows - min(total_inflows, 0.75 * total_outflows)

    Args:
        total_outflows: Total stressed cash outflows over 30 days.
        total_inflows: Total stressed cash inflows over 30 days.

    Returns:
        Net cash outflows (non-negative).

    Raises:
        ValueError: If either input is negative.
    """
    if total_outflows < 0.0 or total_inflows < 0.0:
        raise ValueError("cash flows must be non-negative")
    capped_inflows = min(total_inflows, INFLOW_CAP_FRACTION * total_outflows)
    return total_outflows - capped_inflows


@dataclass(frozen=True)
class LCRResult:
    """Liquidity Coverage Ratio result.

    Attributes:
        hqla: Stock of HQLA after haircuts and caps.
        net_cash_outflows: Net 30-day stressed cash outflows.
        lcr: HQLA / net cash outflows (as a ratio; 1.0 = 100%).
        lcr_pct: LCR as a percentage.
        is_compliant: True if LCR >= 100%.
    """

    hqla: float
    net_cash_outflows: float
    lcr: float
    lcr_pct: float
    is_compliant: bool


def liquidity_coverage_ratio(
    level1: float,
    level2a_pre_haircut: float,
    level2b_pre_haircut: float,
    total_outflows: float,
    total_inflows: float,
) -> LCRResult:
    """Compute the Liquidity Coverage Ratio.

    Args:
        level1: Level 1 HQLA (no haircut).
        level2a_pre_haircut: Level 2A HQLA before haircut.
        level2b_pre_haircut: Level 2B HQLA before haircut.
        total_outflows: Total stressed 30-day cash outflows.
        total_inflows: Total stressed 30-day cash inflows.

    Returns:
        An :class:`LCRResult`. When net cash outflows are zero the LCR is
        reported as ``inf`` and treated as compliant.
    """
    hqla = stock_of_hqla(level1, level2a_pre_haircut, level2b_pre_haircut)
    net = net_cash_outflows(total_outflows, total_inflows)

    if net <= 0.0:
        return LCRResult(
            hqla=hqla.total_hqla,
            net_cash_outflows=0.0,
            lcr=float("inf"),
            lcr_pct=float("inf"),
            is_compliant=True,
        )

    lcr = hqla.total_hqla / net
    return LCRResult(
        hqla=hqla.total_hqla,
        net_cash_outflows=round(net, 6),
        lcr=round(lcr, 6),
        lcr_pct=round(lcr * 100.0, 4),
        is_compliant=lcr >= LCR_MINIMUM,
    )
