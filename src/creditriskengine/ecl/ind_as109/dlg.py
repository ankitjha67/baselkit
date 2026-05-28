"""
RBI ECL Master Direction 2026 — Default Loss Guarantee (DLG) treatment.

Reference: RBI/DOR/2026-27/398, Paragraph 88, in conjunction with RBI Master
Directions on DLG dated November 28, 2025.

A bank may consider DLG cover when determining ECL provisions across
all stages, subject to:
    - The DLG arrangement being integral to the contractual terms
      of the loan, and
    - The DLG not being recognised separately.

On every invocation of the DLG, the cover reduces. The bank must
recompute ECL after each invocation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DLGAdjustment:
    """Result of applying a DLG adjustment to gross ECL.

    Attributes:
        gross_ecl: Model ECL before DLG adjustment.
        dlg_absorbed: Portion of ECL absorbed by the DLG cover.
        net_ecl: ECL after DLG absorption.
        remaining_dlg_capacity: Remaining DLG capacity after this
            absorption (not the gross DLG, but what is still available
            for future invocations).
    """

    gross_ecl: float
    dlg_absorbed: float
    net_ecl: float
    remaining_dlg_capacity: float


def ecl_with_dlg(
    gross_ecl: float,
    dlg_remaining_capacity: float,
    portfolio_ead: float = 0.0,
    dlg_cap_pct: float | None = None,
) -> DLGAdjustment:
    """Adjust ECL for Default Loss Guarantee cover.

    The DLG can absorb up to its remaining capacity (and, if specified,
    up to a percentage cap of portfolio EAD). The net ECL borne by the
    bank is reduced accordingly, but never below zero.

    Args:
        gross_ecl: Model-computed gross ECL on the portfolio.
        dlg_remaining_capacity: Currently available DLG capacity
            (after prior invocations have been deducted).
        portfolio_ead: Portfolio exposure at default (required only
            if ``dlg_cap_pct`` is provided).
        dlg_cap_pct: Optional cap on DLG cover as a fraction of
            portfolio EAD. If provided, the absorption is also
            limited to ``dlg_cap_pct * portfolio_ead``.

    Returns:
        :class:`DLGAdjustment` with absorbed, net, and remaining capacity.

    Reference:
        RBI/DOR/2026-27/398 Paragraph 88.
    """
    if gross_ecl <= 0:
        return DLGAdjustment(
            gross_ecl=gross_ecl,
            dlg_absorbed=0.0,
            net_ecl=max(gross_ecl, 0.0),
            remaining_dlg_capacity=dlg_remaining_capacity,
        )

    max_absorption = max(dlg_remaining_capacity, 0.0)
    if dlg_cap_pct is not None and portfolio_ead > 0:
        max_absorption = min(max_absorption, dlg_cap_pct * portfolio_ead)

    absorbed = min(gross_ecl, max_absorption)
    net = gross_ecl - absorbed
    remaining_after = max(dlg_remaining_capacity - absorbed, 0.0)

    return DLGAdjustment(
        gross_ecl=gross_ecl,
        dlg_absorbed=absorbed,
        net_ecl=net,
        remaining_dlg_capacity=remaining_after,
    )
