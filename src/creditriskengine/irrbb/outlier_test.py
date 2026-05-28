"""
IRRBB Supervisory Outlier Test (SOT).

Reference:
    - EBA RTS on Supervisory Outlier Test (EBA/RTS/2022/09).
    - CRR3 Art. 84.
    - BCBS d368.

Two outlier tests:
    - EVE SOT: max Delta-EVE loss across the 6 shocks must not exceed
      15% of Tier 1 capital.
    - NII SOT: Delta-NII decline under the two parallel shocks must not
      exceed 2.5% of Tier 1 capital (EBA large-decline threshold).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from creditriskengine.irrbb.eve import InterestRateShock

logger = logging.getLogger(__name__)


EVE_SOT_THRESHOLD: float = 0.15
"""EVE outlier threshold: 15% of Tier 1 capital (EBA RTS/2022/09)."""

NII_SOT_THRESHOLD: float = 0.025
"""NII large-decline threshold: 2.5% of Tier 1 capital (EBA RTS/2022/09)."""


@dataclass(frozen=True)
class OutlierTestResult:
    """Supervisory Outlier Test result.

    Attributes:
        worst_eve_shock: The shock producing the largest EVE loss.
        worst_eve_loss: The largest EVE loss (positive magnitude).
        eve_ratio: worst_eve_loss / Tier 1 capital.
        eve_breach: Whether the 15% EVE threshold is breached.
        nii_decline: NII decline (positive magnitude).
        nii_ratio: nii_decline / Tier 1 capital.
        nii_breach: Whether the 2.5% NII threshold is breached.
    """

    worst_eve_shock: InterestRateShock | None
    worst_eve_loss: float
    eve_ratio: float
    eve_breach: bool
    nii_decline: float
    nii_ratio: float
    nii_breach: bool


def supervisory_outlier_test(
    delta_eve: dict[InterestRateShock, float],
    tier1_capital: float,
    delta_nii: float = 0.0,
) -> OutlierTestResult:
    """Run the IRRBB Supervisory Outlier Test.

    Args:
        delta_eve: Mapping of shock → Delta-EVE (from
            :func:`creditriskengine.irrbb.eve.eve_sensitivity`).
        tier1_capital: Tier 1 capital.
        delta_nii: Delta-NII under the binding parallel shock (negative
            for a decline).

    Returns:
        :class:`OutlierTestResult`.

    Raises:
        ValueError: If tier1_capital is non-positive.

    Reference:
        EBA RTS/2022/09, CRR3 Art. 84.
    """
    if tier1_capital <= 0:
        raise ValueError("tier1_capital must be positive")

    # Worst (most negative) Delta-EVE
    worst_shock: InterestRateShock | None = None
    worst_loss = 0.0
    for shock, value in delta_eve.items():
        loss = -value  # loss is the negative of Delta-EVE
        if loss > worst_loss:
            worst_loss = loss
            worst_shock = shock

    eve_ratio = worst_loss / tier1_capital
    eve_breach = eve_ratio > EVE_SOT_THRESHOLD

    nii_decline = max(-delta_nii, 0.0)
    nii_ratio = nii_decline / tier1_capital
    nii_breach = nii_ratio > NII_SOT_THRESHOLD

    return OutlierTestResult(
        worst_eve_shock=worst_shock,
        worst_eve_loss=worst_loss,
        eve_ratio=eve_ratio,
        eve_breach=eve_breach,
        nii_decline=nii_decline,
        nii_ratio=nii_ratio,
        nii_breach=nii_breach,
    )
