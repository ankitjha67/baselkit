"""
RBI ECL Master Direction 2026 — PD and LGD regulatory floors and backstops.

Reference: RBI/DOR/2026-27/398, Paragraphs 96-98.
Effective: April 1, 2027.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


RBI_PD_FLOOR: float = 0.0003
"""12-month PD regulatory floor of 0.03%.

Reference: RBI/DOR/2026-27/398 Paragraph 96.
"""

RBI_LGD_BACKSTOP_SECURED: float = 0.65
"""LGD backstop for the secured portion of an exposure: 65%.

Reference: RBI/DOR/2026-27/398 Paragraph 97.
"""

RBI_LGD_BACKSTOP_UNSECURED: float = 0.70
"""LGD backstop for the unsecured portion of an exposure: 70%.

Reference: RBI/DOR/2026-27/398 Paragraph 97.
"""

RBI_LGD_ELIGIBLE_COLLATERAL: float = 0.30
"""LGD backstop for exposures with eligible collateral: 30%.

Eligible collateral includes cash, gold, government securities,
LIC policy, KVP, NSC.

Reference: RBI/DOR/2026-27/398 Paragraph 98.
"""


def apply_rbi_pd_floor(pd: float) -> float:
    """Apply the RBI PD floor of 0.03%.

    Reference: RBI/DOR/2026-27/398 Paragraph 96.

    Args:
        pd: Model-estimated probability of default.

    Returns:
        Floored PD value (at least ``RBI_PD_FLOOR``).
    """
    return max(pd, RBI_PD_FLOOR)


def apply_rbi_lgd_backstop(
    lgd: float,
    is_secured: bool,
    has_eligible_collateral: bool = False,
) -> float:
    """Apply the RBI LGD backstop per Paragraphs 97-98.

    Hierarchy:
        - If eligible collateral (Paragraph 98): floor at 30%.
        - Else if secured (Paragraph 97): floor at 65%.
        - Else (unsecured): floor at 70%.

    Args:
        lgd: Model-estimated loss given default.
        is_secured: Whether the exposure is secured.
        has_eligible_collateral: Whether the exposure is collateralised
            by eligible collateral (cash, gold, government securities,
            LIC policy, KVP, NSC) per Paragraph 98.

    Returns:
        LGD value floored at the applicable RBI backstop.
    """
    if has_eligible_collateral:
        return max(lgd, RBI_LGD_ELIGIBLE_COLLATERAL)
    if is_secured:
        return max(lgd, RBI_LGD_BACKSTOP_SECURED)
    return max(lgd, RBI_LGD_BACKSTOP_UNSECURED)
