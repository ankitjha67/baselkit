"""
NBFC Ind AS 109 prudential backstop and SBR NPA glide-path.

References:
    - DOR(NBFC).CC.PD.No.109/22.10.106/2019-20 (March 13, 2020) —
      Implementation of Ind AS by NBFCs / ARCs: prudential backstop
      (higher of Ind AS 109 ECL vs IRACP with Impairment Reserve).
    - RBI/2021-22/112 DOR.CRE.REC.No.60/03.10.001/2021-22
      (October 22, 2021) — Scale-Based Regulation for NBFCs.
    - DOR.STR.REC.40/21.04.048/2022-23 (June 6, 2022) —
      NBFC-UL differential standard-asset provisioning.
    - Master Direction (NBFC-SBR), October 19, 2023.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NBFC Layer (Scale-Based Regulation)
# ---------------------------------------------------------------------------


class NBFCLayer(StrEnum):
    """NBFC layers under Scale-Based Regulation.

    Reference: DOR.CRE.REC.No.60/03.10.001/2021-22, October 22, 2021.
    """

    BASE = "base"
    """Base Layer: NBFC-BL (formerly NBFC-ND-NSI)."""

    MIDDLE = "middle"
    """Middle Layer: NBFC-ML."""

    UPPER = "upper"
    """Upper Layer: NBFC-UL (systemically important)."""

    TOP = "top"
    """Top Layer: NBFC-TL (extreme systemic significance)."""


# ---------------------------------------------------------------------------
# Prudential backstop per March 13, 2020 circular
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NBFCBackstopResult:
    """Result of the NBFC Ind AS 109 prudential backstop calculation.

    Per Annex paragraph 2 of DOR(NBFC).CC.PD.No.109/22.10.106/2019-20:
    the NBFC books Ind AS 109 ECL through P&L. The excess of IRACP
    provision over Ind AS 109 ECL is transferred from net profit to an
    Impairment Reserve which is:
        (i)   not reckoned for net NPA computation,
        (ii)  not treated as Tier-1 capital,
        (iii) reviewable by RBI on continuity.

    Attributes:
        ind_as_109_ecl: Ind AS 109 ECL impairment allowance.
        iracp_provision: IRACP-style provision amount.
        impairment_reserve_transfer: Excess of IRACP over ECL (≥ 0).
        booked_to_pl: Amount charged to P&L (= Ind AS 109 ECL).
        total_floor: max(Ind AS 109 ECL, IRACP provision).
    """

    ind_as_109_ecl: float
    iracp_provision: float
    impairment_reserve_transfer: float
    booked_to_pl: float
    total_floor: float


def apply_nbfc_backstop(
    ind_as_109_ecl: float,
    iracp_provision: float,
) -> NBFCBackstopResult:
    """Apply the NBFC prudential backstop.

    Reference:
        DOR(NBFC).CC.PD.No.109/22.10.106/2019-20 (March 13, 2020),
        Annex paragraph 2.

    Args:
        ind_as_109_ecl: Ind AS 109 ECL impairment allowance.
        iracp_provision: IRACP-style provision (from the legacy
            income recognition and asset classification norms).

    Returns:
        :class:`NBFCBackstopResult` with both legs, impairment
        reserve transfer, and the binding floor.
    """
    excess = max(iracp_provision - ind_as_109_ecl, 0.0)
    return NBFCBackstopResult(
        ind_as_109_ecl=ind_as_109_ecl,
        iracp_provision=iracp_provision,
        impairment_reserve_transfer=excess,
        booked_to_pl=ind_as_109_ecl,
        total_floor=max(ind_as_109_ecl, iracp_provision),
    )


# ---------------------------------------------------------------------------
# SBR NPA-DPD glide-path
# ---------------------------------------------------------------------------


def npa_dpd_threshold(
    as_of: date,
    already_on_90_day_norm: bool = False,
) -> int:
    """Return the NPA DPD threshold for an NBFC on a given date.

    NBFCs that were previously on a 180-day overdue norm follow a
    glide-path to 90-day NPA recognition:

        Before March 31, 2024: > 180 days
        March 31, 2024 onwards: > 150 days
        March 31, 2025 onwards: > 120 days
        March 31, 2026 onwards: > 90 days

    NBFCs already on the 90-day norm (banks, and NBFCs that adopted
    90-day voluntarily) skip the glide-path.

    Reference:
        RBI/2021-22/112 DOR.CRE.REC.No.60/03.10.001/2021-22
        (October 22, 2021), SBR Master Direction October 19, 2023.

    Args:
        as_of: Reporting / classification date.
        already_on_90_day_norm: Whether the NBFC is already on 90-day
            NPA recognition.

    Returns:
        DPD threshold (e.g., 90, 120, 150, or 180).
    """
    if already_on_90_day_norm:
        return 90
    if as_of < date(2024, 3, 31):
        return 180
    if as_of < date(2025, 3, 31):
        return 150
    if as_of < date(2026, 3, 31):
        return 120
    return 90


# ---------------------------------------------------------------------------
# NBFC-UL differential standard-asset provisioning
# ---------------------------------------------------------------------------


NBFC_UL_STANDARD_RATES: dict[str, float] = {
    "housing_individual": 0.0025,
    "housing_teaser": 0.0200,
    "housing_teaser_post_reset": 0.0040,
    "cre_rh": 0.0075,
    "cre": 0.0100,
    "small_micro_enterprise": 0.0025,
    "medium_enterprise": 0.0040,
    "other": 0.0040,
}
"""NBFC-UL differential standard-asset provisioning rates.

Reference: DOR.STR.REC.40/21.04.048/2022-23 (June 6, 2022),
effective October 1, 2022.
"""


def nbfc_ul_standard_asset_provision(
    funded_outstanding: float,
    sector: str = "other",
    teaser_one_year_post_reset: bool = False,
) -> float:
    """Calculate NBFC-UL standard-asset provision.

    Reference:
        DOR.STR.REC.40/21.04.048/2022-23 (June 6, 2022).

    Args:
        funded_outstanding: Funded outstanding amount.
        sector: Sector key from ``NBFC_UL_STANDARD_RATES``.
        teaser_one_year_post_reset: For teaser housing, whether one
            year has elapsed since rate reset (reverts to 0.40%).

    Returns:
        Provision amount.
    """
    if funded_outstanding <= 0:
        return 0.0
    if sector == "housing_teaser" and teaser_one_year_post_reset:
        return funded_outstanding * NBFC_UL_STANDARD_RATES["housing_teaser_post_reset"]
    rate = NBFC_UL_STANDARD_RATES.get(sector, NBFC_UL_STANDARD_RATES["other"])
    return funded_outstanding * rate
