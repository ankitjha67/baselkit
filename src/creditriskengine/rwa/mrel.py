"""Minimum Requirement for own funds and Eligible Liabilities (MREL).

Reference:
    - Directive (EU) 2014/59 (BRRD) as amended by (EU) 2019/879 (BRRD2),
      Articles 45-45m.
    - Regulation (EU) 806/2014 (SRMR) as amended by (EU) 2019/877 (SRMR2).
    - SRB "Minimum Requirement for Own Funds and Eligible Liabilities (MREL)
      — SRB Policy" (annual).

MREL is the EU resolution requirement that a bank hold enough own funds and
bail-inable liabilities to absorb losses and recapitalise in resolution. It
is set as the higher of two requirements — one expressed against the Total
Risk Exposure Amount (TREA) and one against the leverage Total Exposure
Measure (TEM) — and calibrated as the sum of a Loss Absorption Amount (LAA)
and a Recapitalisation Amount (RCA):

    LAA = Pillar 1 + Pillar 2 Requirement (P2R)
    RCA = Pillar 1 + P2R + Market Confidence Charge (MCC)
    MREL = LAA + RCA

For resolution entities of EU G-SIIs the MREL is floored at the TLAC
minimums (18% of TREA / 6.75% of TEM). The Combined Buffer Requirement is
met with CET1 on top of the TREA MREL.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Pillar 1 minimums.
PILLAR1_TREA: float = 0.08  # 8% of TREA
PILLAR1_TEM: float = 0.03  # 3% leverage ratio requirement

# TLAC floors for G-SII resolution entities (from 2022).
GSII_TREA_FLOOR: float = 0.18
GSII_TEM_FLOOR: float = 0.0675


def mrel_trea_requirement(
    p2r: float,
    market_confidence_charge: float = 0.0,
    pillar1: float = PILLAR1_TREA,
    is_gsii: bool = False,
) -> float:
    """MREL requirement as a fraction of TREA (BRRD2 Art. 45c).

        LAA  = pillar1 + p2r
        RCA  = pillar1 + p2r + market_confidence_charge
        MREL = LAA + RCA

    Floored at the 18% TLAC minimum for G-SII resolution entities.

    Args:
        p2r: Pillar 2 Requirement as a fraction of TREA.
        market_confidence_charge: MCC added to the recapitalisation amount
            (e.g. the Combined Buffer Requirement net of the countercyclical
            buffer, per the resolution authority's policy).
        pillar1: Pillar 1 TREA minimum (default 8%).
        is_gsii: If True, floor the requirement at the 18% TLAC minimum.

    Returns:
        MREL requirement as a fraction of TREA.

    Raises:
        ValueError: If ``p2r``, ``market_confidence_charge`` or ``pillar1``
            is negative.
    """
    if min(p2r, market_confidence_charge, pillar1) < 0.0:
        raise ValueError("p2r, market_confidence_charge and pillar1 must be non-negative")
    laa = pillar1 + p2r
    rca = pillar1 + p2r + market_confidence_charge
    requirement = laa + rca
    if is_gsii:
        return max(requirement, GSII_TREA_FLOOR)
    return requirement


def mrel_tem_requirement(
    leverage_p2r: float = 0.0,
    pillar1: float = PILLAR1_TEM,
    is_gsii: bool = False,
) -> float:
    """MREL requirement as a fraction of the leverage exposure (TEM).

        LAA  = pillar1 + leverage_p2r
        RCA  = pillar1 + leverage_p2r
        MREL = LAA + RCA

    Floored at the 6.75% TLAC minimum for G-SII resolution entities.

    Args:
        leverage_p2r: Any leverage-based Pillar 2 add-on (default 0).
        pillar1: Leverage-ratio Pillar 1 minimum (default 3%).
        is_gsii: If True, floor the requirement at the 6.75% TLAC minimum.

    Returns:
        MREL requirement as a fraction of TEM.

    Raises:
        ValueError: If ``leverage_p2r`` or ``pillar1`` is negative.
    """
    if min(leverage_p2r, pillar1) < 0.0:
        raise ValueError("leverage_p2r and pillar1 must be non-negative")
    requirement = 2.0 * (pillar1 + leverage_p2r)
    if is_gsii:
        return max(requirement, GSII_TEM_FLOOR)
    return requirement


@dataclass(frozen=True)
class MRELResult:
    """MREL adequacy assessment.

    Attributes:
        eligible_mrel: Eligible own funds and bail-inable liabilities.
        trea: Total Risk Exposure Amount.
        tem: Total Exposure Measure (leverage exposure).
        trea_ratio: eligible_mrel / TREA.
        tem_ratio: eligible_mrel / TEM.
        required_trea_pct: Required MREL as a fraction of TREA.
        required_tem_pct: Required MREL as a fraction of TEM.
        trea_shortfall: MREL shortfall against the TREA requirement.
        tem_shortfall: MREL shortfall against the TEM requirement.
        binding_constraint: Which requirement demands more MREL ("trea" or
            "tem").
        is_compliant: True if both requirements are met.
    """

    eligible_mrel: float
    trea: float
    tem: float
    trea_ratio: float
    tem_ratio: float
    required_trea_pct: float
    required_tem_pct: float
    trea_shortfall: float
    tem_shortfall: float
    binding_constraint: str
    is_compliant: bool


def assess_mrel(
    eligible_mrel: float,
    trea: float,
    tem: float,
    p2r: float,
    market_confidence_charge: float = 0.0,
    leverage_p2r: float = 0.0,
    is_gsii: bool = False,
) -> MRELResult:
    """Assess eligible MREL against the TREA and TEM requirements.

    Args:
        eligible_mrel: Eligible own funds and bail-inable liabilities.
        trea: Total Risk Exposure Amount (must be positive).
        tem: Total Exposure Measure (must be positive).
        p2r: Pillar 2 Requirement as a fraction of TREA.
        market_confidence_charge: MCC added to the TREA recapitalisation
            amount.
        leverage_p2r: Any leverage-based Pillar 2 add-on.
        is_gsii: Whether the entity is an EU G-SII resolution entity (floors
            the requirements at the TLAC minimums).

    Returns:
        An :class:`MRELResult`.

    Raises:
        ValueError: If ``trea`` or ``tem`` is not positive, or
            ``eligible_mrel`` is negative.
    """
    if trea <= 0.0:
        raise ValueError("trea must be positive")
    if tem <= 0.0:
        raise ValueError("tem must be positive")
    if eligible_mrel < 0.0:
        raise ValueError("eligible_mrel must be non-negative")

    req_trea = mrel_trea_requirement(p2r, market_confidence_charge, is_gsii=is_gsii)
    req_tem = mrel_tem_requirement(leverage_p2r, is_gsii=is_gsii)

    trea_ratio = eligible_mrel / trea
    tem_ratio = eligible_mrel / tem

    trea_required_amt = req_trea * trea
    tem_required_amt = req_tem * tem
    trea_shortfall = max(trea_required_amt - eligible_mrel, 0.0)
    tem_shortfall = max(tem_required_amt - eligible_mrel, 0.0)

    binding = "trea" if trea_required_amt >= tem_required_amt else "tem"

    return MRELResult(
        eligible_mrel=round(eligible_mrel, 6),
        trea=round(trea, 6),
        tem=round(tem, 6),
        trea_ratio=round(trea_ratio, 6),
        tem_ratio=round(tem_ratio, 6),
        required_trea_pct=round(req_trea, 6),
        required_tem_pct=round(req_tem, 6),
        trea_shortfall=round(trea_shortfall, 6),
        tem_shortfall=round(tem_shortfall, 6),
        binding_constraint=binding,
        is_compliant=trea_ratio >= req_trea and tem_ratio >= req_tem,
    )
