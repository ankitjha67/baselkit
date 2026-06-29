"""Total Loss-Absorbing Capacity (TLAC) for G-SIBs (FSB / BCBS).

Reference:
    - FSB "Principles on Loss-absorbing and Recapitalisation Capacity of
      G-SIBs in Resolution — Total Loss-absorbing Capacity (TLAC) Term
      Sheet" (9 November 2015).
    - BCBS "TLAC holdings" standard (October 2016).

A global systemically important bank (G-SIB) must hold TLAC of at least
the higher of two minimums:

    - 18% of risk-weighted assets (RWA), and
    - 6.75% of the Basel III leverage ratio exposure measure.

(During the conformance period 2019-2021 the minimums were 16% / 6%.)

Regulatory capital counts toward TLAC, except that CET1 used to meet the
combined buffer requirement (capital conservation + G-SIB + countercyclical)
cannot also be counted toward the TLAC minimum — buffers sit on top of
TLAC. Eligible long-term subordinated debt makes up the remainder.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Steady-state minimums (from 1 January 2022).
TLAC_RWA_MINIMUM: float = 0.18
TLAC_LEVERAGE_MINIMUM: float = 0.0675

# Conformance-period minimums (1 January 2019 - 31 December 2021).
TLAC_RWA_MINIMUM_CONFORMANCE: float = 0.16
TLAC_LEVERAGE_MINIMUM_CONFORMANCE: float = 0.06


def available_tlac(
    cet1: float,
    additional_tier1: float,
    tier2: float,
    eligible_tlac_debt: float,
    buffer_requirement_pct: float,
    rwa: float,
) -> float:
    """Available TLAC after excluding buffer CET1 (FSB Term Sheet s.6).

    TLAC comprises regulatory capital (CET1 + AT1 + Tier 2) plus eligible
    TLAC debt, less the CET1 required to meet the combined buffer
    requirement (which must sit on top of the TLAC minimum)::

        available = CET1 + AT1 + T2 + eligible_debt
                    - buffer_requirement_pct * RWA

    Args:
        cet1: Common Equity Tier 1 capital.
        additional_tier1: Additional Tier 1 capital.
        tier2: Tier 2 capital.
        eligible_tlac_debt: Eligible external TLAC-eligible debt.
        buffer_requirement_pct: Combined buffer requirement as a fraction
            of RWA (CConB + G-SIB surcharge + CCyB).
        rwa: Risk-weighted assets.

    Returns:
        Available TLAC (non-negative).

    Raises:
        ValueError: If any capital amount is negative, RWA is negative, or
            the buffer requirement is outside [0, 1].
    """
    if min(cet1, additional_tier1, tier2, eligible_tlac_debt, rwa) < 0.0:
        raise ValueError("capital amounts and RWA must be non-negative")
    if not 0.0 <= buffer_requirement_pct <= 1.0:
        raise ValueError("buffer_requirement_pct must be in [0, 1]")

    buffer_cet1 = buffer_requirement_pct * rwa
    total = cet1 + additional_tier1 + tier2 + eligible_tlac_debt
    return max(total - buffer_cet1, 0.0)


@dataclass(frozen=True)
class TLACResult:
    """TLAC adequacy assessment.

    Attributes:
        tlac_available: Available TLAC resources.
        rwa: Risk-weighted assets.
        leverage_exposure: Leverage ratio exposure measure.
        rwa_ratio: TLAC / RWA.
        leverage_ratio: TLAC / leverage exposure.
        rwa_minimum: Applicable RWA-based minimum (fraction).
        leverage_minimum: Applicable leverage-based minimum (fraction).
        rwa_shortfall: TLAC shortfall against the RWA minimum (0 if met).
        leverage_shortfall: Shortfall against the leverage minimum.
        binding_constraint: Which minimum requires more TLAC ("rwa" or
            "leverage").
        is_compliant: True if both minimums are met.
    """

    tlac_available: float
    rwa: float
    leverage_exposure: float
    rwa_ratio: float
    leverage_ratio: float
    rwa_minimum: float
    leverage_minimum: float
    rwa_shortfall: float
    leverage_shortfall: float
    binding_constraint: str
    is_compliant: bool


def tlac_ratios(
    tlac_available: float,
    rwa: float,
    leverage_exposure: float,
    conformance_period: bool = False,
) -> TLACResult:
    """Assess TLAC against the RWA and leverage minimums.

    Args:
        tlac_available: Available TLAC resources (see :func:`available_tlac`).
        rwa: Risk-weighted assets (must be positive).
        leverage_exposure: Basel III leverage ratio exposure measure
            (must be positive).
        conformance_period: If True, applies the 16% / 6% conformance-period
            minimums instead of the steady-state 18% / 6.75%.

    Returns:
        A :class:`TLACResult`.

    Raises:
        ValueError: If ``rwa`` or ``leverage_exposure`` is not positive, or
            ``tlac_available`` is negative.
    """
    if rwa <= 0.0:
        raise ValueError("rwa must be positive")
    if leverage_exposure <= 0.0:
        raise ValueError("leverage_exposure must be positive")
    if tlac_available < 0.0:
        raise ValueError("tlac_available must be non-negative")

    rwa_min = TLAC_RWA_MINIMUM_CONFORMANCE if conformance_period else TLAC_RWA_MINIMUM
    lev_min = (
        TLAC_LEVERAGE_MINIMUM_CONFORMANCE if conformance_period else TLAC_LEVERAGE_MINIMUM
    )

    rwa_ratio = tlac_available / rwa
    lev_ratio = tlac_available / leverage_exposure

    rwa_required = rwa_min * rwa
    lev_required = lev_min * leverage_exposure
    rwa_shortfall = max(rwa_required - tlac_available, 0.0)
    lev_shortfall = max(lev_required - tlac_available, 0.0)

    binding = "rwa" if rwa_required >= lev_required else "leverage"

    return TLACResult(
        tlac_available=round(tlac_available, 6),
        rwa=round(rwa, 6),
        leverage_exposure=round(leverage_exposure, 6),
        rwa_ratio=round(rwa_ratio, 6),
        leverage_ratio=round(lev_ratio, 6),
        rwa_minimum=rwa_min,
        leverage_minimum=lev_min,
        rwa_shortfall=round(rwa_shortfall, 6),
        leverage_shortfall=round(lev_shortfall, 6),
        binding_constraint=binding,
        is_compliant=rwa_ratio >= rwa_min and lev_ratio >= lev_min,
    )
