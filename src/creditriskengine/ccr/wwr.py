"""
Wrong-Way Risk (WWR) — general and specific.

Reference:
    - BCBS CRE52.18, CRE53.48-53 (wrong-way risk).
    - Basel III CCR framework.
    - Gregory (2015) — The xVA Challenge, Ch. 17.

Wrong-way risk arises when exposure to a counterparty is adversely
correlated with the counterparty's credit quality:
    - General WWR (GWWR): correlation with macro/market factors.
    - Specific WWR (SWWR): legal/structural link (e.g., counterparty
      posts its own debt as collateral, or exposure is FX-linked to a
      counterparty's domestic currency).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def specific_wwr_flag(
    collateral_is_own_issuance: bool = False,
    exposure_fx_linked_to_counterparty_sovereign: bool = False,
    counterparty_is_related_party: bool = False,
) -> bool:
    """Flag specific wrong-way risk per BCBS CRE52.18.

    SWWR is present when there is a direct legal or structural link
    between the exposure and the counterparty's creditworthiness.

    Args:
        collateral_is_own_issuance: Counterparty posts its own (or a
            related entity's) securities as collateral.
        exposure_fx_linked_to_counterparty_sovereign: Exposure value
            is linked to the FX rate of the counterparty's domestic
            sovereign.
        counterparty_is_related_party: Counterparty is a related
            party to the underlying reference.

    Returns:
        ``True`` if specific WWR is identified.

    Reference:
        BCBS CRE52.18.
    """
    return (
        collateral_is_own_issuance
        or exposure_fx_linked_to_counterparty_sovereign
        or counterparty_is_related_party
    )


def alpha_wrong_way_multiplier(
    base_alpha: float = 1.4,
    correlation: float = 0.0,
    stress_factor: float = 1.0,
) -> float:
    """Adjust the EAD alpha multiplier for general wrong-way risk.

    Higher positive exposure-credit correlation warrants an alpha
    above the regulatory floor of 1.4. The adjustment scales linearly
    with correlation and a stress factor:

        alpha_adj = base_alpha * (1 + correlation * (stress_factor - 1))

    capped at a prudent maximum of 2.5.

    Args:
        base_alpha: Base alpha (default 1.4 per CRE52.52).
        correlation: Exposure-credit correlation in [-1, 1]. Positive
            values indicate wrong-way risk.
        stress_factor: Multiplicative stress applied at full
            correlation (>= 1.0).

    Returns:
        Adjusted alpha multiplier in [base_alpha, 2.5].

    Reference:
        BCBS CRE53.48 (alpha and WWR), Gregory (2015).
    """
    if correlation <= 0:
        return base_alpha
    adj = base_alpha * (1.0 + correlation * (stress_factor - 1.0))
    return float(min(max(adj, base_alpha), 2.5))


def conditional_epe_wwr(
    base_epe: float,
    exposure_credit_correlation: float,
    counterparty_pd: float,
) -> float:
    """Approximate WWR-adjusted EPE conditional on counterparty default.

    When exposure and counterparty credit are positively correlated,
    the expected exposure conditional on default exceeds the
    unconditional EPE. A simple one-factor adjustment:

        EPE_cond = base_EPE * (1 + rho * lambda)

    where lambda reflects the default-conditional shift, scaled by the
    inverse-normal of the PD to capture tail severity.

    Args:
        base_epe: Unconditional EPE.
        exposure_credit_correlation: Correlation rho in [-1, 1].
        counterparty_pd: Counterparty PD in (0, 1).

    Returns:
        WWR-adjusted conditional EPE (>= base_epe for positive rho).

    Reference:
        BCBS CRE53.48-53, Gregory (2015) Ch. 17.
    """
    from scipy.stats import norm

    if exposure_credit_correlation <= 0 or not 0.0 < counterparty_pd < 1.0:
        return base_epe

    # Default-conditional severity scaler: magnitude of the standardised
    # default threshold, bounded for numerical stability.
    severity = abs(norm.ppf(counterparty_pd))
    lambda_shift = min(severity / 3.0, 1.0)
    factor = 1.0 + exposure_credit_correlation * lambda_shift
    return float(base_epe * factor)
