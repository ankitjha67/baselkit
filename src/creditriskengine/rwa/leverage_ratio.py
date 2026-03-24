"""
Basel III leverage ratio framework per BCBS d424, CRE80.

The leverage ratio is a non-risk-based backstop measure that constrains
excessive balance sheet growth relative to Tier 1 capital:

    Leverage Ratio = Tier 1 Capital / Total Exposure Measure

The minimum requirement is 3%.  G-SIBs face an additional buffer equal to
50% of their risk-weighted G-SIB surcharge.

Total Exposure Measure comprises:
    (a) on-balance-sheet exposures,
    (b) derivative exposures (SA-CCR per CRE52),
    (c) securities financing transaction (SFT) exposures,
    (d) off-balance-sheet items (with a 10% CCF floor, CRE80.38).
"""

import logging

logger = logging.getLogger(__name__)

# CRE80.3 — minimum leverage ratio
MINIMUM_LEVERAGE_RATIO_PCT: float = 0.03

# CRE80.5 — G-SIB buffer is 50% of risk-weighted G-SIB surcharge
GSIB_BUFFER_MULTIPLIER: float = 0.50

# CRE80.38 — CCF floor for off-balance-sheet items
OBS_CCF_FLOOR: float = 0.10

# CRE52.52 — SA-CCR alpha multiplier
SA_CCR_ALPHA: float = 1.4


def leverage_ratio(
    tier1_capital: float,
    total_exposure_measure: float,
) -> float:
    """Compute the Basel III leverage ratio.

    Formula (CRE80.2):
        Leverage Ratio = Tier 1 Capital / Total Exposure Measure

    Args:
        tier1_capital: Tier 1 capital (CET1 + AT1).
        total_exposure_measure: Aggregate exposure measure.

    Returns:
        Leverage ratio as a decimal (e.g. 0.05 for 5%).

    Raises:
        ValueError: If total_exposure_measure is zero or negative.
    """
    if total_exposure_measure <= 0.0:
        raise ValueError(
            f"Total exposure measure must be positive, got {total_exposure_measure}"
        )
    ratio = tier1_capital / total_exposure_measure
    logger.debug(
        "Leverage ratio: tier1=%.2f exposure=%.2f ratio=%.4f (%.2f%%)",
        tier1_capital, total_exposure_measure, ratio, ratio * 100,
    )
    return ratio


def total_exposure_measure(
    on_balance_sheet: float,
    derivative_exposures: float,
    sft_exposures: float,
    off_balance_sheet_items: float,
    ccf_obs: float = 1.0,
) -> float:
    """Aggregate the total exposure measure per CRE80.10.

    Total Exposure = On-BS + Derivatives + SFTs + OBS

    When ``ccf_obs`` is provided it is applied as a blanket multiplier to
    the already-CCF-adjusted off-balance-sheet amount.  Use 1.0 (default)
    when off_balance_sheet_items has already been credit-converted.

    Args:
        on_balance_sheet: On-balance-sheet exposures (CRE80.13).
        derivative_exposures: Derivative exposure amount (CRE80.18).
        sft_exposures: SFT exposure amount (CRE80.30).
        off_balance_sheet_items: Off-balance-sheet exposure (CRE80.36).
        ccf_obs: Additional CCF multiplier for OBS items (default 1.0).

    Returns:
        Aggregate total exposure measure.
    """
    obs_adjusted = off_balance_sheet_items * ccf_obs
    tem = on_balance_sheet + derivative_exposures + sft_exposures + obs_adjusted
    logger.debug(
        "Total exposure measure: on_bs=%.2f deriv=%.2f sft=%.2f obs=%.2f "
        "(ccf=%.2f) => TEM=%.2f",
        on_balance_sheet, derivative_exposures, sft_exposures,
        off_balance_sheet_items, ccf_obs, tem,
    )
    return tem


def derivative_exposure_sa_ccr(
    replacement_cost: float,
    potential_future_exposure: float,
    collateral_held: float = 0.0,
    alpha: float = SA_CCR_ALPHA,
) -> float:
    """Compute derivative exposure under SA-CCR for the leverage ratio.

    Formula (CRE52.52, CRE80.18):
        EAD = alpha * (RC + PFE)

    Where RC is replacement cost (net of collateral if applicable) and PFE
    is the potential future exposure add-on.  Collateral held reduces the
    RC component.

    Args:
        replacement_cost: Current replacement cost of derivatives.
        potential_future_exposure: PFE add-on per CRE52.
        collateral_held: Cash / securities collateral received (default 0).
        alpha: SA-CCR alpha multiplier (default 1.4, CRE52.52).

    Returns:
        Derivative exposure amount (EAD).

    Raises:
        ValueError: If alpha is not positive.
    """
    if alpha <= 0.0:
        raise ValueError(f"Alpha must be positive, got {alpha}")
    rc_net = max(replacement_cost - collateral_held, 0.0)
    ead = alpha * (rc_net + potential_future_exposure)
    logger.debug(
        "SA-CCR derivative exposure: RC=%.2f collateral=%.2f RC_net=%.2f "
        "PFE=%.2f alpha=%.2f => EAD=%.2f",
        replacement_cost, collateral_held, rc_net,
        potential_future_exposure, alpha, ead,
    )
    return ead


def off_balance_sheet_exposure(
    notional: float,
    ccf: float,
) -> float:
    """Compute off-balance-sheet exposure with the leverage ratio CCF floor.

    Per CRE80.38 the CCF for OBS items under the leverage ratio is the
    higher of 10% and the applicable CCF from the standardised approach.

    Args:
        notional: Notional / nominal amount of the OBS item.
        ccf: Credit conversion factor from the standardised approach.

    Returns:
        Credit-equivalent OBS exposure.

    Raises:
        ValueError: If notional is negative.
    """
    if notional < 0.0:
        raise ValueError(f"Notional must be non-negative, got {notional}")
    effective_ccf = max(ccf, OBS_CCF_FLOOR)
    exposure = notional * effective_ccf
    logger.debug(
        "OBS exposure: notional=%.2f ccf=%.4f effective_ccf=%.4f => %.2f",
        notional, ccf, effective_ccf, exposure,
    )
    return exposure


def meets_leverage_requirement(
    ratio: float,
    minimum_pct: float = MINIMUM_LEVERAGE_RATIO_PCT,
    gsib_buffer_pct: float = 0.0,
) -> bool:
    """Check whether the leverage ratio meets the regulatory minimum.

    Per CRE80.3 the minimum is 3%.  G-SIBs must additionally hold a
    leverage ratio buffer equal to 50% of their risk-weighted G-SIB
    surcharge (CRE80.5).

    Args:
        ratio: Computed leverage ratio (decimal).
        minimum_pct: Minimum leverage ratio requirement (default 3%).
        gsib_buffer_pct: G-SIB leverage ratio buffer (decimal, e.g. 0.005
            for a 1% G-SIB surcharge => 50% * 0.01 = 0.005).

    Returns:
        True if the ratio meets or exceeds the requirement.
    """
    required = minimum_pct + gsib_buffer_pct
    compliant = ratio >= required
    logger.debug(
        "Leverage compliance: ratio=%.4f required=%.4f (min=%.4f + gsib=%.4f) => %s",
        ratio, required, minimum_pct, gsib_buffer_pct,
        "PASS" if compliant else "FAIL",
    )
    return compliant


def leverage_ratio_summary(
    tier1_capital: float,
    on_bs: float,
    derivatives: float,
    sfts: float,
    obs_items: float,
    jurisdiction_config: dict[str, float] | None = None,
) -> dict[str, float | bool]:
    """Produce a full leverage ratio summary.

    Aggregates all components, computes the ratio, and checks compliance.

    The optional ``jurisdiction_config`` dict may contain:
        - ``minimum_pct``: override for the minimum ratio (default 0.03).
        - ``gsib_surcharge_pct``: risk-weighted G-SIB surcharge; the
          leverage buffer is 50% of this value (CRE80.5).

    Args:
        tier1_capital: Tier 1 capital.
        on_bs: On-balance-sheet exposures.
        derivatives: Derivative exposure amount.
        sfts: SFT exposure amount.
        obs_items: Off-balance-sheet exposure (already credit-converted).
        jurisdiction_config: Optional jurisdiction-specific overrides.

    Returns:
        Dict containing:
            tier1_capital, on_balance_sheet, derivative_exposures,
            sft_exposures, off_balance_sheet, total_exposure_measure,
            leverage_ratio, minimum_pct, gsib_buffer_pct, required_ratio,
            meets_requirement.
    """
    config = jurisdiction_config or {}
    min_pct = config.get("minimum_pct", MINIMUM_LEVERAGE_RATIO_PCT)
    gsib_surcharge = config.get("gsib_surcharge_pct", 0.0)
    gsib_buffer = gsib_surcharge * GSIB_BUFFER_MULTIPLIER

    tem = total_exposure_measure(on_bs, derivatives, sfts, obs_items)
    ratio = leverage_ratio(tier1_capital, tem)
    compliant = meets_leverage_requirement(ratio, min_pct, gsib_buffer)

    return {
        "tier1_capital": tier1_capital,
        "on_balance_sheet": on_bs,
        "derivative_exposures": derivatives,
        "sft_exposures": sfts,
        "off_balance_sheet": obs_items,
        "total_exposure_measure": tem,
        "leverage_ratio": ratio,
        "minimum_pct": min_pct,
        "gsib_buffer_pct": gsib_buffer,
        "required_ratio": min_pct + gsib_buffer,
        "meets_requirement": compliant,
    }
