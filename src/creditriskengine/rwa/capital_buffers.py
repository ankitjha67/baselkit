"""
Capital buffer calculations per BCBS d424, RBC40.1-40.8.

Capital buffers sit on top of minimum capital requirements and determine
distribution constraints when breached.

Components:
    - Capital Conservation Buffer (CConB): 2.5% CET1
    - Countercyclical Capital Buffer (CCyB): 0-2.5% CET1 (jurisdiction-set)
    - G-SIB Surcharge: 1.0%-3.5% CET1 based on systemic importance bucket
    - D-SIB Surcharge: jurisdiction-specific

The combined buffer requirement is the sum of all applicable buffers.
Breach triggers restrictions on distributions via the Maximum Distributable
Amount (MDA) framework.

Reference:
    BCBS d424, Section IV (Capital Buffers)
    BCBS d445 (G-SIB framework, November 2022)
    BCBS d368 (D-SIB framework, October 2012)
"""

import logging
from typing import Any

from creditriskengine.core.exceptions import ConfigurationError, ValidationError

logger = logging.getLogger(__name__)

# G-SIB surcharge by bucket per BCBS d445, paragraph 30.
# Bucket 1 -> 1.0%, Bucket 2 -> 1.5%, ..., Bucket 5 -> 3.5%.
_GSIB_BUCKET_SURCHARGES: dict[int, float] = {
    1: 0.010,
    2: 0.015,
    3: 0.020,
    4: 0.025,
    5: 0.035,
}

# Minimum capital ratios per BCBS d424, RBC20.2-20.4.
_MINIMUM_CET1_PCT: float = 0.045
_MINIMUM_TIER1_PCT: float = 0.06
_MINIMUM_TOTAL_CAPITAL_PCT: float = 0.08

# MDA quartile schedule per BCBS d424, RBC40.5.
# Maps (quartile_index) -> fraction of earnings that may be distributed.
# Quartile 4 = highest shortfall (within 0-25% of buffer), Quartile 1 = lowest.
_MDA_QUARTILES: list[float] = [1.0, 0.6, 0.4, 0.0]


def capital_conservation_buffer(
    jurisdiction_config: dict[str, Any],
) -> float:
    """Return the Capital Conservation Buffer (CConB) rate.

    The CConB is designed to ensure banks build up capital outside
    periods of stress that can be drawn down when losses are incurred.

    Reference: BCBS d424, RBC40.1.

    Args:
        jurisdiction_config: Parsed regulatory YAML config dict.

    Returns:
        CConB rate as a decimal (e.g. 0.025 for 2.5%).

    Raises:
        ConfigurationError: If capital_requirements section is missing.
    """
    cap_req = jurisdiction_config.get("capital_requirements")
    if cap_req is None:
        raise ConfigurationError(
            "Missing 'capital_requirements' section in jurisdiction config"
        )

    # US replaces CConB with Stress Capital Buffer (SCB) — use SCB minimum
    # if the stress_capital_buffer section is present and replaces CConB.
    scb = cap_req.get("stress_capital_buffer")
    if scb and scb.get("replaces") == "capital_conservation_buffer":
        rate = float(scb.get("minimum_pct", 0.025))
        logger.debug(
            "Using Stress Capital Buffer minimum as CConB proxy: %.4f", rate
        )
        return rate

    rate = float(cap_req.get("capital_conservation_buffer_pct", 0.025))
    logger.debug("Capital conservation buffer: %.4f", rate)
    return rate


def countercyclical_buffer(
    jurisdiction_config: dict[str, Any],
    ccyb_rate: float | None = None,
) -> float:
    """Return the Countercyclical Capital Buffer (CCyB) rate.

    The CCyB protects the banking sector from periods of excess
    aggregate credit growth. It is set by national authorities
    within the BCBS range of 0-2.5%.

    Reference: BCBS d424, RBC40.2; BCBS d172.

    Args:
        jurisdiction_config: Parsed regulatory YAML config dict.
        ccyb_rate: Explicit CCyB rate override. If provided, it is
            validated against the jurisdiction's permitted range.
            If None, the jurisdiction's current/default rate is used.

    Returns:
        CCyB rate as a decimal (e.g. 0.01 for 1.0%).

    Raises:
        ConfigurationError: If capital_requirements section is missing.
        ValidationError: If ccyb_rate is outside the permitted range.
    """
    cap_req = jurisdiction_config.get("capital_requirements")
    if cap_req is None:
        raise ConfigurationError(
            "Missing 'capital_requirements' section in jurisdiction config"
        )

    # Determine permitted range
    ccyb_range = cap_req.get("countercyclical_buffer_range", {})
    range_min = float(ccyb_range.get("min", 0.0))
    range_max = float(ccyb_range.get("max", 0.025))

    if ccyb_rate is not None:
        if ccyb_rate < range_min or ccyb_rate > range_max:
            raise ValidationError(
                f"CCyB rate {ccyb_rate:.4f} outside permitted range "
                f"[{range_min:.4f}, {range_max:.4f}]"
            )
        logger.debug("Countercyclical buffer (explicit): %.4f", ccyb_rate)
        return ccyb_rate

    # Use jurisdiction default if a fixed rate is specified
    fixed_rate = cap_req.get("countercyclical_buffer_pct")
    if fixed_rate is not None:
        rate = float(fixed_rate)
        logger.debug("Countercyclical buffer (jurisdiction default): %.4f", rate)
        return rate

    # Default to zero if no rate specified
    logger.debug("Countercyclical buffer: defaulting to 0.0")
    return 0.0


def gsib_surcharge(bucket: int) -> float:
    """Return the G-SIB surcharge for a given systemic importance bucket.

    G-SIB buckets range from 1 (lowest) to 5 (highest), with higher
    buckets carrying progressively larger surcharges to reduce the
    probability of failure of systemically important banks.

    Reference: BCBS d445, paragraph 30 (bucket allocation);
               BCBS d424, RBC40.3.

    Args:
        bucket: G-SIB bucket (1 through 5). Bucket 0 or None
            indicates the bank is not a G-SIB (returns 0.0).

    Returns:
        G-SIB surcharge as a decimal (e.g. 0.01 for 1.0%).

    Raises:
        ValidationError: If bucket is outside the valid range (0-5).
    """
    if bucket == 0:
        return 0.0

    if bucket not in _GSIB_BUCKET_SURCHARGES:
        raise ValidationError(
            f"Invalid G-SIB bucket {bucket}; must be 0-5"
        )

    surcharge = _GSIB_BUCKET_SURCHARGES[bucket]
    logger.debug("G-SIB surcharge for bucket %d: %.4f", bucket, surcharge)
    return surcharge


def dsib_surcharge(
    jurisdiction_config: dict[str, Any],
) -> float:
    """Return the D-SIB (Domestic Systemically Important Bank) surcharge.

    The D-SIB surcharge is set by national authorities for banks that
    are systemically important at the domestic level but may not be
    G-SIBs.

    Reference: BCBS d368 (D-SIB framework, October 2012).

    Args:
        jurisdiction_config: Parsed regulatory YAML config dict.

    Returns:
        D-SIB surcharge as a decimal. Returns 0.0 if not applicable
        or not configured for the jurisdiction.
    """
    cap_req = jurisdiction_config.get("capital_requirements", {})

    # Check for systemic_risk_buffer (EU terminology) or dsib_surcharge
    srb = cap_req.get("systemic_risk_buffer", {})
    if srb.get("enabled"):
        # The actual rate is institution-specific; return 0.0 as placeholder
        # since it is set by the NCA per institution.
        logger.debug(
            "Systemic risk buffer enabled; rate is institution-specific"
        )
        return 0.0

    dsib_pct = cap_req.get("dsib_surcharge_pct")
    if dsib_pct is not None:
        rate = float(dsib_pct)
        logger.debug("D-SIB surcharge: %.4f", rate)
        return rate

    logger.debug("D-SIB surcharge: not applicable, returning 0.0")
    return 0.0


def combined_buffer_requirement(
    cconb: float,
    ccyb: float,
    gsib: float,
    dsib: float,
) -> float:
    """Calculate the combined buffer requirement (CBR).

    The CBR is the sum of all applicable capital buffers and determines
    the CET1 capital a bank must hold above the regulatory minimum
    to avoid distribution restrictions.

    Reference: BCBS d424, RBC40.4.

    Args:
        cconb: Capital conservation buffer rate.
        ccyb: Countercyclical capital buffer rate.
        gsib: G-SIB surcharge rate.
        dsib: D-SIB surcharge rate.

    Returns:
        Combined buffer requirement as a decimal.

    Raises:
        ValidationError: If any buffer component is negative.
    """
    components = {"CConB": cconb, "CCyB": ccyb, "G-SIB": gsib, "D-SIB": dsib}
    for name, value in components.items():
        if value < 0:
            raise ValidationError(f"{name} buffer cannot be negative: {value}")

    cbr = cconb + ccyb + gsib + dsib
    logger.debug(
        "Combined buffer requirement: %.4f "
        "(CConB=%.4f, CCyB=%.4f, G-SIB=%.4f, D-SIB=%.4f)",
        cbr, cconb, ccyb, gsib, dsib,
    )
    return cbr


def minimum_capital_requirements(
    cet1_ratio: float,
    tier1_ratio: float,
    total_ratio: float,
    combined_buffer: float,
) -> dict[str, Any]:
    """Check whether capital ratios meet minimum requirements plus buffers.

    Evaluates CET1, Tier 1, and Total Capital ratios against the
    Basel III minima (4.5%, 6%, 8%) plus the combined buffer requirement.

    Reference: BCBS d424, RBC20.2-20.4 (minimums);
               BCBS d424, RBC40.4 (combined buffer).

    Args:
        cet1_ratio: Common Equity Tier 1 ratio (e.g. 0.12 for 12%).
        tier1_ratio: Tier 1 capital ratio.
        total_ratio: Total capital ratio.
        combined_buffer: Combined buffer requirement from
            :func:`combined_buffer_requirement`.

    Returns:
        Dict with keys:
            - meets_minimum: True if all minimums (excl. buffers) are met.
            - meets_buffer: True if all minimums + buffers are met.
            - cet1_target: Minimum CET1 + combined buffer.
            - tier1_target: Minimum Tier1 + combined buffer.
            - total_target: Minimum total + combined buffer.
            - cet1_surplus: CET1 ratio minus CET1 target.
            - tier1_surplus: Tier1 ratio minus Tier1 target.
            - total_surplus: Total ratio minus total target.
            - binding_constraint: Which ratio is the tightest.

    Raises:
        ValidationError: If any input ratio is negative.
    """
    if cet1_ratio < 0 or tier1_ratio < 0 or total_ratio < 0:
        raise ValidationError("Capital ratios cannot be negative")
    if combined_buffer < 0:
        raise ValidationError("Combined buffer cannot be negative")

    cet1_target = _MINIMUM_CET1_PCT + combined_buffer
    tier1_target = _MINIMUM_TIER1_PCT + combined_buffer
    total_target = _MINIMUM_TOTAL_CAPITAL_PCT + combined_buffer

    cet1_surplus = cet1_ratio - cet1_target
    tier1_surplus = tier1_ratio - tier1_target
    total_surplus = total_ratio - total_target

    meets_minimum = (
        cet1_ratio >= _MINIMUM_CET1_PCT
        and tier1_ratio >= _MINIMUM_TIER1_PCT
        and total_ratio >= _MINIMUM_TOTAL_CAPITAL_PCT
    )
    meets_buffer = (
        cet1_surplus >= 0 and tier1_surplus >= 0 and total_surplus >= 0
    )

    # Determine binding constraint (smallest surplus)
    surpluses = {
        "cet1": cet1_surplus,
        "tier1": tier1_surplus,
        "total": total_surplus,
    }
    binding_constraint = min(surpluses, key=surpluses.get)  # type: ignore[arg-type]

    result: dict[str, Any] = {
        "meets_minimum": meets_minimum,
        "meets_buffer": meets_buffer,
        "cet1_target": cet1_target,
        "tier1_target": tier1_target,
        "total_target": total_target,
        "cet1_surplus": cet1_surplus,
        "tier1_surplus": tier1_surplus,
        "total_surplus": total_surplus,
        "binding_constraint": binding_constraint,
    }

    logger.debug(
        "Capital adequacy check: meets_minimum=%s meets_buffer=%s "
        "binding=%s surplus=%.4f",
        meets_minimum, meets_buffer,
        binding_constraint, surpluses[binding_constraint],
    )
    return result


def maximum_distributable_amount(
    cet1_ratio: float,
    minimum_cet1: float,
    combined_buffer: float,
    net_income: float,
) -> dict[str, Any]:
    """Calculate the Maximum Distributable Amount (MDA).

    When a bank's CET1 ratio falls within the combined buffer zone
    (i.e. above the minimum but below minimum + buffer), distribution
    constraints apply. The buffer zone is divided into four quartiles,
    each allowing a progressively smaller fraction of earnings to be
    distributed.

    Reference: BCBS d424, RBC40.5-40.8.

    Args:
        cet1_ratio: Current CET1 ratio.
        minimum_cet1: Minimum CET1 requirement (typically 0.045).
        combined_buffer: Combined buffer requirement.
        net_income: Net income available for distribution.

    Returns:
        Dict with keys:
            - in_buffer_zone: True if CET1 is in the buffer range.
            - quartile: Which quartile (1-4) the bank falls in,
              or 0 if above buffer / below minimum.
            - max_payout_ratio: Fraction of earnings distributable.
            - mda: Maximum distributable amount in currency units.
            - buffer_used_pct: Percentage of buffer consumed.

    Raises:
        ValidationError: If combined_buffer is not positive.
    """
    if combined_buffer <= 0:
        raise ValidationError(
            "Combined buffer must be positive for MDA calculation"
        )

    buffer_floor = minimum_cet1
    buffer_ceiling = minimum_cet1 + combined_buffer

    # Below minimum — no distributions permitted, not in buffer zone
    if cet1_ratio < buffer_floor:
        logger.warning(
            "CET1 ratio %.4f below minimum %.4f — "
            "distributions fully restricted",
            cet1_ratio, buffer_floor,
        )
        return {
            "in_buffer_zone": False,
            "quartile": 0,
            "max_payout_ratio": 0.0,
            "mda": 0.0,
            "buffer_used_pct": 1.0,
        }

    # Above buffer ceiling — no restrictions
    if cet1_ratio >= buffer_ceiling:
        logger.debug("CET1 ratio %.4f above buffer ceiling %.4f — no MDA restriction",
                      cet1_ratio, buffer_ceiling)
        return {
            "in_buffer_zone": False,
            "quartile": 0,
            "max_payout_ratio": 1.0,
            "mda": max(net_income, 0.0),
            "buffer_used_pct": 0.0,
        }

    # Within buffer zone — determine quartile
    cet1_above_min = cet1_ratio - buffer_floor
    position_in_buffer = cet1_above_min / combined_buffer
    buffer_used_pct = 1.0 - position_in_buffer

    # Quartile 1: 75-100% of buffer available (top quartile)
    # Quartile 2: 50-75%
    # Quartile 3: 25-50%
    # Quartile 4: 0-25% (bottom quartile, most constrained)
    if position_in_buffer > 0.75:
        quartile = 1
    elif position_in_buffer > 0.50:
        quartile = 2
    elif position_in_buffer > 0.25:
        quartile = 3
    else:
        quartile = 4

    max_payout_ratio = _MDA_QUARTILES[quartile - 1]
    mda = max(net_income * max_payout_ratio, 0.0)

    logger.info(
        "MDA constraint: CET1=%.4f quartile=%d payout_ratio=%.1f%% mda=%.2f",
        cet1_ratio, quartile, max_payout_ratio * 100, mda,
    )
    return {
        "in_buffer_zone": True,
        "quartile": quartile,
        "max_payout_ratio": max_payout_ratio,
        "mda": mda,
        "buffer_used_pct": buffer_used_pct,
    }


def capital_adequacy_summary(
    cet1_ratio: float,
    tier1_ratio: float,
    total_ratio: float,
    jurisdiction_config: dict[str, Any],
    gsib_bucket: int | None = None,
    ccyb_rate: float = 0.0,
) -> dict[str, Any]:
    """Produce a full capital adequacy assessment.

    Combines all buffer calculations, minimum checks, and MDA
    determination into a single summary dict suitable for reporting.

    Reference: BCBS d424, RBC20 (minimums) and RBC40 (buffers).

    Args:
        cet1_ratio: CET1 capital ratio.
        tier1_ratio: Tier 1 capital ratio.
        total_ratio: Total capital ratio.
        jurisdiction_config: Parsed regulatory YAML config dict.
        gsib_bucket: G-SIB bucket (1-5), or None / 0 if not a G-SIB.
        ccyb_rate: Countercyclical buffer rate. Defaults to 0.0.

    Returns:
        Dict with keys:
            - jurisdiction: Jurisdiction name from config.
            - buffers: Dict of individual buffer components.
            - combined_buffer: Total combined buffer.
            - requirements: Output of :func:`minimum_capital_requirements`.
            - cet1_ratio / tier1_ratio / total_ratio: Input ratios echoed.
    """
    jurisdiction_name = jurisdiction_config.get("jurisdiction", "unknown")

    cconb = capital_conservation_buffer(jurisdiction_config)
    ccyb = countercyclical_buffer(jurisdiction_config, ccyb_rate or None)
    gsib = gsib_surcharge(gsib_bucket or 0)
    dsib_rate = dsib_surcharge(jurisdiction_config)

    cbr = combined_buffer_requirement(cconb, ccyb, gsib, dsib_rate)
    requirements = minimum_capital_requirements(
        cet1_ratio, tier1_ratio, total_ratio, cbr,
    )

    buffers = {
        "capital_conservation_buffer": cconb,
        "countercyclical_buffer": ccyb,
        "gsib_surcharge": gsib,
        "dsib_surcharge": dsib_rate,
    }

    summary: dict[str, Any] = {
        "jurisdiction": jurisdiction_name,
        "cet1_ratio": cet1_ratio,
        "tier1_ratio": tier1_ratio,
        "total_ratio": total_ratio,
        "buffers": buffers,
        "combined_buffer": cbr,
        "requirements": requirements,
    }

    logger.info(
        "Capital adequacy summary: jurisdiction=%s CET1=%.2f%% "
        "combined_buffer=%.2f%% meets_buffer=%s",
        jurisdiction_name,
        cet1_ratio * 100,
        cbr * 100,
        requirements["meets_buffer"],
    )
    return summary
