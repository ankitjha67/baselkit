"""US FR Y-14M/Q (Federal Reserve Board) reporting schedules.

FR Y-14M (Monthly): Detailed loan-level data on residential mortgage,
home equity, and credit card portfolios.

FR Y-14Q (Quarterly): Summary schedules including:
- Schedule H.1: Wholesale Credit Risk -- Corporate loan data
- Schedule H.2: Wholesale Credit Risk -- Commercial real estate
- Schedule A: Summary -- Capital components
- Schedule B: Losses -- Projected credit losses

References:
    - Federal Reserve FR Y-14M instructions (current as of 2024)
    - Federal Reserve FR Y-14Q instructions
    - 12 CFR Part 252 (Enhanced Prudential Standards)
"""

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Standard NAICS sector groupings for wholesale reporting
NAICS_SECTOR_GROUPS: dict[str, str] = {
    "11": "Agriculture, Forestry, Fishing and Hunting",
    "21": "Mining, Quarrying, and Oil and Gas Extraction",
    "22": "Utilities",
    "23": "Construction",
    "31-33": "Manufacturing",
    "42": "Wholesale Trade",
    "44-45": "Retail Trade",
    "48-49": "Transportation and Warehousing",
    "51": "Information",
    "52": "Finance and Insurance",
    "53": "Real Estate and Rental and Leasing",
    "54": "Professional, Scientific, and Technical Services",
    "55": "Management of Companies and Enterprises",
    "56": "Administrative and Support and Waste Management",
    "61": "Educational Services",
    "62": "Health Care and Social Assistance",
    "71": "Arts, Entertainment, and Recreation",
    "72": "Accommodation and Food Services",
    "81": "Other Services",
    "92": "Public Administration",
}

# CRE property types for Schedule H.2
CRE_PROPERTY_TYPES: list[str] = [
    "Multifamily",
    "Office",
    "Retail",
    "Industrial/Warehouse",
    "Hotel/Motel",
    "Healthcare",
    "Mixed Use",
    "Land",
    "Other",
]

# CCAR projection horizons (standard 9 quarters)
DEFAULT_HORIZON_QUARTERS: int = 9


@dataclass
class FRY14QScheduleH1Row:
    """Single row for FR Y-14Q Schedule H.1 -- Wholesale Corporate."""

    obligor_id: str
    obligor_name: str = ""
    industry_code: str = ""  # NAICS
    committed_exposure: float = 0.0
    utilized_exposure: float = 0.0
    internal_rating: str = ""
    pd: float = 0.0
    lgd: float = 0.0
    ead: float = 0.0
    maturity_years: float = 0.0
    risk_weight_pct: float = 0.0
    expected_loss: float = 0.0
    facility_type: str = ""


@dataclass
class FRY14QScheduleH2Row:
    """Single row for FR Y-14Q Schedule H.2 -- Commercial Real Estate."""

    loan_id: str
    property_type: str = ""
    property_location: str = ""  # state or MSA
    committed_exposure: float = 0.0
    utilized_exposure: float = 0.0
    appraised_value: float = 0.0
    ltv_ratio: float = 0.0
    dscr: float = 0.0  # Debt service coverage ratio
    internal_rating: str = ""
    pd: float = 0.0
    lgd: float = 0.0
    ead: float = 0.0
    expected_loss: float = 0.0
    maturity_years: float = 0.0


@dataclass
class FRY14LossProjectionRow:
    """Single row in projected loss schedule."""

    quarter_label: str  # e.g., "Q1 2025"
    quarter_index: int = 0
    beginning_balance: float = 0.0
    gross_charge_offs: float = 0.0
    recoveries: float = 0.0
    net_charge_offs: float = 0.0
    provision_expense: float = 0.0
    ending_balance: float = 0.0
    cumulative_loss_rate: float = 0.0


@dataclass
class FRY14Schedule:
    """Generic FR Y-14 schedule container."""

    schedule_id: str
    schedule_name: str
    reporting_date: str
    bhc_name: str  # Bank Holding Company
    rows: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _validate_exposure_fields(
    exposure: dict[str, Any],
    required_fields: list[str],
    context: str,
) -> bool:
    """Check that required fields are present in an exposure dict.

    Args:
        exposure: Exposure data dict.
        required_fields: List of required field names.
        context: Description for log messages.

    Returns:
        True if all required fields are present, False otherwise.
    """
    missing = [f for f in required_fields if f not in exposure]
    if missing:
        logger.warning(
            "%s: missing fields %s in exposure %s",
            context,
            missing,
            exposure.get("obligor_id", exposure.get("loan_id", "unknown")),
        )
        return False
    return True


def generate_schedule_h1(
    wholesale_exposures: list[dict[str, Any]],
    reporting_date: str,
    bhc_name: str = "",
) -> FRY14Schedule:
    """Generate FR Y-14Q Schedule H.1 -- Wholesale Corporate.

    Args:
        wholesale_exposures: List of dicts, each representing a wholesale
            corporate obligor.  Expected keys: ``obligor_id``,
            ``obligor_name``, ``industry_code``, ``committed_exposure``,
            ``utilized_exposure``, ``internal_rating``, ``pd``, ``lgd``,
            ``ead``, ``maturity_years``, ``facility_type``.
        reporting_date: Reporting reference date (ISO format string).
        bhc_name: Bank Holding Company name.

    Returns:
        Populated :class:`FRY14Schedule` for Schedule H.1.
    """
    logger.info(
        "Generating FR Y-14Q Schedule H.1 for %s (%d exposures)",
        reporting_date,
        len(wholesale_exposures),
    )
    rows: list[FRY14QScheduleH1Row] = []

    for exp in wholesale_exposures:
        _validate_exposure_fields(
            exp,
            ["obligor_id", "committed_exposure", "pd", "lgd"],
            "Schedule H.1",
        )
        pd_val = exp.get("pd", 0.0)
        lgd_val = exp.get("lgd", 0.0)
        ead_val = exp.get("ead", exp.get("committed_exposure", 0.0))
        el = pd_val * lgd_val * ead_val
        rw = exp.get("risk_weight_pct", 0.0)

        rows.append(
            FRY14QScheduleH1Row(
                obligor_id=exp.get("obligor_id", ""),
                obligor_name=exp.get("obligor_name", ""),
                industry_code=exp.get("industry_code", ""),
                committed_exposure=exp.get("committed_exposure", 0.0),
                utilized_exposure=exp.get("utilized_exposure", 0.0),
                internal_rating=exp.get("internal_rating", ""),
                pd=pd_val,
                lgd=lgd_val,
                ead=ead_val,
                maturity_years=exp.get("maturity_years", 0.0),
                risk_weight_pct=rw,
                expected_loss=el,
                facility_type=exp.get("facility_type", ""),
            )
        )

    total_committed = sum(r.committed_exposure for r in rows)
    total_ead = sum(r.ead for r in rows)
    total_el = sum(r.expected_loss for r in rows)

    schedule = FRY14Schedule(
        schedule_id="H.1",
        schedule_name="Wholesale Credit Risk — Corporate",
        reporting_date=reporting_date,
        bhc_name=bhc_name,
        rows=rows,
        metadata={
            "obligor_count": len(rows),
            "total_committed_exposure": total_committed,
            "total_ead": total_ead,
            "total_expected_loss": total_el,
            "exposure_weighted_pd": (
                sum(r.pd * r.ead for r in rows) / total_ead
                if total_ead > 0
                else 0.0
            ),
            "exposure_weighted_lgd": (
                sum(r.lgd * r.ead for r in rows) / total_ead
                if total_ead > 0
                else 0.0
            ),
        },
    )
    logger.info(
        "Schedule H.1: %d obligors, total EAD=%.2f, total EL=%.2f",
        len(rows),
        total_ead,
        total_el,
    )
    return schedule


def generate_schedule_h2(
    cre_exposures: list[dict[str, Any]],
    reporting_date: str,
    bhc_name: str = "",
) -> FRY14Schedule:
    """Generate FR Y-14Q Schedule H.2 -- Commercial Real Estate.

    Args:
        cre_exposures: List of dicts, each representing a CRE loan.
            Expected keys: ``loan_id``, ``property_type``,
            ``property_location``, ``committed_exposure``,
            ``utilized_exposure``, ``appraised_value``, ``ltv_ratio``,
            ``dscr``, ``internal_rating``, ``pd``, ``lgd``, ``ead``,
            ``maturity_years``.
        reporting_date: Reporting reference date (ISO format string).
        bhc_name: Bank Holding Company name.

    Returns:
        Populated :class:`FRY14Schedule` for Schedule H.2.
    """
    logger.info(
        "Generating FR Y-14Q Schedule H.2 for %s (%d exposures)",
        reporting_date,
        len(cre_exposures),
    )
    rows: list[FRY14QScheduleH2Row] = []

    for exp in cre_exposures:
        _validate_exposure_fields(
            exp,
            ["loan_id", "committed_exposure", "pd", "lgd"],
            "Schedule H.2",
        )
        pd_val = exp.get("pd", 0.0)
        lgd_val = exp.get("lgd", 0.0)
        ead_val = exp.get("ead", exp.get("committed_exposure", 0.0))
        el = pd_val * lgd_val * ead_val

        appraised = exp.get("appraised_value", 0.0)
        utilized = exp.get("utilized_exposure", 0.0)
        ltv = exp.get("ltv_ratio", 0.0)
        if ltv == 0.0 and appraised > 0:
            ltv = utilized / appraised

        rows.append(
            FRY14QScheduleH2Row(
                loan_id=exp.get("loan_id", ""),
                property_type=exp.get("property_type", ""),
                property_location=exp.get("property_location", ""),
                committed_exposure=exp.get("committed_exposure", 0.0),
                utilized_exposure=utilized,
                appraised_value=appraised,
                ltv_ratio=ltv,
                dscr=exp.get("dscr", 0.0),
                internal_rating=exp.get("internal_rating", ""),
                pd=pd_val,
                lgd=lgd_val,
                ead=ead_val,
                expected_loss=el,
                maturity_years=exp.get("maturity_years", 0.0),
            )
        )

    total_committed = sum(r.committed_exposure for r in rows)
    total_ead = sum(r.ead for r in rows)
    total_el = sum(r.expected_loss for r in rows)

    # Property type breakdown
    property_breakdown: dict[str, float] = {}
    for r in rows:
        pt = r.property_type or "Unknown"
        property_breakdown[pt] = property_breakdown.get(pt, 0.0) + r.ead

    schedule = FRY14Schedule(
        schedule_id="H.2",
        schedule_name="Wholesale Credit Risk — Commercial Real Estate",
        reporting_date=reporting_date,
        bhc_name=bhc_name,
        rows=rows,
        metadata={
            "loan_count": len(rows),
            "total_committed_exposure": total_committed,
            "total_ead": total_ead,
            "total_expected_loss": total_el,
            "avg_ltv": (
                sum(r.ltv_ratio * r.ead for r in rows) / total_ead
                if total_ead > 0
                else 0.0
            ),
            "avg_dscr": (
                sum(r.dscr * r.ead for r in rows) / total_ead
                if total_ead > 0
                else 0.0
            ),
            "property_type_breakdown": property_breakdown,
        },
    )
    logger.info(
        "Schedule H.2: %d loans, total EAD=%.2f, total EL=%.2f",
        len(rows),
        total_ead,
        total_el,
    )
    return schedule


def generate_loss_schedule(
    projected_losses: dict[str, float],
    reporting_date: str,
    bhc_name: str = "",
    horizon_quarters: int = DEFAULT_HORIZON_QUARTERS,
) -> FRY14Schedule:
    """Generate FR Y-14Q Schedule B -- Projected Losses (9 quarters for CCAR).

    Args:
        projected_losses: Mapping of quarter label (e.g., ``"Q1 2025"``) to
            projected net charge-off amount.  Must contain at least
            ``horizon_quarters`` entries, or remaining quarters will be
            zero-filled.
        reporting_date: Reporting reference date (ISO format string).
        bhc_name: Bank Holding Company name.
        horizon_quarters: Number of projection quarters (default 9 per CCAR).

    Returns:
        Populated :class:`FRY14Schedule` for Schedule B.
    """
    logger.info(
        "Generating FR Y-14Q Schedule B for %s (horizon=%d quarters)",
        reporting_date,
        horizon_quarters,
    )

    # Sort quarter labels chronologically; expect format "Q<n> <yyyy>"
    sorted_quarters = sorted(
        projected_losses.keys(),
        key=lambda q: (
            int(q.split()[-1]) if len(q.split()) == 2 else 0,
            int(q[1]) if len(q) >= 2 and q[1].isdigit() else 0,
        ),
    )

    # Pad to horizon if fewer quarters supplied
    if len(sorted_quarters) < horizon_quarters:
        logger.warning(
            "Only %d quarter(s) supplied; padding to %d with zero losses",
            len(sorted_quarters),
            horizon_quarters,
        )
        for i in range(len(sorted_quarters), horizon_quarters):
            label = f"Q{(i % 4) + 1} proj-{i}"
            sorted_quarters.append(label)
            projected_losses[label] = 0.0

    rows: list[FRY14LossProjectionRow] = []
    cumulative_loss = 0.0
    total_beginning_balance = sum(projected_losses.values()) * 10  # placeholder

    for idx, q_label in enumerate(sorted_quarters[:horizon_quarters]):
        nco = projected_losses.get(q_label, 0.0)
        cumulative_loss += nco
        beginning = total_beginning_balance - cumulative_loss + nco

        rows.append(
            FRY14LossProjectionRow(
                quarter_label=q_label,
                quarter_index=idx + 1,
                beginning_balance=beginning,
                gross_charge_offs=nco * 1.1,  # gross ~ 110% of net
                recoveries=nco * 0.1,
                net_charge_offs=nco,
                provision_expense=nco,
                ending_balance=beginning - nco,
                cumulative_loss_rate=(
                    cumulative_loss / beginning if beginning > 0 else 0.0
                ),
            )
        )

    total_nco = sum(r.net_charge_offs for r in rows)
    peak_loss_rate = max(r.cumulative_loss_rate for r in rows) if rows else 0.0

    schedule = FRY14Schedule(
        schedule_id="B",
        schedule_name="Projected Credit Losses",
        reporting_date=reporting_date,
        bhc_name=bhc_name,
        rows=rows,
        metadata={
            "horizon_quarters": horizon_quarters,
            "total_net_charge_offs": total_nco,
            "peak_cumulative_loss_rate": peak_loss_rate,
            "scenario": "baseline",
        },
    )
    logger.info(
        "Schedule B: %d quarters, total NCO=%.2f, peak loss rate=%.4f",
        len(rows),
        total_nco,
        peak_loss_rate,
    )
    return schedule


def schedule_to_dict(schedule: FRY14Schedule) -> dict[str, Any]:
    """Convert schedule to nested dict.

    Args:
        schedule: A populated :class:`FRY14Schedule`.

    Returns:
        Nested dict suitable for JSON serialization.
    """
    return {
        "schedule_id": schedule.schedule_id,
        "schedule_name": schedule.schedule_name,
        "reporting_date": schedule.reporting_date,
        "bhc_name": schedule.bhc_name,
        "metadata": schedule.metadata,
        "row_count": len(schedule.rows),
        "rows": [asdict(row) for row in schedule.rows],
    }
