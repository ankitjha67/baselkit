"""FR 2052a report generation and aggregation.

Provides utilities to build, aggregate, and export FR 2052a submissions.
Supports both granular record-level output and summary aggregations
useful for liquidity risk dashboards and LCR/NSFR estimation.

References:
    - FR 2052a Instructions, General Instructions (pp. 9--15)
    - FR 2052a Instructions, Appendix I: Data Format, Tables, and Fields
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from creditriskengine.reporting.fr2052a.products import (
    ALL_PRODUCTS,
)
from creditriskengine.reporting.fr2052a.schemas import FR2052aRecord
from creditriskengine.reporting.fr2052a.types import (
    FR2052aTable,
    ReporterCategory,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Submission container
# ---------------------------------------------------------------------------

@dataclass
class FR2052aSubmission:
    """Container for a complete FR 2052a filing.

    Attributes:
        reporting_entity: Legal entity name.
        as_of_date: Report as-of date (ISO format).
        reporter_category: Banking organisation category.
        records: All record rows across all tables.
        metadata: Additional submission metadata.
    """

    reporting_entity: str
    as_of_date: str
    reporter_category: ReporterCategory
    records: list[FR2052aRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def record_count(self) -> int:
        """Total number of records."""
        return len(self.records)

    @property
    def tables_covered(self) -> set[FR2052aTable]:
        """Set of tables with at least one record."""
        return {r.table for r in self.records}

    def records_for_table(self, table: FR2052aTable) -> list[FR2052aRecord]:
        """Return records filtered to a specific table."""
        return [r for r in self.records if r.table == table]


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_submission(
    reporting_entity: str,
    as_of_date: str,
    reporter_category: ReporterCategory,
    records: list[FR2052aRecord],
) -> FR2052aSubmission:
    """Build an FR 2052a submission from a list of records.

    Stamps each record with the entity and date, then constructs
    the submission container.

    Args:
        reporting_entity: Legal entity name.
        as_of_date: Report as-of date (ISO format).
        reporter_category: Banking organisation category.
        records: List of validated records.

    Returns:
        A populated :class:`FR2052aSubmission`.
    """
    logger.info(
        "Building FR 2052a submission: entity=%s, date=%s, records=%d",
        reporting_entity,
        as_of_date,
        len(records),
    )
    return FR2052aSubmission(
        reporting_entity=reporting_entity,
        as_of_date=as_of_date,
        reporter_category=reporter_category,
        records=records,
        metadata={
            "record_count": len(records),
            "tables_covered": sorted(
                t.value for t in {r.table for r in records}
            ),
        },
    )


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def submission_to_records(
    submission: FR2052aSubmission,
) -> list[dict[str, Any]]:
    """Convert a submission to a list of dicts (JSON-serialisable).

    Args:
        submission: A populated FR 2052a submission.

    Returns:
        List of dicts, one per record, suitable for JSON/CSV export.
    """
    rows: list[dict[str, Any]] = []
    for record in submission.records:
        d = record.model_dump(exclude_none=True)
        # Replace enum values with their string representation
        for key, val in d.items():
            if hasattr(val, "value"):
                d[key] = val.value
        rows.append(d)
    return rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_by_product(
    records: list[FR2052aRecord],
) -> dict[str, dict[str, Any]]:
    """Aggregate maturity amounts by product code.

    Args:
        records: List of FR 2052a records.

    Returns:
        Dict keyed by product code (e.g. ``'I.A.1'``) with aggregated
        maturity_amount, record_count, and product_name.
    """
    agg: dict[str, dict[str, Any]] = {}
    for record in records:
        code = f"{record.table.value}.{record.product_id}"
        if code not in agg:
            product_def = ALL_PRODUCTS.get(code)
            agg[code] = {
                "product_code": code,
                "product_name": (
                    product_def.name if product_def else record.product_name
                ),
                "total_maturity_amount": 0.0,
                "record_count": 0,
            }
        agg[code]["total_maturity_amount"] += record.maturity_amount
        agg[code]["record_count"] += 1

    return agg


def _sum_table_amounts(
    records: list[FR2052aRecord],
    table: FR2052aTable,
) -> float:
    """Sum maturity amounts for all records in a table."""
    return sum(r.maturity_amount for r in records if r.table == table)


def generate_summary(
    submission: FR2052aSubmission,
) -> dict[str, Any]:
    """Generate a high-level summary of an FR 2052a submission.

    Produces aggregate inflow/outflow totals, table-level breakdowns,
    and product-level aggregations suitable for liquidity dashboards.

    Args:
        submission: A populated FR 2052a submission.

    Returns:
        Summary dict with keys: reporting_entity, as_of_date,
        total_inflows, total_outflows, net_liquidity_position,
        table_breakdown, product_breakdown, record_count.
    """
    records = submission.records

    # Inflow tables
    inflow_tables = {
        FR2052aTable.INFLOWS_ASSETS,
        FR2052aTable.INFLOWS_UNSECURED,
        FR2052aTable.INFLOWS_SECURED,
        FR2052aTable.INFLOWS_OTHER,
    }

    # Outflow tables
    outflow_tables = {
        FR2052aTable.OUTFLOWS_WHOLESALE,
        FR2052aTable.OUTFLOWS_SECURED,
        FR2052aTable.OUTFLOWS_DEPOSITS,
        FR2052aTable.OUTFLOWS_OTHER,
    }

    total_inflows = sum(
        r.maturity_amount for r in records if r.table in inflow_tables
    )
    total_outflows = sum(
        r.maturity_amount for r in records if r.table in outflow_tables
    )

    # Table-level breakdown
    table_breakdown: dict[str, dict[str, Any]] = {}
    for table in FR2052aTable:
        table_records = [r for r in records if r.table == table]
        if table_records:
            table_breakdown[table.value] = {
                "record_count": len(table_records),
                "total_maturity_amount": sum(
                    r.maturity_amount for r in table_records
                ),
            }

    # Product-level breakdown
    product_breakdown = aggregate_by_product(records)

    # Maturity profile (within 30 days)
    from creditriskengine.reporting.fr2052a.types import maturity_bucket_midpoint_days
    within_30d_inflows = sum(
        r.maturity_amount
        for r in records
        if r.table in inflow_tables
        and r.maturity_bucket is not None
        and (mp := maturity_bucket_midpoint_days(r.maturity_bucket)) is not None
        and mp <= 30
    )
    within_30d_outflows = sum(
        r.maturity_amount
        for r in records
        if r.table in outflow_tables
        and r.maturity_bucket is not None
        and (mp := maturity_bucket_midpoint_days(r.maturity_bucket)) is not None
        and mp <= 30
    )

    summary: dict[str, Any] = {
        "reporting_entity": submission.reporting_entity,
        "as_of_date": submission.as_of_date,
        "reporter_category": submission.reporter_category.value,
        "record_count": len(records),
        "total_inflows": total_inflows,
        "total_outflows": total_outflows,
        "net_liquidity_position": total_inflows - total_outflows,
        "within_30d_inflows": within_30d_inflows,
        "within_30d_outflows": within_30d_outflows,
        "within_30d_net": within_30d_inflows - within_30d_outflows,
        "table_breakdown": table_breakdown,
        "product_breakdown": product_breakdown,
    }

    logger.info(
        "FR 2052a summary: entity=%s, date=%s, inflows=%.2f, outflows=%.2f, "
        "net=%.2f",
        submission.reporting_entity,
        submission.as_of_date,
        total_inflows,
        total_outflows,
        total_inflows - total_outflows,
    )
    return summary
