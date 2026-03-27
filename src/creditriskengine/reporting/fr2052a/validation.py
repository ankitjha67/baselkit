"""FR 2052a validation engine.

Provides record-level and submission-level validation including:

- Required field completeness per product definition
- Cross-field consistency (e.g. internal flag requires internal_counterparty)
- Counterparty applicability checks per Appendix II-b
- Collateral class requirement checks per Appendix II-c
- Forward start exclusion checks per Appendix II-d
- Maturity amount minimum threshold (0.01 in millions = 10,000 currency units)

References:
    - FR 2052a Instructions, Appendix II-a through II-d
    - FR 2052a Instructions, Field Definitions
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from creditriskengine.reporting.fr2052a.products import (
    ALL_PRODUCTS,
    FR2052aProduct,
)
from creditriskengine.reporting.fr2052a.schemas import FR2052aRecord
from creditriskengine.reporting.fr2052a.types import FR2052aTable

logger = logging.getLogger(__name__)

# Minimum reportable amount per FR 2052a: 10,000 currency units = 0.01 millions
MIN_REPORTABLE_AMOUNT: float = 0.01


class FR2052aValidationError(Exception):
    """Raised when FR 2052a validation fails."""


@dataclass
class FR2052aValidationResult:
    """Validation result for a record or submission.

    Attributes:
        is_valid: True if no errors were found.
        errors: List of error descriptions.
        warnings: List of non-fatal warning descriptions.
    """

    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        """Add an error and mark result as invalid."""
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str) -> None:
        """Add a non-fatal warning."""
        self.warnings.append(msg)

    def merge(self, other: FR2052aValidationResult) -> None:
        """Merge another result into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        if not other.is_valid:
            self.is_valid = False


def _get_product_for_record(
    record: FR2052aRecord,
) -> FR2052aProduct | None:
    """Look up the product definition for a record."""
    code = f"{record.table.value}.{record.product_id}"
    return ALL_PRODUCTS.get(code)


def _validate_required_fields(
    record: FR2052aRecord,
    product: FR2052aProduct,
    result: FR2052aValidationResult,
) -> None:
    """Check that required fields are populated based on product rules."""
    code = product.code

    # Counterparty
    has_counterparty = getattr(record, "counterparty", None) is not None
    if product.counterparty_required and not has_counterparty:
        result.add_error(
            f"{code}: Counterparty is required but not provided."
        )

    # Collateral class
    has_collateral = getattr(record, "collateral_class", None) is not None
    if product.collateral_required and not has_collateral:
        result.add_error(
            f"{code}: Collateral class is required but not provided."
        )

    # Maturity bucket
    if record.maturity_bucket is None:
        result.add_error(f"{code}: Maturity bucket is required.")


def _validate_forward_start(
    record: FR2052aRecord,
    product: FR2052aProduct,
    result: FR2052aValidationResult,
) -> None:
    """Check forward start field exclusion rules per Appendix II-d."""
    code = product.code
    has_fs = (
        record.forward_start_bucket is not None
        or record.forward_start_amount is not None
    )

    if product.forward_start_excluded and has_fs:
        result.add_warning(
            f"{code}: Forward start fields should not be populated "
            f"for this product (Appendix II-d)."
        )


def _validate_internal_fields(
    record: FR2052aRecord,
    result: FR2052aValidationResult,
) -> None:
    """Check internal transaction field consistency."""
    if record.internal and not record.internal_counterparty:
        result.add_error(
            "Internal flag is True but internal_counterparty is not set."
        )
    if not record.internal and record.internal_counterparty:
        result.add_warning(
            "Internal counterparty is set but internal flag is False."
        )


def _validate_amount_threshold(
    record: FR2052aRecord,
    result: FR2052aValidationResult,
) -> None:
    """Check that amounts meet the minimum reporting threshold."""
    if 0 < abs(record.maturity_amount) < MIN_REPORTABLE_AMOUNT:
        result.add_warning(
            f"Maturity amount {record.maturity_amount} is below the "
            f"minimum reportable threshold ({MIN_REPORTABLE_AMOUNT} millions "
            f"= 10,000 currency units)."
        )


def _validate_collateral_value(
    record: FR2052aRecord,
    product: FR2052aProduct,
    result: FR2052aValidationResult,
) -> None:
    """Check collateral value consistency."""
    cv: object = getattr(record, "collateral_value", None)
    if isinstance(cv, (int, float)) and cv < 0:
        result.add_error(
            f"{product.code}: Collateral value cannot be negative."
        )


def _validate_sub_product(
    record: FR2052aRecord,
    product: FR2052aProduct,
    result: FR2052aValidationResult,
) -> None:
    """Check sub-product value is valid for the product."""
    sub = getattr(record, "sub_product", None)
    if sub is not None and product.sub_products and sub not in product.sub_products:
        result.add_error(
            f"{product.code}: Sub-product '{sub}' is not valid. "
            f"Allowed: {product.sub_products}"
        )


def validate_record(record: FR2052aRecord) -> FR2052aValidationResult:
    """Validate a single FR 2052a record.

    Checks required fields, cross-field consistency, and product-specific
    rules based on the FR 2052a Instructions and Appendices.

    Args:
        record: An FR 2052a record to validate.

    Returns:
        Validation result with errors and warnings.
    """
    result = FR2052aValidationResult()

    product = _get_product_for_record(record)
    if product is None:
        result.add_error(
            f"Unknown product: table={record.table.value}, "
            f"product_id={record.product_id}"
        )
        return result

    _validate_required_fields(record, product, result)
    _validate_forward_start(record, product, result)
    _validate_internal_fields(record, result)
    _validate_amount_threshold(record, result)
    _validate_collateral_value(record, product, result)
    _validate_sub_product(record, product, result)

    return result


def validate_submission(
    records: list[FR2052aRecord],
    reporting_entity: str | None = None,
    as_of_date: str | None = None,
) -> FR2052aValidationResult:
    """Validate a complete FR 2052a submission.

    Performs record-level validation on all records plus submission-level
    checks:

    - Consistent reporting entity across all records
    - Consistent as-of date across all records
    - At least one record present
    - Coverage of required tables

    Args:
        records: List of FR 2052a records forming the submission.
        reporting_entity: Expected entity name (optional cross-check).
        as_of_date: Expected as-of date (optional cross-check).

    Returns:
        Aggregated validation result.
    """
    result = FR2052aValidationResult()

    if not records:
        result.add_error("Submission contains no records.")
        return result

    # As-of date format validation (ISO YYYY-MM-DD)
    iso_date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    for record in records:
        if not iso_date_re.match(record.as_of_date):
            result.add_error(
                f"Invalid as-of date format '{record.as_of_date}'. "
                f"Expected ISO format YYYY-MM-DD."
            )
            break

    # Submission-level consistency
    entities = {r.reporting_entity for r in records}
    if len(entities) > 1:
        result.add_warning(
            f"Multiple reporting entities found: {entities}. "
            f"Ensure this is intentional (e.g. material entity reporting)."
        )

    dates = {r.as_of_date for r in records}
    if len(dates) > 1:
        result.add_error(
            f"Inconsistent as-of dates: {dates}. "
            f"All records must share the same as-of date."
        )

    if reporting_entity is not None:
        mismatched = [
            r for r in records if r.reporting_entity != reporting_entity
        ]
        if mismatched:
            result.add_error(
                f"{len(mismatched)} record(s) have a reporting entity "
                f"that does not match expected '{reporting_entity}'."
            )

    if as_of_date is not None:
        mismatched_dates = [r for r in records if r.as_of_date != as_of_date]
        if mismatched_dates:
            result.add_error(
                f"{len(mismatched_dates)} record(s) have an as-of date "
                f"that does not match expected '{as_of_date}'."
            )

    # Record-level validation
    for i, record in enumerate(records):
        rec_result = validate_record(record)
        if not rec_result.is_valid:
            for err in rec_result.errors:
                result.add_error(f"Record {i}: {err}")
        for warn in rec_result.warnings:
            result.add_warning(f"Record {i}: {warn}")

    # Table coverage check -- all 8 flow tables are expected in a
    # complete FR 2052a submission.
    tables_present = {r.table for r in records}
    flow_tables = {
        FR2052aTable.INFLOWS_ASSETS,
        FR2052aTable.INFLOWS_UNSECURED,
        FR2052aTable.INFLOWS_SECURED,
        FR2052aTable.INFLOWS_OTHER,
        FR2052aTable.OUTFLOWS_WHOLESALE,
        FR2052aTable.OUTFLOWS_SECURED,
        FR2052aTable.OUTFLOWS_DEPOSITS,
        FR2052aTable.OUTFLOWS_OTHER,
    }
    missing_flow = flow_tables - tables_present
    if missing_flow:
        result.add_warning(
            f"Flow tables missing from submission: "
            f"{sorted(t.value for t in missing_flow)}. "
            f"Review whether these are applicable."
        )

    logger.info(
        "FR 2052a validation: %d records, %d errors, %d warnings",
        len(records),
        len(result.errors),
        len(result.warnings),
    )
    return result
