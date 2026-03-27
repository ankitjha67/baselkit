"""FR 2052a Complex Institution Liquidity Monitoring Report.

Implements the Federal Reserve's FR 2052a reporting framework (OMB 7100-0361)
for monitoring liquidity risk at large financial institutions.

The module provides:

- **Types & Enumerations**: All FR 2052a counterparty types, product codes,
  collateral classes (asset categories), maturity buckets, settlement types,
  and other field enumerations.
- **Data Schemas**: Pydantic models for each schedule row (Inflows, Outflows,
  Supplemental) with built-in field validation.
- **Product Definitions**: Complete product catalog with metadata, required
  fields, and sub-product mappings for all 100+ FR 2052a products.
- **Validation Engine**: Cross-field validation, completeness checks, and
  regulatory consistency rules.
- **Report Generator**: Aggregation, formatting, and submission-ready output.

References:
    - FR 2052a Instructions (April 2022 revision)
    - 12 CFR Part 249 (Regulation WW -- Liquidity Risk Measurement Standards)
    - 12 CFR Part 252 (Enhanced Prudential Standards)
"""

from creditriskengine.reporting.fr2052a.products import (
    ALL_PRODUCTS,
    FR2052aProduct,
    get_product,
    get_products_for_table,
)
from creditriskengine.reporting.fr2052a.report import (
    FR2052aSubmission,
    aggregate_by_product,
    build_submission,
    generate_summary,
    submission_to_records,
)
from creditriskengine.reporting.fr2052a.schemas import (
    FR2052aRecord,
    InflowAssetRecord,
    InflowOtherRecord,
    InflowSecuredRecord,
    InflowUnsecuredRecord,
    OutflowDepositRecord,
    OutflowOtherRecord,
    OutflowSecuredRecord,
    OutflowWholesaleRecord,
    SupplementalBalanceSheetRecord,
    SupplementalDCRecord,
    SupplementalFXRecord,
    SupplementalInfoRecord,
    SupplementalLRMRecord,
)
from creditriskengine.reporting.fr2052a.types import (
    AccountingDesignation,
    AssetCategory,
    CollateralLevel,
    CounterpartyType,
    EncumbranceType,
    FR2052aCurrency,
    FR2052aTable,
    FXSettlement,
    InsuredType,
    MaturityBucket,
    MaturityOptionality,
    ReporterCategory,
    SecuredSettlement,
)
from creditriskengine.reporting.fr2052a.validation import (
    FR2052aValidationError,
    FR2052aValidationResult,
    validate_record,
    validate_submission,
)

__all__ = [
    # Types / Enums
    "AccountingDesignation",
    "AssetCategory",
    "CollateralLevel",
    "CounterpartyType",
    "EncumbranceType",
    "FR2052aCurrency",
    "FR2052aTable",
    "InsuredType",
    "MaturityBucket",
    "MaturityOptionality",
    "ReporterCategory",
    "SecuredSettlement",
    "FXSettlement",
    # Schemas
    "FR2052aRecord",
    "InflowAssetRecord",
    "InflowUnsecuredRecord",
    "InflowSecuredRecord",
    "InflowOtherRecord",
    "OutflowWholesaleRecord",
    "OutflowSecuredRecord",
    "OutflowDepositRecord",
    "OutflowOtherRecord",
    "SupplementalDCRecord",
    "SupplementalLRMRecord",
    "SupplementalBalanceSheetRecord",
    "SupplementalInfoRecord",
    "SupplementalFXRecord",
    # Products
    "FR2052aProduct",
    "get_product",
    "get_products_for_table",
    "ALL_PRODUCTS",
    # Validation
    "validate_record",
    "validate_submission",
    "FR2052aValidationError",
    "FR2052aValidationResult",
    # Report
    "FR2052aSubmission",
    "build_submission",
    "submission_to_records",
    "aggregate_by_product",
    "generate_summary",
]
