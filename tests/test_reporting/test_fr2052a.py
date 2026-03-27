"""Comprehensive tests for the FR 2052a Complex Institution Liquidity
Monitoring Report module.

Tests cover:
- All enumeration types and constants
- All Pydantic schema models (13 table types)
- Product catalog completeness and lookup
- Record-level and submission-level validation
- Report generation, aggregation, and summary
- HQLA classification utilities
- Maturity bucket midpoint calculations
- Edge cases and error handling
"""

from __future__ import annotations

import pytest

from creditriskengine.reporting.fr2052a.products import (
    ALL_PRODUCTS,
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
    TABLE_SCHEMA_MAP,
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
    record_for_table,
)
from creditriskengine.reporting.fr2052a.types import (
    HQLA_LEVEL_1,
    HQLA_LEVEL_2A,
    HQLA_LEVEL_2B,
    AccountingDesignation,
    AssetCategory,
    CapacitySubProduct,
    CollateralLevel,
    CollateralSwapSubProduct,
    CounterpartyType,
    CustomerShortsSubProduct,
    DCSubProduct,
    DCSubProduct2,
    EncumbranceType,
    ExceptionalCBSubProduct,
    FirmShortsSubProduct,
    FR2052aCurrency,
    FR2052aTable,
    FXSettlement,
    InsuredType,
    LossAbsorbency,
    MaturityBucket,
    MaturityOptionality,
    ReporterCategory,
    ReserveSubProduct,
    SecuredSettlement,
    SyntheticSubProduct,
    hqla_level,
    maturity_bucket_midpoint_days,
)
from creditriskengine.reporting.fr2052a.validation import (
    FR2052aValidationResult,
    validate_record,
    validate_submission,
)

# =====================================================================
# Helper fixtures
# =====================================================================


def _base_kwargs(
    table: FR2052aTable = FR2052aTable.INFLOWS_ASSETS,
    product_id: int = 1,
) -> dict:
    """Common kwargs for constructing any FR2052aRecord."""
    return {
        "reporting_entity": "TestBank",
        "as_of_date": "2024-03-31",
        "table": table,
        "product_id": product_id,
        "maturity_bucket": MaturityBucket.DAY_30,
        "maturity_amount": 100.0,
    }


def _make_inflow_asset(**overrides) -> InflowAssetRecord:
    kwargs = _base_kwargs(FR2052aTable.INFLOWS_ASSETS, 1)
    kwargs["collateral_class"] = AssetCategory.A_1_Q
    kwargs["market_value"] = 100.0
    kwargs.update(overrides)
    return InflowAssetRecord(**kwargs)


def _make_outflow_deposit(**overrides) -> OutflowDepositRecord:
    kwargs = _base_kwargs(FR2052aTable.OUTFLOWS_DEPOSITS, 1)
    kwargs["counterparty"] = CounterpartyType.RETAIL
    kwargs["insured"] = InsuredType.FDIC
    kwargs.update(overrides)
    return OutflowDepositRecord(**kwargs)


def _make_outflow_wholesale(**overrides) -> OutflowWholesaleRecord:
    kwargs = _base_kwargs(FR2052aTable.OUTFLOWS_WHOLESALE, 8)
    kwargs["counterparty"] = CounterpartyType.NON_FINANCIAL_CORPORATE
    kwargs.update(overrides)
    return OutflowWholesaleRecord(**kwargs)


# =====================================================================
# 1. Enum / Type Tests
# =====================================================================


class TestEnumerations:
    """Test all FR 2052a enumeration types."""

    def test_reporter_categories(self) -> None:
        assert len(ReporterCategory) == 9
        assert ReporterCategory.CATEGORY_I.value == "category_i"
        assert ReporterCategory.FBO_CATEGORY_IV.value == "fbo_category_iv"

    def test_fr2052a_tables(self) -> None:
        assert len(FR2052aTable) == 13
        assert FR2052aTable.INFLOWS_ASSETS.value == "I.A"
        assert FR2052aTable.SUPPLEMENTAL_FX.value == "S.FX"

    def test_currencies(self) -> None:
        assert len(FR2052aCurrency) == 7
        assert FR2052aCurrency.USD.value == "USD"
        assert FR2052aCurrency.JPY.value == "JPY"

    def test_counterparty_types(self) -> None:
        assert len(CounterpartyType) == 19
        assert CounterpartyType.RETAIL.value == "Retail"
        assert CounterpartyType.MUNICIPALITIES_VRDN.value == (
            "Municipalities for VRDN Structures"
        )
        assert CounterpartyType.DEBT_ISSUING_SPE.value == (
            "Debt Issuing Special Purpose Entity"
        )

    def test_asset_categories_completeness(self) -> None:
        # HQLA Level 1: 12 categories
        assert len(HQLA_LEVEL_1) == 12
        # HQLA Level 2a: 7 categories
        assert len(HQLA_LEVEL_2A) == 7
        # HQLA Level 2b: 4 categories
        assert len(HQLA_LEVEL_2B) == 4
        # Total asset categories should be > 70
        assert len(AssetCategory) > 70

    def test_hqla_level_function(self) -> None:
        assert hqla_level(AssetCategory.A_0_Q) == "1"
        assert hqla_level(AssetCategory.A_1_Q) == "1"
        assert hqla_level(AssetCategory.G_1_Q) == "2a"
        assert hqla_level(AssetCategory.CB_3_Q) == "2a"
        assert hqla_level(AssetCategory.E_1_Q) == "2b"
        assert hqla_level(AssetCategory.IG_1_Q) == "2b"
        assert hqla_level(AssetCategory.L_1) is None
        assert hqla_level(AssetCategory.Z_1) is None

    def test_maturity_buckets(self) -> None:
        assert MaturityBucket.OPEN.value == "Open"
        assert MaturityBucket.DAY_1.value == "Day 1"
        assert MaturityBucket.DAY_60.value == "Day 60"
        assert MaturityBucket.DAYS_61_67.value == "61 - 67 Days"
        assert MaturityBucket.YR_GT_5.value == ">5 Yr"
        assert MaturityBucket.PERPETUAL.value == "Perpetual"
        # 60 daily + 14 range + Open + Perpetual = 76
        assert len(MaturityBucket) == 76

    def test_maturity_bucket_midpoint(self) -> None:
        assert maturity_bucket_midpoint_days(MaturityBucket.OPEN) is None
        assert maturity_bucket_midpoint_days(MaturityBucket.PERPETUAL) is None
        assert maturity_bucket_midpoint_days(MaturityBucket.DAY_1) == 1.0
        assert maturity_bucket_midpoint_days(MaturityBucket.DAY_30) == 30.0
        assert maturity_bucket_midpoint_days(MaturityBucket.DAYS_83_90) == 86.5
        assert maturity_bucket_midpoint_days(MaturityBucket.YR_GT_5) == 2190.0

    def test_settlement_types(self) -> None:
        assert len(SecuredSettlement) == 4
        assert SecuredSettlement.FICC.value == "FICC"
        assert len(FXSettlement) == 3
        assert FXSettlement.CLS.value == "CLS"

    def test_insured_types(self) -> None:
        assert len(InsuredType) == 3
        assert InsuredType.FDIC.value == "FDIC"

    def test_encumbrance_types(self) -> None:
        assert len(EncumbranceType) == 7
        assert EncumbranceType.SECURITIES_FINANCING.value == (
            "Securities Financing Transaction"
        )

    def test_collateral_levels(self) -> None:
        assert len(CollateralLevel) == 4
        assert CollateralLevel.FULLY_COLLATERALIZED.value == (
            "Fully Collateralized"
        )

    def test_accounting_designations(self) -> None:
        assert len(AccountingDesignation) == 4
        assert AccountingDesignation.AFS.value == "Available-for-Sale"

    def test_loss_absorbency(self) -> None:
        assert len(LossAbsorbency) == 2
        assert LossAbsorbency.TLAC.value == "TLAC"

    def test_maturity_optionality(self) -> None:
        assert len(MaturityOptionality) == 5
        assert MaturityOptionality.EVERGREEN.value == "Evergreen"

    def test_sub_product_enums(self) -> None:
        assert len(CapacitySubProduct) == 10
        assert CapacitySubProduct.FHLB.value == "FHLB"
        assert len(ReserveSubProduct) == 9
        assert len(CollateralSwapSubProduct) == 10
        assert len(ExceptionalCBSubProduct) == 9
        assert len(CustomerShortsSubProduct) == 2
        assert len(FirmShortsSubProduct) == 4
        assert len(SyntheticSubProduct) == 9
        assert len(DCSubProduct) == 5
        assert len(DCSubProduct2) == 5


# =====================================================================
# 2. Schema / Pydantic Model Tests
# =====================================================================


class TestSchemas:
    """Test all Pydantic schema models."""

    def test_base_record(self) -> None:
        rec = FR2052aRecord(
            reporting_entity="TestBank",
            as_of_date="2024-03-31",
            table=FR2052aTable.INFLOWS_ASSETS,
            product_id=1,
        )
        assert rec.reporting_entity == "TestBank"
        assert rec.converted is False
        assert rec.internal is False

    def test_inflow_asset_record(self) -> None:
        rec = _make_inflow_asset()
        assert rec.table == FR2052aTable.INFLOWS_ASSETS
        assert rec.product_id == 1
        assert rec.collateral_class == AssetCategory.A_1_Q
        assert rec.market_value == 100.0
        assert rec.treasury_control is False

    def test_inflow_asset_with_all_fields(self) -> None:
        rec = _make_inflow_asset(
            lendable_value=95.0,
            treasury_control=True,
            accounting_designation=AccountingDesignation.AFS,
            encumbrance_type=EncumbranceType.SECURITIES_FINANCING,
            effective_maturity_bucket=MaturityBucket.DAY_7,
            sub_product="FRB",
            collateral_value=100.0,
        )
        assert rec.lendable_value == 95.0
        assert rec.treasury_control is True
        assert rec.accounting_designation == AccountingDesignation.AFS

    def test_inflow_unsecured_record(self) -> None:
        kwargs = _base_kwargs(FR2052aTable.INFLOWS_UNSECURED, 1)
        kwargs["counterparty"] = CounterpartyType.BANK
        kwargs["risk_weight"] = 0.20
        rec = InflowUnsecuredRecord(**kwargs)
        assert rec.counterparty == CounterpartyType.BANK
        assert rec.risk_weight == 0.20

    def test_inflow_secured_record(self) -> None:
        kwargs = _base_kwargs(FR2052aTable.INFLOWS_SECURED, 1)
        kwargs["counterparty"] = CounterpartyType.BROKER_DEALER
        kwargs["collateral_class"] = AssetCategory.A_1_Q
        kwargs["collateral_value"] = 105.0
        kwargs["settlement"] = SecuredSettlement.TRIPARTY
        kwargs["unencumbered"] = True
        rec = InflowSecuredRecord(**kwargs)
        assert rec.settlement == SecuredSettlement.TRIPARTY
        assert rec.unencumbered is True
        assert rec.collateral_value == 105.0

    def test_inflow_other_record(self) -> None:
        kwargs = _base_kwargs(FR2052aTable.INFLOWS_OTHER, 1)
        rec = InflowOtherRecord(**kwargs)
        assert rec.table == FR2052aTable.INFLOWS_OTHER

    def test_outflow_wholesale_record(self) -> None:
        rec = _make_outflow_wholesale()
        assert rec.table == FR2052aTable.OUTFLOWS_WHOLESALE
        assert rec.product_id == 8

    def test_outflow_wholesale_with_loss_absorbency(self) -> None:
        rec = _make_outflow_wholesale(
            product_id=11,
            loss_absorbency=LossAbsorbency.TLAC,
        )
        assert rec.loss_absorbency == LossAbsorbency.TLAC

    def test_outflow_secured_record(self) -> None:
        kwargs = _base_kwargs(FR2052aTable.OUTFLOWS_SECURED, 1)
        kwargs["counterparty"] = CounterpartyType.BANK
        kwargs["collateral_class"] = AssetCategory.A_1_Q
        kwargs["collateral_value"] = 100.0
        kwargs["settlement"] = SecuredSettlement.FICC
        kwargs["rehypothecated"] = True
        rec = OutflowSecuredRecord(**kwargs)
        assert rec.rehypothecated is True

    def test_outflow_deposit_record(self) -> None:
        rec = _make_outflow_deposit()
        assert rec.insured == InsuredType.FDIC
        assert rec.trigger is False

    def test_outflow_deposit_with_trigger(self) -> None:
        rec = _make_outflow_deposit(trigger=True)
        assert rec.trigger is True

    def test_outflow_other_record(self) -> None:
        kwargs = _base_kwargs(FR2052aTable.OUTFLOWS_OTHER, 1)
        rec = OutflowOtherRecord(**kwargs)
        assert rec.table == FR2052aTable.OUTFLOWS_OTHER

    def test_supplemental_dc_record(self) -> None:
        kwargs = _base_kwargs(FR2052aTable.SUPPLEMENTAL_DC, 1)
        kwargs["counterparty"] = CounterpartyType.BANK
        kwargs["collateral_level"] = CollateralLevel.FULLY_COLLATERALIZED
        kwargs["netting_eligible"] = True
        kwargs["sub_product"] = "Rehypothecatable Collateral Unencumbered"
        kwargs["sub_product_2"] = "OTC - Bilateral"
        rec = SupplementalDCRecord(**kwargs)
        assert rec.collateral_level == CollateralLevel.FULLY_COLLATERALIZED
        assert rec.netting_eligible is True

    def test_supplemental_lrm_record(self) -> None:
        kwargs = _base_kwargs(FR2052aTable.SUPPLEMENTAL_LRM, 6)
        kwargs["market_value"] = 500.0
        rec = SupplementalLRMRecord(**kwargs)
        assert rec.market_value == 500.0

    def test_supplemental_balance_sheet_record(self) -> None:
        kwargs = _base_kwargs(FR2052aTable.SUPPLEMENTAL_BALANCE_SHEET, 1)
        kwargs["market_value"] = 200.0
        kwargs["collection_reference"] = "I.A."
        kwargs["product_reference"] = 1
        rec = SupplementalBalanceSheetRecord(**kwargs)
        assert rec.collection_reference == "I.A."

    def test_supplemental_info_record(self) -> None:
        kwargs = _base_kwargs(FR2052aTable.SUPPLEMENTAL_INFORMATIONAL, 1)
        kwargs["market_value"] = 1000.0
        rec = SupplementalInfoRecord(**kwargs)
        assert rec.market_value == 1000.0

    def test_supplemental_fx_record(self) -> None:
        kwargs = _base_kwargs(FR2052aTable.SUPPLEMENTAL_FX, 1)
        kwargs["counterparty"] = CounterpartyType.BANK
        kwargs["settlement"] = FXSettlement.CLS
        kwargs["currency_buy"] = FR2052aCurrency.EUR
        kwargs["currency_sell"] = FR2052aCurrency.USD
        kwargs["buy_amount"] = 10.0
        kwargs["sell_amount"] = 11.5
        rec = SupplementalFXRecord(**kwargs)
        assert rec.currency_buy == FR2052aCurrency.EUR
        assert rec.sell_amount == 11.5

    def test_table_schema_map_completeness(self) -> None:
        for table in FR2052aTable:
            assert table in TABLE_SCHEMA_MAP, (
                f"Missing schema for table {table.value}"
            )

    def test_record_for_table_factory(self) -> None:
        rec = record_for_table(
            FR2052aTable.INFLOWS_ASSETS,
            reporting_entity="TestBank",
            as_of_date="2024-03-31",
            product_id=1,
            collateral_class=AssetCategory.A_1_Q,
            market_value=50.0,
        )
        assert isinstance(rec, InflowAssetRecord)
        assert rec.market_value == 50.0

    def test_record_forbids_extra_fields(self) -> None:
        with pytest.raises(ValueError):
            FR2052aRecord(
                reporting_entity="Test",
                as_of_date="2024-01-01",
                table=FR2052aTable.INFLOWS_ASSETS,
                product_id=1,
                unknown_field="oops",
            )

    def test_negative_risk_weight_rejected(self) -> None:
        with pytest.raises(ValueError):
            kwargs = _base_kwargs(FR2052aTable.INFLOWS_UNSECURED, 1)
            kwargs["risk_weight"] = -0.5
            InflowUnsecuredRecord(**kwargs)


# =====================================================================
# 3. Product Catalog Tests
# =====================================================================


class TestProducts:
    """Test product definitions and catalog."""

    def test_product_catalog_populated(self) -> None:
        assert len(ALL_PRODUCTS) > 100

    def test_product_code_property(self) -> None:
        p = get_product("I.A.1")
        assert p.code == "I.A.1"
        assert p.name == "Unencumbered Assets"

    def test_get_product_all_tables(self) -> None:
        """Verify at least one product exists per table."""
        for table in FR2052aTable:
            products = get_products_for_table(table)
            assert len(products) > 0, (
                f"No products found for table {table.value}"
            )

    def test_get_product_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown FR 2052a product"):
            get_product("X.Y.99")

    def test_inflow_asset_products(self) -> None:
        products = get_products_for_table(FR2052aTable.INFLOWS_ASSETS)
        assert len(products) == 7
        pids = [p.product_id for p in products]
        assert pids == [1, 2, 3, 4, 5, 6, 7]

    def test_inflow_unsecured_products(self) -> None:
        products = get_products_for_table(FR2052aTable.INFLOWS_UNSECURED)
        assert len(products) == 8

    def test_inflow_secured_products(self) -> None:
        products = get_products_for_table(FR2052aTable.INFLOWS_SECURED)
        assert len(products) == 10

    def test_inflow_other_products(self) -> None:
        products = get_products_for_table(FR2052aTable.INFLOWS_OTHER)
        assert len(products) == 9

    def test_outflow_wholesale_products(self) -> None:
        products = get_products_for_table(FR2052aTable.OUTFLOWS_WHOLESALE)
        assert len(products) == 19

    def test_outflow_secured_products(self) -> None:
        products = get_products_for_table(FR2052aTable.OUTFLOWS_SECURED)
        assert len(products) == 11

    def test_outflow_deposit_products(self) -> None:
        products = get_products_for_table(FR2052aTable.OUTFLOWS_DEPOSITS)
        assert len(products) == 15

    def test_outflow_other_products(self) -> None:
        products = get_products_for_table(FR2052aTable.OUTFLOWS_OTHER)
        assert len(products) == 22

    def test_supplemental_dc_products(self) -> None:
        products = get_products_for_table(FR2052aTable.SUPPLEMENTAL_DC)
        assert len(products) == 21

    def test_supplemental_lrm_products(self) -> None:
        products = get_products_for_table(FR2052aTable.SUPPLEMENTAL_LRM)
        assert len(products) == 10

    def test_supplemental_balance_sheet_products(self) -> None:
        products = get_products_for_table(
            FR2052aTable.SUPPLEMENTAL_BALANCE_SHEET
        )
        assert len(products) == 6

    def test_supplemental_info_products(self) -> None:
        products = get_products_for_table(
            FR2052aTable.SUPPLEMENTAL_INFORMATIONAL
        )
        assert len(products) == 6

    def test_supplemental_fx_products(self) -> None:
        products = get_products_for_table(FR2052aTable.SUPPLEMENTAL_FX)
        assert len(products) == 3

    def test_capacity_sub_products(self) -> None:
        p = get_product("I.A.2")
        assert "FRB" in p.sub_products
        assert "FHLB" in p.sub_products
        assert len(p.sub_products) == 10

    def test_collateral_swap_sub_products(self) -> None:
        p_inflow = get_product("I.S.4")
        assert "Level 1 Pledged" in p_inflow.sub_products
        p_outflow = get_product("O.S.4")
        assert "Level 1 Received" in p_outflow.sub_products

    def test_forward_start_exclusions(self) -> None:
        p = get_product("I.A.1")
        assert p.forward_start_excluded is True
        p2 = get_product("I.S.1")
        assert p2.forward_start_excluded is False

    def test_collateral_required_flags(self) -> None:
        p = get_product("I.A.1")
        assert p.collateral_required is True
        p2 = get_product("I.U.1")
        assert p2.collateral_required is False

    def test_counterparty_required_flags(self) -> None:
        p = get_product("I.A.1")
        assert p.counterparty_required is False
        p2 = get_product("O.O.4")
        assert p2.counterparty_required is True


# =====================================================================
# 4. Validation Tests
# =====================================================================


class TestValidation:
    """Test record-level and submission-level validation."""

    def test_valid_inflow_asset(self) -> None:
        rec = _make_inflow_asset()
        result = validate_record(rec)
        assert result.is_valid
        assert not result.errors

    def test_missing_maturity_bucket(self) -> None:
        rec = _make_inflow_asset(maturity_bucket=None)
        result = validate_record(rec)
        assert not result.is_valid
        assert any("Maturity bucket" in e for e in result.errors)

    def test_missing_collateral_class(self) -> None:
        rec = _make_inflow_asset(collateral_class=None)
        result = validate_record(rec)
        assert not result.is_valid
        assert any("Collateral class" in e for e in result.errors)

    def test_internal_without_counterparty(self) -> None:
        rec = _make_inflow_asset(internal=True)
        result = validate_record(rec)
        assert not result.is_valid
        assert any("internal_counterparty" in e for e in result.errors)

    def test_internal_with_counterparty(self) -> None:
        rec = _make_inflow_asset(
            internal=True,
            internal_counterparty="SubBank",
        )
        result = validate_record(rec)
        assert result.is_valid

    def test_internal_counterparty_without_flag(self) -> None:
        rec = _make_inflow_asset(
            internal=False,
            internal_counterparty="SubBank",
        )
        result = validate_record(rec)
        assert result.is_valid  # Warning only
        assert any("internal flag" in w.lower() for w in result.warnings)

    def test_forward_start_exclusion_warning(self) -> None:
        rec = _make_inflow_asset(
            forward_start_bucket=MaturityBucket.DAY_5,
            forward_start_amount=10.0,
        )
        result = validate_record(rec)
        assert any("Forward start" in w for w in result.warnings)

    def test_below_threshold_warning(self) -> None:
        rec = _make_inflow_asset(maturity_amount=0.005)
        result = validate_record(rec)
        assert any("threshold" in w for w in result.warnings)

    def test_zero_amount_no_warning(self) -> None:
        rec = _make_inflow_asset(maturity_amount=0.0)
        result = validate_record(rec)
        assert not any("threshold" in w for w in result.warnings)

    def test_unknown_product(self) -> None:
        kwargs = _base_kwargs()
        kwargs["product_id"] = 99
        rec = InflowAssetRecord(**kwargs)
        result = validate_record(rec)
        assert not result.is_valid
        assert any("Unknown product" in e for e in result.errors)

    def test_invalid_sub_product(self) -> None:
        kwargs = _base_kwargs(FR2052aTable.INFLOWS_ASSETS, 2)
        kwargs["collateral_class"] = AssetCategory.A_1_Q
        kwargs["market_value"] = 100.0
        kwargs["sub_product"] = "InvalidCentralBank"
        rec = InflowAssetRecord(**kwargs)
        result = validate_record(rec)
        assert not result.is_valid
        assert any("Sub-product" in e for e in result.errors)

    def test_valid_sub_product(self) -> None:
        kwargs = _base_kwargs(FR2052aTable.INFLOWS_ASSETS, 2)
        kwargs["collateral_class"] = AssetCategory.A_1_Q
        kwargs["market_value"] = 100.0
        kwargs["sub_product"] = "FRB"
        rec = InflowAssetRecord(**kwargs)
        result = validate_record(rec)
        assert result.is_valid

    def test_negative_collateral_value(self) -> None:
        rec = _make_inflow_asset(collateral_value=-5.0)
        result = validate_record(rec)
        assert not result.is_valid
        assert any("negative" in e.lower() for e in result.errors)

    # --- Submission-level validation ---

    def test_valid_submission(self) -> None:
        records = [
            _make_inflow_asset(),
            _make_outflow_deposit(),
            _make_outflow_wholesale(),
        ]
        result = validate_submission(records)
        assert result.is_valid

    def test_empty_submission(self) -> None:
        result = validate_submission([])
        assert not result.is_valid
        assert any("no records" in e.lower() for e in result.errors)

    def test_inconsistent_dates(self) -> None:
        r1 = _make_inflow_asset(as_of_date="2024-03-31")
        r2 = _make_outflow_deposit(as_of_date="2024-04-01")
        result = validate_submission([r1, r2])
        assert not result.is_valid
        assert any("as-of date" in e.lower() for e in result.errors)

    def test_multiple_entities_warning(self) -> None:
        r1 = _make_inflow_asset(reporting_entity="BankA")
        r2 = _make_outflow_deposit(reporting_entity="BankB")
        result = validate_submission([r1, r2])
        assert any("entities" in w.lower() for w in result.warnings)

    def test_entity_mismatch_check(self) -> None:
        r1 = _make_inflow_asset(reporting_entity="WrongBank")
        result = validate_submission(
            [r1], reporting_entity="TestBank"
        )
        assert not result.is_valid

    def test_date_mismatch_check(self) -> None:
        r1 = _make_inflow_asset(as_of_date="2024-03-30")
        result = validate_submission(
            [r1], as_of_date="2024-03-31"
        )
        assert not result.is_valid

    def test_missing_flow_tables_warning(self) -> None:
        records = [_make_inflow_asset()]
        result = validate_submission(records)
        assert any("Flow tables" in w for w in result.warnings)

    def test_invalid_date_format(self) -> None:
        r1 = _make_inflow_asset(as_of_date="2024/03/31")
        result = validate_submission([r1])
        assert not result.is_valid
        assert any("date format" in e.lower() for e in result.errors)

    def test_validation_result_merge(self) -> None:
        r1 = FR2052aValidationResult()
        r1.add_error("Error 1")
        r2 = FR2052aValidationResult()
        r2.add_warning("Warning 1")
        r1.merge(r2)
        assert not r1.is_valid
        assert len(r1.errors) == 1
        assert len(r1.warnings) == 1


# =====================================================================
# 5. Report Generation Tests
# =====================================================================


class TestReportGeneration:
    """Test report building, aggregation, and summary."""

    def _build_test_submission(self) -> FR2052aSubmission:
        records = [
            _make_inflow_asset(maturity_amount=500.0),
            _make_inflow_asset(
                product_id=3,
                collateral_class=AssetCategory.A_0_Q,
                maturity_amount=200.0,
                sub_product="FRB",
            ),
            _make_outflow_deposit(maturity_amount=300.0),
            _make_outflow_wholesale(maturity_amount=100.0),
        ]
        return build_submission(
            reporting_entity="TestBank",
            as_of_date="2024-03-31",
            reporter_category=ReporterCategory.CATEGORY_I,
            records=records,
        )

    def test_build_submission(self) -> None:
        sub = self._build_test_submission()
        assert sub.reporting_entity == "TestBank"
        assert sub.record_count == 4
        assert FR2052aTable.INFLOWS_ASSETS in sub.tables_covered
        assert FR2052aTable.OUTFLOWS_DEPOSITS in sub.tables_covered

    def test_records_for_table(self) -> None:
        sub = self._build_test_submission()
        ia_records = sub.records_for_table(FR2052aTable.INFLOWS_ASSETS)
        assert len(ia_records) == 2

    def test_submission_to_records(self) -> None:
        sub = self._build_test_submission()
        rows = submission_to_records(sub)
        assert len(rows) == 4
        assert all(isinstance(r, dict) for r in rows)
        assert rows[0]["reporting_entity"] == "TestBank"

    def test_aggregate_by_product(self) -> None:
        sub = self._build_test_submission()
        agg = aggregate_by_product(sub.records)
        assert "I.A.1" in agg
        assert agg["I.A.1"]["total_maturity_amount"] == 500.0
        assert agg["I.A.1"]["record_count"] == 1
        assert "I.A.3" in agg
        assert agg["I.A.3"]["total_maturity_amount"] == 200.0

    def test_generate_summary(self) -> None:
        sub = self._build_test_submission()
        summary = generate_summary(sub)
        assert summary["reporting_entity"] == "TestBank"
        assert summary["as_of_date"] == "2024-03-31"
        assert summary["total_inflows"] == 700.0
        assert summary["total_outflows"] == 400.0
        assert summary["net_liquidity_position"] == 300.0
        assert summary["record_count"] == 4

    def test_summary_30d_buckets(self) -> None:
        sub = self._build_test_submission()
        summary = generate_summary(sub)
        # All records have DAY_30 maturity bucket (within 30 days)
        assert summary["within_30d_inflows"] == 700.0
        assert summary["within_30d_outflows"] == 400.0
        assert summary["within_30d_net"] == 300.0

    def test_summary_table_breakdown(self) -> None:
        sub = self._build_test_submission()
        summary = generate_summary(sub)
        tb = summary["table_breakdown"]
        assert "I.A" in tb
        assert tb["I.A"]["record_count"] == 2
        assert tb["I.A"]["total_maturity_amount"] == 700.0

    def test_summary_product_breakdown(self) -> None:
        sub = self._build_test_submission()
        summary = generate_summary(sub)
        pb = summary["product_breakdown"]
        assert "I.A.1" in pb
        assert pb["I.A.1"]["product_name"] == "Unencumbered Assets"

    def test_empty_submission_summary(self) -> None:
        sub = FR2052aSubmission(
            reporting_entity="EmptyBank",
            as_of_date="2024-03-31",
            reporter_category=ReporterCategory.CATEGORY_IV,
        )
        summary = generate_summary(sub)
        assert summary["total_inflows"] == 0.0
        assert summary["total_outflows"] == 0.0
        assert summary["record_count"] == 0

    def test_submission_metadata(self) -> None:
        sub = self._build_test_submission()
        assert sub.metadata["record_count"] == 4
        assert "I.A" in sub.metadata["tables_covered"]


# =====================================================================
# 6. Integration / End-to-End Tests
# =====================================================================


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_workflow(self) -> None:
        """Build records, validate, build submission, generate summary."""
        # Build records across multiple tables
        records: list[FR2052aRecord] = [
            InflowAssetRecord(
                reporting_entity="MegaBank",
                as_of_date="2024-06-30",
                product_id=1,
                maturity_bucket=MaturityBucket.OPEN,
                maturity_amount=5000.0,
                collateral_class=AssetCategory.A_1_Q,
                market_value=5000.0,
                treasury_control=True,
                accounting_designation=AccountingDesignation.AFS,
            ),
            InflowAssetRecord(
                reporting_entity="MegaBank",
                as_of_date="2024-06-30",
                product_id=3,
                maturity_bucket=MaturityBucket.OPEN,
                maturity_amount=2000.0,
                collateral_class=AssetCategory.A_0_Q,
                market_value=2000.0,
                sub_product="FRB",
            ),
            InflowSecuredRecord(
                reporting_entity="MegaBank",
                as_of_date="2024-06-30",
                product_id=1,
                maturity_bucket=MaturityBucket.DAY_7,
                maturity_amount=1000.0,
                counterparty=CounterpartyType.BANK,
                collateral_class=AssetCategory.A_1_Q,
                collateral_value=1050.0,
                settlement=SecuredSettlement.TRIPARTY,
            ),
            OutflowDepositRecord(
                reporting_entity="MegaBank",
                as_of_date="2024-06-30",
                product_id=1,
                maturity_bucket=MaturityBucket.OPEN,
                maturity_amount=3000.0,
                counterparty=CounterpartyType.RETAIL,
                insured=InsuredType.FDIC,
            ),
            OutflowWholesaleRecord(
                reporting_entity="MegaBank",
                as_of_date="2024-06-30",
                product_id=8,
                maturity_bucket=MaturityBucket.DAY_30,
                maturity_amount=500.0,
                counterparty=CounterpartyType.NON_FINANCIAL_CORPORATE,
            ),
            SupplementalDCRecord(
                reporting_entity="MegaBank",
                as_of_date="2024-06-30",
                product_id=1,
                maturity_bucket=MaturityBucket.OPEN,
                maturity_amount=200.0,
                counterparty=CounterpartyType.BANK,
                collateral_level=CollateralLevel.FULLY_COLLATERALIZED,
                sub_product_2="OTC - Bilateral",
            ),
        ]

        # Validate all records
        for rec in records:
            result = validate_record(rec)
            assert result.is_valid, f"Record validation failed: {result.errors}"

        # Validate submission
        sub_result = validate_submission(
            records,
            reporting_entity="MegaBank",
            as_of_date="2024-06-30",
        )
        assert sub_result.is_valid, (
            f"Submission validation failed: {sub_result.errors}"
        )

        # Build submission
        submission = build_submission(
            reporting_entity="MegaBank",
            as_of_date="2024-06-30",
            reporter_category=ReporterCategory.CATEGORY_I,
            records=records,
        )
        assert submission.record_count == 6

        # Generate summary
        summary = generate_summary(submission)
        assert summary["total_inflows"] == 8000.0
        assert summary["total_outflows"] == 3500.0
        assert summary["net_liquidity_position"] == 4500.0
        assert summary["record_count"] == 6

        # Export
        exported = submission_to_records(submission)
        assert len(exported) == 6
        assert all("reporting_entity" in r for r in exported)

    def test_all_table_schemas_constructible(self) -> None:
        """Verify every table can create a valid minimal record."""
        for table, cls in TABLE_SCHEMA_MAP.items():
            products = get_products_for_table(table)
            assert products, f"No products for {table.value}"
            pid = products[0].product_id
            kwargs: dict[str, object] = {
                "reporting_entity": "TestBank",
                "as_of_date": "2024-01-01",
                "table": table,
                "product_id": pid,
                "maturity_bucket": MaturityBucket.DAY_1,
                "maturity_amount": 1.0,
            }
            rec = cls(**kwargs)
            assert rec.table == table
