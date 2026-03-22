"""Tests for jurisdiction-configurable risk weight tables — BCBS d424, CRE20."""

from pathlib import Path

import pytest

from creditriskengine.core.types import (
    CreditQualityStep,
    SAExposureClass,
)
from creditriskengine.rwa.standardized.risk_weights import (
    RiskWeightRegistry,
    load_risk_weight_registry,
)


def _make_config(
    sa_risk_weights: dict | None = None,
) -> dict:
    """Build a minimal config dict for testing."""
    if sa_risk_weights is None:
        sa_risk_weights = {
            "sovereign": {
                "cqs_1": 0.0,
                "cqs_2": 20.0,
                "cqs_3": 50.0,
                "cqs_4": 100.0,
                "cqs_5": 100.0,
                "cqs_6": 150.0,
                "unrated": 100.0,
            },
            "bank_ecra": {
                "cqs_1": 20.0,
                "cqs_2": 30.0,
                "cqs_3": 50.0,
                "cqs_4": 100.0,
                "cqs_5": 100.0,
                "cqs_6": 150.0,
                "unrated": 50.0,
            },
            "corporate": {
                "cqs_1": 20.0,
                "cqs_2": 50.0,
                "cqs_3": 75.0,
                "cqs_4": 100.0,
                "cqs_5": 150.0,
                "cqs_6": 150.0,
                "unrated": 100.0,
            },
            "rre_whole_loan": [
                {"ltv_upper": 0.50, "rw": 20.0},
                {"ltv_upper": 0.60, "rw": 25.0},
                {"ltv_upper": 0.80, "rw": 30.0},
                {"ltv_upper": 0.90, "rw": 40.0},
                {"ltv_upper": 1.00, "rw": 50.0},
            ],
            "rre_cashflow": [
                {"ltv_upper": 0.50, "rw": 30.0},
                {"ltv_upper": 0.60, "rw": 35.0},
                {"ltv_upper": 0.80, "rw": 45.0},
                {"ltv_upper": 1.00, "rw": 70.0},
            ],
            "cre_not_cashflow": [
                {"ltv_upper": 0.60, "rw": 60.0},
                {"ltv_upper": 0.80, "rw": 80.0},
            ],
            "cre_ipre": [
                {"ltv_upper": 0.60, "rw": 70.0},
                {"ltv_upper": 0.80, "rw": 90.0},
            ],
        }
    return {"sa_risk_weights": sa_risk_weights}


class TestRiskWeightRegistryInit:
    """Test registry construction from config."""

    def test_empty_config(self) -> None:
        reg = RiskWeightRegistry({})
        # No tables loaded; querying should raise
        with pytest.raises(KeyError):
            reg.get_risk_weight(SAExposureClass.SOVEREIGN, CreditQualityStep.CQS_1)

    def test_no_sa_risk_weights_key(self) -> None:
        reg = RiskWeightRegistry({"other_key": "value"})
        with pytest.raises(KeyError):
            reg.get_risk_weight(SAExposureClass.SOVEREIGN, CreditQualityStep.CQS_1)

    def test_loads_cqs_tables(self) -> None:
        reg = RiskWeightRegistry(_make_config())
        rw = reg.get_risk_weight(SAExposureClass.SOVEREIGN, CreditQualityStep.CQS_1)
        assert rw == pytest.approx(0.0)

    def test_loads_ltv_tables(self) -> None:
        reg = RiskWeightRegistry(_make_config())
        rw = reg.get_rre_risk_weight(0.45)
        assert rw == pytest.approx(20.0)

    def test_unrecognised_cqs_key_skipped(self) -> None:
        cfg = _make_config()
        cfg["sa_risk_weights"]["sovereign"]["bogus_key"] = 999.0
        reg = RiskWeightRegistry(cfg)
        # Should still work, just skip the unrecognised key
        rw = reg.get_risk_weight(SAExposureClass.SOVEREIGN, CreditQualityStep.CQS_1)
        assert rw == pytest.approx(0.0)


class TestGetRiskWeight:
    """Test get_risk_weight for CQS-keyed classes."""

    def test_sovereign_cqs_2(self) -> None:
        reg = RiskWeightRegistry(_make_config())
        rw = reg.get_risk_weight(SAExposureClass.SOVEREIGN, CreditQualityStep.CQS_2)
        assert rw == pytest.approx(20.0)

    def test_bank_uses_ecra_table(self) -> None:
        reg = RiskWeightRegistry(_make_config())
        rw = reg.get_risk_weight(SAExposureClass.BANK, CreditQualityStep.CQS_1)
        assert rw == pytest.approx(20.0)

    def test_securities_firm_uses_ecra_table(self) -> None:
        reg = RiskWeightRegistry(_make_config())
        rw = reg.get_risk_weight(
            SAExposureClass.SECURITIES_FIRM, CreditQualityStep.CQS_3
        )
        assert rw == pytest.approx(50.0)

    def test_corporate_cqs_5(self) -> None:
        reg = RiskWeightRegistry(_make_config())
        rw = reg.get_risk_weight(SAExposureClass.CORPORATE, CreditQualityStep.CQS_5)
        assert rw == pytest.approx(150.0)

    def test_corporate_sme_uses_corporate_table(self) -> None:
        reg = RiskWeightRegistry(_make_config())
        rw = reg.get_risk_weight(
            SAExposureClass.CORPORATE_SME, CreditQualityStep.CQS_3
        )
        assert rw == pytest.approx(75.0)

    def test_unrated_defaults_to_unrated_cqs(self) -> None:
        reg = RiskWeightRegistry(_make_config())
        rw = reg.get_risk_weight(SAExposureClass.SOVEREIGN, cqs=None)
        assert rw == pytest.approx(100.0)

    def test_missing_cqs_returns_100(self) -> None:
        # Config has no CQS_4 explicitly missing (but we do have it)
        # Let's create a config without CQS_4
        cfg = _make_config()
        del cfg["sa_risk_weights"]["sovereign"]["cqs_4"]
        reg = RiskWeightRegistry(cfg)
        rw = reg.get_risk_weight(SAExposureClass.SOVEREIGN, CreditQualityStep.CQS_4)
        assert rw == pytest.approx(100.0)

    def test_unknown_exposure_class_raises(self) -> None:
        reg = RiskWeightRegistry(_make_config())
        with pytest.raises(KeyError, match="No risk weight table"):
            reg.get_risk_weight(SAExposureClass.EQUITY, CreditQualityStep.CQS_1)


class TestResidentialMortgage:
    """Test RRE risk weight lookups."""

    def test_rre_requires_ltv(self) -> None:
        reg = RiskWeightRegistry(_make_config())
        with pytest.raises(ValueError, match="LTV required"):
            reg.get_risk_weight(SAExposureClass.RESIDENTIAL_MORTGAGE)

    def test_rre_whole_loan_low_ltv(self) -> None:
        reg = RiskWeightRegistry(_make_config())
        rw = reg.get_risk_weight(
            SAExposureClass.RESIDENTIAL_MORTGAGE, ltv=0.45
        )
        assert rw == pytest.approx(20.0)

    def test_rre_whole_loan_mid_ltv(self) -> None:
        reg = RiskWeightRegistry(_make_config())
        rw = reg.get_risk_weight(
            SAExposureClass.RESIDENTIAL_MORTGAGE, ltv=0.75
        )
        assert rw == pytest.approx(30.0)

    def test_rre_cashflow_dependent(self) -> None:
        reg = RiskWeightRegistry(_make_config())
        rw = reg.get_risk_weight(
            SAExposureClass.RESIDENTIAL_MORTGAGE,
            ltv=0.45,
            cashflow_dependent=True,
        )
        assert rw == pytest.approx(30.0)

    def test_rre_beyond_last_bucket(self) -> None:
        reg = RiskWeightRegistry(_make_config())
        rw = reg.get_rre_risk_weight(1.5)
        assert rw == pytest.approx(50.0)  # last bucket's RW


class TestCommercialRealEstate:
    """Test CRE risk weight lookups."""

    def test_cre_requires_ltv(self) -> None:
        reg = RiskWeightRegistry(_make_config())
        with pytest.raises(ValueError, match="LTV required"):
            reg.get_risk_weight(SAExposureClass.COMMERCIAL_REAL_ESTATE)

    def test_cre_not_cashflow_low_ltv(self) -> None:
        reg = RiskWeightRegistry(_make_config())
        rw = reg.get_risk_weight(
            SAExposureClass.COMMERCIAL_REAL_ESTATE, ltv=0.50
        )
        assert rw == pytest.approx(60.0)

    def test_cre_ipre(self) -> None:
        reg = RiskWeightRegistry(_make_config())
        rw = reg.get_risk_weight(
            SAExposureClass.COMMERCIAL_REAL_ESTATE,
            ltv=0.50,
            cashflow_dependent=True,
        )
        assert rw == pytest.approx(70.0)


class TestLookupLtv:
    """Test internal _lookup_ltv method edge cases."""

    def test_missing_ltv_table_raises(self) -> None:
        reg = RiskWeightRegistry(_make_config())
        with pytest.raises(KeyError, match="LTV table.*not found"):
            reg._lookup_ltv("nonexistent_table", 0.5)


class TestLoadRiskWeightRegistry:
    """Test file-based loading."""

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        from creditriskengine.core.types import Jurisdiction

        with pytest.raises(FileNotFoundError, match="Jurisdiction config not found"):
            load_risk_weight_registry(Jurisdiction.BCBS, config_dir=tmp_path)

    def test_loads_from_yaml(self, tmp_path: Path) -> None:
        import yaml

        from creditriskengine.core.types import Jurisdiction

        config = _make_config()
        config_path = tmp_path / "bcbs.yaml"
        config_path.write_text(yaml.dump(config))

        reg = load_risk_weight_registry(Jurisdiction.BCBS, config_dir=tmp_path)
        rw = reg.get_risk_weight(SAExposureClass.SOVEREIGN, CreditQualityStep.CQS_1)
        assert rw == pytest.approx(0.0)
