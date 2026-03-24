"""Tests for creditriskengine.regulatory.schema module."""

from __future__ import annotations

import pytest

from creditriskengine.core.exceptions import ConfigurationError
from creditriskengine.regulatory.schema import (
    JURISDICTION_SCHEMA,
    sanitize_exposure_inputs,
    validate_config,
    validate_config_strict,
    validate_lgd_range,
    validate_pd_range,
    validate_risk_weight_range,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_config() -> dict:
    """Return a minimal valid jurisdiction config."""
    return {
        "jurisdiction": "UK",
        "regulator": "PRA",
        "framework": "Basel 3.1",
        "effective_date": "2026-01-01",
        "credit_risk": {
            "approaches_permitted": ["SA", "FIRB", "AIRB"],
            "standardised_approach": {"risk_weights": {}},
            "irb_approach": {
                "pd_floor_bps": 5,
                "lgd_floors": {"senior_unsecured": 0.25},
                "maturity_default_years": 2.5,
            },
        },
        "output_floor": {
            "enabled": True,
            "final_pct": 0.725,
        },
        "default_definition": {
            "dpd_threshold": 90,
            "cure_period_months": 3,
        },
    }


# ---------------------------------------------------------------------------
# JURISDICTION_SCHEMA
# ---------------------------------------------------------------------------


class TestJurisdictionSchema:
    def test_schema_is_dict(self) -> None:
        assert isinstance(JURISDICTION_SCHEMA, dict)

    def test_required_top_level_keys(self) -> None:
        assert "required_top_level" in JURISDICTION_SCHEMA
        assert "jurisdiction" in JURISDICTION_SCHEMA["required_top_level"]
        assert "regulator" in JURISDICTION_SCHEMA["required_top_level"]
        assert "framework" in JURISDICTION_SCHEMA["required_top_level"]
        assert "effective_date" in JURISDICTION_SCHEMA["required_top_level"]

    def test_credit_risk_section(self) -> None:
        cr = JURISDICTION_SCHEMA["credit_risk"]
        assert "approaches_permitted" in cr["required"]
        assert "standardised_approach" in cr["required"]
        assert "irb_approach" in cr["required"]

    def test_irb_approach_rules(self) -> None:
        irb = JURISDICTION_SCHEMA["credit_risk"]["irb_approach"]["required"]
        assert irb["pd_floor_bps"]["type"] is int
        assert irb["pd_floor_bps"]["min"] == 1
        assert irb["pd_floor_bps"]["max"] == 100
        assert irb["lgd_floors"]["type"] is dict
        assert irb["maturity_default_years"]["type"] is float

    def test_output_floor_rules(self) -> None:
        of = JURISDICTION_SCHEMA["output_floor"]["required"]
        assert of["enabled"]["type"] is bool
        assert of["final_pct"]["type"] is float
        assert of["final_pct"]["min"] == 0.0
        assert of["final_pct"]["max"] == 1.0

    def test_default_definition_rules(self) -> None:
        dd = JURISDICTION_SCHEMA["default_definition"]["required"]
        assert dd["dpd_threshold"]["type"] is int
        assert dd["cure_period_months"]["type"] is int


# ---------------------------------------------------------------------------
# validate_config — valid
# ---------------------------------------------------------------------------


class TestValidateConfigValid:
    def test_valid_config(self) -> None:
        errors = validate_config(_valid_config())
        assert errors == []

    def test_valid_config_int_maturity(self) -> None:
        """Int should be accepted where float is expected."""
        cfg = _valid_config()
        cfg["credit_risk"]["irb_approach"]["maturity_default_years"] = 3
        errors = validate_config(cfg)
        assert errors == []

    def test_valid_config_int_final_pct(self) -> None:
        """Int 0 or 1 should be accepted for final_pct."""
        cfg = _valid_config()
        cfg["output_floor"]["final_pct"] = 1
        errors = validate_config(cfg)
        assert errors == []


# ---------------------------------------------------------------------------
# validate_config — missing top-level keys
# ---------------------------------------------------------------------------


class TestValidateConfigMissingTopLevel:
    @pytest.mark.parametrize("key", [
        "jurisdiction", "regulator", "framework", "effective_date",
    ])
    def test_missing_top_level_key(self, key: str) -> None:
        cfg = _valid_config()
        del cfg[key]
        errors = validate_config(cfg)
        assert any(key in e for e in errors)

    def test_missing_all_top_level(self) -> None:
        errors = validate_config({})
        # Should report missing: jurisdiction, regulator, framework, effective_date,
        # credit_risk, output_floor, default_definition
        assert len(errors) >= 7


# ---------------------------------------------------------------------------
# validate_config — credit_risk errors
# ---------------------------------------------------------------------------


class TestValidateConfigCreditRisk:
    def test_missing_credit_risk(self) -> None:
        cfg = _valid_config()
        del cfg["credit_risk"]
        errors = validate_config(cfg)
        assert any("credit_risk" in e for e in errors)

    def test_credit_risk_not_dict(self) -> None:
        cfg = _valid_config()
        cfg["credit_risk"] = "not a dict"
        errors = validate_config(cfg)
        assert any("must be a dict" in e for e in errors)

    @pytest.mark.parametrize("key", [
        "approaches_permitted", "standardised_approach", "irb_approach",
    ])
    def test_missing_credit_risk_subkey(self, key: str) -> None:
        cfg = _valid_config()
        del cfg["credit_risk"][key]
        errors = validate_config(cfg)
        assert any(key in e for e in errors)

    def test_irb_approach_not_dict(self) -> None:
        cfg = _valid_config()
        cfg["credit_risk"]["irb_approach"] = "bad"
        errors = validate_config(cfg)
        assert any("irb_approach" in e and "dict" in e for e in errors)

    def test_missing_irb_pd_floor(self) -> None:
        cfg = _valid_config()
        del cfg["credit_risk"]["irb_approach"]["pd_floor_bps"]
        errors = validate_config(cfg)
        assert any("pd_floor_bps" in e for e in errors)

    def test_missing_irb_lgd_floors(self) -> None:
        cfg = _valid_config()
        del cfg["credit_risk"]["irb_approach"]["lgd_floors"]
        errors = validate_config(cfg)
        assert any("lgd_floors" in e for e in errors)

    def test_missing_irb_maturity(self) -> None:
        cfg = _valid_config()
        del cfg["credit_risk"]["irb_approach"]["maturity_default_years"]
        errors = validate_config(cfg)
        assert any("maturity_default_years" in e for e in errors)

    def test_pd_floor_bps_wrong_type(self) -> None:
        cfg = _valid_config()
        cfg["credit_risk"]["irb_approach"]["pd_floor_bps"] = "five"
        errors = validate_config(cfg)
        assert any("pd_floor_bps" in e and "int" in e for e in errors)

    def test_pd_floor_bps_below_min(self) -> None:
        cfg = _valid_config()
        cfg["credit_risk"]["irb_approach"]["pd_floor_bps"] = 0
        errors = validate_config(cfg)
        assert any("pd_floor_bps" in e and ">= 1" in e for e in errors)

    def test_pd_floor_bps_above_max(self) -> None:
        cfg = _valid_config()
        cfg["credit_risk"]["irb_approach"]["pd_floor_bps"] = 101
        errors = validate_config(cfg)
        assert any("pd_floor_bps" in e and "<= 100" in e for e in errors)

    def test_lgd_floors_wrong_type(self) -> None:
        cfg = _valid_config()
        cfg["credit_risk"]["irb_approach"]["lgd_floors"] = "not a dict"
        errors = validate_config(cfg)
        assert any("lgd_floors" in e and "dict" in e for e in errors)

    def test_maturity_wrong_type(self) -> None:
        cfg = _valid_config()
        cfg["credit_risk"]["irb_approach"]["maturity_default_years"] = "two"
        errors = validate_config(cfg)
        assert any("maturity_default_years" in e and "float" in e for e in errors)

    def test_maturity_below_min(self) -> None:
        cfg = _valid_config()
        cfg["credit_risk"]["irb_approach"]["maturity_default_years"] = -1.0
        errors = validate_config(cfg)
        assert any("maturity_default_years" in e and ">= 0" in e for e in errors)

    def test_maturity_above_max(self) -> None:
        cfg = _valid_config()
        cfg["credit_risk"]["irb_approach"]["maturity_default_years"] = 31.0
        errors = validate_config(cfg)
        assert any("maturity_default_years" in e and "<= 30" in e for e in errors)


# ---------------------------------------------------------------------------
# validate_config — output_floor errors
# ---------------------------------------------------------------------------


class TestValidateConfigOutputFloor:
    def test_missing_output_floor(self) -> None:
        cfg = _valid_config()
        del cfg["output_floor"]
        errors = validate_config(cfg)
        assert any("output_floor" in e for e in errors)

    def test_output_floor_not_dict(self) -> None:
        cfg = _valid_config()
        cfg["output_floor"] = 42
        errors = validate_config(cfg)
        assert any("output_floor" in e and "dict" in e for e in errors)

    def test_missing_enabled(self) -> None:
        cfg = _valid_config()
        del cfg["output_floor"]["enabled"]
        errors = validate_config(cfg)
        assert any("enabled" in e for e in errors)

    def test_missing_final_pct(self) -> None:
        cfg = _valid_config()
        del cfg["output_floor"]["final_pct"]
        errors = validate_config(cfg)
        assert any("final_pct" in e for e in errors)

    def test_enabled_wrong_type(self) -> None:
        cfg = _valid_config()
        cfg["output_floor"]["enabled"] = "yes"
        errors = validate_config(cfg)
        assert any("enabled" in e and "bool" in e for e in errors)

    def test_final_pct_wrong_type(self) -> None:
        cfg = _valid_config()
        cfg["output_floor"]["final_pct"] = "high"
        errors = validate_config(cfg)
        assert any("final_pct" in e and "float" in e for e in errors)

    def test_final_pct_below_min(self) -> None:
        cfg = _valid_config()
        cfg["output_floor"]["final_pct"] = -0.1
        errors = validate_config(cfg)
        assert any("final_pct" in e and ">= 0" in e for e in errors)

    def test_final_pct_above_max(self) -> None:
        cfg = _valid_config()
        cfg["output_floor"]["final_pct"] = 1.5
        errors = validate_config(cfg)
        assert any("final_pct" in e and "<= 1" in e for e in errors)


# ---------------------------------------------------------------------------
# validate_config — default_definition errors
# ---------------------------------------------------------------------------


class TestValidateConfigDefaultDefinition:
    def test_missing_default_definition(self) -> None:
        cfg = _valid_config()
        del cfg["default_definition"]
        errors = validate_config(cfg)
        assert any("default_definition" in e for e in errors)

    def test_default_definition_not_dict(self) -> None:
        cfg = _valid_config()
        cfg["default_definition"] = "bad"
        errors = validate_config(cfg)
        assert any("default_definition" in e and "dict" in e for e in errors)

    def test_missing_dpd_threshold(self) -> None:
        cfg = _valid_config()
        del cfg["default_definition"]["dpd_threshold"]
        errors = validate_config(cfg)
        assert any("dpd_threshold" in e for e in errors)

    def test_missing_cure_period(self) -> None:
        cfg = _valid_config()
        del cfg["default_definition"]["cure_period_months"]
        errors = validate_config(cfg)
        assert any("cure_period_months" in e for e in errors)

    def test_dpd_threshold_wrong_type(self) -> None:
        cfg = _valid_config()
        cfg["default_definition"]["dpd_threshold"] = "ninety"
        errors = validate_config(cfg)
        assert any("dpd_threshold" in e and "int" in e for e in errors)

    def test_cure_period_wrong_type(self) -> None:
        cfg = _valid_config()
        cfg["default_definition"]["cure_period_months"] = 3.5
        errors = validate_config(cfg)
        assert any("cure_period_months" in e and "int" in e for e in errors)


# ---------------------------------------------------------------------------
# validate_config_strict
# ---------------------------------------------------------------------------


class TestValidateConfigStrict:
    def test_valid_config_passes(self) -> None:
        validate_config_strict(_valid_config())  # should not raise

    def test_invalid_config_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="Configuration validation failed"):
            validate_config_strict({})

    def test_error_message_contains_details(self) -> None:
        with pytest.raises(ConfigurationError) as exc_info:
            validate_config_strict({})
        msg = str(exc_info.value)
        assert "jurisdiction" in msg
        assert "credit_risk" in msg


# ---------------------------------------------------------------------------
# validate_risk_weight_range
# ---------------------------------------------------------------------------


class TestValidateRiskWeightRange:
    def test_valid_zero(self) -> None:
        validate_risk_weight_range(0.0, "rw")

    def test_valid_mid(self) -> None:
        validate_risk_weight_range(1.5, "rw")

    def test_valid_max(self) -> None:
        validate_risk_weight_range(12.5, "rw")

    def test_valid_int(self) -> None:
        validate_risk_weight_range(1, "rw")

    def test_below_zero(self) -> None:
        with pytest.raises(ConfigurationError, match="between 0 and 12.5"):
            validate_risk_weight_range(-0.01, "rw")

    def test_above_max(self) -> None:
        with pytest.raises(ConfigurationError, match="between 0 and 12.5"):
            validate_risk_weight_range(12.51, "rw")

    def test_non_numeric(self) -> None:
        with pytest.raises(ConfigurationError, match="must be numeric"):
            validate_risk_weight_range("high", "rw")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# validate_pd_range
# ---------------------------------------------------------------------------


class TestValidatePdRange:
    def test_valid_zero(self) -> None:
        validate_pd_range(0.0, "pd")

    def test_valid_one(self) -> None:
        validate_pd_range(1.0, "pd")

    def test_valid_mid(self) -> None:
        validate_pd_range(0.05, "pd")

    def test_valid_int(self) -> None:
        validate_pd_range(0, "pd")

    def test_below_zero(self) -> None:
        with pytest.raises(ConfigurationError, match="between 0 and 1"):
            validate_pd_range(-0.001, "pd")

    def test_above_one(self) -> None:
        with pytest.raises(ConfigurationError, match="between 0 and 1"):
            validate_pd_range(1.001, "pd")

    def test_non_numeric(self) -> None:
        with pytest.raises(ConfigurationError, match="must be numeric"):
            validate_pd_range("low", "pd")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# validate_lgd_range
# ---------------------------------------------------------------------------


class TestValidateLgdRange:
    def test_valid_zero(self) -> None:
        validate_lgd_range(0.0, "lgd")

    def test_valid_one(self) -> None:
        validate_lgd_range(1.0, "lgd")

    def test_valid_mid(self) -> None:
        validate_lgd_range(0.45, "lgd")

    def test_valid_int(self) -> None:
        validate_lgd_range(0, "lgd")

    def test_below_zero(self) -> None:
        with pytest.raises(ConfigurationError, match="between 0 and 1"):
            validate_lgd_range(-0.01, "lgd")

    def test_above_one(self) -> None:
        with pytest.raises(ConfigurationError, match="between 0 and 1"):
            validate_lgd_range(1.01, "lgd")

    def test_non_numeric(self) -> None:
        with pytest.raises(ConfigurationError, match="must be numeric"):
            validate_lgd_range(None, "lgd")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# sanitize_exposure_inputs
# ---------------------------------------------------------------------------


class TestSanitizeExposureInputs:
    def test_valid_inputs_no_warnings(self) -> None:
        result = sanitize_exposure_inputs(0.05, 0.45, 1_000_000, 2.5)
        assert result["pd"] == 0.05
        assert result["lgd"] == 0.45
        assert result["ead"] == 1_000_000.0
        assert result["maturity"] == 2.5
        assert result["warnings"] == []

    def test_pd_clipped_below(self) -> None:
        result = sanitize_exposure_inputs(-0.01, 0.45, 1000, 2.5)
        assert result["pd"] == 0.0
        assert any("PD clipped" in w for w in result["warnings"])

    def test_pd_clipped_above(self) -> None:
        result = sanitize_exposure_inputs(1.5, 0.45, 1000, 2.5)
        assert result["pd"] == 1.0
        assert any("PD clipped" in w for w in result["warnings"])

    def test_lgd_clipped_below(self) -> None:
        result = sanitize_exposure_inputs(0.05, -0.1, 1000, 2.5)
        assert result["lgd"] == 0.0
        assert any("LGD clipped" in w for w in result["warnings"])

    def test_lgd_clipped_above(self) -> None:
        result = sanitize_exposure_inputs(0.05, 1.2, 1000, 2.5)
        assert result["lgd"] == 1.0
        assert any("LGD clipped" in w for w in result["warnings"])

    def test_ead_clipped_below(self) -> None:
        result = sanitize_exposure_inputs(0.05, 0.45, -500, 2.5)
        assert result["ead"] == 0.0
        assert any("EAD clipped" in w for w in result["warnings"])

    def test_maturity_clipped_below(self) -> None:
        result = sanitize_exposure_inputs(0.05, 0.45, 1000, 0.5)
        assert result["maturity"] == 1.0
        assert any("Maturity clipped" in w for w in result["warnings"])

    def test_maturity_clipped_above(self) -> None:
        result = sanitize_exposure_inputs(0.05, 0.45, 1000, 10)
        assert result["maturity"] == 5.0
        assert any("Maturity clipped" in w for w in result["warnings"])

    def test_all_clipped(self) -> None:
        result = sanitize_exposure_inputs(-1, -1, -1, 0.1)
        assert result["pd"] == 0.0
        assert result["lgd"] == 0.0
        assert result["ead"] == 0.0
        assert result["maturity"] == 1.0
        assert len(result["warnings"]) == 4

    def test_boundary_values_no_clip(self) -> None:
        result = sanitize_exposure_inputs(0, 0, 0, 1)
        assert result["pd"] == 0.0
        assert result["lgd"] == 0.0
        assert result["ead"] == 0.0
        assert result["maturity"] == 1.0
        assert result["warnings"] == []

    def test_upper_boundary_values_no_clip(self) -> None:
        result = sanitize_exposure_inputs(1, 1, 999999, 5)
        assert result["pd"] == 1.0
        assert result["lgd"] == 1.0
        assert result["maturity"] == 5.0
        assert result["warnings"] == []

    def test_pd_non_numeric_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="'pd' must be numeric"):
            sanitize_exposure_inputs("bad", 0.45, 1000, 2.5)  # type: ignore[arg-type]

    def test_lgd_non_numeric_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="'lgd' must be numeric"):
            sanitize_exposure_inputs(0.05, "bad", 1000, 2.5)  # type: ignore[arg-type]

    def test_ead_non_numeric_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="'ead' must be numeric"):
            sanitize_exposure_inputs(0.05, 0.45, "bad", 2.5)  # type: ignore[arg-type]

    def test_maturity_non_numeric_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="'maturity' must be numeric"):
            sanitize_exposure_inputs(0.05, 0.45, 1000, "bad")  # type: ignore[arg-type]

    def test_returns_floats(self) -> None:
        result = sanitize_exposure_inputs(0, 0, 0, 1)
        assert isinstance(result["pd"], float)
        assert isinstance(result["lgd"], float)
        assert isinstance(result["ead"], float)
        assert isinstance(result["maturity"], float)
