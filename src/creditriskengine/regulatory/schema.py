"""YAML schema validation for jurisdiction configuration files.

Validates that jurisdiction YAML configs contain required keys with
correct types before they are consumed by calculation engines.
"""

from __future__ import annotations

from typing import Any

from creditriskengine.core.exceptions import ConfigurationError

# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

JURISDICTION_SCHEMA: dict[str, Any] = {
    "required_top_level": [
        "jurisdiction",
        "regulator",
        "framework",
        "effective_date",
    ],
    "credit_risk": {
        "required": [
            "approaches_permitted",
            "standardised_approach",
            "irb_approach",
        ],
        "irb_approach": {
            "required": {
                "pd_floor_bps": {"type": int, "min": 1, "max": 100},
                "lgd_floors": {"type": dict},
                "maturity_default_years": {"type": float, "min": 0.0, "max": 30.0},
            },
        },
    },
    "output_floor": {
        "required": {
            "enabled": {"type": bool},
            "final_pct": {"type": float, "min": 0.0, "max": 1.0},
        },
    },
    "default_definition": {
        "required": {
            "dpd_threshold": {"type": int},
            "cure_period_months": {"type": int},
        },
    },
}


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def validate_config(config: dict[str, Any]) -> list[str]:
    """Validate a jurisdiction config dict against :data:`JURISDICTION_SCHEMA`.

    Returns
    -------
    list[str]
        A list of human-readable error strings.  An empty list means the
        config is valid.
    """
    errors: list[str] = []

    # Top-level required keys
    for key in JURISDICTION_SCHEMA["required_top_level"]:
        if key not in config:
            errors.append(f"Missing required top-level key: '{key}'")

    # credit_risk section
    credit_risk = config.get("credit_risk")
    if credit_risk is None:
        errors.append("Missing required top-level key: 'credit_risk'")
    else:
        if not isinstance(credit_risk, dict):
            errors.append("'credit_risk' must be a dict")
        else:
            for key in JURISDICTION_SCHEMA["credit_risk"]["required"]:
                if key not in credit_risk:
                    errors.append(f"Missing required key in 'credit_risk': '{key}'")

            # irb_approach nested validation
            irb = credit_risk.get("irb_approach")
            if irb is not None:
                if not isinstance(irb, dict):
                    errors.append("'credit_risk.irb_approach' must be a dict")
                else:
                    irb_schema = JURISDICTION_SCHEMA["credit_risk"]["irb_approach"]["required"]
                    for key, rules in irb_schema.items():
                        if key not in irb:
                            errors.append(
                                f"Missing required key in 'credit_risk.irb_approach': '{key}'"
                            )
                        else:
                            value = irb[key]
                            expected_type = rules["type"]
                            # Allow int where float is expected
                            if expected_type is float:
                                if not isinstance(value, (int, float)):
                                    errors.append(
                                        f"'credit_risk.irb_approach.{key}' must be "
                                        f"{expected_type.__name__}, got {type(value).__name__}"
                                    )
                                elif "min" in rules and value < rules["min"]:
                                    errors.append(
                                        f"'credit_risk.irb_approach.{key}' must be "
                                        f">= {rules['min']}, got {value}"
                                    )
                                elif "max" in rules and value > rules["max"]:
                                    errors.append(
                                        f"'credit_risk.irb_approach.{key}' must be "
                                        f"<= {rules['max']}, got {value}"
                                    )
                            else:
                                if not isinstance(value, expected_type):
                                    errors.append(
                                        f"'credit_risk.irb_approach.{key}' must be "
                                        f"{expected_type.__name__}, got {type(value).__name__}"
                                    )
                                elif expected_type is int:
                                    if "min" in rules and value < rules["min"]:
                                        errors.append(
                                            f"'credit_risk.irb_approach.{key}' must be "
                                            f">= {rules['min']}, got {value}"
                                        )
                                    elif "max" in rules and value > rules["max"]:
                                        errors.append(
                                            f"'credit_risk.irb_approach.{key}' must be "
                                            f"<= {rules['max']}, got {value}"
                                        )

    # output_floor section
    output_floor = config.get("output_floor")
    if output_floor is None:
        errors.append("Missing required top-level key: 'output_floor'")
    else:
        if not isinstance(output_floor, dict):
            errors.append("'output_floor' must be a dict")
        else:
            of_schema = JURISDICTION_SCHEMA["output_floor"]["required"]
            for key, rules in of_schema.items():
                if key not in output_floor:
                    errors.append(f"Missing required key in 'output_floor': '{key}'")
                else:
                    value = output_floor[key]
                    expected_type = rules["type"]
                    if expected_type is float:
                        if not isinstance(value, (int, float)):
                            errors.append(
                                f"'output_floor.{key}' must be "
                                f"{expected_type.__name__}, got {type(value).__name__}"
                            )
                        elif "min" in rules and value < rules["min"]:
                            errors.append(
                                f"'output_floor.{key}' must be "
                                f">= {rules['min']}, got {value}"
                            )
                        elif "max" in rules and value > rules["max"]:
                            errors.append(
                                f"'output_floor.{key}' must be "
                                f"<= {rules['max']}, got {value}"
                            )
                    elif expected_type is bool and not isinstance(value, bool):
                        errors.append(
                            f"'output_floor.{key}' must be "
                            f"{expected_type.__name__}, got {type(value).__name__}"
                        )

    # default_definition section
    default_def = config.get("default_definition")
    if default_def is None:
        errors.append("Missing required top-level key: 'default_definition'")
    else:
        if not isinstance(default_def, dict):
            errors.append("'default_definition' must be a dict")
        else:
            dd_schema = JURISDICTION_SCHEMA["default_definition"]["required"]
            for key, rules in dd_schema.items():
                if key not in default_def:
                    errors.append(f"Missing required key in 'default_definition': '{key}'")
                else:
                    value = default_def[key]
                    expected_type = rules["type"]
                    if not isinstance(value, expected_type):
                        errors.append(
                            f"'default_definition.{key}' must be "
                            f"{expected_type.__name__}, got {type(value).__name__}"
                        )

    return errors


def validate_config_strict(config: dict[str, Any]) -> None:
    """Validate config; raise :class:`ConfigurationError` if invalid.

    Parameters
    ----------
    config : dict
        Jurisdiction configuration dictionary.

    Raises
    ------
    ConfigurationError
        If one or more validation errors are found.
    """
    errors = validate_config(config)
    if errors:
        msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ConfigurationError(msg)


# ---------------------------------------------------------------------------
# Range validators
# ---------------------------------------------------------------------------


def validate_risk_weight_range(value: float, field_name: str) -> None:
    """Ensure *value* is a valid risk weight (0 to 12.5 i.e. 1250%).

    Raises
    ------
    ConfigurationError
        If the value is outside the permitted range.
    """
    if not isinstance(value, (int, float)):
        raise ConfigurationError(
            f"'{field_name}' must be numeric, got {type(value).__name__}"
        )
    if value < 0 or value > 12.5:
        raise ConfigurationError(
            f"'{field_name}' must be between 0 and 12.5 (1250%), got {value}"
        )


def validate_pd_range(value: float, field_name: str) -> None:
    """Ensure *value* is a valid probability of default (0 to 1).

    Raises
    ------
    ConfigurationError
        If the value is outside [0, 1].
    """
    if not isinstance(value, (int, float)):
        raise ConfigurationError(
            f"'{field_name}' must be numeric, got {type(value).__name__}"
        )
    if value < 0 or value > 1:
        raise ConfigurationError(
            f"'{field_name}' must be between 0 and 1, got {value}"
        )


def validate_lgd_range(value: float, field_name: str) -> None:
    """Ensure *value* is a valid LGD (0 to 1).

    Raises
    ------
    ConfigurationError
        If the value is outside [0, 1].
    """
    if not isinstance(value, (int, float)):
        raise ConfigurationError(
            f"'{field_name}' must be numeric, got {type(value).__name__}"
        )
    if value < 0 or value > 1:
        raise ConfigurationError(
            f"'{field_name}' must be between 0 and 1, got {value}"
        )


# ---------------------------------------------------------------------------
# Input sanitization
# ---------------------------------------------------------------------------


def sanitize_exposure_inputs(
    pd: float,
    lgd: float,
    ead: float,
    maturity: float,
) -> dict[str, Any]:
    """Validate and clip exposure inputs to regulatory bounds.

    Parameters
    ----------
    pd : float
        Probability of default (clipped to [0, 1]).
    lgd : float
        Loss given default (clipped to [0, 1]).
    ead : float
        Exposure at default (must be >= 0).
    maturity : float
        Effective maturity in years (clipped to [1, 5]).

    Returns
    -------
    dict
        ``{"pd": <float>, "lgd": <float>, "ead": <float>,
        "maturity": <float>, "warnings": [<str>, ...]}``
    """
    warnings: list[str] = []

    # PD
    if not isinstance(pd, (int, float)):
        raise ConfigurationError(f"'pd' must be numeric, got {type(pd).__name__}")
    if pd < 0:
        warnings.append(f"PD clipped from {pd} to 0")
        pd = 0.0
    elif pd > 1:
        warnings.append(f"PD clipped from {pd} to 1")
        pd = 1.0

    # LGD
    if not isinstance(lgd, (int, float)):
        raise ConfigurationError(f"'lgd' must be numeric, got {type(lgd).__name__}")
    if lgd < 0:
        warnings.append(f"LGD clipped from {lgd} to 0")
        lgd = 0.0
    elif lgd > 1:
        warnings.append(f"LGD clipped from {lgd} to 1")
        lgd = 1.0

    # EAD
    if not isinstance(ead, (int, float)):
        raise ConfigurationError(f"'ead' must be numeric, got {type(ead).__name__}")
    if ead < 0:
        warnings.append(f"EAD clipped from {ead} to 0")
        ead = 0.0

    # Maturity
    if not isinstance(maturity, (int, float)):
        raise ConfigurationError(
            f"'maturity' must be numeric, got {type(maturity).__name__}"
        )
    if maturity < 1:
        warnings.append(f"Maturity clipped from {maturity} to 1")
        maturity = 1.0
    elif maturity > 5:
        warnings.append(f"Maturity clipped from {maturity} to 5")
        maturity = 5.0

    return {
        "pd": float(pd),
        "lgd": float(lgd),
        "ead": float(ead),
        "maturity": float(maturity),
        "warnings": warnings,
    }
