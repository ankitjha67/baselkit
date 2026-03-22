"""
Regulatory reporting templates.

Generates structured reports for regulatory submissions
and internal model validation documentation.

References:
- EBA ITS on Supervisory Reporting (COREP)
- BCBS Pillar 3 disclosure requirements (DIS)
"""

import logging
from datetime import date
from typing import Any

logger = logging.getLogger(__name__)


def _worst_rag(*rags: str) -> str:
    """Return the worst RAG status from a set of RAG values."""
    if "red" in rags:
        return "red"
    if "yellow" in rags:
        return "yellow"
    return "green"


def generate_corep_credit_risk_summary(
    jurisdiction: str,
    reporting_date: date,
    sa_rwa: float,
    irb_rwa: float,
    floored_rwa: float,
    total_ead: float,
    total_ecl: float,
) -> dict[str, Any]:
    """Generate COREP-style credit risk summary.

    Args:
        jurisdiction: Reporting jurisdiction.
        reporting_date: As-of date.
        sa_rwa: Total SA RWA.
        irb_rwa: Total IRB RWA.
        floored_rwa: Output-floored RWA.
        total_ead: Total EAD.
        total_ecl: Total ECL provision.

    Returns:
        Structured report dict.
    """
    capital_requirement = floored_rwa * 0.08  # 8% minimum per Basel III
    return {
        "report_type": "COREP_CR",
        "jurisdiction": jurisdiction,
        "reporting_date": reporting_date.isoformat(),
        "total_ead": total_ead,
        "sa_rwa": sa_rwa,
        "irb_rwa": irb_rwa,
        "floored_rwa": floored_rwa,
        "output_floor_binding": floored_rwa > irb_rwa,
        "capital_requirement_8pct": capital_requirement,
        "total_ecl": total_ecl,
        "ecl_to_ead_ratio": total_ecl / total_ead if total_ead > 0 else 0.0,
    }


def generate_pillar3_credit_risk(
    reporting_date: date,
    exposure_classes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Generate Pillar 3 credit risk disclosure template.

    Args:
        reporting_date: As-of date.
        exposure_classes: List of dicts with per-class data
            (class_name, ead, rwa, expected_loss, n_exposures).

    Returns:
        Structured Pillar 3 report.
    """
    total_ead = sum(ec.get("ead", 0) for ec in exposure_classes)
    total_rwa = sum(ec.get("rwa", 0) for ec in exposure_classes)
    total_el = sum(ec.get("expected_loss", 0) for ec in exposure_classes)

    return {
        "report_type": "PILLAR3_CR",
        "reporting_date": reporting_date.isoformat(),
        "total_ead": total_ead,
        "total_rwa": total_rwa,
        "total_expected_loss": total_el,
        "avg_risk_weight": total_rwa / total_ead if total_ead > 0 else 0.0,
        "exposure_classes": exposure_classes,
    }


def generate_model_inventory_entry(
    model_name: str,
    model_type: str,
    asset_class: str,
    validation_date: date,
    auroc: float,
    gini: float,
    psi: float,
    calibration_result: str,
) -> dict[str, Any]:
    """Generate a model inventory entry for MRM reporting.

    Reference: SR 11-7 / SS1/23 model risk management.

    Args:
        model_name: Name of the model.
        model_type: Model type (PD, LGD, EAD).
        asset_class: Asset class the model covers.
        validation_date: Date of last validation.
        auroc: Area under ROC curve.
        gini: Gini coefficient.
        psi: Population Stability Index.
        calibration_result: Calibration test result (green/yellow/red).

    Returns:
        Model inventory entry dict.
    """
    # Traffic light based on thresholds
    discrimination_rag = "green"
    if gini < 0.30:
        discrimination_rag = "yellow"
    if gini < 0.15:
        discrimination_rag = "red"

    stability_rag = "green"
    if psi >= 0.25:
        stability_rag = "red"
    elif psi >= 0.10:
        stability_rag = "yellow"

    return {
        "model_name": model_name,
        "model_type": model_type,
        "asset_class": asset_class,
        "validation_date": validation_date.isoformat(),
        "discrimination": {"auroc": auroc, "gini": gini, "rag": discrimination_rag},
        "stability": {"psi": psi, "rag": stability_rag},
        "calibration": {"result": calibration_result},
        "overall_rag": _worst_rag(
            discrimination_rag, stability_rag, calibration_result
        ),
    }
