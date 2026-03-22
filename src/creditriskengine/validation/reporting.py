"""
Automated validation report generator.

Generates standardized model validation reports combining
discrimination, calibration, and stability results.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def generate_validation_summary(
    model_name: str,
    discrimination_results: dict[str, float],
    calibration_results: dict[str, Any],
    stability_results: dict[str, float],
) -> dict[str, Any]:
    """Generate a validation summary report.

    Args:
        model_name: Name of the model being validated.
        discrimination_results: Results from discrimination tests.
        calibration_results: Results from calibration tests.
        stability_results: Results from stability tests.

    Returns:
        Structured validation summary.
    """
    # Traffic light assessment
    overall = "green"
    if discrimination_results.get("gini", 0) < 0.3:
        overall = "yellow"
    if discrimination_results.get("gini", 0) < 0.15:
        overall = "red"

    psi = stability_results.get("psi", 0)
    if psi >= 0.25:
        overall = "red"
    elif psi >= 0.10 and overall != "red":
        overall = "yellow"

    return {
        "model_name": model_name,
        "overall_assessment": overall,
        "discrimination": discrimination_results,
        "calibration": calibration_results,
        "stability": stability_results,
    }
