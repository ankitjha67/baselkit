"""
Forward-looking information (FLI) and macro overlay adjustments.

Reference: IFRS 9.5.5.17(c), IFRS 9.B5.5.49-B5.5.54.

Incorporates macroeconomic forecasts into ECL estimates through
PD/LGD adjustments based on forward-looking indicators.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def macro_adjustment_factor(
    macro_variable_forecast: np.ndarray,
    macro_variable_baseline: float,
    sensitivity: float,
) -> np.ndarray:
    """Calculate macro adjustment factors for PD/LGD.

    Simple linear sensitivity approach:
        Adj(t) = 1 + sensitivity * (forecast(t) - baseline) / baseline

    Args:
        macro_variable_forecast: Forecasted macro variable values.
        macro_variable_baseline: Baseline (long-run) value.
        sensitivity: Model-estimated sensitivity coefficient.

    Returns:
        Array of adjustment factors (1.0 = no adjustment).
    """
    if macro_variable_baseline == 0:
        return np.ones_like(macro_variable_forecast)

    pct_change = (macro_variable_forecast - macro_variable_baseline) / abs(macro_variable_baseline)
    factors = 1.0 + sensitivity * pct_change
    return np.maximum(factors, 0.0)


def apply_macro_overlay(
    base_pds: np.ndarray,
    adjustment_factors: np.ndarray,
    floor: float = 0.0001,
    cap: float = 1.0,
) -> np.ndarray:
    """Apply macro overlay adjustments to PD term structure.

    Args:
        base_pds: Base PD term structure.
        adjustment_factors: Macro adjustment factors.
        floor: Minimum PD floor.
        cap: Maximum PD cap.

    Returns:
        Adjusted PD term structure.
    """
    adjusted = base_pds * adjustment_factors[:len(base_pds)]
    return np.asarray(np.clip(adjusted, floor, cap))
