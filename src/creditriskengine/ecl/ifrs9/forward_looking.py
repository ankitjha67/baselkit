"""
Forward-looking information (FLI) and macro overlay adjustments.

Reference: IFRS 9.5.5.17(c), IFRS 9.B5.5.49-B5.5.54.

Incorporates macroeconomic forecasts into ECL estimates through
PD/LGD adjustments based on forward-looking indicators.

Supports:
    - Linear sensitivity (single-variable) approach.
    - Multi-variable satellite model with configurable link function.
    - Non-linear (logistic) link for bounded PD adjustments.
    - Forecast-horizon mean-reversion to long-run average (IFRS 9.B5.5.50).
    - LGD macro overlay driven by collateral-value proxies (e.g. HPI).

Regulatory basis:
    - IFRS 9.5.5.17(c) — forward-looking information requirement.
    - IFRS 9.B5.5.49-B5.5.54 — using reasonable and supportable forecasts.
    - IFRS 9.B5.5.50 — revert to historical experience beyond forecast horizon.
    - EBA/GL/2017/06 para 30-34 — macro-economic factor integration.
    - ECB Guide to Internal Models (Feb 2017) Ch. 7 — satellite model expectations.
    - BCBS d350 (2015) — guidance on credit risk and accounting for ECL.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-variable linear approach (original)
# ---------------------------------------------------------------------------


def macro_adjustment_factor(
    macro_variable_forecast: np.ndarray,
    macro_variable_baseline: float,
    sensitivity: float,
) -> np.ndarray:
    """Calculate macro adjustment factors for PD/LGD.

    Simple linear sensitivity approach:
        Adj(t) = 1 + sensitivity * (forecast(t) - baseline) / baseline

    Reference: IFRS 9.B5.5.51 — sensitivity to macro-economic conditions.

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


# ---------------------------------------------------------------------------
# Multi-variable satellite model (advanced)
# ---------------------------------------------------------------------------


@dataclass
class SatelliteModelConfig:
    """Configuration for a macro-economic satellite model.

    A satellite model maps macro-economic variables to credit risk
    parameter adjustments.  The model is:

        z(t) = intercept + Sum_j [beta_j * X_j(t)]

    where X_j(t) is the j-th macro variable at time t and beta_j is
    its coefficient.  The output z(t) is then passed through a link
    function to produce the adjustment factor.

    Reference:
        ECB Guide to Internal Models Ch. 7 — satellite model design.
        EBA/GL/2017/06 para 30-34 — macro factor estimation.

    Attributes:
        variable_names: Ordered list of macro variable names.
        coefficients: Corresponding regression coefficients (beta_j).
        intercept: Model intercept.
        link: Link function: ``"linear"`` (identity), ``"logistic"``
            (bounded sigmoid), or ``"log"`` (multiplicative).
    """

    variable_names: list[str]
    coefficients: list[float]
    intercept: float = 0.0
    link: str = "linear"


def satellite_model_predict(
    config: SatelliteModelConfig,
    macro_forecasts: dict[str, np.ndarray],
    n_periods: int | None = None,
) -> np.ndarray:
    """Predict PD adjustment factors using a multi-variable satellite model.

    Computes the linear predictor z(t) for each period and applies
    the configured link function.

    Link functions:
        - ``"linear"``:   factor(t) = max(z(t), 0)
        - ``"logistic"``: factor(t) = 2 / (1 + exp(-z(t)))
          *(centred at 1.0 so that z=0 → factor=1.0)*
        - ``"log"``:      factor(t) = exp(z(t))

    Reference:
        ECB Guide to Internal Models — satellite model calibration.
        EBA/GL/2017/06 para 33 — multiple macro-economic variables.

    Args:
        config: Satellite model configuration.
        macro_forecasts: Dict mapping variable name to forecast array.
        n_periods: Number of forecast periods.  Inferred from the
            shortest forecast array if not provided.

    Returns:
        Array of PD adjustment factors (shape: n_periods).

    Raises:
        ValueError: If a required variable is missing from *macro_forecasts*
            or if *coefficients* length mismatches *variable_names*.
    """
    if len(config.variable_names) != len(config.coefficients):
        raise ValueError(
            f"variable_names ({len(config.variable_names)}) and "
            f"coefficients ({len(config.coefficients)}) must have the same length"
        )

    # Determine number of periods
    lengths = [len(macro_forecasts[v]) for v in config.variable_names if v in macro_forecasts]
    if n_periods is None:
        if not lengths:
            raise ValueError("No forecast data provided for any variable")
        n_periods = min(lengths)

    z = np.full(n_periods, config.intercept, dtype=np.float64)

    for name, beta in zip(config.variable_names, config.coefficients, strict=True):
        if name not in macro_forecasts:
            raise ValueError(f"Missing forecast for macro variable '{name}'")
        forecast = np.asarray(macro_forecasts[name], dtype=np.float64)[:n_periods]
        z[:len(forecast)] += beta * forecast

    if config.link == "logistic":
        # Centred logistic: z=0 → factor=1.0, range (0, 2)
        factors = 2.0 / (1.0 + np.exp(-z))
    elif config.link == "log":
        factors = np.exp(z)
    else:
        # Linear (identity) with floor at 0
        factors = np.maximum(z, 0.0)

    logger.debug(
        "Satellite model predict: link=%s, n_periods=%d, factor_range=[%.4f, %.4f]",
        config.link, n_periods, float(np.min(factors)), float(np.max(factors)),
    )
    return factors


# ---------------------------------------------------------------------------
# Forecast-horizon mean-reversion (IFRS 9.B5.5.50)
# ---------------------------------------------------------------------------


def mean_reversion_weights(
    total_periods: int,
    forecast_horizon: int,
    reversion_periods: int,
) -> np.ndarray:
    """Calculate blending weights for forecast vs. long-run average.

    IFRS 9.B5.5.50 requires entities to revert to historical loss
    experience for periods beyond the reasonable and supportable
    forecast horizon.

    During the forecast horizon, weight = 1.0 (full forecast).
    During the reversion window, weight linearly decays from 1.0 to 0.0.
    Beyond forecast + reversion, weight = 0.0 (pure long-run).

    Reference:
        IFRS 9.B5.5.50 — reversion to historical experience.
        EBA/GL/2017/06 para 34 — forecast horizon and reversion.

    Args:
        total_periods: Total number of ECL projection periods.
        forecast_horizon: Number of periods with supportable forecasts.
        reversion_periods: Number of periods for linear reversion.

    Returns:
        Array of weights in [0, 1] (shape: total_periods).
    """
    weights = np.zeros(total_periods, dtype=np.float64)

    for t in range(total_periods):
        if t < forecast_horizon:
            weights[t] = 1.0
        elif reversion_periods > 0 and t < forecast_horizon + reversion_periods:
            progress = (t - forecast_horizon) / reversion_periods
            weights[t] = 1.0 - progress
        # else: 0.0 (long-run)

    return weights


def apply_fli_with_reversion(
    base_pds: np.ndarray,
    fli_adjustment_factors: np.ndarray,
    long_run_pds: np.ndarray | float,
    forecast_horizon: int,
    reversion_periods: int = 4,
    floor: float = 0.0001,
    cap: float = 1.0,
) -> np.ndarray:
    """Apply forward-looking adjustments with mean-reversion.

    Blends FLI-adjusted PDs during the forecast window with long-run
    PDs beyond the forecast horizon, using a linear reversion ramp.

    For each period t:
        PD(t) = w(t) × PD_fli(t) + (1 - w(t)) × PD_long_run(t)

    where w(t) is the forecast weight from :func:`mean_reversion_weights`.

    Reference:
        IFRS 9.B5.5.50 — reverting to historical experience.
        EBA/GL/2017/06 para 34 — reasonable and supportable forecasts.

    Args:
        base_pds: Base PD term structure (n_periods,).
        fli_adjustment_factors: Macro adjustment factors (n_periods,).
        long_run_pds: Long-run (through-the-cycle) PD term structure
            or scalar.
        forecast_horizon: Number of periods with supportable forecasts.
        reversion_periods: Number of periods for linear mean-reversion
            (default 4, i.e. 4 years).
        floor: Minimum PD.
        cap: Maximum PD.

    Returns:
        Blended PD term structure after FLI and mean-reversion.
    """
    n = len(base_pds)
    fli_pds = base_pds * fli_adjustment_factors[:n]

    if isinstance(long_run_pds, (int, float)):
        long_run_pds = np.full(n, long_run_pds)
    else:
        long_run_pds = np.asarray(long_run_pds, dtype=np.float64)[:n]

    weights = mean_reversion_weights(n, forecast_horizon, reversion_periods)
    blended = weights[:n] * fli_pds + (1.0 - weights[:n]) * long_run_pds[:n]

    return np.asarray(np.clip(blended, floor, cap))


# ---------------------------------------------------------------------------
# LGD macro overlay
# ---------------------------------------------------------------------------


def lgd_macro_overlay(
    base_lgds: np.ndarray,
    collateral_index_forecast: np.ndarray,
    collateral_index_baseline: float,
    sensitivity: float = 0.5,
    floor: float = 0.0,
    cap: float = 1.0,
) -> np.ndarray:
    """Apply macro-driven LGD adjustments based on collateral index changes.

    Collateral value declines (e.g. house price drops) increase LGD:

        LGD_adj(t) = LGD_base + sensitivity × max(0, baseline − forecast(t)) / baseline

    Reference:
        IFRS 9.B5.5.52 — adjustments for current conditions.
        EBA/GL/2017/16 Art. 181 — downturn LGD and collateral haircuts.
        BoE ACS methodology — HPI-driven LGD stress.

    Args:
        base_lgds: Base LGD term structure (n_periods,).
        collateral_index_forecast: Forecasted collateral index (e.g. HPI).
        collateral_index_baseline: Baseline collateral index value.
        sensitivity: LGD sensitivity to collateral declines (default 0.5).
        floor: Minimum LGD.
        cap: Maximum LGD.

    Returns:
        Adjusted LGD term structure.
    """
    if collateral_index_baseline <= 0:
        return np.asarray(np.clip(base_lgds, floor, cap))

    n = len(base_lgds)
    forecast = np.asarray(collateral_index_forecast, dtype=np.float64)[:n]

    decline_pct = np.maximum(
        (collateral_index_baseline - forecast) / collateral_index_baseline,
        0.0,
    )

    adjusted = base_lgds + sensitivity * decline_pct[:n]
    return np.asarray(np.clip(adjusted, floor, cap))


# ---------------------------------------------------------------------------
# FLI summary for audit and disclosure
# ---------------------------------------------------------------------------


def fli_impact_summary(
    base_pds: np.ndarray,
    adjusted_pds: np.ndarray,
    model_type: str = "linear",
    variables_used: list[str] | None = None,
) -> dict[str, Any]:
    """Generate a summary of FLI adjustments for governance reporting.

    Designed for IFRS 7.35G disclosure and internal model monitoring.

    Reference:
        IFRS 7.35G — ECL measurement disclosure.
        EBA/GL/2017/06 para 35 — FLI impact monitoring.

    Args:
        base_pds: Base PD term structure before FLI.
        adjusted_pds: PD term structure after FLI adjustment.
        model_type: Type of model used (e.g. ``"linear"``,
            ``"satellite_logistic"``).
        variables_used: List of macro variable names used.

    Returns:
        Dict with summary metrics for reporting.
    """
    mean_base = float(np.mean(base_pds))
    mean_adj = float(np.mean(adjusted_pds))
    pct_change = (
        (mean_adj - mean_base) / mean_base * 100.0
        if mean_base > 0
        else 0.0
    )

    return {
        "model_type": model_type,
        "variables_used": variables_used or [],
        "n_periods": len(base_pds),
        "mean_base_pd": round(mean_base, 6),
        "mean_adjusted_pd": round(mean_adj, 6),
        "pct_change": round(pct_change, 2),
        "max_adjustment_factor": round(
            float(np.max(adjusted_pds / np.maximum(base_pds, 1e-10))), 4
        ),
        "min_adjustment_factor": round(
            float(np.min(adjusted_pds / np.maximum(base_pds, 1e-10))), 4
        ),
    }
