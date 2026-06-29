"""Counterparty Credit Risk (CCR) analytics.

Modules:
    exposure : EPE / EEPE / PFE exposure profiles and netting.
    sa_ccr   : Full SA-CCR EAD engine (add-ons, RC, PFE multiplier).
    wwr      : Wrong-way risk (general + specific).

References:
    - BCBS CRE52 (SA-CCR), CRE53 (IMM).
    - BCBS CRE52.18 (wrong-way risk).
    - Basel III counterparty credit risk framework.
"""

from creditriskengine.ccr.exposure import (
    ExposureProfile,
    effective_epe,
    effective_expected_exposure,
    expected_positive_exposure,
    netting_set_exposure,
    potential_future_exposure,
)
from creditriskengine.ccr.sa_ccr import (
    ALPHA,
    AssetClass,
    OptionType,
    SACCRResult,
    SACCRTrade,
    adjusted_notional,
    aggregate_addon,
    maturity_factor,
    pfe_multiplier,
    replacement_cost,
    sa_ccr_ead,
    supervisory_delta,
    supervisory_duration,
)
from creditriskengine.ccr.wwr import (
    alpha_wrong_way_multiplier,
    conditional_epe_wwr,
    specific_wwr_flag,
)

__all__ = [
    "ExposureProfile",
    "expected_positive_exposure",
    "effective_expected_exposure",
    "effective_epe",
    "potential_future_exposure",
    "netting_set_exposure",
    "alpha_wrong_way_multiplier",
    "conditional_epe_wwr",
    "specific_wwr_flag",
    "ALPHA",
    "AssetClass",
    "OptionType",
    "SACCRTrade",
    "SACCRResult",
    "supervisory_duration",
    "adjusted_notional",
    "supervisory_delta",
    "maturity_factor",
    "aggregate_addon",
    "pfe_multiplier",
    "replacement_cost",
    "sa_ccr_ead",
]
