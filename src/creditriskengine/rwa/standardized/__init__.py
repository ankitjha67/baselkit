"""Standardized Approach (SA) for credit risk RWA per BCBS d424 CRE20."""

from creditriskengine.rwa.standardized.credit_risk_sa import (
    assign_sa_risk_weight,
    get_bank_risk_weight,
    get_commercial_re_risk_weight,
    get_corporate_risk_weight,
    get_residential_re_risk_weight,
    get_sovereign_risk_weight,
)
from creditriskengine.rwa.standardized.risk_weights import (
    RiskWeightRegistry,
    load_risk_weight_registry,
)

__all__ = [
    "assign_sa_risk_weight",
    "get_sovereign_risk_weight",
    "get_bank_risk_weight",
    "get_corporate_risk_weight",
    "get_residential_re_risk_weight",
    "get_commercial_re_risk_weight",
    "RiskWeightRegistry",
    "load_risk_weight_registry",
]
