"""
CreditRiskEngine — Production-grade credit risk analytics.

Modules:
    core: Data models and type definitions
    rwa: Risk-weighted asset calculation (SA and IRB)
    ecl: Expected credit loss engines (IFRS 9, CECL)
    models: PD, LGD, EAD modeling pipelines
    validation: Model validation toolkit
    portfolio: Portfolio credit risk models
    reporting: Regulatory reporting
"""

__version__ = "0.3.0"

# Public API — convenient top-level imports
from creditriskengine.core.exposure import Collateral, Exposure
from creditriskengine.core.portfolio import Portfolio
from creditriskengine.core.types import (
    CollateralType,
    CreditQualityStep,
    CreditRiskApproach,
    IFRS9Stage,
    IRBAssetClass,
    IRBRetailSubClass,
    Jurisdiction,
    SAExposureClass,
)

__all__ = [
    "Collateral",
    "CollateralType",
    "CreditQualityStep",
    "CreditRiskApproach",
    "Exposure",
    "IFRS9Stage",
    "IRBAssetClass",
    "IRBRetailSubClass",
    "Jurisdiction",
    "Portfolio",
    "SAExposureClass",
]
