"""Core types, data models, and configuration for creditriskengine.

Provides regulatory type enumerations, exposure/portfolio data models,
configuration loading, and the exception hierarchy.
"""

from creditriskengine.core.exceptions import (
    CalculationError,
    ConfigurationError,
    CreditRiskEngineError,
    DataError,
    RegulatoryError,
    ValidationError,
)
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
    "CreditRiskEngineError",
    "ConfigurationError",
    "ValidationError",
    "RegulatoryError",
    "CalculationError",
    "DataError",
    "Jurisdiction",
    "CreditRiskApproach",
    "IRBAssetClass",
    "IRBRetailSubClass",
    "SAExposureClass",
    "CreditQualityStep",
    "IFRS9Stage",
    "CollateralType",
    "Exposure",
    "Collateral",
    "Portfolio",
]
