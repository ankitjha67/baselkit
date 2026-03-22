"""Custom exception hierarchy for creditriskengine."""


class CreditRiskEngineError(Exception):
    """Base exception for all creditriskengine errors."""


class ConfigurationError(CreditRiskEngineError):
    """Raised when configuration is invalid or missing."""


class JurisdictionError(CreditRiskEngineError):
    """Raised when jurisdiction-specific config is not found."""


class ValidationError(CreditRiskEngineError):
    """Raised when input validation fails."""


class RegulatoryError(CreditRiskEngineError):
    """Raised when regulatory constraints are violated."""


class CalculationError(CreditRiskEngineError):
    """Raised when a calculation fails or produces invalid results."""


class DataError(CreditRiskEngineError):
    """Raised when input data is malformed or insufficient."""
