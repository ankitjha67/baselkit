"""
Abstract base classes for RWA calculators.

Provides the common interface and result container for all credit risk
RWA calculation approaches (SA, F-IRB, A-IRB, supervisory slotting).
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from creditriskengine.core.exposure import Exposure

logger = logging.getLogger(__name__)

__all__ = ["RWAResult", "BaseRWACalculator"]


@dataclass
class RWAResult:
    """Result of an RWA calculation.

    Attributes:
        exposure_id: Unique identifier for the exposure.
        risk_weight: Risk weight as a percentage (e.g. 75.0 means 75%).
        rwa: Risk-weighted asset amount (= EAD * RW / 100).
        ead: Exposure at Default used in the calculation.
        capital_requirement: Minimum capital = 8% of RWA (BCBS CRE31.4).
        approach: Calculation approach used (e.g. 'foundation_irb').
        asset_class: IRB asset class or SA exposure class, if applicable.
        details: Intermediate calculation values (K, correlation, MA, etc.).
    """

    exposure_id: str
    risk_weight: float  # percentage e.g. 75.0
    rwa: float  # risk-weighted asset amount
    ead: float
    capital_requirement: float  # 8% of RWA
    approach: str
    asset_class: str | None = None
    details: dict[str, float] | None = field(default_factory=dict)


class BaseRWACalculator(ABC):
    """Abstract RWA calculator interface.

    All concrete calculators (SA, F-IRB, A-IRB, slotting) must implement
    the :meth:`calculate` method. Portfolio-level helpers are provided
    by the base class.
    """

    @abstractmethod
    def calculate(self, exposure: Exposure) -> RWAResult:
        """Calculate RWA for a single exposure.

        Args:
            exposure: Fully populated :class:`Exposure` instance.

        Returns:
            :class:`RWAResult` with risk weight and capital requirement.
        """
        ...

    def calculate_portfolio(
        self, exposures: list[Exposure]
    ) -> list[RWAResult]:
        """Calculate RWA for a portfolio of exposures.

        Args:
            exposures: List of :class:`Exposure` instances.

        Returns:
            List of :class:`RWAResult`, one per exposure.
        """
        results: list[RWAResult] = []
        for exp in exposures:
            try:
                results.append(self.calculate(exp))
            except Exception:
                logger.exception(
                    "RWA calculation failed for exposure %s", exp.exposure_id
                )
                raise
        return results

    def total_rwa(self, exposures: list[Exposure]) -> float:
        """Compute aggregate RWA for a portfolio.

        Args:
            exposures: List of :class:`Exposure` instances.

        Returns:
            Sum of individual RWA amounts.
        """
        return sum(r.rwa for r in self.calculate_portfolio(exposures))
