"""Portfolio container for credit exposures."""

from typing import Iterator, Optional

from pydantic import BaseModel, Field

from creditriskengine.core.exposure import Exposure
from creditriskengine.core.types import CreditRiskApproach, Jurisdiction


class Portfolio(BaseModel):
    """Container for a portfolio of credit exposures.

    Provides iteration, filtering, and aggregation over exposures.
    """

    name: str = "Unnamed Portfolio"
    jurisdiction: Optional[Jurisdiction] = None
    approach: Optional[CreditRiskApproach] = None
    exposures: list[Exposure] = Field(default_factory=list)

    def add_exposure(self, exposure: Exposure) -> None:
        """Add an exposure to the portfolio."""
        self.exposures.append(exposure)

    def total_ead(self) -> float:
        """Total Exposure at Default across all exposures."""
        return sum(e.ead for e in self.exposures)

    def __len__(self) -> int:
        return len(self.exposures)

    def __iter__(self) -> Iterator[Exposure]:
        return iter(self.exposures)

    def filter_by_approach(self, approach: CreditRiskApproach) -> list[Exposure]:
        """Return exposures using a specific calculation approach."""
        return [e for e in self.exposures if e.approach == approach]

    def filter_defaulted(self) -> list[Exposure]:
        """Return defaulted exposures."""
        return [e for e in self.exposures if e.is_defaulted]

    def filter_non_defaulted(self) -> list[Exposure]:
        """Return non-defaulted exposures."""
        return [e for e in self.exposures if not e.is_defaulted]
