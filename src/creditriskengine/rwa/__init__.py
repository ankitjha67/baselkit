"""RWA (Risk-Weighted Assets) calculation package.

Supports Standardized Approach (SA), Foundation IRB (F-IRB),
and Advanced IRB (A-IRB) per BCBS d424.
"""

from creditriskengine.rwa.base import BaseRWACalculator, RWAResult
from creditriskengine.rwa.output_floor import OutputFloorCalculator, get_output_floor_pct

__all__ = [
    "BaseRWACalculator",
    "RWAResult",
    "OutputFloorCalculator",
    "get_output_floor_pct",
]
