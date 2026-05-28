"""Interest Rate Risk in the Banking Book (IRRBB).

Modules:
    eve : Economic Value of Equity sensitivity to the 6 BCBS shocks.
    nii : Net Interest Income sensitivity.
    outlier_test : Supervisory Outlier Test (15% Tier 1 EVE / 5% NII).

References:
    - BCBS d368 (Standards on IRRBB, April 2016).
    - EBA/GL/2018/02 and EBA RTS on SOT (EBA/RTS/2022/09).
    - CRR3 Art. 84 (IRRBB outlier test).
"""

from creditriskengine.irrbb.eve import (
    SHOCK_SCENARIOS,
    InterestRateShock,
    eve_sensitivity,
    repricing_gap,
)
from creditriskengine.irrbb.nii import nii_sensitivity
from creditriskengine.irrbb.outlier_test import (
    OutlierTestResult,
    supervisory_outlier_test,
)

__all__ = [
    "InterestRateShock",
    "SHOCK_SCENARIOS",
    "repricing_gap",
    "eve_sensitivity",
    "nii_sensitivity",
    "OutlierTestResult",
    "supervisory_outlier_test",
]
