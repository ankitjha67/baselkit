"""ESG ratings integration for credit risk.

Modules:
    ratings : Vendor-agnostic ESG rating → PD adjustment adapter.

References:
    - MSCI ESG Ratings (AAA-CCC, industry-relative).
    - Sustainalytics ESG Risk Rating (0-40+, absolute, higher = worse).
    - S&P Global ESG scores.
    - EBA Report on ESG Risks Management (October 2023).
"""

from creditriskengine.esg.ratings import (
    ESGProvider,
    esg_pd_multiplier,
    normalise_esg_score,
)

__all__ = [
    "ESGProvider",
    "normalise_esg_score",
    "esg_pd_multiplier",
]
