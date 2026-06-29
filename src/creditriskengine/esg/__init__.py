"""ESG ratings integration and ESG risk management for credit risk.

Modules:
    ratings         : Vendor-agnostic ESG rating -> PD adjustment adapter.
    risk_management : EBA/GL/2025/01 ESG materiality assessment and
                      CRD Art. 76(2) transition-plan monitoring.

References:
    - MSCI ESG Ratings (AAA-CCC, industry-relative).
    - Sustainalytics ESG Risk Rating (0-40+, absolute, higher = worse).
    - S&P Global ESG scores.
    - EBA Report on ESG Risks Management (October 2023).
    - EBA Guidelines on the management of ESG risks (EBA/GL/2025/01),
      applicable from 11 January 2026.
"""

from creditriskengine.esg.ratings import (
    ESGProvider,
    esg_pd_multiplier,
    normalise_esg_score,
)
from creditriskengine.esg.risk_management import (
    ESGFactor,
    ESGRiskDriver,
    MaterialityLevel,
    MaterialityMethod,
    MaterialityResult,
    TimeHorizon,
    TransitionPlanStatus,
    assess_esg_materiality,
    recommended_method,
    transition_plan_alignment,
)

__all__ = [
    "ESGProvider",
    "normalise_esg_score",
    "esg_pd_multiplier",
    "TimeHorizon",
    "MaterialityMethod",
    "ESGFactor",
    "MaterialityLevel",
    "ESGRiskDriver",
    "MaterialityResult",
    "TransitionPlanStatus",
    "recommended_method",
    "assess_esg_materiality",
    "transition_plan_alignment",
]
