"""AI/ML model governance and responsible AI validation.

References:
    - EU AI Act (Regulation 2024/1689), in force August 1, 2024.
    - PRA SS1/23 Model Risk Management Principles (May 2024).
    - Fed SR 11-7 / OCC 2011-12 Model Risk Management.
    - CFPB Circulars 2022-03, 2023-03 — adverse action with ML.
    - MAS FEAT Principles / Veritas Toolkit 2.0 (June 2023).
    - NIST AI RMF 1.0 (January 2023).
    - CJEU SCHUFA Ruling C-634/21 (December 2023) — GDPR Art. 22.
"""

from creditriskengine.validation.ai_governance.drift import (
    detect_psi_drift,
    psi,
)
from creditriskengine.validation.ai_governance.fairness import (
    demographic_parity_difference,
    disparate_impact_ratio,
    equal_opportunity_difference,
)

__all__ = [
    "demographic_parity_difference",
    "disparate_impact_ratio",
    "equal_opportunity_difference",
    "psi",
    "detect_psi_drift",
]
