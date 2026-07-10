"""Emerging-market asset classification frameworks.

Modules:
    china     : NFRA five-tier risk classification.
    indonesia : OJK five-tier collectability + provisioning.
    brazil    : CMN 4.966 three-stage ECL classification.

References:
    - NFRA Measures for the Risk Classification of Financial Assets of
      Commercial Banks (February 2023, effective July 1, 2023).
    - OJK POJK 40/POJK.03/2019 (asset quality assessment).
    - Resolucao CMN 4.966/21 (Brazil ECL, effective 1 January 2025).
"""

from creditriskengine.ecl.emerging.brazil import (
    CMN_4966_DEFAULT_DPD,
    CMN_4966_SICR_DPD_BACKSTOP,
    classify_cmn_4966_stage,
    uses_simplified_model,
)
from creditriskengine.ecl.emerging.china import (
    NFRAFiveTier,
    classify_nfra_five_tier,
    nfra_tier_to_ifrs9_stage,
)
from creditriskengine.ecl.emerging.indonesia import (
    OJKCollectability,
    classify_ojk_collectability,
    ojk_minimum_provision,
    ojk_to_ifrs9_stage,
)

__all__ = [
    "CMN_4966_SICR_DPD_BACKSTOP",
    "CMN_4966_DEFAULT_DPD",
    "classify_cmn_4966_stage",
    "uses_simplified_model",
    "NFRAFiveTier",
    "classify_nfra_five_tier",
    "nfra_tier_to_ifrs9_stage",
    "OJKCollectability",
    "classify_ojk_collectability",
    "ojk_minimum_provision",
    "ojk_to_ifrs9_stage",
]
