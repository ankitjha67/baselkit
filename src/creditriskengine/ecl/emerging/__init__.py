"""Emerging-market asset classification frameworks.

Modules:
    china     : NFRA five-tier risk classification.
    indonesia : OJK five-tier collectability + provisioning.

References:
    - NFRA Measures for the Risk Classification of Financial Assets of
      Commercial Banks (February 2023, effective July 1, 2023).
    - OJK POJK 40/POJK.03/2019 (asset quality assessment).
"""

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
    "NFRAFiveTier",
    "classify_nfra_five_tier",
    "nfra_tier_to_ifrs9_stage",
    "OJKCollectability",
    "classify_ojk_collectability",
    "ojk_minimum_provision",
    "ojk_to_ifrs9_stage",
]
