"""Climate & ESG risk analytics package.

Modules:
    ngfs_scenarios  : NGFS Phase V climate scenario library.
    physical_risk   : Physical risk PD/LGD adjustments.
    transition_risk : Carbon-price-driven PD impact.
    financed_emissions : PCAF methodology for Scope 1/2/3.
    green_asset_ratio  : EBA GAR and BTAR calculation.
    crypto           : BCBS SCO60 crypto-asset capital.

References:
    - NGFS Climate Scenarios, Phase V (November 2024).
    - BCBS Principles for Management of Climate-Related Financial Risks (June 2022).
    - BCBS d597 Climate Disclosure Framework (June 2025).
    - EBA/GL/2025/01 ESG Risk Management Guidelines (January 9, 2025).
    - EBA ITS Pillar 3 ESG Disclosures (Reg 2022/2453).
    - PCAF Global GHG Accounting Standard for the Financial Industry (2022).
    - BCBS SCO60 Prudential Treatment of Cryptoasset Exposures (July 2024).
"""

from creditriskengine.climate.crypto import (
    CryptoAssetGroup,
    CryptoCapitalResult,
    crypto_asset_rwa,
)
from creditriskengine.climate.financed_emissions import (
    PCAFScore,
    financed_emissions,
)
from creditriskengine.climate.green_asset_ratio import green_asset_ratio
from creditriskengine.climate.ngfs_scenarios import (
    NGFSScenario,
    get_ngfs_scenario,
    list_ngfs_scenarios,
)
from creditriskengine.climate.physical_risk import physical_risk_lgd_haircut
from creditriskengine.climate.transition_risk import transition_risk_pd_multiplier

__all__ = [
    "NGFSScenario",
    "get_ngfs_scenario",
    "list_ngfs_scenarios",
    "physical_risk_lgd_haircut",
    "transition_risk_pd_multiplier",
    "financed_emissions",
    "PCAFScore",
    "green_asset_ratio",
    "CryptoAssetGroup",
    "CryptoCapitalResult",
    "crypto_asset_rwa",
]
