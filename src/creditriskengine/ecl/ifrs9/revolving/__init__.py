"""IFRS 9 ECL for revolving credit facilities.

Provides behavioral life determination, multi-approach CCF models,
revolving EAD term structures, and ECL calculation with drawn/undrawn
decomposition per IFRS 7 B8E.

Covers credit cards, overdrafts, HELOCs, corporate revolvers, working
capital facilities, and margin lending -- with multi-jurisdiction
provision floors (CBUAE, RBI, MAS, SAMA, APRA).

References:
    - IFRS 9 paragraphs 5.5.20, B5.5.31, B5.5.39-40
    - IFRS 7 paragraph B8E (drawn/undrawn presentation)
    - BCBS d424 (Basel III final reforms)
    - CRR3 (EU Regulation 2024/1623)
"""

from creditriskengine.ecl.ifrs9.revolving.behavioral_life import (
    determine_behavioral_life,
    effective_life_months,
    segment_behavioral_life,
)
from creditriskengine.ecl.ifrs9.revolving.ccf import (
    airb_ccf_floor,
    apply_ccf_with_floor,
    behavioral_ccf,
    ccf_pit_adjustment,
    eadf_ccf,
    regulatory_ccf_firb,
    regulatory_ccf_sa,
)
from creditriskengine.ecl.ifrs9.revolving.ead_profile import (
    RevolvingEADProfile,
    ead_drawn_undrawn_split,
    revolving_ead_term_structure,
)
from creditriskengine.ecl.ifrs9.revolving.ecl_revolving import (
    RevolvingECLResult,
    calculate_revolving_ecl,
    revolving_ecl_scenario_weighted,
)
from creditriskengine.ecl.ifrs9.revolving.product_config import (
    PRODUCT_CONFIGS,
    RevolvingProductConfig,
    get_product_config,
)
from creditriskengine.ecl.ifrs9.revolving.provision_floors import (
    ProvisionFloor,
    apply_provision_floor,
    get_provision_floors,
)
from creditriskengine.ecl.ifrs9.revolving.types import (
    BehavioralLifeMethod,
    CCFMethod,
    RevolvingProductType,
)

__all__ = [
    # Types
    "BehavioralLifeMethod",
    "CCFMethod",
    "RevolvingProductType",
    # Product config
    "PRODUCT_CONFIGS",
    "RevolvingProductConfig",
    "get_product_config",
    # Behavioral life
    "determine_behavioral_life",
    "effective_life_months",
    "segment_behavioral_life",
    # CCF
    "airb_ccf_floor",
    "apply_ccf_with_floor",
    "behavioral_ccf",
    "ccf_pit_adjustment",
    "eadf_ccf",
    "regulatory_ccf_firb",
    "regulatory_ccf_sa",
    # EAD profile
    "RevolvingEADProfile",
    "ead_drawn_undrawn_split",
    "revolving_ead_term_structure",
    # ECL engine
    "RevolvingECLResult",
    "calculate_revolving_ecl",
    "revolving_ecl_scenario_weighted",
    # Provision floors
    "ProvisionFloor",
    "apply_provision_floor",
    "get_provision_floors",
]
