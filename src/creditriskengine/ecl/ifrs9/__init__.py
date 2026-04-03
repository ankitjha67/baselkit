"""IFRS 9 expected credit loss framework.

Provides staging, SICR assessment, lifetime PD term structures,
scenario weighting, 12-month/lifetime ECL calculation, and revolving
credit ECL with behavioral life, CCF models, and drawn/undrawn split.
"""

from creditriskengine.ecl.ifrs9.ecl_calc import calculate_ecl, ecl_12_month, ecl_lifetime
from creditriskengine.ecl.ifrs9.lifetime_pd import (
    cumulative_pd_from_annual,
    flat_pd_term_structure,
    marginal_pd_from_cumulative,
)
from creditriskengine.ecl.ifrs9.overlays import (
    ManagementOverlay,
    OverlayResult,
    OverlayType,
    apply_overlays,
    overlay_impact_summary,
    validate_overlay,
)
from creditriskengine.ecl.ifrs9.revolving import (
    RevolvingECLResult,
    RevolvingProductType,
    calculate_revolving_ecl,
    determine_behavioral_life,
    regulatory_ccf_sa,
    revolving_ecl_scenario_weighted,
)
from creditriskengine.ecl.ifrs9.scenarios import (
    Scenario,
    ScenarioSetMetadata,
    SensitivityResult,
    scenario_sensitivity_analysis,
    validate_scenario_governance,
    weighted_ecl,
)
from creditriskengine.ecl.ifrs9.sicr import assess_sicr
from creditriskengine.ecl.ifrs9.staging import assign_stage
from creditriskengine.ecl.ifrs9.ttc_to_pit import ttc_to_pit_pd

__all__ = [
    "calculate_ecl",
    "ecl_12_month",
    "ecl_lifetime",
    "assign_stage",
    "assess_sicr",
    "cumulative_pd_from_annual",
    "marginal_pd_from_cumulative",
    "flat_pd_term_structure",
    "ttc_to_pit_pd",
    "Scenario",
    "weighted_ecl",
    # Scenario governance
    "ScenarioSetMetadata",
    "SensitivityResult",
    "validate_scenario_governance",
    "scenario_sensitivity_analysis",
    # Management overlays
    "ManagementOverlay",
    "OverlayType",
    "OverlayResult",
    "apply_overlays",
    "overlay_impact_summary",
    "validate_overlay",
    # Revolving credit ECL
    "calculate_revolving_ecl",
    "revolving_ecl_scenario_weighted",
    "RevolvingECLResult",
    "RevolvingProductType",
    "determine_behavioral_life",
    "regulatory_ccf_sa",
]
