"""Ind AS 109 expected credit loss framework (Indian GAAP, converged with IFRS 9).

Wraps IFRS 9 ECL functions with India-specific defaults (RBI norms),
including IRAC asset classification, provisioning floors, restructured
account handling, and the RBI ECL Master Direction 2026 framework
(RBI/DOR/2026-27/398, effective April 1, 2027).
"""

from creditriskengine.ecl.ind_as109.borrower_classification import (
    apply_borrower_level_staging,
)
from creditriskengine.ecl.ind_as109.collateral_valuation import (
    validate_collateral_revaluation,
)
from creditriskengine.ecl.ind_as109.dlg import DLGAdjustment, ecl_with_dlg
from creditriskengine.ecl.ind_as109.ind_as_ecl import (
    RBI_AGRI_SHORT_CROP_DPD,
    RBI_DEFAULT_DPD_THRESHOLD,
    RBI_PROVISION_RATES,
    RBI_REVOLVING_SICR_OVERLIMIT_DAYS,
    RBI_SICR_DPD_BACKSTOP,
    IRACAssetClass,
    assess_sicr_rbi,
    assign_stage_ind_as,
    calculate_ecl_ind_as,
    calculate_ecl_ind_as_2026,
    calculate_ecl_ind_as_auto,
    classify_irac,
    determine_upgrade_eligibility,
    irac_to_ifrs9_stage,
    rbi_minimum_provision,
    restructured_account_stage,
)

# IRACP standard-asset provisioning + Resolution Framework + Out-of-order CC/OD
from creditriskengine.ecl.ind_as109.iracp import (
    IRACP_STANDARD_RATES,
    RF_RESTRUCTURED_ADDON_RATE,
    RF_SLIPPAGE_ADDON_RATE,
    StandardAssetSector,
    is_out_of_order,
    resolution_framework_addon,
    standard_asset_provision,
)

# NBFC backstop + SBR glide-path + NBFC-UL rates
from creditriskengine.ecl.ind_as109.nbfc_backstop import (
    NBFC_UL_STANDARD_RATES,
    NBFCBackstopResult,
    NBFCLayer,
    apply_nbfc_backstop,
    nbfc_ul_standard_asset_provision,
    npa_dpd_threshold,
)
from creditriskengine.ecl.ind_as109.pd_lgd_floors import (
    RBI_LGD_BACKSTOP_SECURED,
    RBI_LGD_BACKSTOP_UNSECURED,
    RBI_LGD_ELIGIBLE_COLLATERAL,
    RBI_PD_FLOOR,
    apply_rbi_lgd_backstop,
    apply_rbi_pd_floor,
)
from creditriskengine.ecl.ind_as109.provision_floors_2026 import (
    RBI_DCCO_INFRA_QUARTERLY_RATE,
    RBI_DCCO_NON_INFRA_QUARTERLY_RATE,
    RBI_ECL_FLOOR_STAGE_1_2,
    classify_rbi_exposure_category,
    collateral_category_for,
    dcco_additional_provision,
    rbi_ecl_floor_2026,
)
from creditriskengine.ecl.ind_as109.transition import (
    CAPITAL_ADD_BACK_SCHEDULE,
    RBI_ECL_EFFECTIVE_DATE,
    RBI_EIR_MIGRATION_DEADLINE,
    capital_add_back_factor,
    eir_required,
    is_ecl_framework_effective,
)
from creditriskengine.ecl.ind_as109.types import (
    RBICollateralCategory,
    RBIExposureCategory,
)

__all__ = [
    # Legacy IRAC framework
    "assign_stage_ind_as",
    "calculate_ecl_ind_as",
    "IRACAssetClass",
    "classify_irac",
    "irac_to_ifrs9_stage",
    "rbi_minimum_provision",
    "restructured_account_stage",
    "RBI_PROVISION_RATES",
    "RBI_DEFAULT_DPD_THRESHOLD",
    "RBI_SICR_DPD_BACKSTOP",
    "RBI_AGRI_SHORT_CROP_DPD",
    # RBI ECL Master Direction 2026
    "RBIExposureCategory",
    "RBICollateralCategory",
    "RBI_ECL_EFFECTIVE_DATE",
    "RBI_EIR_MIGRATION_DEADLINE",
    "RBI_PD_FLOOR",
    "RBI_LGD_BACKSTOP_SECURED",
    "RBI_LGD_BACKSTOP_UNSECURED",
    "RBI_LGD_ELIGIBLE_COLLATERAL",
    "RBI_DCCO_INFRA_QUARTERLY_RATE",
    "RBI_DCCO_NON_INFRA_QUARTERLY_RATE",
    "RBI_REVOLVING_SICR_OVERLIMIT_DAYS",
    "RBI_ECL_FLOOR_STAGE_1_2",
    "CAPITAL_ADD_BACK_SCHEDULE",
    "apply_rbi_pd_floor",
    "apply_rbi_lgd_backstop",
    "rbi_ecl_floor_2026",
    "classify_rbi_exposure_category",
    "collateral_category_for",
    "dcco_additional_provision",
    "apply_borrower_level_staging",
    "validate_collateral_revaluation",
    "DLGAdjustment",
    "ecl_with_dlg",
    "capital_add_back_factor",
    "is_ecl_framework_effective",
    "eir_required",
    "assess_sicr_rbi",
    "determine_upgrade_eligibility",
    "calculate_ecl_ind_as_2026",
    "calculate_ecl_ind_as_auto",
    # IRACP standard-asset provisioning
    "StandardAssetSector",
    "IRACP_STANDARD_RATES",
    "standard_asset_provision",
    "RF_RESTRUCTURED_ADDON_RATE",
    "RF_SLIPPAGE_ADDON_RATE",
    "resolution_framework_addon",
    "is_out_of_order",
    # NBFC backstop
    "NBFCLayer",
    "NBFCBackstopResult",
    "NBFC_UL_STANDARD_RATES",
    "apply_nbfc_backstop",
    "nbfc_ul_standard_asset_provision",
    "npa_dpd_threshold",
]
