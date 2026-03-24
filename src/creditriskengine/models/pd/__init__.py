"""PD (Probability of Default) modeling framework.

Provides logistic scorecard building, WoE/IV binning, master scale
construction, and PD calibration utilities.
"""

from creditriskengine.models.pd.binning import (
    BinResult,
    apply_woe_transform,
    calculate_iv,
    calculate_woe,
    monotonic_binning,
    optimal_binning,
    quantile_binning,
)
from creditriskengine.models.pd.scorecard import (
    ScorecardBuilder,
    build_master_scale,
    calibrate_pd_anchor_point,
    score_to_pd,
    scorecard_to_pd,
)
from creditriskengine.models.pd.structural import (
    distance_to_default,
    implied_asset_value,
    merton_default_probability,
)
from creditriskengine.models.pd.term_structure import (
    forward_pd,
    interpolate_pd_term_structure,
    pd_term_structure_from_hazard,
    pd_term_structure_from_transitions,
)
from creditriskengine.models.pd.transition_matrix import (
    default_column,
    estimate_transition_matrix,
    generator_matrix,
    multi_period_transition_matrix,
    validate_transition_matrix,
)
from creditriskengine.models.pd.zscore import (
    altman_z_score,
    altman_z_score_emerging,
    altman_z_score_private,
    z_score_zone,
)

__all__ = [
    "ScorecardBuilder",
    "build_master_scale",
    "calibrate_pd_anchor_point",
    "score_to_pd",
    "scorecard_to_pd",
    "BinResult",
    "calculate_woe",
    "calculate_iv",
    "quantile_binning",
    "monotonic_binning",
    "optimal_binning",
    "apply_woe_transform",
    # Term structure
    "pd_term_structure_from_hazard",
    "pd_term_structure_from_transitions",
    "interpolate_pd_term_structure",
    "forward_pd",
    # Transition matrix
    "estimate_transition_matrix",
    "multi_period_transition_matrix",
    "generator_matrix",
    "validate_transition_matrix",
    "default_column",
    # Structural model
    "distance_to_default",
    "merton_default_probability",
    "implied_asset_value",
    # Z-Score
    "altman_z_score",
    "altman_z_score_private",
    "altman_z_score_emerging",
    "z_score_zone",
]
