"""Credit Conversion Factor models for revolving credit.

Supports four CCF approaches:

1. **Regulatory SA** -- Basel III/CRR3 standardized CCFs with
   jurisdiction-specific values (including APRA's 40% outlier).
2. **Regulatory F-IRB** -- Supervisory CCFs per BCBS CRE32.29-32.32.
3. **Behavioral** -- PIT bank-estimated CCFs from default-weighted
   drawdown observations.
4. **EADF** -- EAD Factor approach avoiding LEQ singularity.

IFRS 9 requires CCFs consistent with expectations of drawdowns
(B5.5.31) and reflecting forward-looking conditions -- banks cannot
simply adopt Basel regulatory CCFs without PIT adjustment.

References:
    - BCBS d424 (December 2017), CRE32.29-32.32
    - CRR3 (EU Regulation 2024/1623), Art. 166(8b), Art. 495d
    - APRA APS 112 (40% CCF for UCCs)
    - Araten & Jacobs (2001) on LEQ methodology
    - Tong et al. (2016) on bimodal CCF distribution
"""

from __future__ import annotations

import numpy as np

from creditriskengine.core.types import Jurisdiction
from creditriskengine.ecl.ifrs9.revolving.types import (
    CCFMethod,
    RevolvingProductType,
)

# -----------------------------------------------------------------------
# Basel III / CRR3 Standardized CCFs (per BCBS d424 Table 2)
# -----------------------------------------------------------------------

_SA_CCFS: dict[str, float] = {
    "unconditionally_cancellable": 0.10,
    "committed_any_maturity": 0.40,
    "nif_ruf": 0.50,
    "transaction_related": 0.50,
    "trade_related": 0.20,
    "direct_credit_substitutes": 1.00,
}

# Jurisdiction-specific overrides
_SA_CCF_OVERRIDES: dict[Jurisdiction, dict[str, float]] = {
    Jurisdiction.AUSTRALIA: {
        "unconditionally_cancellable": 0.40,
    },
}

# CRR3 transitional: 0% permitted until 31 Dec 2029 for UCCs
_CRR3_TRANSITIONAL_UCC_CCF: float = 0.0

# F-IRB supervisory CCFs (aligned with SA for revolving)
_FIRB_CCFS: dict[str, float] = {
    "unconditionally_cancellable": 0.40,
    "committed_other": 0.75,
    "transaction_related": 0.50,
    "trade_related": 0.20,
    "nif": 0.75,
    "direct_credit_substitutes": 1.00,
}

# A-IRB CCF input floor: 50% of applicable SA CCF (CRR3 Art. 166(8b))
AIRB_CCF_FLOOR_FACTOR: float = 0.50


def _product_to_facility(product_type: RevolvingProductType) -> str:
    """Map product type to SA facility classification."""
    facility_map: dict[RevolvingProductType, str] = {
        RevolvingProductType.CREDIT_CARD: "unconditionally_cancellable",
        RevolvingProductType.OVERDRAFT: "unconditionally_cancellable",
        RevolvingProductType.HELOC: "committed_any_maturity",
        RevolvingProductType.CORPORATE_REVOLVER: "committed_any_maturity",
        RevolvingProductType.WORKING_CAPITAL: "committed_any_maturity",
        RevolvingProductType.MARGIN_LENDING: "committed_any_maturity",
    }
    return facility_map.get(product_type, "committed_any_maturity")


def regulatory_ccf_sa(
    product_type: RevolvingProductType,
    jurisdiction: Jurisdiction = Jurisdiction.BCBS,
    use_crr3_transitional: bool = False,
) -> float:
    """Return the Basel III / CRR3 Standardized Approach CCF.

    Args:
        product_type: Revolving product classification.
        jurisdiction: Reporting jurisdiction.  APRA uses 40% for UCCs.
        use_crr3_transitional: If True and jurisdiction is EU, returns
            the CRR3 transitional 0% CCF for UCCs (valid until 2029).

    Returns:
        Standardized CCF as a decimal (e.g., 0.10 = 10%).
    """
    facility = _product_to_facility(product_type)

    if (
        use_crr3_transitional
        and jurisdiction == Jurisdiction.EU
        and facility == "unconditionally_cancellable"
    ):
        return _CRR3_TRANSITIONAL_UCC_CCF

    overrides = _SA_CCF_OVERRIDES.get(jurisdiction, {})
    if facility in overrides:
        return overrides[facility]

    return _SA_CCFS.get(facility, 0.40)


def regulatory_ccf_firb(
    product_type: RevolvingProductType,
) -> float:
    """Return the Foundation IRB supervisory CCF.

    Args:
        product_type: Revolving product classification.

    Returns:
        F-IRB CCF as a decimal.
    """
    facility = _product_to_facility(product_type)
    firb_key = {
        "unconditionally_cancellable": "unconditionally_cancellable",
        "committed_any_maturity": "committed_other",
    }.get(facility, "committed_other")
    return _FIRB_CCFS.get(firb_key, 0.75)


def airb_ccf_floor(
    product_type: RevolvingProductType,
    jurisdiction: Jurisdiction = Jurisdiction.BCBS,
) -> float:
    """Return the A-IRB CCF input floor (50% of SA CCF).

    Per CRR3 Art. 166(8b), A-IRB own-estimate CCFs must be at least
    50% of the applicable SA-CCF.

    Args:
        product_type: Revolving product classification.
        jurisdiction: Reporting jurisdiction.

    Returns:
        Minimum permissible A-IRB CCF.
    """
    sa_ccf = regulatory_ccf_sa(product_type, jurisdiction)
    return sa_ccf * AIRB_CCF_FLOOR_FACTOR


def behavioral_ccf(
    ead_at_default: np.ndarray,
    drawn_at_observation: np.ndarray,
    undrawn_at_observation: np.ndarray,
) -> float:
    """Estimate behavioral CCF from observed defaults (LEQ method).

    CCF = (EAD_default - Drawn_observation) / Undrawn_observation

    Per Araten & Jacobs (2001), the observation is typically 12 months
    before default.  The result is the mean CCF across the sample,
    clipped to [0, 1].

    Args:
        ead_at_default: Array of EAD amounts at default.
        drawn_at_observation: Array of drawn amounts at observation date.
        undrawn_at_observation: Array of undrawn amounts at observation.

    Returns:
        Mean behavioral CCF clipped to [0, 1].

    Raises:
        ValueError: If input arrays have different lengths or are empty.
    """
    ead_arr = np.asarray(ead_at_default, dtype=float)
    drawn_arr = np.asarray(drawn_at_observation, dtype=float)
    undrawn_arr = np.asarray(undrawn_at_observation, dtype=float)

    if ead_arr.shape != drawn_arr.shape or ead_arr.shape != undrawn_arr.shape:
        raise ValueError("Input arrays must have the same length.")
    if ead_arr.size == 0:
        raise ValueError("Input arrays must not be empty.")

    mask = undrawn_arr > 0
    if not np.any(mask):
        return 0.0

    ccfs = (ead_arr[mask] - drawn_arr[mask]) / undrawn_arr[mask]
    return float(np.clip(np.mean(ccfs), 0.0, 1.0))


def eadf_ccf(
    ead_at_default: np.ndarray,
    limit_at_observation: np.ndarray,
) -> float:
    """Estimate CCF using the EAD Factor (EADF) approach.

    EADF = EAD_default / Limit_observation

    This avoids the singularity problem in the LEQ approach when the
    undrawn amount approaches zero.

    Args:
        ead_at_default: Array of EAD amounts at default.
        limit_at_observation: Array of credit limits at observation.

    Returns:
        Mean EADF clipped to [0, 1].

    Raises:
        ValueError: If inputs have different lengths or are empty.
    """
    ead_arr = np.asarray(ead_at_default, dtype=float)
    limit_arr = np.asarray(limit_at_observation, dtype=float)

    if ead_arr.shape != limit_arr.shape:
        raise ValueError("Input arrays must have the same length.")
    if ead_arr.size == 0:
        raise ValueError("Input arrays must not be empty.")

    mask = limit_arr > 0
    if not np.any(mask):
        return 0.0

    eadfs = ead_arr[mask] / limit_arr[mask]
    return float(np.clip(np.mean(eadfs), 0.0, 1.0))


def ccf_pit_adjustment(
    ttc_ccf: float,
    z_factor: float,
    sensitivity: float = 0.5,
) -> float:
    """Convert a TTC CCF to a PIT CCF using macro conditions.

    IFRS 9 requires CCFs reflecting current and forecast conditions
    (B5.5.31), not through-the-cycle averages.  This applies a simple
    linear adjustment:

        CCF_PIT = CCF_TTC × (1 + sensitivity × z_factor)

    where z_factor > 0 indicates deteriorating conditions (higher
    expected drawdowns) and z_factor < 0 indicates improving conditions.

    Args:
        ttc_ccf: Through-the-cycle CCF (e.g., 0.50).
        z_factor: Macroeconomic index (standard normal).  Positive
            values indicate stress.
        sensitivity: Elasticity of CCF to macro conditions (default 0.5).

    Returns:
        PIT-adjusted CCF clipped to [0, 1].
    """
    pit = ttc_ccf * (1.0 + sensitivity * z_factor)
    return float(np.clip(pit, 0.0, 1.0))


def apply_ccf_with_floor(
    ccf: float,
    product_type: RevolvingProductType,
    method: CCFMethod,
    jurisdiction: Jurisdiction = Jurisdiction.BCBS,
) -> float:
    """Apply jurisdiction-aware CCF floor.

    For A-IRB behavioral estimates, the floor is 50% of the applicable
    SA CCF.  For regulatory approaches, the CCF is returned unchanged.

    Args:
        ccf: Input CCF value.
        product_type: Revolving product classification.
        method: CCF estimation method.
        jurisdiction: Reporting jurisdiction.

    Returns:
        CCF with applicable floor applied.
    """
    if method in (CCFMethod.BEHAVIORAL, CCFMethod.EADF):
        floor = airb_ccf_floor(product_type, jurisdiction)
        return max(ccf, floor)
    return ccf
