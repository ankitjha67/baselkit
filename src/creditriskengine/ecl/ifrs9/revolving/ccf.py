"""Credit Conversion Factor models for revolving credit.

Delegates regulatory CCF lookups to the canonical tables in
:mod:`creditriskengine.models.ead.ead_model` and adds revolving-
specific behavioral estimation and PIT adjustment on top.

Supports four CCF approaches:

1. **Regulatory SA** -- via :func:`ead_model.get_sa_ccf` with
   jurisdiction overrides (APRA 40%, CRR3 transitional).
2. **Regulatory F-IRB** -- via :func:`ead_model.get_supervisory_ccf`.
3. **Behavioral** -- PIT bank-estimated CCFs from default-weighted
   drawdown observations (LEQ method per Araten & Jacobs 2001).
4. **EADF** -- EAD Factor approach avoiding LEQ singularity.

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
from creditriskengine.models.ead.ead_model import (
    get_airb_ccf_floor,
    get_sa_ccf,
    get_supervisory_ccf,
)

# -----------------------------------------------------------------------
# Product type → facility type mapping
# -----------------------------------------------------------------------

_PRODUCT_TO_SA_FACILITY: dict[RevolvingProductType, str] = {
    RevolvingProductType.CREDIT_CARD: "unconditionally_cancellable",
    RevolvingProductType.OVERDRAFT: "unconditionally_cancellable",
    RevolvingProductType.HELOC: "committed_any_maturity",
    RevolvingProductType.CORPORATE_REVOLVER: "committed_any_maturity",
    RevolvingProductType.WORKING_CAPITAL: "committed_any_maturity",
    RevolvingProductType.MARGIN_LENDING: "committed_any_maturity",
}

_PRODUCT_TO_FIRB_FACILITY: dict[RevolvingProductType, str] = {
    RevolvingProductType.CREDIT_CARD: "committed_unconditionally_cancellable",
    RevolvingProductType.OVERDRAFT: "committed_unconditionally_cancellable",
    RevolvingProductType.HELOC: "committed_other",
    RevolvingProductType.CORPORATE_REVOLVER: "committed_other",
    RevolvingProductType.WORKING_CAPITAL: "committed_other",
    RevolvingProductType.MARGIN_LENDING: "committed_other",
}

_JURISDICTION_TO_CODE: dict[Jurisdiction, str] = {
    Jurisdiction.BCBS: "bcbs",
    Jurisdiction.EU: "eu",
    Jurisdiction.UK: "uk",
    Jurisdiction.US: "us",
    Jurisdiction.INDIA: "india",
    Jurisdiction.SINGAPORE: "singapore",
    Jurisdiction.HONG_KONG: "hong_kong",
    Jurisdiction.JAPAN: "japan",
    Jurisdiction.AUSTRALIA: "australia",
    Jurisdiction.CANADA: "canada",
    Jurisdiction.UAE: "uae",
    Jurisdiction.SAUDI_ARABIA: "saudi_arabia",
}


# -----------------------------------------------------------------------
# Regulatory CCF lookups (delegate to ead_model)
# -----------------------------------------------------------------------

def regulatory_ccf_sa(
    product_type: RevolvingProductType,
    jurisdiction: Jurisdiction = Jurisdiction.BCBS,
    use_crr3_transitional: bool = False,
) -> float:
    """Return the Basel III / CRR3 Standardized Approach CCF.

    Delegates to :func:`models.ead.ead_model.get_sa_ccf` -- the single
    source of truth for SA CCF tables and jurisdiction overrides.

    Args:
        product_type: Revolving product classification.
        jurisdiction: Reporting jurisdiction.
        use_crr3_transitional: If True and jurisdiction is EU, returns
            CRR3 transitional 0% CCF for UCCs (valid until 2029).

    Returns:
        Standardized CCF as a decimal (e.g., 0.10 = 10%).
    """
    facility = _PRODUCT_TO_SA_FACILITY.get(
        product_type, "committed_any_maturity"
    )
    jur_code = _JURISDICTION_TO_CODE.get(jurisdiction, "bcbs")
    return get_sa_ccf(facility, jur_code, use_crr3_transitional)


def regulatory_ccf_firb(
    product_type: RevolvingProductType,
) -> float:
    """Return the Foundation IRB supervisory CCF.

    Delegates to :func:`models.ead.ead_model.get_supervisory_ccf`.

    Args:
        product_type: Revolving product classification.

    Returns:
        F-IRB CCF as a decimal.
    """
    facility = _PRODUCT_TO_FIRB_FACILITY.get(
        product_type, "committed_other"
    )
    return get_supervisory_ccf(facility)


def airb_ccf_floor(
    product_type: RevolvingProductType,
    jurisdiction: Jurisdiction = Jurisdiction.BCBS,
) -> float:
    """Return the A-IRB CCF input floor (50% of SA CCF).

    Delegates to :func:`models.ead.ead_model.get_airb_ccf_floor`.

    Args:
        product_type: Revolving product classification.
        jurisdiction: Reporting jurisdiction.

    Returns:
        Minimum permissible A-IRB CCF.
    """
    facility = _PRODUCT_TO_SA_FACILITY.get(
        product_type, "committed_any_maturity"
    )
    jur_code = _JURISDICTION_TO_CODE.get(jurisdiction, "bcbs")
    return get_airb_ccf_floor(facility, jur_code)


# -----------------------------------------------------------------------
# Behavioral CCF estimation
# -----------------------------------------------------------------------

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

    Avoids the singularity problem in LEQ when undrawn approaches zero.

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


# -----------------------------------------------------------------------
# PIT adjustment
# -----------------------------------------------------------------------

def ccf_pit_adjustment(
    ttc_ccf: float,
    z_factor: float,
    sensitivity: float = 0.5,
) -> float:
    """Convert a TTC CCF to a PIT CCF using macro conditions.

    IFRS 9 requires CCFs reflecting current and forecast conditions
    (B5.5.31), not through-the-cycle averages.

        CCF_PIT = CCF_TTC x (1 + sensitivity x z_factor)

    Args:
        ttc_ccf: Through-the-cycle CCF (e.g., 0.50).
        z_factor: Macroeconomic index (standard normal).  Positive
            values indicate stress (higher expected drawdowns).
        sensitivity: Elasticity of CCF to macro conditions (default 0.5).

    Returns:
        PIT-adjusted CCF clipped to [0, 1].
    """
    pit = ttc_ccf * (1.0 + sensitivity * z_factor)
    return float(np.clip(pit, 0.0, 1.0))


# -----------------------------------------------------------------------
# Floor application
# -----------------------------------------------------------------------

def apply_ccf_with_floor(
    ccf: float,
    product_type: RevolvingProductType,
    method: CCFMethod,
    jurisdiction: Jurisdiction = Jurisdiction.BCBS,
) -> float:
    """Apply jurisdiction-aware CCF floor.

    For behavioral/EADF estimates, the floor is 50% of the applicable
    SA CCF (A-IRB input floor per CRR3 Art. 166(8b)).  For regulatory
    approaches, the CCF is returned unchanged.

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
