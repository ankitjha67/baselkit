"""Credit Valuation Adjustment risk charge — BCBS d424, CVA25-27.

Implements:
- BA-CVA: Basic Approach (CVA25)
- SA-CVA: Standardised Approach (CVA26)

BA-CVA formula (CVA25.1)::

    K_BA-CVA = beta * K_reduced + sqrt(1 - beta**2) * K_hedged

    K_reduced = sqrt(
        sum((SCVAc - SNHc)**2 for c in counterparties)
        + rho**2 * (sum(SCVAc - SNHc for c in counterparties))**2
    )

    K_hedged = sqrt(
        sum(SCVAc**2 for c in counterparties)
        + rho**2 * (sum(SCVAc for c in counterparties))**2
    )

where:
- SCVAc = stand-alone CVA capital for counterparty c
- SNHc = notional of single-name hedges for counterparty c
- rho = supervisory inter-counterparty correlation = 0.5
- beta = supervisory parameter = 0.25
"""

import logging
import math
from dataclasses import dataclass
from typing import Final

logger = logging.getLogger(__name__)


# ============================================================
# Data classes
# ============================================================

@dataclass(frozen=True)
class CVACounterparty:
    """Counterparty data for CVA calculation.

    Attributes:
        counterparty_id: Unique identifier.
        ead: Total EAD across all netting sets with this counterparty.
        credit_spread: Market credit spread in basis points.
        maturity_years: Effective maturity (regulatory, capped at 5y).
        sector: Sector classification for supervisory RW lookup.
        rating: Credit quality step (1-6) if externally rated.
        is_exempt: Whether exempt from CVA charge (sovereigns, MDBs, etc.).
    """

    counterparty_id: str
    ead: float
    credit_spread: float
    maturity_years: float = 2.5
    sector: str = "corporate_ig"
    rating: int | None = None
    is_exempt: bool = False


@dataclass(frozen=True)
class CVAHedge:
    """CVA hedge instrument (e.g. single-name CDS).

    Attributes:
        counterparty_id: Counterparty being hedged.
        notional: Hedge notional amount.
        maturity_years: Remaining maturity of the hedge.
    """

    counterparty_id: str
    notional: float
    maturity_years: float


# ============================================================
# Supervisory parameters — CVA25
# ============================================================

RHO: Final[float] = 0.50   # Inter-counterparty correlation (CVA25.3)
BETA: Final[float] = 0.25  # Supervisory parameter (CVA25.5)

# Supervisory risk weights by sector — CVA25, Table 1
# Values are expressed as decimals (e.g. 0.05 = 5 %).
SECTOR_RW: Final[dict[str, float]] = {
    "sovereign_ig": 0.005,
    "sovereign_hy": 0.02,
    "sovereign_unrated": 0.02,
    "financial_ig": 0.05,
    "financial_hy": 0.12,
    "financial_unrated": 0.12,
    "corporate_ig": 0.03,
    "corporate_hy": 0.07,
    "corporate_unrated": 0.07,
    "basic_materials_ig": 0.015,
    "basic_materials_hy": 0.06,
    "consumer_ig": 0.025,
    "consumer_hy": 0.05,
    "technology_ig": 0.02,
    "technology_hy": 0.06,
    "other_ig": 0.03,
    "other_hy": 0.07,
    "other_unrated": 0.07,
}

# Discount factor scalar — used to approximate the EE profile
_ALPHA: Final[float] = 1.4  # Per BCBS CVA25.7


# ============================================================
# Supervisory discount factor
# ============================================================

def _supervisory_discount_factor(maturity_years: float) -> float:
    """Supervisory discount factor per CVA25.4.

    DF = (1 - exp(-0.05 * M)) / (0.05 * M)

    where M is the effective maturity in years.

    Args:
        maturity_years: Effective maturity in years.

    Returns:
        Discount factor (0, 1].
    """
    if maturity_years <= 0:
        return 1.0
    rate = 0.05
    exponent = rate * maturity_years
    if exponent < 1e-10:
        return 1.0
    return (1.0 - math.exp(-exponent)) / exponent


# ============================================================
# Standalone CVA capital
# ============================================================

def scva_standalone(counterparty: CVACounterparty) -> float:
    """Standalone CVA capital for a single counterparty.

    Per CVA25.2::

        SCVAc = (2/3) * RWc * Mc * EADc * DFc

    where:
    - RWc is the supervisory risk weight for the counterparty's sector
    - Mc is the effective maturity
    - EADc is the exposure at default
    - DFc is the supervisory discount factor

    The factor (2/3) accounts for the regulatory expectation that
    CVA risk is approximately 2/3 of the full mark-to-market loss.

    Args:
        counterparty: Counterparty data.

    Returns:
        Standalone CVA capital charge (monetary amount).
    """
    if counterparty.is_exempt:
        return 0.0

    rw = SECTOR_RW.get(counterparty.sector)
    if rw is None:
        logger.warning(
            "Sector '%s' not found in SECTOR_RW table for counterparty '%s'; "
            "using 'other_unrated' (7%%)",
            counterparty.sector,
            counterparty.counterparty_id,
        )
        rw = SECTOR_RW["other_unrated"]

    df = _supervisory_discount_factor(counterparty.maturity_years)
    scva = (2.0 / 3.0) * rw * counterparty.maturity_years * counterparty.ead * df

    logger.debug(
        "SCVAc for '%s': RW=%.4f, M=%.1f, EAD=%.2f, DF=%.4f -> SCVA=%.2f",
        counterparty.counterparty_id,
        rw,
        counterparty.maturity_years,
        counterparty.ead,
        df,
        scva,
    )
    return scva


# ============================================================
# BA-CVA — CVA25
# ============================================================

def ba_cva_capital(
    counterparties: list[CVACounterparty],
    hedges: list[CVAHedge] | None = None,
) -> float:
    """Calculate BA-CVA capital charge per CVA25.

    The BA-CVA formula aggregates standalone CVA charges across
    counterparties with supervisory correlation::

        K_reduced = sqrt(
            sum((SCVAc - SNHc)**2)
            + rho**2 * (sum(SCVAc - SNHc))**2
        )

        K_hedged = sqrt(
            sum(SCVAc**2)
            + rho**2 * (sum(SCVAc))**2
        )

        K_BA-CVA = beta * K_reduced + sqrt(1 - beta**2) * K_hedged

    ``K_hedged`` captures the systematic component without hedge
    recognition; ``K_reduced`` captures the benefit of single-name
    hedges.

    Args:
        counterparties: List of counterparty exposures.
        hedges: Optional list of single-name CVA hedges.

    Returns:
        BA-CVA capital charge (monetary amount).
    """
    if not counterparties:
        return 0.0

    # Build hedge lookup: counterparty_id -> total effective hedge notional
    hedge_map: dict[str, float] = {}
    if hedges:
        for h in hedges:
            # Adjust hedge notional by supervisory discount factor
            h_df = _supervisory_discount_factor(h.maturity_years)
            h_rw = SECTOR_RW.get("corporate_ig", 0.03)  # default sector RW
            # Effective hedge: (2/3) * RW * M * Notional * DF
            effective = (2.0 / 3.0) * h_rw * h.maturity_years * h.notional * h_df
            hedge_map[h.counterparty_id] = hedge_map.get(h.counterparty_id, 0.0) + effective

    # Compute standalone and net amounts
    scva_values: list[float] = []
    net_values: list[float] = []

    for cp in counterparties:
        if cp.is_exempt:
            continue
        scva_c = scva_standalone(cp)
        snh_c = hedge_map.get(cp.counterparty_id, 0.0)
        scva_values.append(scva_c)
        net_values.append(scva_c - snh_c)

    if not scva_values:
        return 0.0

    # K_reduced — with hedge benefit
    sum_net_sq = sum(v ** 2 for v in net_values)
    sum_net = sum(net_values)
    k_reduced = math.sqrt(sum_net_sq + RHO ** 2 * sum_net ** 2)

    # K_hedged — systematic component (no hedge recognition)
    sum_scva_sq = sum(v ** 2 for v in scva_values)
    sum_scva = sum(scva_values)
    k_hedged = math.sqrt(sum_scva_sq + RHO ** 2 * sum_scva ** 2)

    # BA-CVA capital
    k_ba_cva = BETA * k_reduced + math.sqrt(1.0 - BETA ** 2) * k_hedged

    logger.info(
        "BA-CVA capital: K_reduced=%.2f, K_hedged=%.2f, K_BA-CVA=%.2f "
        "(%d counterparties, %d hedges)",
        k_reduced,
        k_hedged,
        k_ba_cva,
        len(scva_values),
        len(hedges) if hedges else 0,
    )
    return k_ba_cva


# ============================================================
# SA-CVA — CVA26 (simplified)
# ============================================================

# SA-CVA risk factor correlation parameters (CVA26, Table 2)
_SA_CVA_RHO_SAME_SECTOR: Final[float] = 0.75
_SA_CVA_RHO_CROSS_SECTOR: Final[float] = 0.25
_SA_CVA_GAMMA: Final[float] = 0.50  # Cross-bucket correlation

# SA-CVA delta risk weights by bucket — CVA26, Table 3
_SA_CVA_DELTA_RW: Final[dict[str, float]] = {
    "sovereign_ig": 0.005,
    "sovereign_hy": 0.02,
    "financial_ig": 0.04,
    "financial_hy": 0.10,
    "corporate_ig": 0.03,
    "corporate_hy": 0.06,
    "other": 0.05,
}


def sa_cva_capital(
    counterparties: list[CVACounterparty],
    hedges: list[CVAHedge] | None = None,
) -> float:
    """Calculate SA-CVA capital charge per CVA26 (simplified).

    SA-CVA uses a sensitivity-based approach similar to FRTB-SA,
    applied to CVA sensitivities.  This implementation provides a
    simplified version using delta risk charges only.

    The delta CVA risk charge for each bucket *b* is::

        K_b = sqrt(
            sum_i sum_j rho_ij * WS_i * WS_j
        )

    where ``WS_i = RW_i * s_i`` (weighted sensitivity) and ``rho_ij``
    is the prescribed correlation between counterparties *i* and *j*
    within the same bucket.

    Cross-bucket aggregation::

        K_delta = sqrt(sum_b sum_c gamma_bc * K_b * K_c)

    This simplified version groups counterparties by sector and
    applies the intra-bucket and inter-bucket correlations.

    Args:
        counterparties: List of counterparty exposures.
        hedges: Optional hedges (applied as offsetting sensitivities).

    Returns:
        SA-CVA capital charge (monetary amount).
    """
    if not counterparties:
        return 0.0

    # Build hedge map
    hedge_map: dict[str, float] = {}
    if hedges:
        for h in hedges:
            hedge_map[h.counterparty_id] = (
                hedge_map.get(h.counterparty_id, 0.0) + h.notional
            )

    # Group counterparties by sector bucket
    buckets: dict[str, list[float]] = {}
    for cp in counterparties:
        if cp.is_exempt:
            continue

        # Compute CVA sensitivity as proxy: EAD * maturity * DF
        df = _supervisory_discount_factor(cp.maturity_years)
        sensitivity = cp.ead * cp.maturity_years * df

        # Subtract hedge notional (simplified)
        hedge_notional = hedge_map.get(cp.counterparty_id, 0.0)
        net_sensitivity = max(sensitivity - hedge_notional * df, 0.0)

        # Determine bucket from sector
        sector = cp.sector
        rw = _SA_CVA_DELTA_RW.get(sector, _SA_CVA_DELTA_RW["other"])
        weighted_sensitivity = rw * net_sensitivity

        # Map to canonical bucket name
        bucket_key = sector.rsplit("_", 1)[0] if "_" in sector else sector
        if bucket_key not in buckets:
            buckets[bucket_key] = []
        buckets[bucket_key].append(weighted_sensitivity)

    # Intra-bucket aggregation
    bucket_charges: dict[str, float] = {}
    for bucket_key, ws_list in buckets.items():
        n = len(ws_list)
        variance = 0.0
        for i in range(n):
            for j in range(n):
                rho_ij = 1.0 if i == j else _SA_CVA_RHO_SAME_SECTOR
                variance += rho_ij * ws_list[i] * ws_list[j]
        bucket_charges[bucket_key] = math.sqrt(max(variance, 0.0))

    # Inter-bucket aggregation
    bucket_keys = list(bucket_charges.keys())
    total_variance = 0.0
    for i, bk_i in enumerate(bucket_keys):
        for j, bk_j in enumerate(bucket_keys):
            gamma = 1.0 if i == j else _SA_CVA_GAMMA
            total_variance += gamma * bucket_charges[bk_i] * bucket_charges[bk_j]

    k_sa_cva = math.sqrt(max(total_variance, 0.0))

    logger.info(
        "SA-CVA capital: K_SA-CVA=%.2f (%d counterparties across %d buckets)",
        k_sa_cva,
        sum(len(v) for v in buckets.values()),
        len(buckets),
    )
    return k_sa_cva
