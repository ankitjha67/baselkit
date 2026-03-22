"""
Concentration risk analytics.

Single-name concentration, sector/geographic HHI,
and the Granularity Adjustment (GA) per BCBS.

References:
- BCBS d424: Pillar 2 concentration risk
- Gordy (2003): A risk-factor model foundation for ratings-based capital rules
- Gordy & Lütkebohmert (2013): Granularity adjustment for regulatory capital
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def single_name_concentration(
    eads: np.ndarray,
) -> dict[str, float]:
    """Measure single-name concentration in a portfolio.

    Returns HHI and top-N exposure shares.

    Args:
        eads: EAD per obligor.

    Returns:
        Dict with hhi, top_1_share, top_5_share, top_10_share, n_obligors.
    """
    eads = np.asarray(eads, dtype=np.float64)
    total = float(np.sum(eads))
    if total <= 0:
        return {
            "hhi": 0.0,
            "top_1_share": 0.0,
            "top_5_share": 0.0,
            "top_10_share": 0.0,
            "n_obligors": 0,
        }

    shares = eads / total
    hhi = float(np.sum(shares ** 2))

    sorted_eads = np.sort(eads)[::-1]
    n = len(eads)

    top_1 = float(sorted_eads[0] / total) if n >= 1 else 0.0
    top_5 = float(np.sum(sorted_eads[:5]) / total) if n >= 5 else float(np.sum(sorted_eads) / total)
    top_10 = (
        float(np.sum(sorted_eads[:10]) / total) if n >= 10
        else float(np.sum(sorted_eads) / total)
    )

    return {
        "hhi": hhi,
        "top_1_share": top_1,
        "top_5_share": top_5,
        "top_10_share": top_10,
        "n_obligors": n,
    }


def sector_concentration(
    eads: np.ndarray,
    sector_labels: np.ndarray,
) -> dict[str, float | dict[str, float]]:
    """Measure sector concentration via HHI.

    Args:
        eads: EAD per exposure.
        sector_labels: Sector label per exposure.

    Returns:
        Dict with sector_hhi and sector_shares.
    """
    eads = np.asarray(eads, dtype=np.float64)
    total = float(np.sum(eads))
    if total <= 0:
        return {"sector_hhi": 0.0, "sector_shares": {}}

    unique_sectors = np.unique(sector_labels)
    shares: dict[str, float] = {}
    for sector in unique_sectors:
        mask = sector_labels == sector
        shares[str(sector)] = float(np.sum(eads[mask]) / total)

    hhi = sum(s ** 2 for s in shares.values())
    return {"sector_hhi": hhi, "sector_shares": shares}


def granularity_adjustment(
    eads: np.ndarray,
    pds: np.ndarray,
    lgds: np.ndarray,
    rho: float,
) -> float:
    """Gordy (2003) Granularity Adjustment.

    GA = (1/2) × C_3 × HHI_adj

    Where C_3 captures the curvature of the loss distribution
    and HHI_adj is the EAD-weighted HHI of PD×LGD contributions.

    Simplified single-factor version for Pillar 2 add-on estimation.

    Args:
        eads: EAD per obligor.
        pds: PD per obligor.
        lgds: LGD per obligor.
        rho: Common asset correlation.

    Returns:
        Granularity adjustment as a fraction of total EAD.
    """
    eads = np.asarray(eads, dtype=np.float64)
    pds = np.asarray(pds, dtype=np.float64)
    lgds = np.asarray(lgds, dtype=np.float64)

    total_ead = float(np.sum(eads))
    if total_ead <= 0:
        return 0.0

    # EAD-weighted expected loss contributions
    el_contributions = eads * pds * lgds
    total_el = float(np.sum(el_contributions))
    if total_el <= 0:
        return 0.0

    # HHI of loss contributions
    shares = el_contributions / total_el
    hhi = float(np.sum(shares ** 2))

    # Simplified GA: proportional to HHI and variance
    # Higher correlation → less idiosyncratic risk → smaller GA
    idiosyncratic_factor = (1.0 - rho)
    ga = 0.5 * hhi * idiosyncratic_factor * total_el / total_ead

    return ga
