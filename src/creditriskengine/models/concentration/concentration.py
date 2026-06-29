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
import math

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
    confidence: float = 0.999,
) -> float:
    """Martin-Wilde / Gordy (2003) granularity adjustment for the ASRF model.

    Computes the undiversified idiosyncratic risk add-on to the ASRF VaR
    using the Martin & Wilde (2002) second-order expansion:

        GA = -(1/2) * (1 / h(x_q)) * d/dx[ h(x) * sigma^2(x) / mu'(x) ]_{x = x_q}

    where, under the single-factor Vasicek model with systematic factor
    X ~ N(0, 1) and common asset correlation ``rho``:

        p_i(x) = Phi( (Phi^{-1}(PD_i) - sqrt(rho) * x) / sqrt(1 - rho) )
        mu(x)  = sum_i s_i * LGD_i * p_i(x)            (conditional EL rate)
        sigma^2(x) = sum_i s_i^2 * LGD_i^2 * p_i(x) * (1 - p_i(x))
        h(x)   = standard normal density of X
        x_q    = Phi^{-1}(1 - confidence)             (stress state)
        s_i    = EAD_i / sum(EAD)

    The portfolio variance term ``sigma^2`` scales like 1/n for a
    homogeneous portfolio, so the GA vanishes as the book becomes
    infinitely granular and grows with name concentration.

    Args:
        eads: EAD per obligor.
        pds: PD per obligor (decimals in (0, 1)).
        lgds: LGD per obligor (decimals).
        rho: Common asset correlation in [0, 1).
        confidence: VaR confidence level (default 0.999, Basel IRB).

    Returns:
        Granularity adjustment as a fraction of total EAD (non-negative).
    """
    from scipy.stats import norm

    eads = np.asarray(eads, dtype=np.float64)
    pds = np.asarray(pds, dtype=np.float64)
    lgds = np.asarray(lgds, dtype=np.float64)

    total_ead = float(np.sum(eads))
    if total_ead <= 0:
        return 0.0
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0, 1)")

    # No default risk in the book -> no idiosyncratic add-on.
    if float(np.sum(eads * pds * lgds)) <= 0.0:
        return 0.0

    shares = eads / total_ead
    # Guard PDs away from {0, 1} so the inverse-normal is finite.
    pd_clip = np.clip(pds, 1e-9, 1.0 - 1e-9)
    inv_pd = norm.ppf(pd_clip)
    sqrt_rho = math.sqrt(rho)
    sqrt_1mrho = math.sqrt(1.0 - rho) if rho < 1.0 else 1e-12

    def conditional_pd(x: float) -> np.ndarray:
        return norm.cdf((inv_pd - sqrt_rho * x) / sqrt_1mrho)

    def mu_prime(x: float) -> float:
        g = (inv_pd - sqrt_rho * x) / sqrt_1mrho
        dp_dx = norm.pdf(g) * (-sqrt_rho / sqrt_1mrho)
        return float(np.sum(shares * lgds * dp_dx))

    def numerator(x: float) -> float:
        # F(x) = h(x) * sigma^2(x) / mu'(x)
        p = conditional_pd(x)
        var = float(np.sum(shares**2 * lgds**2 * p * (1.0 - p)))
        mp = mu_prime(x)
        if abs(mp) < 1e-15:
            return 0.0
        return float(norm.pdf(x)) * var / mp

    x_q = float(norm.ppf(1.0 - confidence))
    h_xq = float(norm.pdf(x_q))
    if h_xq < 1e-300:
        return 0.0

    # Central finite difference for the outer derivative.
    step = 1e-4
    derivative = (numerator(x_q + step) - numerator(x_q - step)) / (2.0 * step)
    ga = -0.5 * derivative / h_xq
    return max(ga, 0.0)
