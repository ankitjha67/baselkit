"""
IRB Risk Weight Formulas — BCBS d424 (December 2017).

This module implements the regulatory risk weight functions for all IRB
asset classes. Each function documents its exact source paragraph from
the Basel III framework.

CRITICAL: These formulas directly affect bank capital requirements.
Every parameter, threshold, and formula must be verified against the
referenced Basel Committee text.
"""

import logging
import math

from scipy.stats import norm

logger = logging.getLogger(__name__)

# PD floor per BCBS CRE32.13: 0.03% (3 basis points)
PD_FLOOR: float = 0.0003


# ============================================================
# CORRELATION FUNCTIONS
# ============================================================


def asset_correlation_corporate(pd: float) -> float:
    """Asset correlation for corporate, sovereign, and bank exposures.

    Formula (BCBS CRE31.5):
        R = 0.12 * (1 - exp(-50 * PD)) / (1 - exp(-50))
          + 0.24 * (1 - (1 - exp(-50 * PD)) / (1 - exp(-50)))

    Produces R in [0.12, 0.24]:
    - R -> 0.24 as PD -> 0
    - R -> 0.12 as PD -> 1

    Args:
        pd: Probability of Default (annualized, in [0.0003, 1.0])

    Returns:
        Asset correlation R in [0.12, 0.24]
    """
    exp_factor = (1.0 - math.exp(-50.0 * pd)) / (1.0 - math.exp(-50.0))
    return 0.12 * exp_factor + 0.24 * (1.0 - exp_factor)


def sme_firm_size_adjustment(turnover_eur_millions: float) -> float:
    """SME firm-size adjustment to correlation for corporate exposures.

    Formula (BCBS CRE31.6):
        Adjustment = -0.04 * (1 - (min(max(S, 5), 50) - 5) / 45)

    Where S = annual turnover in EUR millions.
    - S is floored at EUR 5M and capped at EUR 50M
    - Maximum reduction is 0.04 (at S = EUR 5M)
    - No adjustment when S >= EUR 50M

    Args:
        turnover_eur_millions: Annual sales/turnover in EUR millions

    Returns:
        Correlation adjustment (negative value to subtract from R)
    """
    s = min(max(turnover_eur_millions, 5.0), 50.0)
    return -0.04 * (1.0 - (s - 5.0) / 45.0)


def asset_correlation_residential_mortgage(pd: float) -> float:
    """Asset correlation for residential mortgage exposures.

    Formula (BCBS CRE31.8):
        R = 0.15

    Fixed correlation of 0.15 per the 2017 Basel III reform.

    Args:
        pd: Probability of Default (unused, kept for interface consistency)

    Returns:
        Fixed correlation of 0.15
    """
    return 0.15


def asset_correlation_qrre(pd: float) -> float:
    """Asset correlation for Qualifying Revolving Retail Exposures.

    Formula (BCBS CRE31.9):
        R = 0.04

    Fixed correlation for all QRRE exposures.

    Args:
        pd: Probability of Default

    Returns:
        Fixed correlation of 0.04
    """
    return 0.04


def asset_correlation_other_retail(pd: float) -> float:
    """Asset correlation for Other Retail exposures.

    Formula (BCBS CRE31.10):
        R = 0.03 * (1 - exp(-35 * PD)) / (1 - exp(-35))
          + 0.16 * (1 - (1 - exp(-35 * PD)) / (1 - exp(-35)))

    Range: [0.03, 0.16]

    Args:
        pd: Probability of Default

    Returns:
        Asset correlation in [0.03, 0.16]
    """
    exp_factor = (1.0 - math.exp(-35.0 * pd)) / (1.0 - math.exp(-35.0))
    return 0.03 * exp_factor + 0.16 * (1.0 - exp_factor)


# ============================================================
# MATURITY ADJUSTMENT
# BCBS d424, CRE31.7
# ============================================================


def maturity_adjustment(pd: float, maturity: float) -> float:
    """Maturity adjustment factor for corporate, sovereign, bank exposures.

    Formula (BCBS CRE31.7):
        b = (0.11852 - 0.05478 * ln(PD))^2
        MA = (1 + (M - 2.5) * b) / (1 - 1.5 * b)

    Where M = effective maturity in years.

    For F-IRB: M = 2.5 years fixed (BCBS CRE32.47).
    For A-IRB: M = max(1, effective maturity), capped at 5 years.
    Retail exposures: NO maturity adjustment.

    Args:
        pd: Probability of Default (>= 0.0003)
        maturity: Effective maturity M in years

    Returns:
        Maturity adjustment factor (multiplier to capital requirement K)
    """
    pd_calc = max(pd, PD_FLOOR)
    b = (0.11852 - 0.05478 * math.log(pd_calc)) ** 2
    return (1.0 + (maturity - 2.5) * b) / (1.0 - 1.5 * b)


# ============================================================
# IRB CAPITAL REQUIREMENT AND RISK WEIGHT
# BCBS d424, CRE31.4
# ============================================================


def irb_capital_requirement_k(
    pd: float,
    lgd: float,
    correlation: float,
) -> float:
    """IRB capital requirement K (before maturity adjustment).

    Formula (BCBS CRE31.4):
        K = LGD * [N((1-R)^(-0.5) * G(PD) + (R/(1-R))^0.5 * G(0.999)) - PD]

    Where:
    - N() = standard normal CDF
    - G() = standard normal inverse CDF
    - R = asset correlation
    - PD = probability of default (floored at 0.03%)
    - LGD = loss given default

    The 0.999 confidence level = 99.9th percentile.

    Args:
        pd: Probability of Default (>= 0.0003 floor applied)
        lgd: Loss Given Default (in [0, 1])
        correlation: Asset correlation R

    Returns:
        Capital requirement K as a fraction of EAD
    """
    if not 0.0 < correlation < 1.0:
        raise ValueError(f"Correlation must be in (0, 1), got {correlation}")

    pd_floored = max(pd, PD_FLOOR)

    g_pd = norm.ppf(pd_floored)
    g_999 = norm.ppf(0.999)

    conditional_pd = norm.cdf(
        (1.0 / math.sqrt(1.0 - correlation)) * g_pd
        + math.sqrt(correlation / (1.0 - correlation)) * g_999
    )

    k = lgd * (float(conditional_pd) - pd_floored)
    return max(k, 0.0)


def irb_risk_weight(
    pd: float,
    lgd: float,
    asset_class: str,
    maturity: float = 2.5,
    turnover_eur_millions: float | None = None,
    is_qrre_transactor: bool = False,
    ead: float = 1.0,
) -> float:
    """Full IRB risk weight computation.

    Formula (BCBS CRE31.4-31.10):
        RW = K * 12.5 * MA    (corporate/sovereign/bank)
        RW = K * 12.5          (retail)

    The 12.5 multiplier converts K to risk weight because
    Capital = 8% * RWA = K * EAD, so RW = K * 12.5.

    PD Floor (CRE32.13): 0.03% for all non-defaulted exposures.

    Args:
        pd: Probability of Default
        lgd: Loss Given Default
        asset_class: One of 'corporate', 'sovereign', 'bank',
                     'residential_mortgage', 'qrre', 'other_retail'
        maturity: Effective maturity in years (non-retail only)
        turnover_eur_millions: For SME firm-size correlation adjustment
        is_qrre_transactor: If True, apply 0.75× RW scalar per CRE31.9 fn 15
        ead: Exposure at Default (default 1.0)

    Returns:
        Risk weight as a percentage (e.g., 75.0 means 75%)
    """
    pd_floored = max(pd, PD_FLOOR)

    # Defaulted exposures: K = max(0, LGD - EL_BE)
    if pd >= 1.0:
        return 0.0

    # Step 1: Determine asset correlation
    if asset_class in ("corporate", "sovereign", "bank"):
        r = asset_correlation_corporate(pd_floored)
        if asset_class == "corporate" and turnover_eur_millions is not None:
            r += sme_firm_size_adjustment(turnover_eur_millions)
            r = max(r, 0.0)
    elif asset_class == "residential_mortgage":
        r = asset_correlation_residential_mortgage(pd_floored)
    elif asset_class == "qrre":
        r = asset_correlation_qrre(pd_floored)
    elif asset_class == "other_retail":
        r = asset_correlation_other_retail(pd_floored)
    else:
        raise ValueError(f"Unknown asset class: {asset_class}")

    # Step 2: Capital requirement K
    k = irb_capital_requirement_k(pd_floored, lgd, r)

    # Step 3: Maturity adjustment (non-retail only)
    if asset_class in ("corporate", "sovereign", "bank"):
        m = max(1.0, min(maturity, 5.0))
        ma = maturity_adjustment(pd_floored, m)
        k *= ma

    # Step 4: Convert to risk weight
    rw = k * 12.5

    # Step 5: QRRE transactor scalar (BCBS CRE31.9, footnote 15)
    # Transactors are obligors who repay balances in full each month.
    # They receive a 0.75× multiplier on the risk weight.
    if asset_class == "qrre" and is_qrre_transactor:
        rw *= 0.75

    logger.debug(
        "IRB RW: asset_class=%s pd=%.4f lgd=%.2f R=%.4f K=%.6f RW=%.2f%%",
        asset_class, pd_floored, lgd, r, k, rw * 100.0,
    )

    return rw * 100.0


# ============================================================
# DOUBLE DEFAULT — SUBSTITUTION APPROACH
# BCBS d424, CRE32.38-41
# ============================================================


def double_default_rw(
    pd_obligor: float,
    pd_guarantor: float,
    lgd: float,
    maturity: float = 2.5,
    asset_class: str = "corporate",
) -> float:
    """Double default risk weight using the substitution approach.

    When a guarantee or credit derivative exists, the bank may substitute
    the guarantor's PD for the obligor's PD while retaining the asset
    correlation derived from the obligor's asset class.

    Formula (BCBS CRE32.38-41):
        1. Effective PD = max(PD_guarantor, PD_FLOOR)
        2. Correlation R = derived from obligor's asset class using PD_obligor
        3. K = IRB capital requirement using effective PD, obligor's R, and LGD
        4. Apply maturity adjustment for non-retail asset classes
        5. RW = K * 12.5

    The guarantor PD is floored at 0.03% per CRE32.13.

    Args:
        pd_obligor: Probability of Default of the original obligor.
        pd_guarantor: Probability of Default of the guarantor.
        lgd: Loss Given Default (in [0, 1]).
        maturity: Effective maturity in years (non-retail only).
        asset_class: One of 'corporate', 'sovereign', 'bank',
                     'residential_mortgage', 'qrre', 'other_retail'.

    Returns:
        Risk weight as a percentage (e.g., 75.0 means 75%).
    """
    pd_obligor_floored = max(pd_obligor, PD_FLOOR)
    pd_eff = max(pd_guarantor, PD_FLOOR)

    # Defaulted guarantor: no benefit
    if pd_guarantor >= 1.0:
        return 0.0

    # Step 1: Asset correlation from obligor's asset class and PD
    if asset_class in ("corporate", "sovereign", "bank"):
        r = asset_correlation_corporate(pd_obligor_floored)
    elif asset_class == "residential_mortgage":
        r = asset_correlation_residential_mortgage(pd_obligor_floored)
    elif asset_class == "qrre":
        r = asset_correlation_qrre(pd_obligor_floored)
    elif asset_class == "other_retail":
        r = asset_correlation_other_retail(pd_obligor_floored)
    else:
        raise ValueError(f"Unknown asset class: {asset_class}")

    # Step 2: Capital requirement using guarantor PD with obligor correlation
    k = irb_capital_requirement_k(pd_eff, lgd, r)

    # Step 3: Maturity adjustment (non-retail only)
    if asset_class in ("corporate", "sovereign", "bank"):
        m = max(1.0, min(maturity, 5.0))
        ma = maturity_adjustment(pd_eff, m)
        k *= ma

    # Step 4: Convert to risk weight
    rw = k * 12.5

    logger.debug(
        "Double default RW: asset_class=%s pd_obligor=%.4f pd_guarantor=%.4f "
        "pd_eff=%.4f lgd=%.2f R=%.4f K=%.6f RW=%.2f%%",
        asset_class, pd_obligor_floored, pd_guarantor, pd_eff, lgd, r, k,
        rw * 100.0,
    )

    return rw * 100.0


# ============================================================
# EQUITY IRB — SIMPLE RISK WEIGHT METHOD
# BCBS d424, CRE33
# ============================================================


def equity_irb_rw(
    pd: float,
    equity_type: str = "listed",
) -> float:
    """IRB simple risk weight method for equity exposures.

    Formula (BCBS CRE33):
        RW = max(floor, 2.5 * corporate_IRB_RW(PD, LGD=90%, M=5))

    Where:
    - Listed equity: floor = 200%
    - Private/unlisted equity: floor = 300%
    - LGD is fixed at 90% per CRE33
    - Maturity M is fixed at 5 years per CRE33
    - The corporate IRB RW uses the standard corporate correlation

    Args:
        pd: Probability of Default.
        equity_type: One of 'listed' or 'private'.

    Returns:
        Risk weight as a percentage (e.g., 200.0 means 200%).

    Raises:
        ValueError: If equity_type is not 'listed' or 'private'.
    """
    if equity_type not in ("listed", "private"):
        raise ValueError(
            f"equity_type must be 'listed' or 'private', got '{equity_type}'"
        )

    # Fixed parameters per CRE33
    lgd = 0.90
    maturity = 5.0

    # Corporate IRB RW with fixed LGD=90% and M=5
    corporate_rw = irb_risk_weight(
        pd=pd, lgd=lgd, asset_class="corporate", maturity=maturity
    )

    # Apply 2.5× multiplier
    scaled_rw = 2.5 * corporate_rw

    # Apply floor based on equity type
    floor = 200.0 if equity_type == "listed" else 300.0

    rw = max(floor, scaled_rw)

    logger.debug(
        "Equity IRB RW: type=%s pd=%.4f corporate_rw=%.2f%% "
        "scaled=%.2f%% floor=%.2f%% final=%.2f%%",
        equity_type, max(pd, PD_FLOOR), corporate_rw, scaled_rw, floor, rw,
    )

    return rw
