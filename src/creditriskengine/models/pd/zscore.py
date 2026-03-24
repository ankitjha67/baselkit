"""Altman Z-Score models for default prediction.

Implements the Altman (1968) Z-Score family of models for predicting
corporate bankruptcy. Provides the original model for public
manufacturers, the Z'-Score for private firms, and the Z''-Score
for non-manufacturers and emerging-market firms.

References:
    - Altman, E.I. (1968): Financial Ratios, Discriminant Analysis and
      the Prediction of Corporate Bankruptcy
    - Altman, E.I. (2005): An Emerging Market Credit Scoring System
    - BCBS d350: IRB approach — benchmarking models
"""

import logging

logger = logging.getLogger(__name__)


def altman_z_score(
    working_capital: float,
    total_assets: float,
    retained_earnings: float,
    ebit: float,
    market_equity: float,
    total_liabilities: float,
    sales: float,
) -> float:
    """Compute the original Altman Z-Score for public manufacturers.

    Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5

    where:
        X1 = Working Capital / Total Assets
        X2 = Retained Earnings / Total Assets
        X3 = EBIT / Total Assets
        X4 = Market Value of Equity / Total Liabilities
        X5 = Sales / Total Assets

    Reference: Altman (1968), Table 1.

    Args:
        working_capital: Current assets minus current liabilities.
        total_assets: Total assets.
        retained_earnings: Cumulative retained earnings.
        ebit: Earnings before interest and taxes.
        market_equity: Market value of equity.
        total_liabilities: Total liabilities.
        sales: Net sales revenue.

    Returns:
        Original Altman Z-Score.

    Raises:
        ValueError: If total_assets or total_liabilities are non-positive.
    """
    if total_assets <= 0:
        raise ValueError("total_assets must be positive.")
    if total_liabilities <= 0:
        raise ValueError("total_liabilities must be positive.")

    x1 = working_capital / total_assets
    x2 = retained_earnings / total_assets
    x3 = ebit / total_assets
    x4 = market_equity / total_liabilities
    x5 = sales / total_assets

    z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5

    logger.debug(
        "Altman Z-Score (original): X1=%.4f, X2=%.4f, X3=%.4f, "
        "X4=%.4f, X5=%.4f -> Z=%.4f",
        x1,
        x2,
        x3,
        x4,
        x5,
        z,
    )

    return z


def altman_z_score_private(
    working_capital: float,
    total_assets: float,
    retained_earnings: float,
    ebit: float,
    book_equity: float,
    total_liabilities: float,
    sales: float,
) -> float:
    """Compute the Altman Z'-Score for private firms.

    Z' = 0.717*X1 + 0.847*X2 + 3.107*X3 + 0.420*X4 + 0.998*X5

    where:
        X1 = Working Capital / Total Assets
        X2 = Retained Earnings / Total Assets
        X3 = EBIT / Total Assets
        X4 = Book Value of Equity / Total Liabilities
        X5 = Sales / Total Assets

    Reference: Altman (1968), revised for private firms.

    Args:
        working_capital: Current assets minus current liabilities.
        total_assets: Total assets.
        retained_earnings: Cumulative retained earnings.
        ebit: Earnings before interest and taxes.
        book_equity: Book value of equity.
        total_liabilities: Total liabilities.
        sales: Net sales revenue.

    Returns:
        Altman Z'-Score for private firms.

    Raises:
        ValueError: If total_assets or total_liabilities are non-positive.
    """
    if total_assets <= 0:
        raise ValueError("total_assets must be positive.")
    if total_liabilities <= 0:
        raise ValueError("total_liabilities must be positive.")

    x1 = working_capital / total_assets
    x2 = retained_earnings / total_assets
    x3 = ebit / total_assets
    x4 = book_equity / total_liabilities
    x5 = sales / total_assets

    z = 0.717 * x1 + 0.847 * x2 + 3.107 * x3 + 0.420 * x4 + 0.998 * x5

    logger.debug(
        "Altman Z'-Score (private): X1=%.4f, X2=%.4f, X3=%.4f, "
        "X4=%.4f, X5=%.4f -> Z'=%.4f",
        x1,
        x2,
        x3,
        x4,
        x5,
        z,
    )

    return z


def altman_z_score_emerging(
    working_capital: float,
    total_assets: float,
    retained_earnings: float,
    ebit: float,
    book_equity: float,
    total_liabilities: float,
) -> float:
    """Compute the Altman Z''-Score for emerging-market firms.

    Z'' = 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4

    where:
        X1 = Working Capital / Total Assets
        X2 = Retained Earnings / Total Assets
        X3 = EBIT / Total Assets
        X4 = Book Value of Equity / Total Liabilities

    This variant removes the sales/assets ratio to eliminate
    industry sensitivity, making it suitable for non-manufacturers
    and emerging-market firms.

    Reference: Altman (1968); Altman (2005), Emerging Market Scoring.

    Args:
        working_capital: Current assets minus current liabilities.
        total_assets: Total assets.
        retained_earnings: Cumulative retained earnings.
        ebit: Earnings before interest and taxes.
        book_equity: Book value of equity.
        total_liabilities: Total liabilities.

    Returns:
        Altman Z''-Score for emerging-market firms.

    Raises:
        ValueError: If total_assets or total_liabilities are non-positive.
    """
    if total_assets <= 0:
        raise ValueError("total_assets must be positive.")
    if total_liabilities <= 0:
        raise ValueError("total_liabilities must be positive.")

    x1 = working_capital / total_assets
    x2 = retained_earnings / total_assets
    x3 = ebit / total_assets
    x4 = book_equity / total_liabilities

    z = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4

    logger.debug(
        "Altman Z''-Score (emerging): X1=%.4f, X2=%.4f, X3=%.4f, "
        "X4=%.4f -> Z''=%.4f",
        x1,
        x2,
        x3,
        x4,
        z,
    )

    return z


def z_score_zone(
    z_score: float,
    model: str = "original",
) -> str:
    """Classify a Z-Score into safe, grey, or distress zone.

    Zone cutoffs per Altman (1968):
        - Original:  safe > 2.99,  grey 1.81-2.99,  distress < 1.81
        - Private:   safe > 2.9,   grey 1.23-2.9,   distress < 1.23
        - Emerging:  safe > 2.6,   grey 1.1-2.6,    distress < 1.1

    Args:
        z_score: Computed Z-Score value.
        model: Model variant — one of "original", "private", "emerging".

    Returns:
        Zone classification: "safe", "grey", or "distress".

    Raises:
        ValueError: If model is not a recognised variant.
    """
    cutoffs = {
        "original": (2.99, 1.81),
        "private": (2.9, 1.23),
        "emerging": (2.6, 1.1),
    }

    if model not in cutoffs:
        raise ValueError(
            f"Unknown model variant '{model}'. "
            f"Must be one of: {', '.join(cutoffs.keys())}."
        )

    upper, lower = cutoffs[model]

    if z_score > upper:
        zone = "safe"
    elif z_score >= lower:
        zone = "grey"
    else:
        zone = "distress"

    logger.debug(
        "Z-Score zone (%s): Z=%.4f -> %s (cutoffs: safe>%.2f, distress<%.2f)",
        model,
        z_score,
        zone,
        upper,
        lower,
    )

    return zone
