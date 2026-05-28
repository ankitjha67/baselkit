"""EU Green Asset Ratio (GAR) and Banking Book Taxonomy Alignment Ratio (BTAR).

Reference:
    - EU Taxonomy Regulation (EU) 2020/852.
    - Commission Delegated Regulation (EU) 2021/2178, Article 8.
    - EBA ITS on Pillar 3 ESG Disclosures (Reg 2022/2453).
    - EBA Advice on GAR KPIs (March 2021).

GAR = Taxonomy-aligned exposures / Total covered assets

Covered assets exclude: sovereigns, central bank exposures,
trading book, interbank on-demand. Numerator requires:
    1. Substantial contribution to >= 1 environmental objective.
    2. Do No Significant Harm (DNSH) to the remaining 5 objectives.
    3. Minimum safeguards (UNGP, OECD MNE Guidelines, ILO).

BTAR extends the numerator to non-NFRD counterparties, using
bank-estimated alignment from counterparty data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GARResult:
    """Green Asset Ratio calculation result.

    Attributes:
        taxonomy_aligned_amount: Sum of taxonomy-aligned exposures.
        covered_assets_total: Total covered assets (denominator).
        gar_pct: Green Asset Ratio as a percentage.
        excluded_sovereign: Sovereign/central bank exposures excluded.
        excluded_trading_book: Trading book exposures excluded.
    """

    taxonomy_aligned_amount: float
    covered_assets_total: float
    gar_pct: float
    excluded_sovereign: float
    excluded_trading_book: float


def green_asset_ratio(
    taxonomy_aligned: float,
    total_assets: float,
    sovereign_and_central_bank: float = 0.0,
    trading_book: float = 0.0,
    interbank_on_demand: float = 0.0,
) -> GARResult:
    """Calculate the EU Green Asset Ratio.

    GAR = taxonomy_aligned / (total_assets - exclusions)

    Exclusions per EBA ITS: sovereigns, central bank, trading book,
    interbank on-demand deposits.

    Args:
        taxonomy_aligned: Sum of taxonomy-aligned exposures (after
            substantial-contribution + DNSH + safeguards checks).
        total_assets: Bank's total on-balance-sheet assets.
        sovereign_and_central_bank: Exposures to sovereigns and
            central banks (excluded from denominator).
        trading_book: Trading-book exposures (excluded).
        interbank_on_demand: Interbank on-demand exposures (excluded).

    Returns:
        :class:`GARResult` with the computed ratio.

    Reference:
        EBA ITS Pillar 3 ESG Disclosures (Reg 2022/2453), Template 7.
    """
    exclusions = sovereign_and_central_bank + trading_book + interbank_on_demand
    covered = total_assets - exclusions

    if covered <= 0:
        return GARResult(
            taxonomy_aligned_amount=taxonomy_aligned,
            covered_assets_total=0.0,
            gar_pct=0.0,
            excluded_sovereign=sovereign_and_central_bank,
            excluded_trading_book=trading_book,
        )

    gar = (taxonomy_aligned / covered) * 100.0

    return GARResult(
        taxonomy_aligned_amount=taxonomy_aligned,
        covered_assets_total=covered,
        gar_pct=round(gar, 4),
        excluded_sovereign=sovereign_and_central_bank,
        excluded_trading_book=trading_book,
    )
