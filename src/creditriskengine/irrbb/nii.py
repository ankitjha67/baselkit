"""
Net Interest Income (NII) sensitivity for IRRBB.

Reference:
    - BCBS d368 (IRRBB, April 2016), NII perspective.
    - EBA/GL/2018/02.

NII sensitivity measures the change in projected net interest income
over a (typically 12-month) horizon under a rate shock, driven by the
repricing of rate-sensitive assets and liabilities.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def nii_sensitivity(
    rate_sensitive_assets: float,
    rate_sensitive_liabilities: float,
    shock_bps: float,
    horizon_years: float = 1.0,
    asset_repricing_fraction: float = 1.0,
    liability_repricing_fraction: float = 1.0,
) -> float:
    """Change in NII under a parallel rate shock.

    Delta-NII = shock * (RSA * asset_repricing - RSL * liability_repricing)
                * horizon

    A positive Delta-NII under an up-shock indicates an asset-sensitive
    balance sheet (assets reprice faster than liabilities).

    Args:
        rate_sensitive_assets: Rate-sensitive assets repricing in the
            horizon.
        rate_sensitive_liabilities: Rate-sensitive liabilities repricing.
        shock_bps: Parallel rate shock in basis points (can be negative).
        horizon_years: NII projection horizon (default 1 year).
        asset_repricing_fraction: Fraction of assets that reprice
            (default 1.0).
        liability_repricing_fraction: Fraction of liabilities that
            reprice (default 1.0).

    Returns:
        Delta-NII over the horizon.

    Reference:
        BCBS d368 (NII perspective).
    """
    shock = shock_bps / 10_000.0
    repricing_gap = (
        rate_sensitive_assets * asset_repricing_fraction
        - rate_sensitive_liabilities * liability_repricing_fraction
    )
    return shock * repricing_gap * horizon_years
