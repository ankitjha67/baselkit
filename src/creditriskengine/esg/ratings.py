"""
Vendor-agnostic ESG rating → PD adjustment adapter.

Reference:
    - MSCI ESG Ratings methodology (AAA-CCC, industry-relative).
    - Sustainalytics ESG Risk Rating (absolute risk score 0-40+).
    - S&P Global ESG scores.
    - EBA Report on ESG Risks Management (October 2023).

Maps heterogeneous ESG ratings to a common normalised risk score in
[0, 1] (0 = best ESG, 1 = worst), then derives a modest PD multiplier.
The multiplier is intentionally bounded to avoid double-counting
governance factors already captured in financial PD models (EBA 2023).
"""

from __future__ import annotations

import logging
from enum import StrEnum

logger = logging.getLogger(__name__)


class ESGProvider(StrEnum):
    """Supported ESG rating providers."""

    MSCI = "msci"
    SUSTAINALYTICS = "sustainalytics"
    SP_GLOBAL = "sp_global"


# MSCI letter grade → normalised risk score (0 best … 1 worst).
_MSCI_SCALE: dict[str, float] = {
    "AAA": 0.00,
    "AA": 0.15,
    "A": 0.30,
    "BBB": 0.50,
    "BB": 0.65,
    "B": 0.80,
    "CCC": 1.00,
}

# Default PD multiplier bounds (EBA: ESG is a modest overlay, not a
# wholesale repricing of PD).
_PD_MULT_MIN: float = 0.90
_PD_MULT_MAX: float = 1.30


def normalise_esg_score(
    provider: ESGProvider,
    raw_score: float | str,
) -> float:
    """Normalise a provider-specific ESG rating to [0, 1].

    0.0 = best ESG profile (lowest ESG risk), 1.0 = worst.

    Provider conventions:
        - MSCI: letter grade AAA (best) → CCC (worst).
        - Sustainalytics: ESG Risk Rating 0 (negligible) → 40+ (severe);
          higher is worse. Capped at 50 for normalisation.
        - S&P Global: ESG score 0-100 where higher is BETTER; inverted
          here so that 0 maps to worst and 100 to best.

    Args:
        provider: ESG rating provider.
        raw_score: Provider-specific rating (letter for MSCI, numeric
            otherwise).

    Returns:
        Normalised ESG risk score in [0, 1].

    Raises:
        ValueError: If the rating is invalid for the provider.
    """
    if provider == ESGProvider.MSCI:
        if not isinstance(raw_score, str):
            raise ValueError("MSCI rating must be a letter grade string")
        grade = raw_score.upper().strip()
        if grade not in _MSCI_SCALE:
            raise ValueError(
                f"Unknown MSCI grade '{grade}'. "
                f"Valid: {sorted(_MSCI_SCALE.keys())}"
            )
        return _MSCI_SCALE[grade]

    value = float(raw_score)
    if provider == ESGProvider.SUSTAINALYTICS:
        # 0 (best) … 50 (worst) → clip and scale
        return min(max(value, 0.0), 50.0) / 50.0

    # S&P Global: 0-100 where higher is better → invert
    clipped = min(max(value, 0.0), 100.0)
    return 1.0 - clipped / 100.0


def esg_pd_multiplier(
    normalised_score: float,
    max_uplift: float = 0.30,
    max_relief: float = 0.10,
) -> float:
    """Derive a PD multiplier from a normalised ESG risk score.

    Linearly interpolates between a relief (best ESG) and an uplift
    (worst ESG) around a neutral score of 0.5:

        score 0.0 → 1 - max_relief   (best ESG, modest PD relief)
        score 0.5 → 1.0              (neutral)
        score 1.0 → 1 + max_uplift   (worst ESG, PD uplift)

    The bounds keep the ESG overlay modest per EBA (2023) guidance,
    avoiding double-counting of governance signals already embedded in
    financial-statement-driven PD models.

    Args:
        normalised_score: ESG risk score in [0, 1] (0 best, 1 worst).
        max_uplift: Maximum PD uplift at the worst ESG score.
        max_relief: Maximum PD relief at the best ESG score.

    Returns:
        PD multiplier (e.g., 1.15 = +15% PD).

    Raises:
        ValueError: If normalised_score is outside [0, 1].
    """
    if not 0.0 <= normalised_score <= 1.0:
        raise ValueError("normalised_score must be in [0, 1]")

    if normalised_score <= 0.5:
        # Relief region: 0.0 → (1 - max_relief), 0.5 → 1.0
        frac = normalised_score / 0.5
        return (1.0 - max_relief) + frac * max_relief
    # Uplift region: 0.5 → 1.0, 1.0 → (1 + max_uplift)
    frac = (normalised_score - 0.5) / 0.5
    return 1.0 + frac * max_uplift
