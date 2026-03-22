"""
Output floor mechanism per BCBS d424, RBC25.2-25.4.

The output floor ensures that total RWA under internal models cannot
fall below a specified percentage of total SA RWA.

Total_RWA_floored = max(Total_RWA_internal, floor_pct * Total_RWA_standardized)

Phase-in schedules differ by jurisdiction.
"""

import logging
from datetime import date
from typing import Optional

from creditriskengine.core.types import Jurisdiction

logger = logging.getLogger(__name__)


# Phase-in schedules: {jurisdiction: [(effective_date, floor_pct), ...]}
# Sorted by date ascending.
OUTPUT_FLOOR_SCHEDULES: dict[str, list[tuple[str, float]]] = {
    # BCBS d424, RBC25.3
    "bcbs": [
        ("2023-01-01", 0.50),
        ("2024-01-01", 0.55),
        ("2025-01-01", 0.60),
        ("2026-01-01", 0.65),
        ("2027-01-01", 0.70),
        ("2028-01-01", 0.725),
    ],
    # EU CRR3 Art. 92a
    "eu": [
        ("2025-01-01", 0.50),
        ("2026-01-01", 0.55),
        ("2027-01-01", 0.60),
        ("2028-01-01", 0.65),
        ("2029-01-01", 0.70),
        ("2030-01-01", 0.725),
    ],
    # UK PRA PS1/26
    "uk": [
        ("2027-01-01", 0.50),
        ("2028-01-01", 0.55),
        ("2029-01-01", 0.60),
        ("2030-01-01", 0.65),
        ("2031-01-01", 0.70),
        ("2032-01-01", 0.725),
    ],
    # India RBI — 80% floor (more conservative)
    "india": [
        ("2023-01-01", 0.80),
    ],
    # Australia APRA — 72.5% immediately from 2023
    "australia": [
        ("2023-01-01", 0.725),
    ],
    # Canada OSFI — 72.5% immediately from Q2 2024
    "canada": [
        ("2024-04-01", 0.725),
    ],
    # Singapore MAS — per BCBS timeline
    "singapore": [
        ("2024-01-01", 0.50),
        ("2025-01-01", 0.55),
        ("2026-01-01", 0.60),
        ("2027-01-01", 0.65),
        ("2028-01-01", 0.70),
        ("2029-01-01", 0.725),
    ],
    # Hong Kong HKMA — per BCBS timeline
    "hong_kong": [
        ("2024-01-01", 0.50),
        ("2025-01-01", 0.55),
        ("2026-01-01", 0.60),
        ("2027-01-01", 0.65),
        ("2028-01-01", 0.70),
        ("2029-01-01", 0.725),
    ],
    # Japan JFSA — per BCBS timeline, effective March 2025
    "japan": [
        ("2025-03-01", 0.50),
        ("2026-03-01", 0.55),
        ("2027-03-01", 0.60),
        ("2028-03-01", 0.65),
        ("2029-03-01", 0.70),
        ("2030-03-01", 0.725),
    ],
    # China NFRA — per BCBS schedule
    "china": [
        ("2024-01-01", 0.50),
        ("2025-01-01", 0.55),
        ("2026-01-01", 0.60),
        ("2027-01-01", 0.65),
        ("2028-01-01", 0.70),
        ("2029-01-01", 0.725),
    ],
}

# EU transitional cap: max RWA increase from floor at 25% (CRR3 Art. 92a(3))
EU_TRANSITIONAL_CAP_ENABLED: bool = True
EU_MAX_RWA_INCREASE_PCT: float = 0.25


def get_output_floor_pct(
    jurisdiction: Jurisdiction,
    reporting_date: date,
) -> float:
    """Get the applicable output floor percentage for a jurisdiction and date.

    Args:
        jurisdiction: Regulatory jurisdiction.
        reporting_date: Reporting/calculation date.

    Returns:
        Floor percentage (e.g., 0.725 for 72.5%).
        Returns 0.0 if floor is not yet effective.
    """
    schedule = OUTPUT_FLOOR_SCHEDULES.get(jurisdiction.value)
    if schedule is None:
        # Default to BCBS schedule
        schedule = OUTPUT_FLOOR_SCHEDULES["bcbs"]

    floor_pct = 0.0
    for effective_str, pct in schedule:
        effective = date.fromisoformat(effective_str)
        if reporting_date >= effective:
            floor_pct = pct
        else:
            break

    return floor_pct


class OutputFloorCalculator:
    """Calculate output-floored RWA.

    The output floor ensures IRB RWA does not fall below
    a percentage of SA RWA.

    Reference: BCBS d424, RBC25.2-25.4.

    Example:
        >>> calc = OutputFloorCalculator(Jurisdiction.EU, date(2026, 6, 30))
        >>> result = calc.calculate(irb_rwa=800.0, sa_rwa=1200.0)
        >>> result["floored_rwa"]
        800.0  # if 55% * 1200 = 660 < 800
    """

    def __init__(
        self,
        jurisdiction: Jurisdiction,
        reporting_date: date,
    ) -> None:
        self.jurisdiction = jurisdiction
        self.reporting_date = reporting_date
        self.floor_pct = get_output_floor_pct(jurisdiction, reporting_date)

    def calculate(
        self,
        irb_rwa: float,
        sa_rwa: float,
        pre_floor_irb_rwa: Optional[float] = None,
    ) -> dict[str, float]:
        """Apply the output floor.

        Formula:
            floored_rwa = max(irb_rwa, floor_pct * sa_rwa)

        For EU: applies transitional cap limiting RWA increase to 25%.

        Args:
            irb_rwa: Total RWA from internal models.
            sa_rwa: Total RWA from standardized approach.
            pre_floor_irb_rwa: Pre-floor IRB RWA for EU transitional cap.

        Returns:
            Dict with floored_rwa, floor_pct, floor_rwa, is_binding, add_on.
        """
        floor_rwa = self.floor_pct * sa_rwa
        is_binding = floor_rwa > irb_rwa
        floored_rwa = max(irb_rwa, floor_rwa)
        add_on = max(0.0, floor_rwa - irb_rwa)

        # EU transitional cap: CRR3 Art. 92a(3)
        if (
            self.jurisdiction == Jurisdiction.EU
            and EU_TRANSITIONAL_CAP_ENABLED
            and is_binding
        ):
            base_rwa = pre_floor_irb_rwa if pre_floor_irb_rwa is not None else irb_rwa
            max_increase = base_rwa * EU_MAX_RWA_INCREASE_PCT
            if add_on > max_increase:
                add_on = max_increase
                floored_rwa = irb_rwa + add_on
                logger.info(
                    "EU transitional cap applied: add-on capped at %.2f (25%% of %.2f)",
                    max_increase, base_rwa,
                )

        logger.debug(
            "Output floor: jurisdiction=%s date=%s floor_pct=%.1f%% "
            "irb_rwa=%.2f sa_rwa=%.2f floor_rwa=%.2f floored_rwa=%.2f binding=%s",
            self.jurisdiction.value, self.reporting_date, self.floor_pct * 100,
            irb_rwa, sa_rwa, floor_rwa, floored_rwa, is_binding,
        )

        return {
            "floored_rwa": floored_rwa,
            "floor_pct": self.floor_pct,
            "floor_rwa": floor_rwa,
            "is_binding": is_binding,
            "add_on": add_on,
            "irb_rwa": irb_rwa,
            "sa_rwa": sa_rwa,
        }
