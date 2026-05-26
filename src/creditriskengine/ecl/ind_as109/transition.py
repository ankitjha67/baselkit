"""
RBI ECL Master Direction 2026 — Transition timeline support.

Reference: RBI/DOR/2026-27/398, Paragraphs 2, 19-21, 108, 114.

Implements:
    - Effective date (April 1, 2027).
    - EIR migration deadline (March 31, 2030) for legacy exposures.
    - Capital add-back phase-in schedule over FY 2027-28 to FY 2030-31.
"""

from __future__ import annotations

import logging
from datetime import date

logger = logging.getLogger(__name__)


RBI_ECL_EFFECTIVE_DATE: date = date(2027, 4, 1)
"""Date from which the RBI ECL Master Direction 2026 takes effect.

Reference: RBI/DOR/2026-27/398 Paragraph 2.
"""

RBI_EIR_MIGRATION_DEADLINE: date = date(2030, 3, 31)
"""Final date by which all legacy exposures must transition to EIR
method (Effective Interest Rate) for income recognition.

Reference: RBI/DOR/2026-27/398 Paragraphs 21, 114.
"""


CAPITAL_ADD_BACK_SCHEDULE: dict[int, float] = {
    2028: 0.80,  # 4/5 — FY 2027-28
    2029: 0.60,  # 3/5 — FY 2028-29
    2030: 0.40,  # 2/5 — FY 2029-30
    2031: 0.20,  # 1/5 — FY 2030-31
}
"""Capital add-back phase-in schedule per Paragraph 108.

Keys are fiscal year endings (Indian FY ends March 31).
For FY 2027-28 (ending March 2028), banks may add back 4/5 (80%) of
the transitional adjustment to CET1 capital; this declines to 0%
from FY 2031-32 onwards.

Reference: RBI/DOR/2026-27/398 Paragraph 108.
"""


def is_ecl_framework_effective(reporting_date: date) -> bool:
    """Check whether the ECL Master Direction is in effect on a given date.

    Args:
        reporting_date: Reference date for the check.

    Returns:
        ``True`` if ``reporting_date`` is on or after April 1, 2027.

    Reference:
        RBI/DOR/2026-27/398 Paragraph 2.
    """
    return reporting_date >= RBI_ECL_EFFECTIVE_DATE


def capital_add_back_factor(reporting_fy: int) -> float:
    """Return the capital add-back fraction for the reporting fiscal year.

    Per Paragraph 108, banks may add back a declining fraction of the
    transitional ECL adjustment to CET1 capital over a four-year
    phase-in window:

        FY 2027-28 (ending 2028): 4/5
        FY 2028-29 (ending 2029): 3/5
        FY 2029-30 (ending 2030): 2/5
        FY 2030-31 (ending 2031): 1/5
        FY 2031-32 onwards:       0/5

    Args:
        reporting_fy: Calendar year of fiscal year ending (e.g., 2028
            for FY 2027-28).

    Returns:
        Add-back fraction in ``[0.0, 0.8]``.

    Reference:
        RBI/DOR/2026-27/398 Paragraph 108.
    """
    return CAPITAL_ADD_BACK_SCHEDULE.get(reporting_fy, 0.0)


def eir_required(
    origination_date: date,
    reporting_date: date,
) -> bool:
    """Determine whether EIR-based income recognition is mandatory.

    Per Paragraph 50:
        - New exposures originated on or after April 1, 2027: EIR mandatory.
        - Existing exposures as of March 31, 2027: contractual rate
          permissible until full EIR migration by March 31, 2030.

    Args:
        origination_date: Date the exposure was originated.
        reporting_date: Current reporting date.

    Returns:
        ``True`` if EIR is mandatory.

    Reference:
        RBI/DOR/2026-27/398 Paragraphs 21, 50, 114.
    """
    # New originations are subject to EIR immediately
    if origination_date >= RBI_ECL_EFFECTIVE_DATE:
        return reporting_date >= RBI_ECL_EFFECTIVE_DATE
    # Legacy exposures must migrate by the deadline
    return reporting_date > RBI_EIR_MIGRATION_DEADLINE
