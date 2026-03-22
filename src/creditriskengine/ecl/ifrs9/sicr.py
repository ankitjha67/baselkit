"""
Significant Increase in Credit Risk (SICR) assessment.

Reference: IFRS 9.5.5.9-5.5.12, IFRS 9.B5.5.15-B5.5.22.

SICR is assessed by comparing lifetime PD at reporting date vs
lifetime PD at origination. Uses a relative change threshold.
30 DPD backstop is rebuttable per IFRS 9.B5.5.19.
"""

import logging

logger = logging.getLogger(__name__)


def assess_sicr(
    current_lifetime_pd: float,
    origination_lifetime_pd: float,
    days_past_due: int = 0,
    relative_threshold: float = 2.0,
    absolute_threshold: float = 0.005,
    dpd_backstop: int = 30,
    use_dpd_backstop: bool = True,
) -> bool:
    """Assess whether Significant Increase in Credit Risk has occurred.

    Quantitative assessment (IFRS 9.5.5.9):
        SICR if lifetime PD has increased significantly since origination.
        Typically: relative change > threshold OR absolute change > threshold.

    Qualitative backstop (IFRS 9.B5.5.19):
        30 DPD rebuttable presumption of SICR.

    Args:
        current_lifetime_pd: Current lifetime PD estimate.
        origination_lifetime_pd: Lifetime PD at origination date.
        days_past_due: Current days past due.
        relative_threshold: Relative PD increase threshold (default 2.0 = 200%).
        absolute_threshold: Absolute PD increase threshold (default 50 bps).
        dpd_backstop: DPD backstop (default 30 days).
        use_dpd_backstop: Whether to apply DPD backstop.

    Returns:
        True if SICR is triggered.
    """
    # DPD backstop
    if use_dpd_backstop and days_past_due > dpd_backstop:
        logger.debug("SICR triggered by DPD backstop: %d > %d", days_past_due, dpd_backstop)
        return True

    # Guard against zero origination PD
    if origination_lifetime_pd <= 0:
        return current_lifetime_pd > absolute_threshold

    # Relative change test
    relative_change = current_lifetime_pd / origination_lifetime_pd
    if relative_change > relative_threshold:
        logger.debug(
            "SICR triggered by relative PD change: %.4f / %.4f = %.2fx > %.2fx",
            current_lifetime_pd, origination_lifetime_pd, relative_change, relative_threshold,
        )
        return True

    # Absolute change test
    absolute_change = current_lifetime_pd - origination_lifetime_pd
    if absolute_change > absolute_threshold:
        logger.debug(
            "SICR triggered by absolute PD change: %.4f - %.4f = %.4f > %.4f",
            current_lifetime_pd, origination_lifetime_pd, absolute_change, absolute_threshold,
        )
        return True

    return False
