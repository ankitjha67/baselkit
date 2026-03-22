"""Commercial Real Estate risk weight logic — BCBS d424, CRE20.87-20.98.

Handles the complexity of CRE risk weights which depend on:
- LTV ratio
- Whether income-producing (cashflow dependent)
- Land ADC status
- Counterparty risk weight (for low-LTV non-income-producing)
- Jurisdiction-specific treatments
"""

import logging
from typing import Final

from creditriskengine.core.types import Jurisdiction

logger = logging.getLogger(__name__)

# ============================================================
# BCBS CRE20 Table 14 — CRE NOT dependent on cashflows
# ============================================================
# Whole-loan approach: (ltv_lower, ltv_upper, risk_weight)
# For LTV <= 60 % the RW is min(60 %, counterparty_rw) — CRE20.87
CRE_NOT_CASHFLOW_RW: Final[list[tuple[float, float, float]]] = [
    (0.0, 0.60, 60.0),   # min(60%, counterparty_rw) per CRE20.87
    (0.60, 0.80, 80.0),  # Whole-loan approach — CRE20.89
    (0.80, float("inf"), -1.0),  # sentinel: use counterparty RW
]

# ============================================================
# BCBS CRE20 Table 15 — Income-Producing RE (IPRE)
# ============================================================
CRE_IPRE_RW: Final[list[tuple[float, float, float]]] = [
    (0.0, 0.60, 70.0),
    (0.60, 0.80, 90.0),
    (0.80, float("inf"), 110.0),
]

# ============================================================
# EU CRR3 Art. 126 — CRE LTV buckets (slightly different)
# ============================================================
# Non-income-producing — CRR3 Art. 126(1)
CRE_EU_NOT_CASHFLOW_RW: Final[list[tuple[float, float, float]]] = [
    (0.0, 0.55, 60.0),   # min(60%, counterparty_rw)
    (0.55, 0.60, 60.0),  # EU adds a 55-60 sub-bucket at 60 %
    (0.60, 0.80, 80.0),
    (0.80, float("inf"), -1.0),  # counterparty RW
]

# Income-producing — CRR3 Art. 126(1)(b)
CRE_EU_IPRE_RW: Final[list[tuple[float, float, float]]] = [
    (0.0, 0.60, 70.0),
    (0.60, 0.80, 90.0),
    (0.80, float("inf"), 110.0),
]

# ============================================================
# Land ADC — CRE20.97
# ============================================================
LAND_ADC_RW: Final[float] = 150.0
LAND_ADC_PRESOLD_RESIDENTIAL_RW: Final[float] = 100.0  # CRR3 Art. 126(2)(e)


def _lookup_ltv_table(
    table: list[tuple[float, float, float]],
    ltv: float,
    counterparty_rw: float,
) -> float:
    """Look up risk weight in an LTV-bucket table.

    Handles the sentinel value ``-1.0`` which indicates that the
    counterparty risk weight should be applied.

    Args:
        table: List of ``(ltv_lower, ltv_upper, rw)`` tuples.
        ltv: Loan-to-value ratio.
        counterparty_rw: Counterparty risk weight (percentage).

    Returns:
        Risk weight as percentage.
    """
    for ltv_lower, ltv_upper, rw in table:
        if ltv_lower < ltv <= ltv_upper:
            if rw < 0:
                return counterparty_rw
            return min(rw, counterparty_rw) if rw == 60.0 else rw
    # LTV exactly 0 or below first bucket
    if ltv <= 0:
        first_rw = table[0][2]
        if first_rw < 0:
            return counterparty_rw
        return min(first_rw, counterparty_rw) if first_rw == 60.0 else first_rw
    # Beyond last bucket
    last_rw = table[-1][2]
    return counterparty_rw if last_rw < 0 else last_rw


def get_cre_risk_weight(
    ltv: float,
    counterparty_rw: float,
    is_income_producing: bool = False,
    is_adc: bool = False,
    is_presold: bool = False,
    jurisdiction: Jurisdiction | None = None,
) -> float:
    """Assign CRE risk weight per BCBS CRE20.87-20.98.

    Decision tree:
    1. Land ADC exposures receive 150 % (CRE20.97), unless pre-sold
       residential (100 % under CRR3 Art. 126(2)(e)).
    2. Income-producing / cashflow-dependent exposures use IPRE table
       (CRE20.89, Table 15).
    3. Non-income-producing exposures use Table 14, with
       ``min(60 %, counterparty_rw)`` for the lowest LTV bucket and
       counterparty RW for the highest.

    Jurisdiction routing:
    - ``EU``: delegates to :func:`get_cre_risk_weight_eu`.
    - All others: BCBS base tables.

    Args:
        ltv: Loan-to-value ratio (e.g. 0.75 for 75 %).
        counterparty_rw: Risk weight of the counterparty (percentage).
        is_income_producing: If True, exposure is income-producing RE.
        is_adc: If True, Land Acquisition/Development/Construction.
        is_presold: If True and ADC, pre-sold residential treatment.
        jurisdiction: Regulatory jurisdiction override.

    Returns:
        Risk weight as percentage.
    """
    # Land ADC — CRE20.97
    if is_adc:
        if is_presold:
            logger.debug(
                "CRE Land ADC pre-sold residential: RW=%.0f%%",
                LAND_ADC_PRESOLD_RESIDENTIAL_RW,
            )
            return LAND_ADC_PRESOLD_RESIDENTIAL_RW
        logger.debug("CRE Land ADC: RW=%.0f%%", LAND_ADC_RW)
        return LAND_ADC_RW

    # EU CRR3 delegation
    if jurisdiction == Jurisdiction.EU:
        return get_cre_risk_weight_eu(
            ltv, counterparty_rw, is_income_producing=is_income_producing
        )

    # BCBS base tables
    table = CRE_IPRE_RW if is_income_producing else CRE_NOT_CASHFLOW_RW

    rw = _lookup_ltv_table(table, ltv, counterparty_rw)
    logger.debug(
        "CRE RW for LTV=%.2f, IPRE=%s, jurisdiction=%s: %.0f%%",
        ltv,
        is_income_producing,
        jurisdiction or "BCBS",
        rw,
    )
    return rw


def get_cre_risk_weight_eu(
    ltv: float,
    counterparty_rw: float,
    is_income_producing: bool = False,
) -> float:
    """EU CRR3 Art. 126 — slightly different LTV buckets.

    The EU implementation under CRR3 Art. 126 introduces:
    - A 55-60 % LTV sub-bucket for non-income-producing CRE.
    - Same IPRE table as BCBS for income-producing.
    - Pre-sold residential ADC at 100 % (Art. 126(2)(e)).

    The ``min(counterparty_rw, 60 %)`` logic for the lowest LTV bucket
    is preserved.

    Args:
        ltv: Loan-to-value ratio.
        counterparty_rw: Risk weight of the counterparty (percentage).
        is_income_producing: If True, use IPRE table.

    Returns:
        Risk weight as percentage.
    """
    table = CRE_EU_IPRE_RW if is_income_producing else CRE_EU_NOT_CASHFLOW_RW

    rw = _lookup_ltv_table(table, ltv, counterparty_rw)
    logger.debug(
        "CRE EU RW for LTV=%.2f, IPRE=%s: %.0f%%",
        ltv,
        is_income_producing,
        rw,
    )
    return rw
