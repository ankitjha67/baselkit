"""
Parallel-run comparator: legacy IRACP vs RBI ECL Master Direction 2026.

Banks transitioning from the existing IRACP regime (in force through
March 31, 2027) to the new Master Direction (effective April 1, 2027)
typically run both frameworks in parallel through FY 2027 to quantify
the provisioning delta and capital impact.

This module provides a single-call comparator that computes both
provisions and returns the difference, percentage change, and
binding-floor diagnostic for each exposure.

References:
    - DOR.STR.REC.9/21.04.048/2025-26 (April 1, 2025) — IRACP Master Circular.
    - RBI/DOR/2026-27/398; DOR.STR.REC.No.6/21.06.011/2026-27
      (April 27, 2026) — ECL Master Direction.
    - Paragraph 108 — Capital transitional add-back (4/5 -> 1/5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from creditriskengine.core.types import IFRS9Stage
from creditriskengine.ecl.ind_as109.ind_as_ecl import (
    IRACAssetClass,
    calculate_ecl_ind_as,
    calculate_ecl_ind_as_2026,
    rbi_minimum_provision,
)
from creditriskengine.ecl.ind_as109.transition import capital_add_back_factor
from creditriskengine.ecl.ind_as109.types import RBIExposureCategory


@dataclass(frozen=True)
class ParallelRunResult:
    """Per-exposure comparison of IRACP vs ECL 2026 provisions.

    Attributes:
        iracp_provision: Legacy IRACP-based provision (model ECL floored
            by IRAC sub-category rate per DOR.STR.REC.9/21.04.048/2025-26).
        ecl_2026_provision: Provision under the new Master Direction
            (PD/LGD floors + model ECL + DLG + Para 82 prudential floor).
        delta: ecl_2026_provision - iracp_provision (positive = uplift
            from new framework).
        pct_change: Percentage change vs IRACP, expressed as a decimal
            (e.g., 0.15 = +15%). Returns 0.0 if IRACP is zero.
        binding_in_2026: Whether the regulatory floor binds (i.e.,
            regulatory floor >= net model ECL) under the new framework.
        capital_add_back_amount: Amount of the delta eligible for CET1
            transitional add-back per Paragraph 108 (declining over
            FY 2027-28 to FY 2030-31).
    """

    iracp_provision: float
    ecl_2026_provision: float
    delta: float
    pct_change: float
    binding_in_2026: bool
    capital_add_back_amount: float


def parallel_run(
    stage: IFRS9Stage,
    pd_12m: float,
    lgd: float,
    ead: float,
    *,
    eir: float = 0.0,
    marginal_pds: np.ndarray | None = None,
    # Legacy IRACP inputs
    irac_class: IRACAssetClass | None = None,
    sector: str = "commercial",
    # 2026 framework inputs
    category: RBIExposureCategory = RBIExposureCategory.OTHER,
    is_secured: bool = True,
    has_eligible_collateral: bool = False,
    years_in_stage3: float = 0.0,
    dlg_remaining_capacity: float = 0.0,
    is_wilful_defaulter: bool = False,
    is_sovereign_slr: bool = False,
    # Transition
    reporting_fy: int = 2028,
) -> ParallelRunResult:
    """Compute IRACP and ECL 2026 provisions side by side.

    Use during FY 2027 parallel-run exercises to quantify the capital
    impact of the new Master Direction. The capital add-back amount
    is determined by the FY in which the bank is reporting:

        FY 2027-28 (reporting_fy=2028): add back 4/5 of the delta
        FY 2028-29 (reporting_fy=2029): add back 3/5
        FY 2029-30 (reporting_fy=2030): add back 2/5
        FY 2030-31 (reporting_fy=2031): add back 1/5
        FY 2031-32 onwards:             no add-back

    Args:
        stage: IFRS 9 / Ind AS 109 stage.
        pd_12m: 12-month PD.
        lgd: Loss given default.
        ead: Exposure at default.
        eir: Effective interest rate.
        marginal_pds: Marginal PD curve for lifetime ECL.
        irac_class: Legacy IRAC classification.
        sector: Sector for legacy IRACP standard-asset rate.
        category: RBI exposure category under the 2026 framework.
        is_secured: Whether the exposure is secured.
        has_eligible_collateral: Eligible collateral flag (Para 98).
        years_in_stage3: Years since Stage 3 classification.
        dlg_remaining_capacity: Remaining DLG capacity.
        is_wilful_defaulter: Wilful defaulter flag (Para 101(4)).
        is_sovereign_slr: Sovereign / SLR carve-out flag (Para 37-38).
        reporting_fy: Fiscal year of reporting (e.g., 2028 for FY 2027-28).

    Returns:
        :class:`ParallelRunResult` with both legs and reconciliation.

    Reference:
        RBI/DOR/2026-27/398 Paragraph 108 (capital transitional add-back).
    """
    iracp = calculate_ecl_ind_as(
        stage=stage,
        pd_12m=pd_12m,
        lgd=lgd,
        ead=ead,
        eir=eir,
        marginal_pds=marginal_pds,
        irac_class=irac_class,
        is_secured=is_secured,
        sector=sector,
    )

    ecl_2026 = calculate_ecl_ind_as_2026(
        stage=stage,
        pd_12m=pd_12m,
        lgd=lgd,
        ead=ead,
        eir=eir,
        marginal_pds=marginal_pds,
        category=category,
        is_secured=is_secured,
        has_eligible_collateral=has_eligible_collateral,
        years_in_stage3=years_in_stage3,
        dlg_remaining_capacity=dlg_remaining_capacity,
        is_wilful_defaulter=is_wilful_defaulter,
        is_sovereign_slr=is_sovereign_slr,
    )

    delta = ecl_2026 - iracp
    pct_change = delta / iracp if iracp > 0 else 0.0

    # Diagnose whether the 2026 regulatory floor is binding (vs model ECL)
    binding_in_2026 = False
    if not is_sovereign_slr and ecl_2026 > 0:
        # If we re-compute with the floor off, does it materially drop?
        # Floor binding is approximated by checking if ECL == floor amount.
        # The exact comparison is brittle due to PD/LGD floor effects, so
        # we use a 1% tolerance on the regulatory floor.
        from creditriskengine.ecl.ind_as109.provision_floors_2026 import (
            rbi_ecl_floor_2026,
        )
        reg_floor = rbi_ecl_floor_2026(
            ead, stage, category,
            is_secured=is_secured,
            years_in_stage3=years_in_stage3,
            is_wilful_defaulter=is_wilful_defaulter,
        )
        binding_in_2026 = ecl_2026 >= reg_floor * 0.999

    add_back_factor = capital_add_back_factor(reporting_fy)
    add_back_amount = max(delta, 0.0) * add_back_factor

    return ParallelRunResult(
        iracp_provision=iracp,
        ecl_2026_provision=ecl_2026,
        delta=delta,
        pct_change=pct_change,
        binding_in_2026=binding_in_2026,
        capital_add_back_amount=add_back_amount,
    )


def portfolio_parallel_run_summary(
    results: list[ParallelRunResult],
) -> dict[str, float]:
    """Aggregate per-exposure parallel-run results into a portfolio summary.

    Useful for board reporting and the Statement on Feedback narrative
    around portfolio-level capital impact.

    Args:
        results: List of per-exposure :class:`ParallelRunResult` items.

    Returns:
        Dict with portfolio totals and average percentage change.
    """
    if not results:
        return {
            "n_exposures": 0,
            "total_iracp": 0.0,
            "total_ecl_2026": 0.0,
            "total_delta": 0.0,
            "weighted_pct_change": 0.0,
            "total_capital_add_back": 0.0,
            "n_binding_in_2026": 0,
        }

    total_iracp = sum(r.iracp_provision for r in results)
    total_2026 = sum(r.ecl_2026_provision for r in results)
    total_delta = total_2026 - total_iracp
    weighted_pct = total_delta / total_iracp if total_iracp > 0 else 0.0
    total_add_back = sum(r.capital_add_back_amount for r in results)
    n_binding = sum(1 for r in results if r.binding_in_2026)

    return {
        "n_exposures": len(results),
        "total_iracp": total_iracp,
        "total_ecl_2026": total_2026,
        "total_delta": total_delta,
        "weighted_pct_change": weighted_pct,
        "total_capital_add_back": total_add_back,
        "n_binding_in_2026": n_binding,
    }


__all__ = [
    "ParallelRunResult",
    "parallel_run",
    "portfolio_parallel_run_summary",
    "rbi_minimum_provision",  # re-export for convenience
]
