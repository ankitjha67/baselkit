"""
Example 03: IFRS 9 ECL Pipeline

Demonstrates a complete IFRS 9 Expected Credit Loss calculation:
1. Stage allocation (SICR assessment)
2. Lifetime PD term structure
3. TTC-to-PIT PD conversion
4. 12-month and lifetime ECL computation
5. Multi-scenario probability weighting
"""

import numpy as np

from creditriskengine.core.types import IFRS9Stage
from creditriskengine.ecl.ifrs9.staging import assign_stage
from creditriskengine.ecl.ifrs9.sicr import assess_sicr
from creditriskengine.ecl.ifrs9.lifetime_pd import (
    cumulative_pd_from_annual,
    marginal_pd_from_cumulative,
)
from creditriskengine.ecl.ifrs9.ttc_to_pit import ttc_to_pit_pd
from creditriskengine.ecl.ifrs9.ecl_calc import ecl_12_month, ecl_lifetime
from creditriskengine.ecl.ifrs9.scenarios import weighted_ecl, Scenario


def main() -> None:
    # -------------------------------------------------------
    # 1. Sample loan data
    # -------------------------------------------------------
    print("=" * 60)
    print("IFRS 9 ECL Pipeline Example")
    print("=" * 60)

    # Loan parameters
    ead = 1_000_000.0
    lgd = 0.40
    eir = 0.05  # 5% effective interest rate
    remaining_life = 5  # years
    origination_pd = 0.008  # 0.8% at origination
    current_pd = 0.015  # 1.5% current (deteriorated)
    dpd = 15  # days past due

    # -------------------------------------------------------
    # 2. SICR Assessment
    # -------------------------------------------------------
    sicr_triggered = assess_sicr(
        origination_pd=origination_pd,
        current_pd=current_pd,
        days_past_due=dpd,
    )
    print(f"\nOrigination PD: {origination_pd:.2%}")
    print(f"Current PD:     {current_pd:.2%}")
    print(f"Relative change: {current_pd / origination_pd:.1f}x")
    print(f"SICR triggered: {sicr_triggered}")

    # -------------------------------------------------------
    # 3. Stage Assignment
    # -------------------------------------------------------
    stage = assign_stage(
        days_past_due=dpd,
        is_credit_impaired=False,
        sicr_triggered=sicr_triggered,
    )
    print(f"Assigned stage: Stage {stage}")

    # -------------------------------------------------------
    # 4. Lifetime PD Term Structure
    # -------------------------------------------------------
    annual_pds = np.array([current_pd] * remaining_life)
    cumulative_pds = cumulative_pd_from_annual(annual_pds)
    marginal_pds = marginal_pd_from_cumulative(cumulative_pds)

    print(f"\nPD Term Structure (flat {current_pd:.2%} annual):")
    print(f"  {'Year':>4}  {'Annual PD':>10}  {'Cumul PD':>10}  {'Marginal PD':>12}")
    for t in range(remaining_life):
        print(f"  {t + 1:>4}  {annual_pds[t]:>10.4%}  "
              f"{cumulative_pds[t]:>10.4%}  {marginal_pds[t]:>12.4%}")

    # -------------------------------------------------------
    # 5. TTC-to-PIT conversion (macro adjustment)
    # -------------------------------------------------------
    rho = 0.15  # asset correlation proxy
    z_factor = -0.5  # mildly adverse macro environment
    pit_pd = ttc_to_pit_pd(pd_ttc=current_pd, rho=rho, z_factor=z_factor)
    print(f"\nTTC PD: {current_pd:.4%}  →  PIT PD (Z={z_factor}): {pit_pd:.4%}")

    # -------------------------------------------------------
    # 6. ECL Calculation
    # -------------------------------------------------------
    # Stage 1: 12-month ECL
    ecl_12m = ecl_12_month(pd_12m=current_pd, lgd=lgd, ead=ead, eir=eir)
    print(f"\nStage 1 (12-month ECL): {ecl_12m:,.2f}")

    # Stage 2: Lifetime ECL
    lgd_array = np.full(remaining_life, lgd)
    ead_array = np.full(remaining_life, ead)  # simplified: no amortisation
    ecl_life = ecl_lifetime(
        marginal_pds=marginal_pds,
        lgd=lgd_array,
        ead=ead_array,
        eir=eir,
    )
    print(f"Stage 2 (Lifetime ECL): {ecl_life:,.2f}")

    # -------------------------------------------------------
    # 7. Multi-Scenario Weighting (IFRS 9.5.5.17)
    # -------------------------------------------------------
    # Three scenarios with different macro conditions
    scenarios = [
        Scenario(name="Base", weight=0.50, ecl=ecl_life),
        Scenario(name="Upside", weight=0.25, ecl=ecl_life * 0.7),
        Scenario(name="Downside", weight=0.25, ecl=ecl_life * 1.5),
    ]

    final_ecl = weighted_ecl(scenarios)
    print(f"\nMulti-scenario ECL:")
    for s in scenarios:
        print(f"  {s.name:>10}: ECL={s.ecl:>12,.2f}  Weight={s.weight:.0%}")
    print(f"  {'Weighted':>10}: ECL={final_ecl:>12,.2f}")
    print(f"\nECL as % of EAD: {final_ecl / ead:.4%}")


if __name__ == "__main__":
    main()
