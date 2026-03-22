"""
Example 01: Basic RWA Calculation

Demonstrates how to compute risk-weighted assets (RWA) using both
the Standardized Approach (SA) and IRB approach for simple exposures.
"""

from creditriskengine.core.types import (
    CreditQualityStep,
    Jurisdiction,
    SAExposureClass,
)
from creditriskengine.rwa.irb.formulas import irb_risk_weight
from creditriskengine.rwa.standardized.credit_risk_sa import (
    assign_sa_risk_weight,
    get_sovereign_risk_weight,
    get_corporate_risk_weight,
    get_residential_re_risk_weight,
)


def main() -> None:
    # -------------------------------------------------------
    # 1. SA Risk Weights for various exposure classes
    # -------------------------------------------------------
    print("=" * 60)
    print("Standardized Approach (SA) Risk Weights")
    print("=" * 60)

    # Sovereign: AAA-rated → 0% (BCBS CRE20.7, Table 1)
    rw = get_sovereign_risk_weight(CreditQualityStep.CQS_1)
    print(f"Sovereign AAA (CQS 1):   {rw:.0f}%")

    # Sovereign: BBB-rated → 50%
    rw = get_sovereign_risk_weight(CreditQualityStep.CQS_3)
    print(f"Sovereign BBB (CQS 3):   {rw:.0f}%")

    # Corporate: A-rated → 50% (BCBS CRE20 Table 7)
    rw = get_corporate_risk_weight(CreditQualityStep.CQS_2)
    print(f"Corporate A (CQS 2):     {rw:.0f}%")

    # Corporate: Unrated → 100%
    rw = get_corporate_risk_weight(CreditQualityStep.UNRATED)
    print(f"Corporate Unrated:       {rw:.0f}%")

    # Corporate: UK PRA unrated investment-grade → 65% (PS9/24)
    rw = get_corporate_risk_weight(
        CreditQualityStep.UNRATED,
        jurisdiction=Jurisdiction.UK,
        is_investment_grade=True,
    )
    print(f"Corporate Unrated IG (UK): {rw:.0f}%")

    # Residential mortgage: LTV 70% → 30% (BCBS CRE20 Table 12)
    rw = get_residential_re_risk_weight(ltv=0.70)
    print(f"RRE LTV=70%:             {rw:.0f}%")

    # Residential mortgage: LTV 85% → 40%
    rw = get_residential_re_risk_weight(ltv=0.85)
    print(f"RRE LTV=85%:             {rw:.0f}%")

    # Master dispatcher example
    rw = assign_sa_risk_weight(
        exposure_class=SAExposureClass.DEFAULTED,
        specific_provisions_pct=0.10,
    )
    print(f"Defaulted (prov<20%):    {rw:.0f}%")

    # -------------------------------------------------------
    # 2. IRB Risk Weights
    # -------------------------------------------------------
    print("\n" + "=" * 60)
    print("IRB Approach Risk Weights")
    print("=" * 60)

    # Corporate: PD=1%, LGD=45%, M=2.5y → ~69%
    rw = irb_risk_weight(pd=0.01, lgd=0.45, asset_class="corporate", maturity=2.5)
    print(f"Corporate PD=1% LGD=45% M=2.5:  {rw:.1f}%")

    # Corporate: PD=0.03% (floor), LGD=45%, M=2.5y → ~14%
    rw = irb_risk_weight(pd=0.0003, lgd=0.45, asset_class="corporate", maturity=2.5)
    print(f"Corporate PD=0.03% LGD=45%:      {rw:.1f}%")

    # Residential mortgage: PD=1%, LGD=15%
    rw = irb_risk_weight(pd=0.01, lgd=0.15, asset_class="residential_mortgage")
    print(f"RRE PD=1% LGD=15%:               {rw:.1f}%")

    # SME corporate with firm-size adjustment
    rw = irb_risk_weight(
        pd=0.01, lgd=0.45, asset_class="corporate",
        maturity=2.5, turnover_eur_millions=10.0,
    )
    print(f"SME Corp PD=1% LGD=45% S=10M:    {rw:.1f}%")

    # QRRE: PD=2%, LGD=85%
    rw = irb_risk_weight(pd=0.02, lgd=0.85, asset_class="qrre")
    print(f"QRRE PD=2% LGD=85%:              {rw:.1f}%")

    # -------------------------------------------------------
    # 3. Compute RWA amounts
    # -------------------------------------------------------
    print("\n" + "=" * 60)
    print("RWA Amounts (EAD = 1,000,000)")
    print("=" * 60)

    ead = 1_000_000.0
    for label, rw_pct in [
        ("SA Corporate A-rated (50%)", 50.0),
        ("SA Retail regulatory (75%)", 75.0),
        ("IRB Corporate PD=1%", irb_risk_weight(0.01, 0.45, "corporate", 2.5)),
    ]:
        rwa = ead * rw_pct / 100.0
        capital = rwa * 0.08  # 8% minimum capital requirement
        print(f"{label:40s}  RWA={rwa:>12,.0f}  Capital={capital:>10,.0f}")


if __name__ == "__main__":
    main()
