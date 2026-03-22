"""
Example 06: Multi-Jurisdiction Regulatory Calculations

Demonstrates computing RWA and capital requirements for the same portfolio
under different regulatory jurisdictions (BCBS, EU CRR3, UK PRA, India RBI).
"""

from creditriskengine.core.types import (
    CreditQualityStep,
    Jurisdiction,
    SAExposureClass,
)
from creditriskengine.rwa.standardized.credit_risk_sa import (
    assign_sa_risk_weight,
    get_corporate_risk_weight,
    get_residential_re_risk_weight,
)
from creditriskengine.rwa.output_floor import get_output_floor_pct


def main() -> None:
    print("=" * 70)
    print("Multi-Jurisdiction Comparison")
    print("=" * 70)

    jurisdictions = [
        Jurisdiction.BCBS,
        Jurisdiction.EU,
        Jurisdiction.UK,
        Jurisdiction.INDIA,
    ]

    # -------------------------------------------------------
    # 1. Corporate Unrated: varies by jurisdiction
    # -------------------------------------------------------
    print("\n--- Corporate Unrated Risk Weights ---")
    print(f"{'Jurisdiction':<15} {'Standard':>10} {'IG Unrated':>12}")
    for j in jurisdictions:
        rw_std = get_corporate_risk_weight(CreditQualityStep.UNRATED, j)
        rw_ig = get_corporate_risk_weight(
            CreditQualityStep.UNRATED, j, is_investment_grade=True,
        )
        print(f"{j.value:<15} {rw_std:>10.0f}% {rw_ig:>11.0f}%")

    # -------------------------------------------------------
    # 2. Residential Mortgage by LTV
    # -------------------------------------------------------
    print("\n--- Residential Mortgage Risk Weights by LTV ---")
    ltv_values = [0.50, 0.60, 0.70, 0.80, 0.90]
    header = f"{'LTV':>5}"
    for j in jurisdictions:
        header += f"  {j.value:>8}"
    print(header)

    for ltv in ltv_values:
        row = f"{ltv:>5.0%}"
        for j in jurisdictions:
            rw = get_residential_re_risk_weight(ltv=ltv, jurisdiction=j)
            row += f"  {rw:>7.0f}%"
        print(row)

    # -------------------------------------------------------
    # 3. EU SME Supporting Factor
    # -------------------------------------------------------
    print("\n--- EU SME Supporting Factor (CRR3 Art. 501) ---")
    rw_corp = get_corporate_risk_weight(CreditQualityStep.UNRATED, Jurisdiction.EU)
    rw_sme = get_corporate_risk_weight(
        CreditQualityStep.UNRATED, Jurisdiction.EU, is_sme=True,
    )
    print(f"EU Corporate Unrated:     {rw_corp:.1f}%")
    print(f"EU SME Corporate Unrated: {rw_sme:.1f}% (×0.7619)")

    # -------------------------------------------------------
    # 4. Output Floor Phase-in
    # -------------------------------------------------------
    print("\n--- Output Floor Phase-In (as of 2026-01-01) ---")
    from datetime import date
    report_date = date(2026, 1, 1)
    print(f"{'Jurisdiction':<15} {'Floor %':>8}")
    for j in jurisdictions:
        try:
            floor = get_output_floor_pct(j, report_date)
            print(f"{j.value:<15} {floor:>7.1%}")
        except Exception:
            print(f"{j.value:<15} {'N/A':>8}")

    # -------------------------------------------------------
    # 5. Portfolio RWA comparison
    # -------------------------------------------------------
    print("\n--- Portfolio RWA Comparison ---")
    portfolio = [
        {"class": SAExposureClass.SOVEREIGN, "cqs": CreditQualityStep.CQS_1,
         "ead": 50_000_000, "label": "Sovereign AAA"},
        {"class": SAExposureClass.CORPORATE, "cqs": CreditQualityStep.CQS_3,
         "ead": 30_000_000, "label": "Corporate BBB"},
        {"class": SAExposureClass.CORPORATE, "cqs": CreditQualityStep.UNRATED,
         "ead": 20_000_000, "label": "Corporate Unrated"},
        {"class": SAExposureClass.RESIDENTIAL_MORTGAGE, "cqs": CreditQualityStep.UNRATED,
         "ead": 40_000_000, "label": "RRE (LTV=70%)"},
        {"class": SAExposureClass.RETAIL_REGULATORY, "cqs": CreditQualityStep.UNRATED,
         "ead": 10_000_000, "label": "Retail Regulatory"},
    ]

    total_ead = sum(p["ead"] for p in portfolio)

    for j in jurisdictions:
        total_rwa = 0.0
        for p in portfolio:
            kwargs = {"exposure_class": p["class"], "cqs": p["cqs"], "jurisdiction": j}
            if p["class"] == SAExposureClass.RESIDENTIAL_MORTGAGE:
                kwargs["ltv"] = 0.70
            rw = assign_sa_risk_weight(**kwargs)
            rwa = p["ead"] * rw / 100.0
            total_rwa += rwa
        avg_rw = total_rwa / total_ead * 100
        capital = total_rwa * 0.08
        print(f"{j.value:<15}  RWA={total_rwa:>14,.0f}  "
              f"Avg RW={avg_rw:>5.1f}%  Capital={capital:>12,.0f}")


if __name__ == "__main__":
    main()
