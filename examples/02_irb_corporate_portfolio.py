"""
Example 02: IRB Corporate Portfolio

Demonstrates computing RWA for a portfolio of corporate exposures using
IRB formulas, including SME firm-size adjustment and maturity adjustment.
"""

from creditriskengine.rwa.irb.formulas import (
    asset_correlation_corporate,
    irb_risk_weight,
    maturity_adjustment,
    sme_firm_size_adjustment,
)


def main() -> None:
    # Sample corporate portfolio
    portfolio = [
        {"name": "Large Corp A", "pd": 0.005, "lgd": 0.45, "ead": 50_000_000,
         "maturity": 3.0, "turnover": None},
        {"name": "Large Corp B", "pd": 0.02, "lgd": 0.45, "ead": 30_000_000,
         "maturity": 4.5, "turnover": None},
        {"name": "SME Corp C", "pd": 0.015, "lgd": 0.45, "ead": 10_000_000,
         "maturity": 2.5, "turnover": 15.0},
        {"name": "SME Corp D", "pd": 0.03, "lgd": 0.40, "ead": 5_000_000,
         "maturity": 2.0, "turnover": 8.0},
        {"name": "Low Risk E", "pd": 0.001, "lgd": 0.45, "ead": 80_000_000,
         "maturity": 2.5, "turnover": None},
    ]

    print("=" * 90)
    print(f"{'Obligor':<16} {'PD':>6} {'LGD':>6} {'EAD':>14} {'M':>4} "
          f"{'R':>7} {'MA':>6} {'RW%':>7} {'RWA':>14}")
    print("-" * 90)

    total_ead = 0.0
    total_rwa = 0.0

    for exp in portfolio:
        rw = irb_risk_weight(
            pd=exp["pd"],
            lgd=exp["lgd"],
            asset_class="corporate",
            maturity=exp["maturity"],
            turnover_eur_millions=exp["turnover"],
        )
        rwa = exp["ead"] * rw / 100.0
        total_ead += exp["ead"]
        total_rwa += rwa

        # Show intermediate calculations
        r = asset_correlation_corporate(max(exp["pd"], 0.0003))
        if exp["turnover"] is not None:
            r += sme_firm_size_adjustment(exp["turnover"])
        ma = maturity_adjustment(max(exp["pd"], 0.0003),
                                 max(1.0, min(exp["maturity"], 5.0)))

        print(f"{exp['name']:<16} {exp['pd']:>6.2%} {exp['lgd']:>6.0%} "
              f"{exp['ead']:>14,.0f} {exp['maturity']:>4.1f} "
              f"{r:>7.4f} {ma:>6.3f} {rw:>7.1f} {rwa:>14,.0f}")

    print("-" * 90)
    avg_rw = (total_rwa / total_ead * 100) if total_ead > 0 else 0
    capital = total_rwa * 0.08
    print(f"{'TOTAL':<16} {'':>6} {'':>6} {total_ead:>14,.0f} {'':>4} "
          f"{'':>7} {'':>6} {avg_rw:>7.1f} {total_rwa:>14,.0f}")
    print(f"\nMinimum capital requirement (8%): {capital:,.0f}")


if __name__ == "__main__":
    main()
