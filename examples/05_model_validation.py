"""
Example 05: Model Validation

Demonstrates the model validation toolkit for assessing PD model performance:
- Discriminatory power (AUROC, Gini, KS)
- Calibration (Binomial test, Traffic Light)
- Stability (PSI)
"""

import numpy as np

from creditriskengine.validation.discrimination import (
    auroc,
    gini_coefficient,
    ks_statistic,
    information_value,
)
from creditriskengine.validation.calibration import (
    binomial_test,
    brier_score,
    traffic_light_test,
)
from creditriskengine.validation.stability import (
    population_stability_index,
)


def main() -> None:
    print("=" * 60)
    print("Model Validation Toolkit Example")
    print("=" * 60)

    # -------------------------------------------------------
    # 1. Generate synthetic model output
    # -------------------------------------------------------
    rng = np.random.default_rng(42)
    n = 5000
    n_defaults = 150

    # Simulate PD predictions and actual outcomes
    actual = np.zeros(n, dtype=int)
    actual[:n_defaults] = 1
    rng.shuffle(actual)

    # Good model: defaults get higher PDs
    predicted_pd = rng.beta(2, 50, size=n)  # base distribution
    predicted_pd[actual == 1] += rng.beta(5, 10, size=n_defaults) * 0.1
    predicted_pd = np.clip(predicted_pd, 0.001, 0.999)

    # -------------------------------------------------------
    # 2. Discriminatory Power
    # -------------------------------------------------------
    print("\n--- Discriminatory Power ---")

    auc = auroc(actual, predicted_pd)
    gini = gini_coefficient(actual, predicted_pd)
    ks = ks_statistic(actual, predicted_pd)

    print(f"AUROC:            {auc:.4f}")
    print(f"Gini coefficient: {gini:.4f}")
    print(f"KS statistic:     {ks:.4f}")

    # Information Value
    iv = information_value(actual, predicted_pd, n_bins=10)
    print(f"Information Value: {iv:.4f}", end="  ")
    if iv < 0.02:
        print("(Useless)")
    elif iv < 0.1:
        print("(Weak)")
    elif iv < 0.3:
        print("(Medium)")
    elif iv < 0.5:
        print("(Strong)")
    else:
        print("(Suspicious)")

    # -------------------------------------------------------
    # 3. Calibration Tests
    # -------------------------------------------------------
    print("\n--- Calibration ---")

    avg_pd = float(np.mean(predicted_pd))
    observed_dr = n_defaults / n

    print(f"Average predicted PD: {avg_pd:.4%}")
    print(f"Observed default rate: {observed_dr:.4%}")

    # Binomial test
    binom = binomial_test(
        n_observations=n,
        n_defaults=n_defaults,
        predicted_pd=avg_pd,
    )
    print(f"Binomial test z-stat: {binom['z_stat']:.3f}, "
          f"reject H0: {binom['reject_h0']}")

    # Brier score
    bs = brier_score(actual, predicted_pd)
    print(f"Brier score: {bs:.6f}")

    # Traffic light test (BCBS WP14)
    tl = traffic_light_test(
        n_observations=n,
        n_defaults=n_defaults,
        predicted_pd=avg_pd,
    )
    print(f"Traffic light: {tl['zone']}")

    # -------------------------------------------------------
    # 4. Stability (PSI)
    # -------------------------------------------------------
    print("\n--- Stability ---")

    # Compare development vs current score distributions
    dev_scores = rng.normal(500, 50, size=3000)
    current_scores = rng.normal(495, 55, size=3000)  # slight drift

    psi = population_stability_index(dev_scores, current_scores, n_bins=10)
    print(f"PSI: {psi:.4f}", end="  ")
    if psi < 0.10:
        print("(No significant change)")
    elif psi < 0.25:
        print("(Moderate shift — investigate)")
    else:
        print("(Significant shift — action required)")

    # -------------------------------------------------------
    # 5. Overall Assessment
    # -------------------------------------------------------
    print("\n--- Overall Model Assessment ---")
    checks = [
        ("AUROC > 0.70", auc > 0.70),
        ("Gini > 0.40", gini > 0.40),
        ("KS > 0.30", ks > 0.30),
        ("IV in [0.02, 0.50]", 0.02 <= iv <= 0.50),
        ("Binomial: not rejected", not binom["reject_h0"]),
        ("Traffic light: not red", tl["zone"] != "red"),
        ("PSI < 0.25", psi < 0.25),
    ]

    for desc, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {desc}")

    n_passed = sum(1 for _, p in checks if p)
    print(f"\nResult: {n_passed}/{len(checks)} checks passed")


if __name__ == "__main__":
    main()
