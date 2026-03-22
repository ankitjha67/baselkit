"""
Example 04: Scorecard Development

Demonstrates building a PD scorecard using Weight of Evidence (WoE)
transformation, logistic regression, and calibration to a master rating scale.
"""

import numpy as np

from creditriskengine.models.pd.scorecard import (
    build_master_scale,
    assign_rating_grade,
    calibrate_pd_anchor_point,
    logistic_score,
    score_to_pd,
    pd_to_score,
)


def main() -> None:
    print("=" * 60)
    print("PD Scorecard Development Example")
    print("=" * 60)

    # -------------------------------------------------------
    # 1. Simulated logistic regression output
    # -------------------------------------------------------
    # Assume we have a fitted logistic regression with these coefficients
    # (typically from WoE-transformed features)
    intercept = -3.5
    coefficients = np.array([0.8, 0.5, 0.3, -0.2])
    feature_values = np.array([1.2, 0.5, -0.3, 0.8])  # WoE-transformed values

    score = logistic_score(intercept, coefficients, feature_values)
    pd_raw = score_to_pd(score)
    print(f"\nLogistic score: {score:.4f}")
    print(f"Raw PD:         {pd_raw:.4%}")

    # -------------------------------------------------------
    # 2. Industry scorecard scaling
    # -------------------------------------------------------
    # Convert PD to standard scorecard points
    # Parameters: base_score=600, PDO=20 (points to double odds)
    scorecard_points = pd_to_score(
        pd_raw,
        base_score=600.0,
        pdo=20.0,
        base_odds=50.0,
    )
    print(f"Scorecard points: {scorecard_points:.0f}")

    # -------------------------------------------------------
    # 3. Master Rating Scale
    # -------------------------------------------------------
    print(f"\n{'Grade':<8} {'PD Lower':>10} {'PD Upper':>10} {'PD Midpoint':>12}")
    print("-" * 45)

    grade_boundaries = [0.0003, 0.001, 0.005, 0.01, 0.03, 0.05, 0.10, 0.20, 1.0]
    master_scale = build_master_scale(grade_boundaries)

    for grade in master_scale:
        print(f"{grade['label']:<8} {grade['pd_lower']:>10.4%} "
              f"{grade['pd_upper']:>10.4%} {grade['pd_midpoint']:>12.4%}")

    # Assign a rating
    rating = assign_rating_grade(pd_raw, master_scale)
    print(f"\nPD {pd_raw:.4%} → Rating grade: {rating}")

    # -------------------------------------------------------
    # 4. Calibration to Central Tendency
    # -------------------------------------------------------
    # Portfolio of raw PDs from model
    raw_pds = np.array([0.005, 0.008, 0.012, 0.020, 0.035, 0.050, 0.080])
    central_tendency = 0.025  # Long-run average default rate

    calibrated_pds = calibrate_pd_anchor_point(raw_pds, central_tendency)

    print(f"\nCalibration (central tendency = {central_tendency:.2%}):")
    print(f"  {'Raw PD':>8}  {'Calibrated':>10}")
    for raw, cal in zip(raw_pds, calibrated_pds):
        print(f"  {raw:>8.3%}  {cal:>10.3%}")


if __name__ == "__main__":
    main()
