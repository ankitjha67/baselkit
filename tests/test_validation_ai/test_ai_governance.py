"""Tests for creditriskengine.validation.ai_governance package."""

from __future__ import annotations

import numpy as np
import pytest

from creditriskengine.validation.ai_governance import (
    demographic_parity_difference,
    detect_psi_drift,
    disparate_impact_ratio,
    equal_opportunity_difference,
    psi,
)
from creditriskengine.validation.ai_governance.drift import DriftSeverity

# ============================================================================
# Fairness metrics
# ============================================================================


class TestDisparateImpact:
    def test_perfect_parity(self) -> None:
        y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        sens = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        di = disparate_impact_ratio(y_pred, sens)
        assert di == pytest.approx(1.0)

    def test_below_four_fifths_rule(self) -> None:
        # Privileged: 4/5 = 80% favorable
        # Unprivileged: 1/5 = 20% favorable → DI = 0.20/0.80 = 0.25
        y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 0])
        sens = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        di = disparate_impact_ratio(y_pred, sens)
        assert di < 0.80

    def test_all_favorable(self) -> None:
        y_pred = np.ones(10, dtype=int)
        sens = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        assert disparate_impact_ratio(y_pred, sens) == pytest.approx(1.0)

    def test_empty_group_returns_1(self) -> None:
        y_pred = np.array([1, 0, 1])
        sens = np.array([1, 1, 1])  # No unprivileged
        assert disparate_impact_ratio(y_pred, sens) == 1.0

    def test_zero_privileged_rate_returns_0(self) -> None:
        # Privileged group has zero favorable predictions -> ratio 0.0
        y_pred = np.array([0, 0, 1, 1])
        sens = np.array([1, 1, 0, 0])
        assert disparate_impact_ratio(y_pred, sens) == 0.0


class TestDemographicParity:
    def test_perfect_parity(self) -> None:
        y_pred = np.array([1, 0, 1, 0])
        sens = np.array([1, 1, 0, 0])
        dpd = demographic_parity_difference(y_pred, sens)
        assert dpd == pytest.approx(0.0)

    def test_unfavorable_to_unprivileged(self) -> None:
        # Privileged: 100% favorable, Unprivileged: 0% → DPD = -1.0
        y_pred = np.array([1, 1, 0, 0])
        sens = np.array([1, 1, 0, 0])
        dpd = demographic_parity_difference(y_pred, sens)
        assert dpd == pytest.approx(-1.0)

    def test_within_tolerance(self) -> None:
        y_pred = np.array([1, 1, 0, 1, 1, 0, 0, 1])
        sens = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        dpd = demographic_parity_difference(y_pred, sens)
        assert -0.30 <= dpd <= 0.30

    def test_empty_group_returns_zero(self) -> None:
        y_pred = np.array([1, 0, 1])
        sens = np.array([1, 1, 1])  # No unprivileged
        assert demographic_parity_difference(y_pred, sens) == 0.0


class TestEqualOpportunity:
    def test_perfect_equal_opportunity(self) -> None:
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        sens = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        eod = equal_opportunity_difference(y_true, y_pred, sens)
        assert eod == pytest.approx(0.0)

    def test_unequal_tpr(self) -> None:
        # Privileged: TPR = 2/2 = 1.0; Unprivileged: TPR = 0/2 = 0.0
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 0, 0, 0, 0])
        sens = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        eod = equal_opportunity_difference(y_true, y_pred, sens)
        assert eod == pytest.approx(-1.0)

    def test_no_privileged_returns_zero(self) -> None:
        y_true = np.array([1, 0, 1])
        y_pred = np.array([1, 0, 1])
        sens = np.array([0, 0, 0])
        assert equal_opportunity_difference(y_true, y_pred, sens) == 0.0

    def test_group_without_actual_positives_tpr_zero(self) -> None:
        # Unprivileged group has no truly-favorable individuals, so its TPR
        # falls back to 0.0; privileged TPR = 1.0 -> EOD = 0.0 - 1.0 = -1.0
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 0])
        sens = np.array([1, 1, 0, 0])
        eod = equal_opportunity_difference(y_true, y_pred, sens)
        assert eod == pytest.approx(-1.0)


# ============================================================================
# Drift detection (PSI)
# ============================================================================


class TestPSI:
    def test_identical_distributions(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 1000)
        psi_val = psi(data, data, n_bins=10)
        assert psi_val < 0.01

    def test_shifted_distribution_high_psi(self) -> None:
        rng = np.random.default_rng(42)
        expected = rng.normal(0, 1, 1000)
        actual = rng.normal(2, 1, 1000)  # Mean shift of 2
        psi_val = psi(expected, actual, n_bins=10)
        assert psi_val > 0.25

    def test_detect_stable(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 1000)
        psi_val, severity = detect_psi_drift(data, data)
        assert severity == DriftSeverity.STABLE

    def test_detect_material_drift(self) -> None:
        rng = np.random.default_rng(42)
        expected = rng.normal(0, 1, 1000)
        actual = rng.normal(3, 1, 1000)
        psi_val, severity = detect_psi_drift(expected, actual)
        assert severity == DriftSeverity.MATERIAL

    def test_psi_is_non_negative(self) -> None:
        rng = np.random.default_rng(42)
        expected = rng.normal(0, 1, 500)
        actual = rng.normal(0.5, 1.2, 500)
        assert psi(expected, actual) >= 0.0

    def test_detect_minor_drift(self) -> None:
        rng = np.random.default_rng(42)
        expected = rng.normal(0, 1, 2000)
        actual = rng.normal(0.3, 1.05, 2000)
        psi_val, severity = detect_psi_drift(expected, actual)
        assert severity in (DriftSeverity.MINOR, DriftSeverity.STABLE)
