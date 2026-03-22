"""Tests for WoE-IV binning — EBA GL/2017/16."""

import numpy as np
import pytest

from creditriskengine.models.pd.binning import (
    BinResult,
    calculate_iv,
    calculate_woe,
    equal_width_binning,
    monotonic_binning,
    optimal_binning,
    quantile_binning,
    apply_woe_transform,
)


@pytest.fixture
def binary_data():
    """Synthetic binary classification data with signal."""
    rng = np.random.default_rng(42)
    n = 2000
    values = rng.normal(0, 1, n)
    # Higher values -> higher default probability
    prob = 1 / (1 + np.exp(-0.8 * values))
    target = (rng.random(n) < prob).astype(np.int64)
    return values, target


class TestCalculateWoE:
    """Test WoE computation."""

    def test_basic_woe(self) -> None:
        n_events = np.array([10, 30, 60])
        n_non_events = np.array([90, 70, 40])
        woe = calculate_woe(n_events, n_non_events)
        assert woe.shape == (3,)
        # Bin with more events relative to non-events should have higher WoE
        assert woe[2] > woe[0]

    def test_zero_handling(self) -> None:
        """Laplace smoothing should handle zero bins."""
        n_events = np.array([0, 50, 50])
        n_non_events = np.array([100, 0, 100])
        woe = calculate_woe(n_events, n_non_events)
        assert np.all(np.isfinite(woe))


class TestCalculateIV:
    """Test Information Value computation."""

    def test_iv_positive(self) -> None:
        woe = np.array([-0.5, 0.0, 0.5])
        pct_events = np.array([0.2, 0.3, 0.5])
        pct_non_events = np.array([0.4, 0.3, 0.3])
        iv = calculate_iv(woe, pct_events, pct_non_events)
        assert iv > 0.0

    def test_iv_zero_for_equal_distributions(self) -> None:
        woe = np.array([0.0, 0.0, 0.0])
        pct = np.array([1 / 3, 1 / 3, 1 / 3])
        iv = calculate_iv(woe, pct, pct)
        assert iv == pytest.approx(0.0, abs=1e-10)


class TestQuantileBinning:
    """Test equal-frequency binning."""

    def test_basic(self, binary_data) -> None:
        values, target = binary_data
        result = quantile_binning(values, target, n_bins=5)
        assert isinstance(result, BinResult)
        assert len(result.woe_values) == 5
        assert result.iv > 0.0

    def test_bin_counts_sum(self, binary_data) -> None:
        values, target = binary_data
        result = quantile_binning(values, target, n_bins=10)
        assert result.bin_counts.sum() == len(values)


class TestMonotonicBinning:
    """Test monotonic WoE binning."""

    def test_monotonic_woe(self, binary_data) -> None:
        values, target = binary_data
        result = monotonic_binning(values, target, n_bins=10)
        # WoE should be monotonically increasing or decreasing
        diffs = np.diff(result.woe_values)
        assert np.all(diffs >= -1e-10) or np.all(diffs <= 1e-10)

    def test_auto_direction(self, binary_data) -> None:
        values, target = binary_data
        result = monotonic_binning(values, target, increasing=None)
        assert isinstance(result, BinResult)


class TestOptimalBinning:
    """Test decision tree-based optimal binning."""

    def test_basic(self, binary_data) -> None:
        values, target = binary_data
        result = optimal_binning(values, target, max_bins=8)
        assert isinstance(result, BinResult)
        assert len(result.woe_values) <= 8
        assert result.iv > 0.0

    def test_min_bin_pct(self, binary_data) -> None:
        values, target = binary_data
        result = optimal_binning(values, target, max_bins=5, min_bin_pct=0.10)
        min_pct = result.bin_counts.min() / result.bin_counts.sum()
        # Allow small tolerance due to binning mechanics
        assert min_pct >= 0.05


class TestApplyWoETransform:
    """Test WoE transformation of new data."""

    def test_transform_matches_bins(self, binary_data) -> None:
        values, target = binary_data
        result = quantile_binning(values, target, n_bins=5)
        transformed = apply_woe_transform(values, result)
        assert transformed.shape == values.shape
        # Each transformed value should be one of the WoE values
        unique_woe = np.unique(transformed)
        assert len(unique_woe) <= len(result.woe_values) + 1  # +1 for edge cases
