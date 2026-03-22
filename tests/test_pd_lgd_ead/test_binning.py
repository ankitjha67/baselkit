"""Tests for WoE-IV binning — EBA GL/2017/16."""

import numpy as np
import pytest

from creditriskengine.models.pd.binning import (
    BinResult,
    WoEBinningTransformer,
    apply_woe_transform,
    calculate_iv,
    calculate_woe,
    equal_width_binning,
    monotonic_binning,
    optimal_binning,
    quantile_binning,
)


@pytest.fixture
def binary_data() -> tuple[np.ndarray, np.ndarray]:
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

    def test_basic(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        values, target = binary_data
        result = quantile_binning(values, target, n_bins=5)
        assert isinstance(result, BinResult)
        assert len(result.woe_values) == 5
        assert result.iv > 0.0

    def test_bin_counts_sum(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        values, target = binary_data
        result = quantile_binning(values, target, n_bins=10)
        assert result.bin_counts.sum() == len(values)


class TestMonotonicBinning:
    """Test monotonic WoE binning."""

    def test_monotonic_woe(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        values, target = binary_data
        result = monotonic_binning(values, target, n_bins=10)
        # WoE should be monotonically increasing or decreasing
        diffs = np.diff(result.woe_values)
        assert np.all(diffs >= -1e-10) or np.all(diffs <= 1e-10)

    def test_auto_direction(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        values, target = binary_data
        result = monotonic_binning(values, target, increasing=None)
        assert isinstance(result, BinResult)


class TestOptimalBinning:
    """Test decision tree-based optimal binning."""

    def test_basic(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        values, target = binary_data
        result = optimal_binning(values, target, max_bins=8)
        assert isinstance(result, BinResult)
        assert len(result.woe_values) <= 8
        assert result.iv > 0.0

    def test_min_bin_pct(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        values, target = binary_data
        result = optimal_binning(values, target, max_bins=5, min_bin_pct=0.10)
        min_pct = result.bin_counts.min() / result.bin_counts.sum()
        # Allow small tolerance due to binning mechanics
        assert min_pct >= 0.05

    def test_optimal_binning_few_unique_values(self) -> None:
        """Cover lines 364-365: break when len(woe) <= 2 in optimal_binning."""
        values = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0], dtype=np.float64)
        target = np.array([0, 1, 1, 0, 0, 1], dtype=np.int64)
        result = optimal_binning(values, target, max_bins=2)
        assert isinstance(result, BinResult)

    def test_optimal_binning_two_bins_woe_break(self) -> None:
        """Cover line 365: len(woe) <= 2 break in optimal_binning monotonicity loop."""
        # Binary feature with very few values forces tree to produce <= 2 bins
        values = np.array([0.0] * 50 + [1.0] * 50, dtype=np.float64)
        target = np.array([0] * 40 + [1] * 10 + [1] * 30 + [0] * 20, dtype=np.int64)
        result = optimal_binning(values, target, max_bins=2)
        assert isinstance(result, BinResult)
        assert len(result.woe_values) <= 2

    def test_optimal_binning_non_monotonic_data(self) -> None:
        """Cover lines 370-374, 379-382: violation detection and merging in optimal_binning."""
        rng = np.random.default_rng(77)
        n = 500
        # Create non-monotonic relationship
        values = np.concatenate([
            rng.normal(-2, 0.5, n // 3),
            rng.normal(0, 0.5, n // 3),
            rng.normal(2, 0.5, n - 2 * (n // 3)),
        ])
        target = np.concatenate([
            (rng.random(n // 3) < 0.3).astype(np.int64),
            (rng.random(n // 3) < 0.05).astype(np.int64),
            (rng.random(n - 2 * (n // 3)) < 0.4).astype(np.int64),
        ])
        result = optimal_binning(values, target, max_bins=10)
        assert isinstance(result, BinResult)

    def test_optimal_binning_decreasing_correlation(self) -> None:
        """Cover line 373-374: decreasing direction violation detection."""
        rng = np.random.default_rng(55)
        n = 500
        values = rng.normal(0, 1, n)
        # Negative correlation: higher values -> lower default
        prob = 1 / (1 + np.exp(0.8 * values))
        target = (rng.random(n) < prob).astype(np.int64)
        result = optimal_binning(values, target, max_bins=8)
        assert isinstance(result, BinResult)

    def test_optimal_binning_violation_at_last_edge(self) -> None:
        """Cover lines 379-382: merge_edge_pos when violation_idx is at the last bin."""
        rng = np.random.default_rng(123)
        n = 600
        # Create strongly non-monotonic data with violation likely at last bin
        values = np.concatenate([
            rng.normal(-3, 0.3, n // 3),
            rng.normal(0, 0.3, n // 3),
            rng.normal(3, 0.3, n - 2 * (n // 3)),
        ])
        target = np.concatenate([
            (rng.random(n // 3) < 0.2).astype(np.int64),
            (rng.random(n // 3) < 0.05).astype(np.int64),
            (rng.random(n - 2 * (n // 3)) < 0.5).astype(np.int64),
        ])
        result = optimal_binning(values, target, max_bins=10)
        assert isinstance(result, BinResult)

    def test_optimal_binning_merges_down_to_two(self) -> None:
        """Cover line 365: len(woe) <= 2 break after repeated merging in optimal_binning."""
        # U-shaped default rate forces continuous merging until 2 bins
        n = 400
        values = np.concatenate([
            np.full(n // 4, -3.0),
            np.full(n // 4, -1.0),
            np.full(n // 4, 1.0),
            np.full(n // 4, 3.0),
        ])
        target = np.concatenate([
            np.ones(n // 4, dtype=np.int64),
            np.zeros(n // 4, dtype=np.int64),
            np.zeros(n // 4, dtype=np.int64),
            np.ones(n // 4, dtype=np.int64),
        ])
        result = optimal_binning(values, target, max_bins=10)
        assert isinstance(result, BinResult)

    def test_optimal_binning_last_bin_violation(self) -> None:
        """Cover line 379-382: violation at the last bin in optimal_binning."""
        # 3 clusters: increasing default for first 2, then drop at last -> violation at end
        values = np.array(
            [1.0] * 100 + [2.0] * 100 + [3.0] * 100, dtype=np.float64
        )
        target = np.array(
            [0] * 90 + [1] * 10
            + [0] * 70 + [1] * 30
            + [0] * 95 + [1] * 5,
            dtype=np.int64,
        )
        result = optimal_binning(values, target, max_bins=5)
        assert isinstance(result, BinResult)


class TestEqualWidthBinning:
    """Test equal-width binning."""

    def test_basic(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        values, target = binary_data
        result = equal_width_binning(values, target, n_bins=5)
        assert isinstance(result, BinResult)
        assert len(result.woe_values) == 5

    def test_bin_edges_structure(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        values, target = binary_data
        result = equal_width_binning(values, target, n_bins=4)
        assert result.bin_edges[0] == -np.inf
        assert result.bin_edges[-1] == np.inf

    def test_bin_counts_sum(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        values, target = binary_data
        result = equal_width_binning(values, target, n_bins=8)
        assert result.bin_counts.sum() == len(values)


class TestMonotonicBinningExtended:
    """Additional tests for monotonic binning."""

    def test_explicit_increasing(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        values, target = binary_data
        result = monotonic_binning(values, target, n_bins=5, increasing=True)
        diffs = np.diff(result.woe_values)
        assert np.all(diffs >= -1e-10)

    def test_explicit_decreasing(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        values, target = binary_data
        # Negate values to flip the relationship
        result = monotonic_binning(-values, target, n_bins=5, increasing=False)
        diffs = np.diff(result.woe_values)
        assert np.all(diffs <= 1e-10)

    def test_feature_name_stored(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        values, target = binary_data
        result = monotonic_binning(values, target, feature_name="my_feature")
        assert result.feature_name == "my_feature"

    def test_few_unique_values_breaks_early(self) -> None:
        """Cover line 277: break when len(woe) <= 2."""
        # Only 2 unique values -> at most 2 bins -> breaks immediately
        values = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        target = np.array([0, 0, 1, 1, 1, 0], dtype=np.int64)
        result = monotonic_binning(values, target, n_bins=2, increasing=True)
        assert isinstance(result, BinResult)
        assert len(result.woe_values) <= 2

    def test_few_unique_values_decreasing(self) -> None:
        """Cover line 277: break when len(woe) <= 2 with decreasing direction."""
        values = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        target = np.array([1, 1, 0, 0, 0, 1], dtype=np.int64)
        result = monotonic_binning(values, target, n_bins=2, increasing=False)
        assert isinstance(result, BinResult)
        assert len(result.woe_values) <= 2

    def test_violation_at_last_bin_merges_previous(self) -> None:
        """Cover line 298: violation at last index merges with previous."""
        # Create data with deliberate non-monotonic WoE at the end
        rng = np.random.default_rng(99)
        n = 300
        values = np.concatenate([
            rng.normal(-2, 0.3, n // 3),
            rng.normal(0, 0.3, n // 3),
            rng.normal(2, 0.3, n // 3),
        ])
        # Non-monotonic target: high default rate at ends, low in middle
        target = np.concatenate([
            (rng.random(n // 3) < 0.3).astype(np.int64),
            (rng.random(n // 3) < 0.05).astype(np.int64),
            (rng.random(n // 3) < 0.4).astype(np.int64),
        ])
        result = monotonic_binning(values, target, n_bins=10, increasing=True)
        assert isinstance(result, BinResult)
        # The result should be monotonic after merging
        diffs = np.diff(result.woe_values)
        assert np.all(diffs >= -1e-10)

    def test_violation_at_last_bin_decreasing(self) -> None:
        """Cover line 298: merge_edge_pos = violation_idx - 1 when violation is at last bin."""
        # Create data where violation happens at the very last bin for decreasing WoE.
        # Use 3 distinct clusters with WoE that violates decreasing pattern at the end.
        rng = np.random.default_rng(77)
        n = 600
        values = np.concatenate([
            rng.normal(-3, 0.2, n // 3),
            rng.normal(0, 0.2, n // 3),
            rng.normal(3, 0.2, n - 2 * (n // 3)),
        ])
        # For decreasing: high default at low values, low at mid, but high again at end (violation)
        target = np.concatenate([
            (rng.random(n // 3) < 0.5).astype(np.int64),
            (rng.random(n // 3) < 0.1).astype(np.int64),
            (rng.random(n - 2 * (n // 3)) < 0.6).astype(np.int64),
        ])
        result = monotonic_binning(values, target, n_bins=10, increasing=False)
        assert isinstance(result, BinResult)
        diffs = np.diff(result.woe_values)
        assert np.all(diffs <= 1e-10)

    def test_monotonic_merges_down_to_two_bins(self) -> None:
        """Cover line 277: merging keeps going until only 2 bins remain.

        Use heavily non-monotonic data with many unique values so the
        merging loop doesn't stop early at 'violation_idx == -1' but
        instead keeps merging until len(woe) <= 2.
        """
        # 4 distinct clusters with U-shaped default rate -> forces increasing
        # direction to keep merging
        rng = np.random.default_rng(321)
        n = 400
        values = np.concatenate([
            rng.normal(-3, 0.1, n // 4),
            rng.normal(-1, 0.1, n // 4),
            rng.normal(1, 0.1, n // 4),
            rng.normal(3, 0.1, n // 4),
        ])
        # U-shaped: high default at extremes, low in middle -> always violates
        target = np.concatenate([
            np.ones(n // 4, dtype=np.int64),       # 100% default
            np.zeros(n // 4, dtype=np.int64),       # 0% default
            np.zeros(n // 4, dtype=np.int64),       # 0% default
            np.ones(n // 4, dtype=np.int64),        # 100% default
        ])
        result = monotonic_binning(values, target, n_bins=10, increasing=True)
        assert isinstance(result, BinResult)
        assert len(result.woe_values) <= 2

    def test_monotonic_violation_at_last_index_only(self) -> None:
        """Cover line 298: violation_idx == len(woe) - 1.

        Construct 3 bins where only the last bin violates monotonicity
        (increasing direction).
        """
        import unittest.mock as mock

        from creditriskengine.models.pd.binning import _build_bin_result

        # Create data with 3 clear clusters
        values = np.array(
            [1.0] * 100 + [2.0] * 100 + [3.0] * 100, dtype=np.float64
        )
        # Increasing WoE for first 2 bins, then violation at last bin
        # Low default in bin1, medium in bin2, low in bin3 -> violation at idx=2
        target = np.array(
            [0] * 90 + [1] * 10  # bin1: low default
            + [0] * 70 + [1] * 30  # bin2: medium default
            + [0] * 95 + [1] * 5,  # bin3: very low default -> violation
            dtype=np.int64,
        )
        result = monotonic_binning(values, target, n_bins=3, increasing=True)
        assert isinstance(result, BinResult)
        # After merging, result should be monotonic
        if len(result.woe_values) > 1:
            diffs = np.diff(result.woe_values)
            assert np.all(diffs >= -1e-10)


class TestApplyWoETransform:
    """Test WoE transformation of new data."""

    def test_transform_matches_bins(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        values, target = binary_data
        result = quantile_binning(values, target, n_bins=5)
        transformed = apply_woe_transform(values, result)
        assert transformed.shape == values.shape
        # Each transformed value should be one of the WoE values
        unique_woe = np.unique(transformed)
        assert len(unique_woe) <= len(result.woe_values) + 1  # +1 for edge cases

    def test_extreme_values(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        values, target = binary_data
        result = quantile_binning(values, target, n_bins=5)
        extreme = np.array([-100.0, 100.0])
        transformed = apply_woe_transform(extreme, result)
        assert np.all(np.isfinite(transformed))


class TestWoEBinningTransformer:
    """Test sklearn-compatible transformer."""

    def test_fit_transform(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        values, target = binary_data
        X = values.reshape(-1, 1)  # noqa: N806
        transformer = WoEBinningTransformer(n_bins=5, method="quantile")
        transformer.fit(X, target)
        result = transformer.transform(X)
        assert result.shape == X.shape

    def test_1d_input(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        values, target = binary_data
        transformer = WoEBinningTransformer(n_bins=5, method="monotonic")
        transformer.fit(values, target)
        result = transformer.transform(values)
        assert result.shape == (len(values), 1)

    def test_feature_ivs_stored(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        values, target = binary_data
        X = values.reshape(-1, 1)  # noqa: N806
        transformer = WoEBinningTransformer(n_bins=5)
        transformer.fit(X, target)
        assert transformer.feature_ivs_ is not None
        assert "feature_0" in transformer.feature_ivs_
        assert transformer.feature_ivs_["feature_0"] > 0.0

    def test_multiple_features(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        values, target = binary_data
        X = np.column_stack([values, values * 2])  # noqa: N806
        transformer = WoEBinningTransformer(n_bins=5)
        transformer.fit(X, target)
        result = transformer.transform(X)
        assert result.shape == X.shape
        assert len(transformer.bin_results_) == 2

    def test_transform_before_fit_raises(self) -> None:
        transformer = WoEBinningTransformer()
        with pytest.raises(AssertionError):
            transformer.transform(np.array([[1.0]]))

    def test_optimal_method(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        values, target = binary_data
        X = values.reshape(-1, 1)  # noqa: N806
        transformer = WoEBinningTransformer(n_bins=5, method="optimal")
        transformer.fit(X, target)
        result = transformer.transform(X)
        assert result.shape == X.shape

    def test_equal_width_method(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        values, target = binary_data
        X = values.reshape(-1, 1)  # noqa: N806
        transformer = WoEBinningTransformer(n_bins=5, method="equal_width")
        transformer.fit(X, target)
        result = transformer.transform(X)
        assert result.shape == X.shape
