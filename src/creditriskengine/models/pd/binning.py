"""Weight of Evidence (WoE) and Information Value (IV) binning.

Standard methodology for scorecard development per EBA GL/2017/16 Annex.
WoE binning transforms continuous/categorical features into risk-ordered bins
with monotonic WoE patterns, suitable for logistic regression PD models.

References:
    - EBA GL/2017/16: Guidelines on PD estimation
    - Siddiqi (2006): "Credit Risk Scorecards"
"""

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Laplace smoothing constant to avoid division by zero / log(0)
_SMOOTHING: float = 0.5


@dataclass
class BinResult:
    """Result of WoE binning for a single feature."""

    feature_name: str
    bin_edges: NDArray[np.float64]  # n+1 edges for n bins
    woe_values: NDArray[np.float64]  # WoE per bin
    iv: float  # Total Information Value
    event_rate: NDArray[np.float64]  # Default rate per bin
    bin_counts: NDArray[np.int64]  # Observations per bin
    pct_events: NDArray[np.float64]  # % of events per bin
    pct_non_events: NDArray[np.float64]  # % of non-events per bin


# ── Core WoE / IV Calculations ────────────────────────────────────


def calculate_woe(
    n_events: NDArray[np.int64],
    n_non_events: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Calculate Weight of Evidence per bin.

    WoE(i) = ln(Distribution of Events_i / Distribution of Non-Events_i)
            = ln((%events_i) / (%non_events_i))

    Uses Laplace smoothing (add 0.5) to handle zero bins.

    Args:
        n_events: Number of events (defaults) per bin.
        n_non_events: Number of non-events per bin.

    Returns:
        Array of WoE values per bin.
    """
    n_events = np.asarray(n_events, dtype=np.float64)
    n_non_events = np.asarray(n_non_events, dtype=np.float64)

    total_events = np.sum(n_events)
    total_non_events = np.sum(n_non_events)

    # Laplace smoothing
    pct_events = (n_events + _SMOOTHING) / (total_events + _SMOOTHING * len(n_events))
    pct_non_events = (n_non_events + _SMOOTHING) / (
        total_non_events + _SMOOTHING * len(n_non_events)
    )

    return np.asarray(np.log(pct_events / pct_non_events), dtype=np.float64)


def calculate_iv(
    woe: NDArray[np.float64],
    pct_events: NDArray[np.float64],
    pct_non_events: NDArray[np.float64],
) -> float:
    """Calculate Information Value.

    IV = sum( (% Events_i - % Non-Events_i) * WoE_i )

    Interpretation (Siddiqi 2006):
        IV < 0.02: Useless/not predictive
        0.02 <= IV < 0.1: Weak predictor
        0.1 <= IV < 0.3: Medium predictor
        0.3 <= IV < 0.5: Strong predictor
        IV >= 0.5: Suspicious (possible overfitting)

    Args:
        woe: WoE values per bin.
        pct_events: Proportion of total events per bin.
        pct_non_events: Proportion of total non-events per bin.

    Returns:
        Total Information Value.
    """
    woe = np.asarray(woe, dtype=np.float64)
    pct_events = np.asarray(pct_events, dtype=np.float64)
    pct_non_events = np.asarray(pct_non_events, dtype=np.float64)

    return float(np.sum((pct_events - pct_non_events) * woe))


# ── Helper: Build BinResult from edges and data ──────────────────


def _build_bin_result(
    values: NDArray[np.float64],
    target: NDArray[np.int64],
    bin_edges: NDArray[np.float64],
    feature_name: str,
) -> BinResult:
    """Construct a BinResult from bin edges and raw data."""
    bin_indices = np.digitize(values, bin_edges[1:-1])  # 0-based bin index
    n_bins = len(bin_edges) - 1

    n_events = np.zeros(n_bins, dtype=np.int64)
    n_non_events = np.zeros(n_bins, dtype=np.int64)
    bin_counts = np.zeros(n_bins, dtype=np.int64)

    for i in range(n_bins):
        mask = bin_indices == i
        bin_counts[i] = int(np.sum(mask))
        n_events[i] = int(np.sum(target[mask]))
        n_non_events[i] = bin_counts[i] - n_events[i]

    total_events = max(int(np.sum(n_events)), 1)
    total_non_events = max(int(np.sum(n_non_events)), 1)

    pct_events = (n_events.astype(np.float64) + _SMOOTHING) / (
        total_events + _SMOOTHING * n_bins
    )
    pct_non_events = (n_non_events.astype(np.float64) + _SMOOTHING) / (
        total_non_events + _SMOOTHING * n_bins
    )

    woe_values = np.log(pct_events / pct_non_events)
    iv = float(np.sum((pct_events - pct_non_events) * woe_values))

    event_rate = np.where(
        bin_counts > 0,
        n_events.astype(np.float64) / bin_counts.astype(np.float64),
        0.0,
    )

    return BinResult(
        feature_name=feature_name,
        bin_edges=np.asarray(bin_edges, dtype=np.float64),
        woe_values=np.asarray(woe_values, dtype=np.float64),
        iv=iv,
        event_rate=np.asarray(event_rate, dtype=np.float64),
        bin_counts=np.asarray(bin_counts, dtype=np.int64),
        pct_events=np.asarray(pct_events, dtype=np.float64),
        pct_non_events=np.asarray(pct_non_events, dtype=np.float64),
    )


# ── Binning Strategies ────────────────────────────────────────────


def quantile_binning(
    values: NDArray[np.float64],
    target: NDArray[np.int64],
    n_bins: int = 10,
    feature_name: str = "feature",
) -> BinResult:
    """Equal-frequency (quantile) binning.

    Splits the feature into bins with approximately equal observation counts
    using quantile boundaries.

    Args:
        values: Continuous feature values.
        target: Binary target (1 = default/event, 0 = non-event).
        n_bins: Desired number of bins.
        feature_name: Name of the feature.

    Returns:
        BinResult with WoE/IV statistics.
    """
    values = np.asarray(values, dtype=np.float64)
    target = np.asarray(target, dtype=np.int64)

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(values, quantiles)
    # Deduplicate edges (can happen with many tied values)
    edges = np.unique(edges)
    # Ensure -inf / +inf at boundaries
    edges[0] = -np.inf
    edges[-1] = np.inf

    return _build_bin_result(values, target, edges, feature_name)


def equal_width_binning(
    values: NDArray[np.float64],
    target: NDArray[np.int64],
    n_bins: int = 10,
    feature_name: str = "feature",
) -> BinResult:
    """Equal-width binning.

    Splits the feature range into bins of equal width.

    Args:
        values: Continuous feature values.
        target: Binary target (1 = default/event, 0 = non-event).
        n_bins: Desired number of bins.
        feature_name: Name of the feature.

    Returns:
        BinResult with WoE/IV statistics.
    """
    values = np.asarray(values, dtype=np.float64)
    target = np.asarray(target, dtype=np.int64)

    v_min, v_max = float(np.min(values)), float(np.max(values))
    edges = np.linspace(v_min, v_max, n_bins + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf

    return _build_bin_result(values, target, edges, feature_name)


def monotonic_binning(
    values: NDArray[np.float64],
    target: NDArray[np.int64],
    n_bins: int = 10,
    feature_name: str = "feature",
    increasing: bool | None = None,
) -> BinResult:
    """Monotonic WoE binning — merges adjacent bins to enforce monotonicity.

    Algorithm:
        1. Start with fine-grained quantile bins (2 * n_bins).
        2. Check monotonicity of WoE.
        3. Merge adjacent bins that break monotonicity.
        4. Recalculate WoE until monotonic.

    If *increasing* is None, auto-detect direction from the Spearman
    correlation between feature values and the target.

    Args:
        values: Continuous feature values.
        target: Binary target (1 = default/event, 0 = non-event).
        n_bins: Maximum number of output bins.
        feature_name: Name of the feature.
        increasing: If True, enforce WoE increasing; if False, decreasing;
            if None, auto-detect.

    Returns:
        BinResult with monotonic WoE pattern.
    """
    values = np.asarray(values, dtype=np.float64)
    target = np.asarray(target, dtype=np.int64)

    # Auto-detect direction from rank correlation
    if increasing is None:
        correlation = np.corrcoef(values, target.astype(np.float64))[0, 1]
        increasing = bool(correlation >= 0)

    # Start with fine-grained quantile bins
    initial_bins = min(max(2 * n_bins, 20), len(np.unique(values)))
    quantiles = np.linspace(0.0, 1.0, initial_bins + 1)
    edges = np.unique(np.quantile(values, quantiles))
    edges[0] = -np.inf
    edges[-1] = np.inf

    # Iteratively merge bins that violate monotonicity
    max_iterations = len(edges) * 2  # safety bound
    for _ in range(max_iterations):
        result = _build_bin_result(values, target, edges, feature_name)
        woe = result.woe_values

        if len(woe) <= 2:
            break

        # Find first monotonicity violation
        violation_idx = -1
        for i in range(1, len(woe)):
            if increasing and woe[i] < woe[i - 1]:
                violation_idx = i
                break
            if not increasing and woe[i] > woe[i - 1]:
                violation_idx = i
                break

        if violation_idx == -1:
            # Monotonic — done
            break

        # Merge bin at violation_idx with its neighbour (pick the smaller bin)
        if violation_idx < len(woe) - 1:  # noqa: SIM108
            # Remove the edge between violation_idx and violation_idx+1
            merge_edge_pos = violation_idx
        else:
            merge_edge_pos = violation_idx - 1

        # Remove the internal edge (offset by 1 because edges[0] = -inf)
        edges = np.delete(edges, merge_edge_pos + 1)

    # Final build with cleaned edges
    return _build_bin_result(values, target, edges, feature_name)


def optimal_binning(
    values: NDArray[np.float64],
    target: NDArray[np.int64],
    max_bins: int = 10,
    min_bin_pct: float = 0.05,
    feature_name: str = "feature",
) -> BinResult:
    """Optimal binning using decision tree splits.

    Uses ``sklearn.tree.DecisionTreeClassifier`` to find optimal split
    points that maximize information gain, then enforces monotonicity
    via adjacent-bin merging.

    Args:
        values: Continuous feature values.
        target: Binary target (1 = default/event, 0 = non-event).
        max_bins: Maximum number of bins.
        min_bin_pct: Minimum fraction of observations per bin.
        feature_name: Name of the feature.

    Returns:
        BinResult with optimal, monotonic WoE bins.
    """
    from sklearn.tree import DecisionTreeClassifier

    values = np.asarray(values, dtype=np.float64)
    target = np.asarray(target, dtype=np.int64)

    min_samples_leaf = max(int(len(values) * min_bin_pct), 1)

    tree = DecisionTreeClassifier(
        max_leaf_nodes=max_bins,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )
    tree.fit(values.reshape(-1, 1), target)

    # Extract split thresholds from the tree
    thresholds = tree.tree_.threshold[tree.tree_.feature != -2]
    thresholds = np.sort(np.unique(thresholds))

    edges = np.concatenate([[-np.inf], thresholds, [np.inf]])
    edges = np.asarray(edges, dtype=np.float64)

    # Build initial result, then enforce monotonicity via merging
    result = _build_bin_result(values, target, edges, feature_name)

    # Auto-detect direction
    correlation = np.corrcoef(values, target.astype(np.float64))[0, 1]
    is_increasing = bool(correlation >= 0)

    # Merge to enforce monotonicity
    max_iterations = len(edges) * 2
    for _ in range(max_iterations):
        result = _build_bin_result(values, target, edges, feature_name)
        woe = result.woe_values

        if len(woe) <= 2:
            break

        violation_idx = -1
        for i in range(1, len(woe)):
            if is_increasing and woe[i] < woe[i - 1]:
                violation_idx = i
                break
            if not is_increasing and woe[i] > woe[i - 1]:
                violation_idx = i
                break

        if violation_idx == -1:
            break

        merge_edge_pos = (
            violation_idx if violation_idx < len(woe) - 1 else violation_idx - 1
        )
        edges = np.delete(edges, merge_edge_pos + 1)

    return _build_bin_result(values, target, edges, feature_name)


# ── Transform ─────────────────────────────────────────────────────


def apply_woe_transform(
    values: NDArray[np.float64],
    bin_result: BinResult,
) -> NDArray[np.float64]:
    """Apply WoE transformation to new data using fitted bin edges.

    Each observation is assigned the WoE value of the bin it falls into.

    Args:
        values: Feature values to transform.
        bin_result: Fitted BinResult from a binning function.

    Returns:
        Array of WoE-transformed values.
    """
    values = np.asarray(values, dtype=np.float64)
    bin_indices = np.digitize(values, bin_result.bin_edges[1:-1])
    # Clip to valid bin range
    bin_indices = np.clip(bin_indices, 0, len(bin_result.woe_values) - 1)
    return np.asarray(bin_result.woe_values[bin_indices], dtype=np.float64)


# ── Sklearn-compatible Transformer ────────────────────────────────

from sklearn.base import (  # noqa: E402
    BaseEstimator,
    TransformerMixin,
)


class WoEBinningTransformer(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Sklearn-compatible WoE binning transformer.

    Fits WoE bins on training data and transforms features to WoE values.

    Parameters:
        n_bins: Number of initial bins.
        method: Binning method ('quantile', 'equal_width', 'monotonic', 'optimal').
        min_bin_pct: Minimum bin population percentage (for optimal).
    """

    def __init__(
        self,
        n_bins: int = 10,
        method: str = "monotonic",
        min_bin_pct: float = 0.05,
    ) -> None:
        self.n_bins = n_bins
        self.method = method
        self.min_bin_pct = min_bin_pct
        self.bin_results_: list[BinResult] | None = None
        self.feature_ivs_: dict[str, float] | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "WoEBinningTransformer":  # noqa: N803
        """Fit WoE bins for each feature column."""
        X = np.asarray(X, dtype=np.float64)  # noqa: N806
        y = np.asarray(y, dtype=np.int64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)  # noqa: N806
        self.bin_results_ = []
        self.feature_ivs_ = {}
        for j in range(X.shape[1]):
            name = f"feature_{j}"
            if self.method == "monotonic":
                result = monotonic_binning(X[:, j], y, self.n_bins, name)
            elif self.method == "optimal":
                result = optimal_binning(X[:, j], y, self.n_bins, self.min_bin_pct, name)
            elif self.method == "equal_width":
                result = equal_width_binning(X[:, j], y, self.n_bins, name)
            else:
                result = quantile_binning(X[:, j], y, self.n_bins, name)
            self.bin_results_.append(result)
            self.feature_ivs_[name] = result.iv
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        """Transform features to WoE values."""
        assert self.bin_results_ is not None, "Call fit() first"
        X = np.asarray(X, dtype=np.float64)  # noqa: N806
        if X.ndim == 1:
            X = X.reshape(-1, 1)  # noqa: N806
        result = np.zeros_like(X)
        for j, br in enumerate(self.bin_results_):
            result[:, j] = apply_woe_transform(X[:, j], br)
        return result
