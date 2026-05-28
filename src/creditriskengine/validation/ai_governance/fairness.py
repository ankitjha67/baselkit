"""Algorithmic fairness metrics for credit scoring models.

References:
    - EU AI Act (Reg 2024/1689), Art. 10 — data bias examination.
    - CFPB Circular 2022-03 — adverse action with complex algorithms.
    - MAS FEAT Principles / Veritas Toolkit 2.0 (June 2023).
    - EEOC Uniform Guidelines — 4/5 (80%) disparate impact rule.
    - Fairlearn library conventions for metric definitions.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def disparate_impact_ratio(
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
    favorable_label: int = 1,
) -> float:
    """Calculate the disparate impact ratio (4/5 rule).

    DI = P(favorable | unprivileged) / P(favorable | privileged)

    A ratio >= 0.80 passes the EEOC 4/5 rule. Below 0.80 indicates
    potential disparate impact requiring investigation.

    Args:
        y_pred: Binary predictions (1 = favorable, 0 = unfavorable).
        sensitive_attr: Binary protected attribute (1 = privileged,
            0 = unprivileged).
        favorable_label: Which label is the favorable outcome.

    Returns:
        Disparate impact ratio in [0, inf). Returns 0.0 if the
        privileged group has zero favorable predictions.

    Reference:
        EEOC Uniform Guidelines, 29 CFR 1607.4(D).
        CFPB Circular 2022-03.
    """
    y_pred = np.asarray(y_pred)
    sensitive_attr = np.asarray(sensitive_attr)

    privileged_mask = sensitive_attr == 1
    unprivileged_mask = sensitive_attr == 0

    if not np.any(privileged_mask) or not np.any(unprivileged_mask):
        return 1.0

    rate_privileged = np.mean(y_pred[privileged_mask] == favorable_label)
    rate_unprivileged = np.mean(y_pred[unprivileged_mask] == favorable_label)

    if rate_privileged == 0:
        return 0.0

    return float(rate_unprivileged / rate_privileged)


def demographic_parity_difference(
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
    favorable_label: int = 1,
) -> float:
    """Calculate the demographic parity difference.

    DPD = P(favorable | unprivileged) - P(favorable | privileged)

    A value of 0.0 indicates perfect parity. Values within [-0.10, 0.10]
    are generally considered acceptable. Negative values indicate the
    unprivileged group receives fewer favorable outcomes.

    Args:
        y_pred: Binary predictions.
        sensitive_attr: Binary protected attribute (1 = privileged).
        favorable_label: Which label is favorable.

    Returns:
        Demographic parity difference in [-1, 1].

    Reference:
        EU AI Act Art. 10, MAS Veritas Toolkit 2.0.
    """
    y_pred = np.asarray(y_pred)
    sensitive_attr = np.asarray(sensitive_attr)

    privileged_mask = sensitive_attr == 1
    unprivileged_mask = sensitive_attr == 0

    if not np.any(privileged_mask) or not np.any(unprivileged_mask):
        return 0.0

    rate_privileged = np.mean(y_pred[privileged_mask] == favorable_label)
    rate_unprivileged = np.mean(y_pred[unprivileged_mask] == favorable_label)

    return float(rate_unprivileged - rate_privileged)


def equal_opportunity_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
    favorable_label: int = 1,
) -> float:
    """Calculate the equal opportunity difference.

    EOD = TPR(unprivileged) - TPR(privileged)

    Measures whether truly-favorable individuals in both groups are
    equally likely to be correctly predicted as favorable. A gap
    <= 0.05 is generally considered acceptable.

    Args:
        y_true: True binary labels.
        y_pred: Predicted binary labels.
        sensitive_attr: Binary protected attribute (1 = privileged).
        favorable_label: Which label is favorable.

    Returns:
        Equal opportunity difference in [-1, 1].

    Reference:
        Hardt et al. (2016), MAS Veritas Toolkit 2.0.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sensitive_attr = np.asarray(sensitive_attr)

    def _tpr(mask: np.ndarray) -> float:
        actual_positive = y_true[mask] == favorable_label
        if not np.any(actual_positive):
            return 0.0
        predicted_positive = y_pred[mask] == favorable_label
        return float(np.sum(actual_positive & predicted_positive) / np.sum(actual_positive))

    privileged_mask = sensitive_attr == 1
    unprivileged_mask = sensitive_attr == 0

    if not np.any(privileged_mask) or not np.any(unprivileged_mask):
        return 0.0

    return _tpr(unprivileged_mask) - _tpr(privileged_mask)
