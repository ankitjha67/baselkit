"""
Rating transition matrix estimation and analysis.

Provides cohort-based estimation, multi-period projection,
continuous-time generator extraction, and validation utilities.

References:
- BCBS d350: Regulatory treatment of accounting provisions
- EBA GL/2017/16: PD estimation, LGD estimation
- Lando (2004): Credit Risk Modeling
- Israel, Rosenthal & Wei (2001): Finding generators for Markov chains
"""

import logging

import numpy as np
from numpy.linalg import matrix_power
from scipy.linalg import logm

logger = logging.getLogger(__name__)


# -- Cohort Estimation -----------------------------------------------------


def estimate_transition_matrix(
    ratings_start: np.ndarray,
    ratings_end: np.ndarray,
    n_grades: int,
) -> np.ndarray:
    """Estimate a transition matrix using the cohort method.

    Counts transitions from grade *i* at the start of the period to
    grade *j* at the end, then normalises by the number of obligors
    starting in grade *i*.

    Args:
        ratings_start: Array of integer rating grades at period start
            (values in 0 .. n_grades-1).
        ratings_end: Array of integer rating grades at period end.
        n_grades: Total number of rating grades (including default).

    Returns:
        Transition matrix of shape (n_grades, n_grades).
    """
    ratings_start = np.asarray(ratings_start, dtype=int)
    ratings_end = np.asarray(ratings_end, dtype=int)
    if len(ratings_start) != len(ratings_end):
        raise ValueError("ratings_start and ratings_end must have equal length")

    counts = np.zeros((n_grades, n_grades), dtype=np.float64)
    for s, e in zip(ratings_start, ratings_end, strict=True):
        counts[s, e] += 1.0

    row_sums = counts.sum(axis=1, keepdims=True)
    # Avoid division by zero for grades with no observations
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return counts / row_sums


# -- Multi-period Projection -----------------------------------------------


def multi_period_transition_matrix(
    annual_matrix: np.ndarray,
    periods: int,
) -> np.ndarray:
    """Project an annual transition matrix over multiple periods.

    Computes M^t using numpy linear-algebra matrix power.

    Args:
        annual_matrix: Annual transition matrix (N x N).
        periods: Number of periods (integer >= 1).

    Returns:
        Transition matrix raised to the given power.
    """
    annual_matrix = np.asarray(annual_matrix, dtype=np.float64)
    if periods < 1:
        raise ValueError(f"periods must be >= 1, got {periods}")
    return np.asarray(matrix_power(annual_matrix, periods), dtype=np.float64)


# -- Continuous-Time Generator ----------------------------------------------


def generator_matrix(annual_matrix: np.ndarray) -> np.ndarray:
    """Extract the continuous-time generator matrix Q from an annual matrix.

    Under a continuous-time Markov chain the annual matrix P satisfies
    P = exp(Q).  The generator is therefore Q = log(P) (matrix logarithm).

    Note: the matrix logarithm may produce complex results for
    non-embeddable matrices.  This implementation takes the real part
    and logs a warning when the imaginary residual is material.

    Args:
        annual_matrix: Annual transition matrix (N x N).

    Returns:
        Generator matrix Q of shape (N, N).
    """
    annual_matrix = np.asarray(annual_matrix, dtype=np.float64)
    gen = logm(annual_matrix)

    if np.issubdtype(gen.dtype, np.complexfloating):
        imag_norm = np.linalg.norm(gen.imag)
        if imag_norm > 1e-6:
            logger.warning(
                "Generator matrix has non-trivial imaginary part "
                "(norm=%.4e); taking real part only.",
                imag_norm,
            )
        gen = gen.real

    return np.asarray(gen, dtype=np.float64)


# -- Validation -------------------------------------------------------------


def validate_transition_matrix(matrix: np.ndarray) -> list[str]:
    """Validate that a matrix satisfies transition matrix properties.

    Checks:
    1. All elements in [0, 1].
    2. Each row sums to 1 (within tolerance 1e-6).
    3. The last state is absorbing (last row = [0, ..., 0, 1]).

    Args:
        matrix: Candidate transition matrix (N x N).

    Returns:
        List of validation error messages.  Empty list means valid.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    errors: list[str] = []

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        errors.append("Matrix must be square 2-D array")
        return errors

    # Element range
    if np.any(matrix < -1e-10) or np.any(matrix > 1.0 + 1e-10):
        errors.append("All elements must be in [0, 1]")

    # Row sums
    row_sums = matrix.sum(axis=1)
    bad_rows = np.where(np.abs(row_sums - 1.0) > 1e-6)[0]
    if len(bad_rows) > 0:
        errors.append(
            f"Rows {bad_rows.tolist()} do not sum to 1 "
            f"(sums: {row_sums[bad_rows].tolist()})"
        )

    # Absorbing default state (last row)
    n = matrix.shape[0]
    expected_last = np.zeros(n, dtype=np.float64)
    expected_last[-1] = 1.0
    if not np.allclose(matrix[-1], expected_last, atol=1e-6):
        errors.append(
            "Last state is not absorbing: last row must be [0, ..., 0, 1]"
        )

    return errors


# -- Default Column Extraction ----------------------------------------------


def default_column(transition_matrix: np.ndarray) -> np.ndarray:
    """Extract PD-by-grade from the default (last) column of a transition matrix.

    Args:
        transition_matrix: Transition matrix (N x N) where the last
            column is the default state.

    Returns:
        Array of length N with the one-period PD for each grade.
    """
    tm = np.asarray(transition_matrix, dtype=np.float64)
    return tm[:, -1].copy()
