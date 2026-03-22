"""Shared test fixtures for creditriskengine."""

import numpy as np
import pytest


@pytest.fixture
def rng() -> None:
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_corporate_params() -> None:
    """Typical corporate IRB parameters."""
    return {
        "pd": 0.01,
        "lgd": 0.45,
        "ead": 1_000_000.0,
        "maturity": 2.5,
        "asset_class": "corporate",
    }


@pytest.fixture
def sample_retail_params() -> None:
    """Typical retail IRB parameters."""
    return {
        "pd": 0.03,
        "lgd": 0.25,
        "ead": 50_000.0,
        "asset_class": "residential_mortgage",
    }
