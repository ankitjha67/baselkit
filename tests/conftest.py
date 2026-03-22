"""Shared test fixtures for creditriskengine."""

import numpy as np
import pytest

from creditriskengine.core.types import (
    CreditQualityStep,
    CreditRiskApproach,
    IFRS9Stage,
    IRBAssetClass,
    Jurisdiction,
    SAExposureClass,
)


@pytest.fixture
def rng():
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_corporate_params():
    """Typical corporate IRB parameters."""
    return {
        "pd": 0.01,
        "lgd": 0.45,
        "ead": 1_000_000.0,
        "maturity": 2.5,
        "asset_class": "corporate",
    }


@pytest.fixture
def sample_retail_params():
    """Typical retail IRB parameters."""
    return {
        "pd": 0.03,
        "lgd": 0.25,
        "ead": 50_000.0,
        "asset_class": "residential_mortgage",
    }
