"""Granularity Adjustment — spec-aligned re-export module.

Re-exports the Gordy granularity adjustment from the consolidated
concentration module to match the spec's
``models/concentration/granularity.py`` file layout.
"""

from creditriskengine.models.concentration.concentration import (
    granularity_adjustment,
)

__all__ = ["granularity_adjustment"]
