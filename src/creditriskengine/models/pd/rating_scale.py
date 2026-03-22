"""Rating scale / master scale — spec-aligned re-export module.

Re-exports master scale construction and rating assignment functions
from the consolidated scorecard module to match the spec's
``models/pd/rating_scale.py`` file layout.
"""

from creditriskengine.models.pd.scorecard import (
    assign_rating_grade,
    build_master_scale,
)

__all__ = ["build_master_scale", "assign_rating_grade"]
