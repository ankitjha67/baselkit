"""Regulatory LGD floor application — spec-aligned re-export module.

Re-exports the apply_lgd_floor function from the consolidated lgd_model
module to match the spec's ``models/lgd/regulatory_lgd.py`` file layout.
"""

from creditriskengine.models.lgd.lgd_model import apply_lgd_floor

__all__ = ["apply_lgd_floor"]
