"""Downturn LGD estimation — spec-aligned re-export module.

Re-exports the downturn_lgd function from the consolidated lgd_model module
to match the spec's ``models/lgd/downturn_lgd.py`` file layout.
"""

from creditriskengine.models.lgd.lgd_model import downturn_lgd

__all__ = ["downturn_lgd"]
