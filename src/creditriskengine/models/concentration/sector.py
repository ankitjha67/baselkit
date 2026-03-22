"""Sector concentration — spec-aligned re-export module.

Re-exports sector concentration from the consolidated concentration module
to match the spec's ``models/concentration/sector.py`` file layout.
"""

from creditriskengine.models.concentration.concentration import (
    sector_concentration,
)

__all__ = ["sector_concentration"]
