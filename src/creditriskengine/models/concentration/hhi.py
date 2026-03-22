"""HHI (Herfindahl-Hirschman Index) — spec-aligned re-export module.

Re-exports single-name concentration from the consolidated concentration
module to match the spec's ``models/concentration/hhi.py`` file layout.
"""

from creditriskengine.models.concentration.concentration import (
    single_name_concentration,
)

__all__ = ["single_name_concentration"]
