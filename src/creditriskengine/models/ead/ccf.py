"""Credit Conversion Factor (CCF) estimation — spec-aligned re-export module.

Re-exports CCF functions from the consolidated ead_model module to match
the spec's ``models/ead/ccf.py`` file layout.
"""

from creditriskengine.models.ead.ead_model import estimate_ccf

__all__ = ["estimate_ccf"]
