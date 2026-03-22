"""Regulatory / supervisory CCF — spec-aligned re-export module.

Re-exports supervisory CCF functions from the consolidated ead_model module
to match the spec's ``models/ead/regulatory_ccf.py`` file layout.
"""

from creditriskengine.models.ead.ead_model import apply_ccf_floor, get_supervisory_ccf

__all__ = ["get_supervisory_ccf", "apply_ccf_floor"]
