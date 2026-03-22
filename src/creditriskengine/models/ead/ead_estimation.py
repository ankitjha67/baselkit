"""EAD term structure estimation — spec-aligned re-export module.

Re-exports the ead_term_structure function from the consolidated ead_model
module to match the spec's ``models/ead/ead_estimation.py`` file layout.
"""

from creditriskengine.models.ead.ead_model import ead_term_structure

__all__ = ["ead_term_structure"]
