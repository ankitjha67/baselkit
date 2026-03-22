"""Workout LGD estimation — spec-aligned re-export module.

Re-exports the workout_lgd function from the consolidated lgd_model module
to match the spec's ``models/lgd/workout_lgd.py`` file layout.
"""

from creditriskengine.models.lgd.lgd_model import workout_lgd

__all__ = ["workout_lgd"]
