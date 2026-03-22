"""Logistic regression scoring — spec-aligned re-export module.

Re-exports logistic scoring functions from the consolidated scorecard module
to match the spec's ``models/pd/logistic.py`` file layout.
"""

from creditriskengine.models.pd.scorecard import logistic_score, score_to_pd

__all__ = ["logistic_score", "score_to_pd"]
