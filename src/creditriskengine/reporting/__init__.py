"""Regulatory reporting and model documentation.

Provides COREP, Pillar 3, FR Y-14, and model inventory report generation.
"""

from creditriskengine.reporting.pillar3 import (
    generate_cr1_template,
    generate_cr3_crm_overview,
    generate_cr4_sa_overview,
    generate_cr6_irb_overview,
)
from creditriskengine.reporting.reports import (
    generate_corep_credit_risk_summary,
    generate_model_inventory_entry,
    generate_pillar3_credit_risk,
)

__all__ = [
    "generate_corep_credit_risk_summary",
    "generate_pillar3_credit_risk",
    "generate_model_inventory_entry",
    # Pillar 3 disclosure templates
    "generate_cr1_template",
    "generate_cr3_crm_overview",
    "generate_cr4_sa_overview",
    "generate_cr6_irb_overview",
]
