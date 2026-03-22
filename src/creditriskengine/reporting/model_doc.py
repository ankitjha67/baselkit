"""Automated model documentation generator.

Generates standardized model documentation aligned with:
- US SR 11-7 (Model Risk Management)
- ECB Guide to Internal Models
- PRA SS1/23 (Model risk management principles)
- EBA GL/2017/16 (PD/LGD estimation)

Outputs structured model documentation using Jinja2 templates.

References:
    - SR 11-7 / OCC 2011-12: Supervisory Guidance on Model Risk Management
    - ECB Guide to Internal Models, Chapter 1 (General topics)
    - PRA SS1/23: Model risk management principles for banks
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent / "templates"

# Discrimination (Gini) thresholds per EBA GL/2017/16
_GINI_GOOD = 0.40
_GINI_ACCEPTABLE = 0.25

# Stability (PSI) thresholds — industry standard
_PSI_STABLE = 0.10
_PSI_SHIFT = 0.25

# Calibration thresholds — binomial test p-value or similar
_CALIBRATION_GOOD = 0.05


def _rag_discrimination(gini: float) -> str:
    """Return RAG status for Gini-based discrimination."""
    if gini >= _GINI_GOOD:
        return "green"
    if gini >= _GINI_ACCEPTABLE:
        return "yellow"
    return "red"


def _rag_stability(psi: float) -> str:
    """Return RAG status for PSI-based stability."""
    if psi < _PSI_STABLE:
        return "green"
    if psi < _PSI_SHIFT:
        return "yellow"
    return "red"


def _rag_calibration(p_value: float) -> str:
    """Return RAG status for calibration test."""
    if p_value >= _CALIBRATION_GOOD:
        return "green"
    if p_value >= 0.01:
        return "yellow"
    return "red"


def _worst_rag(*rags: str) -> str:
    """Return the worst RAG status from a set of RAG values."""
    if "red" in rags:
        return "red"
    if "yellow" in rags:
        return "yellow"
    return "green"


@dataclass
class ModelDocumentation:
    """Structured model documentation."""

    model_name: str
    model_id: str
    model_type: str  # "PD", "LGD", "EAD", "Scorecard", "ECL"
    model_owner: str = ""
    version: str = "1.0"
    effective_date: date | None = None

    # Model overview
    purpose: str = ""
    scope: str = ""
    regulatory_use: str = ""

    # Development
    methodology: str = ""
    data_description: str = ""
    sample_period: str = ""
    sample_size: int = 0

    # Performance metrics
    discrimination_metrics: dict[str, float] = field(default_factory=dict)
    calibration_metrics: dict[str, float] = field(default_factory=dict)
    stability_metrics: dict[str, float] = field(default_factory=dict)

    # Validation
    validation_status: str = "pending"  # pending, approved, conditional, rejected
    validation_date: date | None = None
    findings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)


def generate_model_card(doc: ModelDocumentation) -> dict[str, Any]:
    """Generate a model card (structured metadata summary).

    Based on Mitchell et al. (2019) 'Model Cards for Model Reporting'
    adapted for credit risk regulatory context.

    The model card provides a concise, standardized snapshot of the model
    including its intended use, performance characteristics, and known
    limitations.

    Args:
        doc: Populated :class:`ModelDocumentation` instance.

    Returns:
        Structured dict containing model card fields.
    """
    logger.info("Generating model card for %s (ID: %s)", doc.model_name, doc.model_id)

    # Derive RAG statuses from metrics where available
    gini = doc.discrimination_metrics.get("gini", 0.0)
    auroc = doc.discrimination_metrics.get("auroc", 0.0)
    psi = doc.stability_metrics.get("psi", 0.0)
    calibration_pvalue = doc.calibration_metrics.get("p_value", 1.0)

    disc_rag = _rag_discrimination(gini)
    stab_rag = _rag_stability(psi)
    cal_rag = _rag_calibration(calibration_pvalue)
    overall_rag = _worst_rag(disc_rag, stab_rag, cal_rag)

    card: dict[str, Any] = {
        "model_details": {
            "name": doc.model_name,
            "id": doc.model_id,
            "type": doc.model_type,
            "version": doc.version,
            "owner": doc.model_owner,
            "effective_date": (
                doc.effective_date.isoformat() if doc.effective_date else None
            ),
        },
        "intended_use": {
            "purpose": doc.purpose,
            "scope": doc.scope,
            "regulatory_use": doc.regulatory_use,
        },
        "development": {
            "methodology": doc.methodology,
            "data_description": doc.data_description,
            "sample_period": doc.sample_period,
            "sample_size": doc.sample_size,
        },
        "performance": {
            "discrimination": {
                "gini": gini,
                "auroc": auroc,
                "rag": disc_rag,
                **{
                    k: v
                    for k, v in doc.discrimination_metrics.items()
                    if k not in ("gini", "auroc")
                },
            },
            "calibration": {
                "rag": cal_rag,
                **doc.calibration_metrics,
            },
            "stability": {
                "psi": psi,
                "rag": stab_rag,
                **{
                    k: v
                    for k, v in doc.stability_metrics.items()
                    if k != "psi"
                },
            },
            "overall_rag": overall_rag,
        },
        "validation": {
            "status": doc.validation_status,
            "date": (
                doc.validation_date.isoformat() if doc.validation_date else None
            ),
            "finding_count": len(doc.findings),
            "findings": doc.findings,
        },
        "limitations": doc.limitations,
    }

    logger.info(
        "Model card for %s: overall RAG=%s, Gini=%.3f, PSI=%.3f",
        doc.model_name,
        overall_rag,
        gini,
        psi,
    )
    return card


def generate_model_doc_report(doc: ModelDocumentation) -> str:
    """Generate full model documentation report as formatted text.

    Uses Jinja2 template if available, falls back to structured text.

    Args:
        doc: Populated :class:`ModelDocumentation` instance.

    Returns:
        Formatted model documentation string (HTML if Jinja2 template
        is available, plain text otherwise).
    """
    logger.info(
        "Generating model documentation report for %s (ID: %s)",
        doc.model_name,
        doc.model_id,
    )

    # Attempt Jinja2 rendering
    try:
        from jinja2 import Environment, FileSystemLoader

        if TEMPLATES_DIR.exists():
            env = Environment(
                loader=FileSystemLoader(str(TEMPLATES_DIR)),
                autoescape=True,
            )
            try:
                template = env.get_template("model_doc.html.j2")
                rendered = template.render(
                    model_name=doc.model_name,
                    model_id=doc.model_id,
                    model_type=doc.model_type,
                    model_owner=doc.model_owner,
                    version=doc.version,
                    effective_date=(
                        doc.effective_date.isoformat() if doc.effective_date else "N/A"
                    ),
                    purpose=doc.purpose,
                    scope=doc.scope,
                    regulatory_use=doc.regulatory_use,
                    methodology=doc.methodology,
                    data_description=doc.data_description,
                    sample_period=doc.sample_period,
                    sample_size=doc.sample_size,
                    discrimination_metrics=doc.discrimination_metrics,
                    calibration_metrics=doc.calibration_metrics,
                    stability_metrics=doc.stability_metrics,
                    validation_status=doc.validation_status,
                    validation_date=(
                        doc.validation_date.isoformat()
                        if doc.validation_date
                        else "N/A"
                    ),
                    findings=doc.findings,
                    limitations=doc.limitations,
                    disc_rag=_rag_discrimination(
                        doc.discrimination_metrics.get("gini", 0.0)
                    ),
                    stab_rag=_rag_stability(
                        doc.stability_metrics.get("psi", 0.0)
                    ),
                    cal_rag=_rag_calibration(
                        doc.calibration_metrics.get("p_value", 1.0)
                    ),
                    overall_rag=_worst_rag(
                        _rag_discrimination(
                            doc.discrimination_metrics.get("gini", 0.0)
                        ),
                        _rag_stability(doc.stability_metrics.get("psi", 0.0)),
                        _rag_calibration(
                            doc.calibration_metrics.get("p_value", 1.0)
                        ),
                    ),
                )
                logger.info("Rendered model doc using Jinja2 template")
                return rendered
            except Exception:
                logger.warning(
                    "Jinja2 template rendering failed; falling back to text",
                    exc_info=True,
                )
    except ImportError:
        logger.info("Jinja2 not available; using plain text fallback")

    # Fallback: structured plain text
    return _generate_plain_text_report(doc)


def _generate_plain_text_report(doc: ModelDocumentation) -> str:
    """Generate plain-text model documentation report.

    Args:
        doc: Populated :class:`ModelDocumentation` instance.

    Returns:
        Plain-text formatted report string.
    """
    sep = "=" * 72
    subsep = "-" * 72
    lines: list[str] = []

    lines.append(sep)
    lines.append(f"MODEL DOCUMENTATION: {doc.model_name}")
    lines.append(sep)
    lines.append("")

    # Section 1: Model Overview
    lines.append("1. MODEL OVERVIEW")
    lines.append(subsep)
    lines.append(f"  Model Name:       {doc.model_name}")
    lines.append(f"  Model ID:         {doc.model_id}")
    lines.append(f"  Model Type:       {doc.model_type}")
    lines.append(f"  Version:          {doc.version}")
    lines.append(f"  Owner:            {doc.model_owner}")
    lines.append(
        f"  Effective Date:   "
        f"{doc.effective_date.isoformat() if doc.effective_date else 'N/A'}"
    )
    lines.append("")

    # Section 2: Intended Use
    lines.append("2. INTENDED USE")
    lines.append(subsep)
    lines.append(f"  Purpose:          {doc.purpose}")
    lines.append(f"  Scope:            {doc.scope}")
    lines.append(f"  Regulatory Use:   {doc.regulatory_use}")
    lines.append("")

    # Section 3: Methodology
    lines.append("3. METHODOLOGY")
    lines.append(subsep)
    lines.append(f"  {doc.methodology}")
    lines.append("")

    # Section 4: Data
    lines.append("4. DATA")
    lines.append(subsep)
    lines.append(f"  Description:      {doc.data_description}")
    lines.append(f"  Sample Period:    {doc.sample_period}")
    lines.append(f"  Sample Size:      {doc.sample_size:,}")
    lines.append("")

    # Section 5: Performance Metrics
    lines.append("5. PERFORMANCE METRICS")
    lines.append(subsep)

    lines.append("  5.1 Discrimination")
    if doc.discrimination_metrics:
        for k, v in doc.discrimination_metrics.items():
            lines.append(f"    {k:20s}: {v:.4f}")
        gini = doc.discrimination_metrics.get("gini", 0.0)
        lines.append(f"    {'RAG':20s}: {_rag_discrimination(gini)}")
    else:
        lines.append("    No discrimination metrics available.")
    lines.append("")

    lines.append("  5.2 Calibration")
    if doc.calibration_metrics:
        for k, v in doc.calibration_metrics.items():
            lines.append(f"    {k:20s}: {v:.4f}")
        p_val = doc.calibration_metrics.get("p_value", 1.0)
        lines.append(f"    {'RAG':20s}: {_rag_calibration(p_val)}")
    else:
        lines.append("    No calibration metrics available.")
    lines.append("")

    lines.append("  5.3 Stability")
    if doc.stability_metrics:
        for k, v in doc.stability_metrics.items():
            lines.append(f"    {k:20s}: {v:.4f}")
        psi = doc.stability_metrics.get("psi", 0.0)
        lines.append(f"    {'RAG':20s}: {_rag_stability(psi)}")
    else:
        lines.append("    No stability metrics available.")
    lines.append("")

    # Section 6: Validation
    lines.append("6. VALIDATION")
    lines.append(subsep)
    lines.append(f"  Status:           {doc.validation_status}")
    lines.append(
        f"  Validation Date:  "
        f"{doc.validation_date.isoformat() if doc.validation_date else 'N/A'}"
    )
    lines.append("")

    if doc.findings:
        lines.append("  Findings:")
        for i, finding in enumerate(doc.findings, 1):
            lines.append(f"    {i}. {finding}")
        lines.append("")

    # Section 7: Limitations
    lines.append("7. LIMITATIONS")
    lines.append(subsep)
    if doc.limitations:
        for i, limitation in enumerate(doc.limitations, 1):
            lines.append(f"  {i}. {limitation}")
    else:
        lines.append("  No known limitations documented.")
    lines.append("")
    lines.append(sep)

    return "\n".join(lines)


def generate_validation_report(
    doc: ModelDocumentation,
    discrimination: dict[str, float] | None = None,
    calibration: dict[str, float] | None = None,
    stability: dict[str, float] | None = None,
) -> str:
    """Generate model validation report aligned with SR 11-7.

    Covers the three pillars of model validation:
    1. Conceptual soundness (methodology review)
    2. Outcomes analysis (back-testing / discrimination / calibration)
    3. Ongoing monitoring (stability)

    Args:
        doc: Populated :class:`ModelDocumentation` instance.
        discrimination: Override discrimination metrics (uses doc metrics
            if not provided).
        calibration: Override calibration metrics.
        stability: Override stability metrics.

    Returns:
        Formatted validation report string.
    """
    logger.info(
        "Generating validation report for %s (ID: %s)",
        doc.model_name,
        doc.model_id,
    )

    disc = discrimination if discrimination is not None else doc.discrimination_metrics
    cal = calibration if calibration is not None else doc.calibration_metrics
    stab = stability if stability is not None else doc.stability_metrics

    gini = disc.get("gini", 0.0)
    auroc = disc.get("auroc", 0.0)
    psi = stab.get("psi", 0.0)
    p_value = cal.get("p_value", 1.0)

    disc_rag = _rag_discrimination(gini)
    stab_rag = _rag_stability(psi)
    cal_rag = _rag_calibration(p_value)
    overall = _worst_rag(disc_rag, stab_rag, cal_rag)

    sep = "=" * 72
    subsep = "-" * 72
    lines: list[str] = []

    lines.append(sep)
    lines.append(f"MODEL VALIDATION REPORT: {doc.model_name}")
    lines.append(sep)
    lines.append("")
    lines.append(f"  Model ID:           {doc.model_id}")
    lines.append(f"  Model Type:         {doc.model_type}")
    lines.append(f"  Version:            {doc.version}")
    lines.append(
        f"  Validation Date:    "
        f"{doc.validation_date.isoformat() if doc.validation_date else 'N/A'}"
    )
    lines.append(f"  Overall Assessment: {overall.upper()}")
    lines.append("")

    # Pillar 1: Conceptual Soundness
    lines.append("1. CONCEPTUAL SOUNDNESS")
    lines.append(subsep)
    lines.append(f"  Methodology: {doc.methodology}")
    lines.append(f"  Scope:       {doc.scope}")
    lines.append(
        "  Assessment:  Review of methodology, assumptions, and theoretical"
    )
    lines.append("               basis for the chosen modelling approach.")
    lines.append("")

    # Pillar 2: Outcomes Analysis
    lines.append("2. OUTCOMES ANALYSIS")
    lines.append(subsep)

    lines.append("  2.1 Discrimination")
    lines.append(f"    AUROC:     {auroc:.4f}")
    lines.append(f"    Gini:      {gini:.4f}")
    for k, v in disc.items():
        if k not in ("gini", "auroc"):
            lines.append(f"    {k:10s}: {v:.4f}")
    lines.append(f"    RAG:       {disc_rag.upper()}")
    lines.append("")

    lines.append("  2.2 Calibration")
    for k, v in cal.items():
        lines.append(f"    {k:10s}: {v:.4f}")
    lines.append(f"    RAG:       {cal_rag.upper()}")
    lines.append("")

    lines.append("  2.3 Stability")
    lines.append(f"    PSI:       {psi:.4f}")
    for k, v in stab.items():
        if k != "psi":
            lines.append(f"    {k:10s}: {v:.4f}")
    lines.append(f"    RAG:       {stab_rag.upper()}")
    lines.append("")

    # Pillar 3: Ongoing Monitoring
    lines.append("3. ONGOING MONITORING RECOMMENDATIONS")
    lines.append(subsep)
    lines.append("  - Quarterly back-testing of discrimination and calibration")
    lines.append("  - Monthly population stability monitoring")
    lines.append("  - Annual full re-validation")
    if overall == "red":
        lines.append(
            "  - URGENT: Model requires immediate remediation or override"
        )
    elif overall == "yellow":
        lines.append(
            "  - ACTION: Model performance is weakening; remediation plan needed"
        )
    lines.append("")

    # Findings
    lines.append("4. FINDINGS")
    lines.append(subsep)
    if doc.findings:
        for i, finding in enumerate(doc.findings, 1):
            lines.append(f"  {i}. {finding}")
    else:
        lines.append("  No material findings.")
    lines.append("")

    # Conclusion
    lines.append("5. CONCLUSION")
    lines.append(subsep)
    status_text = {
        "approved": "Model is APPROVED for continued use.",
        "conditional": "Model is CONDITIONALLY APPROVED pending remediation.",
        "rejected": "Model is REJECTED. Use must be discontinued.",
        "pending": "Validation is PENDING completion.",
    }
    lines.append(f"  {status_text.get(doc.validation_status, 'Status unknown.')}")
    lines.append("")
    lines.append(sep)

    report = "\n".join(lines)
    logger.info(
        "Validation report for %s: overall=%s, status=%s",
        doc.model_name,
        overall,
        doc.validation_status,
    )
    return report


def model_inventory_entry(doc: ModelDocumentation) -> dict[str, Any]:
    """Generate MRM model inventory row.

    Produces a flat dict suitable for inclusion in a firm-wide model
    inventory, as required by SR 11-7 and PRA SS1/23.

    Args:
        doc: Populated :class:`ModelDocumentation` instance.

    Returns:
        Dict with model inventory fields.
    """
    gini = doc.discrimination_metrics.get("gini", 0.0)
    auroc = doc.discrimination_metrics.get("auroc", 0.0)
    psi = doc.stability_metrics.get("psi", 0.0)
    p_value = doc.calibration_metrics.get("p_value", 1.0)

    disc_rag = _rag_discrimination(gini)
    stab_rag = _rag_stability(psi)
    cal_rag = _rag_calibration(p_value)
    overall_rag = _worst_rag(disc_rag, stab_rag, cal_rag)

    entry: dict[str, Any] = {
        "model_name": doc.model_name,
        "model_id": doc.model_id,
        "model_type": doc.model_type,
        "version": doc.version,
        "owner": doc.model_owner,
        "effective_date": (
            doc.effective_date.isoformat() if doc.effective_date else None
        ),
        "purpose": doc.purpose,
        "scope": doc.scope,
        "regulatory_use": doc.regulatory_use,
        "methodology": doc.methodology,
        "sample_period": doc.sample_period,
        "sample_size": doc.sample_size,
        "discrimination": {
            "auroc": auroc,
            "gini": gini,
            "rag": disc_rag,
        },
        "calibration": {
            "p_value": p_value,
            "rag": cal_rag,
            **{k: v for k, v in doc.calibration_metrics.items() if k != "p_value"},
        },
        "stability": {
            "psi": psi,
            "rag": stab_rag,
        },
        "overall_rag": overall_rag,
        "validation_status": doc.validation_status,
        "validation_date": (
            doc.validation_date.isoformat() if doc.validation_date else None
        ),
        "finding_count": len(doc.findings),
        "limitation_count": len(doc.limitations),
    }

    logger.info(
        "Model inventory entry for %s: overall RAG=%s",
        doc.model_name,
        overall_rag,
    )
    return entry
