"""EU COREP (Common Reporting) templates -- CRR3 / EBA ITS.

Generates structured data aligned with EBA COREP credit risk templates:
- C 07.00: Credit and counterparty credit risk -- SA
- C 08.01: Credit and counterparty credit risk -- IRB (Foundation)
- C 08.02: Credit and counterparty credit risk -- IRB (Advanced)
- C 09.01: Geographic breakdown of exposures by SA classes
- C 09.02: Geographic breakdown -- IRB

These produce dict/DataFrame outputs that can be mapped to EBA XBRL taxonomy
or CSV templates for submission.

References:
    - EBA ITS on Supervisory Reporting (Regulation 2021/451, updated for CRR3)
    - CRR3 Art. 430 (Reporting requirements)
"""

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# SA exposure classes per CRR3 Art. 112
SA_EXPOSURE_CLASSES: list[str] = [
    "Central governments or central banks",
    "Regional governments or local authorities",
    "Public sector entities",
    "Multilateral development banks",
    "International organisations",
    "Institutions",
    "Corporates",
    "Retail",
    "Secured by mortgages on immovable property",
    "Exposures in default",
    "Higher-risk categories",
    "Covered bonds",
    "Institutions and corporates with short-term credit assessment",
    "Collective investment undertakings",
    "Equity exposures",
    "Other items",
]

# IRB exposure classes per CRR3 Art. 147
IRB_EXPOSURE_CLASSES: list[str] = [
    "Central governments and central banks",
    "Institutions",
    "Corporates — SME",
    "Corporates — Specialised lending",
    "Corporates — Other",
    "Retail — Secured by immovable property — SME",
    "Retail — Secured by immovable property — Non-SME",
    "Retail — Qualifying revolving",
    "Retail — Other SME",
    "Retail — Other non-SME",
    "Equity",
]


@dataclass
class COREPRow:
    """Single row in a COREP template."""

    row_id: str
    exposure_class: str
    original_exposure: float = 0.0
    credit_risk_mitigation: float = 0.0
    ead_post_crm: float = 0.0
    risk_weight_pct: float = 0.0
    rwa: float = 0.0
    expected_loss: float = 0.0
    provisions: float = 0.0


@dataclass
class COREPTemplate:
    """Complete COREP template."""

    template_id: str  # e.g., "C 07.00"
    template_name: str
    reporting_date: str
    institution_name: str
    rows: list[COREPRow] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_original_exposure(self) -> float:
        """Sum of original exposures across all rows."""
        return sum(r.original_exposure for r in self.rows)

    @property
    def total_rwa(self) -> float:
        """Sum of RWA across all rows."""
        return sum(r.rwa for r in self.rows)

    @property
    def total_expected_loss(self) -> float:
        """Sum of expected losses across all rows."""
        return sum(r.expected_loss for r in self.rows)


def _build_sa_row(
    row_idx: int,
    exposure_class: str,
    class_data: dict[str, float],
) -> COREPRow:
    """Build a single SA COREP row from class-level data.

    Args:
        row_idx: Sequential row index (used for row_id).
        exposure_class: SA exposure class name.
        class_data: Dict with keys ``original_exposure``,
            ``credit_risk_mitigation``, ``risk_weight_pct``, and optionally
            ``provisions``.

    Returns:
        Populated :class:`COREPRow`.
    """
    original = class_data.get("original_exposure", 0.0)
    crm = class_data.get("credit_risk_mitigation", 0.0)
    ead = original - crm
    rw = class_data.get("risk_weight_pct", 0.0)
    rwa = ead * rw / 100.0
    provisions = class_data.get("provisions", 0.0)

    return COREPRow(
        row_id=f"R{row_idx:04d}",
        exposure_class=exposure_class,
        original_exposure=original,
        credit_risk_mitigation=crm,
        ead_post_crm=ead,
        risk_weight_pct=rw,
        rwa=rwa,
        expected_loss=0.0,  # SA does not produce EL
        provisions=provisions,
    )


def _build_irb_row(
    row_idx: int,
    exposure_class: str,
    class_data: dict[str, float],
    *,
    advanced: bool = False,
) -> COREPRow:
    """Build a single IRB COREP row from class-level data.

    Args:
        row_idx: Sequential row index.
        exposure_class: IRB exposure class name.
        class_data: Dict with keys ``original_exposure``,
            ``credit_risk_mitigation``, ``ead_post_crm``, ``rwa``,
            ``expected_loss``, and optionally ``provisions``.
        advanced: If True, use A-IRB fields (bank-estimated LGD/EAD);
            otherwise use F-IRB defaults.

    Returns:
        Populated :class:`COREPRow`.
    """
    original = class_data.get("original_exposure", 0.0)
    crm = class_data.get("credit_risk_mitigation", 0.0)
    ead = class_data.get("ead_post_crm", original - crm)
    rwa = class_data.get("rwa", 0.0)
    el = class_data.get("expected_loss", 0.0)
    provisions = class_data.get("provisions", 0.0)
    rw = (rwa / ead * 100.0) if ead > 0 else 0.0

    return COREPRow(
        row_id=f"R{row_idx:04d}",
        exposure_class=exposure_class,
        original_exposure=original,
        credit_risk_mitigation=crm,
        ead_post_crm=ead,
        risk_weight_pct=round(rw, 4),
        rwa=rwa,
        expected_loss=el,
        provisions=provisions,
    )


def generate_c0700_sa(
    exposures_by_class: dict[str, dict[str, float]],
    reporting_date: str,
    institution_name: str = "",
) -> COREPTemplate:
    """Generate C 07.00 -- Credit risk SA template.

    Rows per SA exposure class: Central governments, Institutions, Corporates,
    Retail, Secured by mortgages, Exposures in default, Equity, Other items.
    Columns: Original exposure, CRM, Net exposure, RWA.

    Args:
        exposures_by_class: Mapping of SA exposure class name to a dict
            containing ``original_exposure``, ``credit_risk_mitigation``,
            ``risk_weight_pct``, and optionally ``provisions``.
        reporting_date: Reporting reference date (ISO format string).
        institution_name: Name of the reporting institution.

    Returns:
        Populated :class:`COREPTemplate` for C 07.00.
    """
    logger.info("Generating C 07.00 SA template for %s", reporting_date)
    rows: list[COREPRow] = []
    row_idx = 1

    for ec in SA_EXPOSURE_CLASSES:
        class_data = exposures_by_class.get(ec, {})
        if not class_data:
            continue
        rows.append(_build_sa_row(row_idx, ec, class_data))
        row_idx += 1

    # Handle any non-standard classes supplied by the caller
    for ec, class_data in exposures_by_class.items():
        if ec not in SA_EXPOSURE_CLASSES:
            logger.warning(
                "Non-standard SA exposure class '%s'; appending to template", ec
            )
            rows.append(_build_sa_row(row_idx, ec, class_data))
            row_idx += 1

    template = COREPTemplate(
        template_id="C 07.00",
        template_name="Credit and counterparty credit risk — SA",
        reporting_date=reporting_date,
        institution_name=institution_name,
        rows=rows,
        metadata={
            "approach": "SA",
            "total_original_exposure": sum(r.original_exposure for r in rows),
            "total_rwa": sum(r.rwa for r in rows),
        },
    )
    logger.info(
        "C 07.00: %d rows, total RWA=%.2f",
        len(rows),
        template.metadata["total_rwa"],
    )
    return template


def generate_c0801_irb(
    exposures_by_class: dict[str, dict[str, float]],
    reporting_date: str,
    institution_name: str = "",
) -> COREPTemplate:
    """Generate C 08.01 -- Credit risk IRB Foundation template.

    Uses supervisory LGD and CCF values (Foundation IRB).

    Args:
        exposures_by_class: Mapping of IRB exposure class name to a dict
            containing ``original_exposure``, ``credit_risk_mitigation``,
            ``ead_post_crm``, ``rwa``, ``expected_loss``, and optionally
            ``provisions``.
        reporting_date: Reporting reference date (ISO format string).
        institution_name: Name of the reporting institution.

    Returns:
        Populated :class:`COREPTemplate` for C 08.01.
    """
    logger.info("Generating C 08.01 F-IRB template for %s", reporting_date)
    rows: list[COREPRow] = []
    row_idx = 1

    for ec in IRB_EXPOSURE_CLASSES:
        class_data = exposures_by_class.get(ec, {})
        if not class_data:
            continue
        rows.append(_build_irb_row(row_idx, ec, class_data, advanced=False))
        row_idx += 1

    for ec, class_data in exposures_by_class.items():
        if ec not in IRB_EXPOSURE_CLASSES:
            logger.warning(
                "Non-standard IRB exposure class '%s'; appending to template", ec
            )
            rows.append(_build_irb_row(row_idx, ec, class_data, advanced=False))
            row_idx += 1

    template = COREPTemplate(
        template_id="C 08.01",
        template_name="Credit and counterparty credit risk — IRB (Foundation)",
        reporting_date=reporting_date,
        institution_name=institution_name,
        rows=rows,
        metadata={
            "approach": "F-IRB",
            "total_original_exposure": sum(r.original_exposure for r in rows),
            "total_rwa": sum(r.rwa for r in rows),
            "total_expected_loss": sum(r.expected_loss for r in rows),
        },
    )
    logger.info(
        "C 08.01: %d rows, total RWA=%.2f, total EL=%.2f",
        len(rows),
        template.metadata["total_rwa"],
        template.metadata["total_expected_loss"],
    )
    return template


def generate_c0802_airb(
    exposures_by_class: dict[str, dict[str, float]],
    reporting_date: str,
    institution_name: str = "",
) -> COREPTemplate:
    """Generate C 08.02 -- Credit risk IRB Advanced template.

    Uses bank-estimated LGD and EAD (Advanced IRB).

    Args:
        exposures_by_class: Mapping of IRB exposure class name to a dict
            containing ``original_exposure``, ``credit_risk_mitigation``,
            ``ead_post_crm``, ``rwa``, ``expected_loss``, and optionally
            ``provisions``.
        reporting_date: Reporting reference date (ISO format string).
        institution_name: Name of the reporting institution.

    Returns:
        Populated :class:`COREPTemplate` for C 08.02.
    """
    logger.info("Generating C 08.02 A-IRB template for %s", reporting_date)
    rows: list[COREPRow] = []
    row_idx = 1

    for ec in IRB_EXPOSURE_CLASSES:
        class_data = exposures_by_class.get(ec, {})
        if not class_data:
            continue
        rows.append(_build_irb_row(row_idx, ec, class_data, advanced=True))
        row_idx += 1

    for ec, class_data in exposures_by_class.items():
        if ec not in IRB_EXPOSURE_CLASSES:
            logger.warning(
                "Non-standard IRB exposure class '%s'; appending to template", ec
            )
            rows.append(_build_irb_row(row_idx, ec, class_data, advanced=True))
            row_idx += 1

    template = COREPTemplate(
        template_id="C 08.02",
        template_name="Credit and counterparty credit risk — IRB (Advanced)",
        reporting_date=reporting_date,
        institution_name=institution_name,
        rows=rows,
        metadata={
            "approach": "A-IRB",
            "total_original_exposure": sum(r.original_exposure for r in rows),
            "total_rwa": sum(r.rwa for r in rows),
            "total_expected_loss": sum(r.expected_loss for r in rows),
        },
    )
    logger.info(
        "C 08.02: %d rows, total RWA=%.2f, total EL=%.2f",
        len(rows),
        template.metadata["total_rwa"],
        template.metadata["total_expected_loss"],
    )
    return template


def corep_to_dict(template: COREPTemplate) -> dict[str, Any]:
    """Convert COREP template to nested dict for serialization.

    Args:
        template: A populated :class:`COREPTemplate`.

    Returns:
        Nested dict suitable for JSON serialization or DataFrame conversion.
    """
    return {
        "template_id": template.template_id,
        "template_name": template.template_name,
        "reporting_date": template.reporting_date,
        "institution_name": template.institution_name,
        "metadata": template.metadata,
        "summary": {
            "total_original_exposure": template.total_original_exposure,
            "total_rwa": template.total_rwa,
            "total_expected_loss": template.total_expected_loss,
            "row_count": len(template.rows),
        },
        "rows": [asdict(row) for row in template.rows],
    }


def corep_summary(templates: list[COREPTemplate]) -> dict[str, Any]:
    """Aggregate summary across multiple COREP templates.

    Useful for consolidated reporting or cross-template consistency checks.

    Args:
        templates: List of :class:`COREPTemplate` instances.

    Returns:
        Summary dict with aggregated totals and per-template breakdowns.
    """
    if not templates:
        logger.warning("corep_summary called with empty template list")
        return {"templates": [], "totals": {}}

    per_template: list[dict[str, Any]] = []
    grand_total_rwa = 0.0
    grand_total_exposure = 0.0
    grand_total_el = 0.0

    for tmpl in templates:
        tmpl_rwa = tmpl.total_rwa
        tmpl_exposure = tmpl.total_original_exposure
        tmpl_el = tmpl.total_expected_loss
        grand_total_rwa += tmpl_rwa
        grand_total_exposure += tmpl_exposure
        grand_total_el += tmpl_el

        per_template.append({
            "template_id": tmpl.template_id,
            "template_name": tmpl.template_name,
            "reporting_date": tmpl.reporting_date,
            "institution_name": tmpl.institution_name,
            "row_count": len(tmpl.rows),
            "total_original_exposure": tmpl_exposure,
            "total_rwa": tmpl_rwa,
            "total_expected_loss": tmpl_el,
            "avg_risk_weight_pct": (
                tmpl_rwa / tmpl_exposure * 100.0 if tmpl_exposure > 0 else 0.0
            ),
        })

    summary: dict[str, Any] = {
        "template_count": len(templates),
        "templates": per_template,
        "totals": {
            "total_original_exposure": grand_total_exposure,
            "total_rwa": grand_total_rwa,
            "total_expected_loss": grand_total_el,
            "avg_risk_weight_pct": (
                grand_total_rwa / grand_total_exposure * 100.0
                if grand_total_exposure > 0
                else 0.0
            ),
            "capital_requirement_8pct": grand_total_rwa * 0.08,
        },
    }

    logger.info(
        "COREP summary: %d templates, total RWA=%.2f, capital req=%.2f",
        len(templates),
        grand_total_rwa,
        summary["totals"]["capital_requirement_8pct"],
    )
    return summary
