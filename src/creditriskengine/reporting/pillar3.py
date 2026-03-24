"""Pillar 3 disclosure templates per BCBS d382.

Generates structured data conforming to the Basel Committee's Pillar 3
disclosure requirements for credit risk.

References:
    - BCBS d382: Pillar 3 disclosure requirements — consolidated and enhanced
    - CRR Part Eight: Disclosure by institutions
    - EBA ITS on public disclosures (EBA/ITS/2020/04)
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def generate_cr1_template(
    total_defaulted: float,
    total_non_defaulted: float,
    specific_provisions: float,
    general_provisions: float,
) -> dict[str, Any]:
    """Generate CR1: Credit quality of assets template.

    CR1 provides a comprehensive view of on-balance-sheet and
    off-balance-sheet credit quality, including defaulted and
    non-defaulted exposures and associated provisions.

    Reference: BCBS d382 Table CR1.

    Args:
        total_defaulted: Gross carrying amount of defaulted exposures.
        total_non_defaulted: Gross carrying amount of non-defaulted exposures.
        specific_provisions: Specific credit risk adjustments (Stage 3 / IAS 39).
        general_provisions: General credit risk adjustments (Stage 1+2 / IAS 39).

    Returns:
        Dict structured per Table CR1 with gross exposures, provisions,
        and net values.
    """
    total_provisions = specific_provisions + general_provisions
    gross_total = total_defaulted + total_non_defaulted
    net_defaulted = total_defaulted - specific_provisions
    net_non_defaulted = total_non_defaulted - general_provisions
    net_total = gross_total - total_provisions

    logger.debug(
        "CR1 template: defaulted=%.2f, non_defaulted=%.2f, "
        "provisions=%.2f, net=%.2f",
        total_defaulted, total_non_defaulted,
        total_provisions, net_total,
    )

    return {
        "template": "CR1",
        "description": "Credit quality of assets",
        "defaulted_exposures": {
            "gross_carrying_amount": total_defaulted,
            "specific_provisions": specific_provisions,
            "net_carrying_amount": net_defaulted,
        },
        "non_defaulted_exposures": {
            "gross_carrying_amount": total_non_defaulted,
            "general_provisions": general_provisions,
            "net_carrying_amount": net_non_defaulted,
        },
        "total": {
            "gross_carrying_amount": gross_total,
            "total_provisions": total_provisions,
            "net_carrying_amount": net_total,
        },
    }


def generate_cr3_crm_overview(
    exposures_unsecured: float,
    exposures_secured_collateral: float,
    exposures_secured_guarantees: float,
    exposures_secured_credit_derivatives: float,
) -> dict[str, Any]:
    """Generate CR3: CRM techniques overview template.

    CR3 provides an overview of credit risk mitigation (CRM) techniques
    applied to on-balance-sheet exposures, broken down by type of
    protection.

    Reference: BCBS d382 Table CR3.

    Args:
        exposures_unsecured: Exposure amount not covered by any CRM.
        exposures_secured_collateral: Exposure secured by eligible collateral.
        exposures_secured_guarantees: Exposure secured by guarantees.
        exposures_secured_credit_derivatives: Exposure secured by credit
            derivatives.

    Returns:
        Dict structured per Table CR3 with exposure breakdowns by
        CRM technique.
    """
    total_secured = (
        exposures_secured_collateral
        + exposures_secured_guarantees
        + exposures_secured_credit_derivatives
    )
    total_exposures = exposures_unsecured + total_secured

    secured_pct = total_secured / total_exposures if total_exposures > 0 else 0.0

    logger.debug(
        "CR3 template: unsecured=%.2f, secured=%.2f (%.1f%%), total=%.2f",
        exposures_unsecured, total_secured, secured_pct * 100, total_exposures,
    )

    return {
        "template": "CR3",
        "description": "CRM techniques — overview",
        "exposures_unsecured": exposures_unsecured,
        "exposures_secured": {
            "collateral": exposures_secured_collateral,
            "guarantees": exposures_secured_guarantees,
            "credit_derivatives": exposures_secured_credit_derivatives,
            "total_secured": total_secured,
        },
        "total_exposures": total_exposures,
        "secured_pct": secured_pct,
    }


def generate_cr4_sa_overview(
    exposure_classes_data: list[dict[str, Any]],
) -> dict[str, Any]:
    """Generate CR4: Standardised approach — credit risk overview.

    CR4 summarises exposures, CRM adjustments, net exposures, and
    risk-weighted assets by SA exposure class.

    Each entry in exposure_classes_data should contain:
        - exposure_class: str
        - gross_exposure: float
        - crm_adjustments: float
        - net_exposure: float
        - rwa: float

    Reference: BCBS d382 Table CR4.

    Args:
        exposure_classes_data: List of dicts, one per SA exposure class.

    Returns:
        Dict structured per Table CR4 with per-class and aggregate data.
    """
    total_gross = 0.0
    total_crm = 0.0
    total_net = 0.0
    total_rwa = 0.0

    rows: list[dict[str, Any]] = []
    for ec in exposure_classes_data:
        gross = ec.get("gross_exposure", 0.0)
        crm = ec.get("crm_adjustments", 0.0)
        net = ec.get("net_exposure", 0.0)
        rwa = ec.get("rwa", 0.0)
        rw = rwa / net if net > 0 else 0.0

        rows.append({
            "exposure_class": ec.get("exposure_class", ""),
            "gross_exposure": gross,
            "crm_adjustments": crm,
            "net_exposure": net,
            "rwa": rwa,
            "risk_weight": rw,
        })

        total_gross += gross
        total_crm += crm
        total_net += net
        total_rwa += rwa

    avg_rw = total_rwa / total_net if total_net > 0 else 0.0

    logger.debug(
        "CR4 template: %d classes, total_net=%.2f, total_rwa=%.2f, avg_rw=%.4f",
        len(rows), total_net, total_rwa, avg_rw,
    )

    return {
        "template": "CR4",
        "description": "Standardised approach — credit risk exposure and CRM effects",
        "exposure_classes": rows,
        "totals": {
            "gross_exposure": total_gross,
            "crm_adjustments": total_crm,
            "net_exposure": total_net,
            "rwa": total_rwa,
            "avg_risk_weight": avg_rw,
        },
    }


def generate_cr6_irb_overview(
    irb_data: list[dict[str, Any]],
) -> dict[str, Any]:
    """Generate CR6: IRB — credit risk exposures by PD grade.

    CR6 provides a granular view of IRB credit risk exposures broken
    down by internal PD grade, showing obligor counts, EAD, average
    risk parameters, RWA, and expected loss.

    Each entry in irb_data should contain:
        - pd_range: str (e.g. '0.00-0.15%')
        - n_obligors: int
        - ead: float
        - avg_pd: float
        - avg_lgd: float
        - avg_rw: float
        - rwa: float
        - el: float

    Reference: BCBS d382 Table CR6.

    Args:
        irb_data: List of dicts, one per PD grade bucket.

    Returns:
        Dict structured per Table CR6 with per-grade and aggregate data.
    """
    total_obligors = 0
    total_ead = 0.0
    total_rwa = 0.0
    total_el = 0.0

    rows: list[dict[str, Any]] = []
    for grade in irb_data:
        n_obligors = grade.get("n_obligors", 0)
        ead = grade.get("ead", 0.0)
        rwa = grade.get("rwa", 0.0)
        el = grade.get("el", 0.0)

        rows.append({
            "pd_range": grade.get("pd_range", ""),
            "n_obligors": n_obligors,
            "ead": ead,
            "avg_pd": grade.get("avg_pd", 0.0),
            "avg_lgd": grade.get("avg_lgd", 0.0),
            "avg_rw": grade.get("avg_rw", 0.0),
            "rwa": rwa,
            "el": el,
        })

        total_obligors += n_obligors
        total_ead += ead
        total_rwa += rwa
        total_el += el

    weighted_avg_pd = (
        sum(g.get("avg_pd", 0.0) * g.get("ead", 0.0) for g in irb_data) / total_ead
        if total_ead > 0 else 0.0
    )
    weighted_avg_lgd = (
        sum(g.get("avg_lgd", 0.0) * g.get("ead", 0.0) for g in irb_data) / total_ead
        if total_ead > 0 else 0.0
    )
    avg_rw = total_rwa / total_ead if total_ead > 0 else 0.0

    logger.debug(
        "CR6 template: %d grades, %d obligors, total_ead=%.2f, "
        "total_rwa=%.2f, total_el=%.2f",
        len(rows), total_obligors, total_ead, total_rwa, total_el,
    )

    return {
        "template": "CR6",
        "description": "IRB — credit risk exposures by portfolio and PD range",
        "pd_grades": rows,
        "totals": {
            "n_obligors": total_obligors,
            "ead": total_ead,
            "weighted_avg_pd": weighted_avg_pd,
            "weighted_avg_lgd": weighted_avg_lgd,
            "avg_risk_weight": avg_rw,
            "rwa": total_rwa,
            "el": total_el,
        },
    }
