"""RBI (Project Finance) Directions, 2025 — DCCO deferment framework.

Reference:
    - Reserve Bank of India (Project Finance) Directions, 2025 —
      notification dated 19 June 2025, effective 1 October 2025. Applies
      to projects achieving financial closure on/after the effective date.

Complements the standard-asset provisioning rates in ``iracp.py``
(construction 1.00% / 1.25% CRE; operational 0.40% other / 0.75% CRE-RH /
1.00% CRE) with the Date of Commencement of Commercial Operations (DCCO)
deferment mechanics:

* A project may defer its original DCCO while retaining "standard"
  classification for up to **3 years (infrastructure)** or **2 years
  (non-infrastructure, incl. CRE / CRE-RH)**.
* Each quarter of deferment attracts an additional provision — over and
  above the applicable standard-asset rate — of **0.375% per quarter
  (infrastructure)** or **0.5625% per quarter (non-infrastructure)**,
  reversed once commercial operations commence.
* Deferment beyond the permitted window means the exposure can no longer
  be treated as standard (restructuring / downgrade applies).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from creditriskengine.ecl.ind_as109.provision_floors_2026 import (
    RBI_DCCO_INFRA_QUARTERLY_RATE,
    RBI_DCCO_NON_INFRA_QUARTERLY_RATE,
)

logger = logging.getLogger(__name__)

# Permitted DCCO deferral while retaining standard classification.
DCCO_MAX_DEFERRAL_YEARS_INFRA: float = 3.0
DCCO_MAX_DEFERRAL_YEARS_NON_INFRA: float = 2.0


def dcco_max_deferral_years(is_infrastructure: bool) -> float:
    """Permitted DCCO extension retaining standard classification.

    Args:
        is_infrastructure: True for infrastructure projects.

    Returns:
        Maximum deferral in years (3.0 infrastructure, 2.0 otherwise).
    """
    return (
        DCCO_MAX_DEFERRAL_YEARS_INFRA
        if is_infrastructure
        else DCCO_MAX_DEFERRAL_YEARS_NON_INFRA
    )


@dataclass(frozen=True)
class DCCODefermentResult:
    """Provisioning outcome of a DCCO deferment assessment.

    Attributes:
        quarters_deferred: Whole quarters of DCCO deferment (rounded up).
        within_permitted_window: True if the deferment is within the
            3-year (infra) / 2-year (non-infra) limit.
        additional_provision_rate: Cumulative deferment add-on rate
            (quarterly rate x quarters). Zero when outside the window
            (the exposure is no longer standard — restructuring norms
            apply instead).
        total_provision_rate: Base standard-asset rate + deferment add-on
            while within the window; NaN-free 0.0 outside it.
        additional_provision: Add-on amount on the funded outstanding.
        total_provision: Total provision amount while standard.
    """

    quarters_deferred: int
    within_permitted_window: bool
    additional_provision_rate: float
    total_provision_rate: float
    additional_provision: float
    total_provision: float


def dcco_deferment_provision(
    funded_outstanding: float,
    deferral_years: float,
    base_provision_rate: float,
    is_infrastructure: bool,
) -> DCCODefermentResult:
    """Provision for a project whose DCCO has been deferred.

    Per the Project Finance Directions 2025, each quarter of DCCO
    deferment attracts an additional provision of 0.375% (infrastructure)
    or 0.5625% (non-infrastructure) of the funded outstanding, on top of
    the applicable standard-asset rate, while the deferment stays within
    the permitted window. Partial quarters count as full quarters.

    Args:
        funded_outstanding: Funded outstanding amount.
        deferral_years: Cumulative DCCO deferral from the original DCCO,
            in years (e.g. 0.5 = two quarters).
        base_provision_rate: Applicable standard-asset construction-phase
            rate (e.g. 0.0100 non-CRE, 0.0125 CRE from
            ``iracp.IRACP_STANDARD_RATES``).
        is_infrastructure: True for infrastructure projects.

    Returns:
        A :class:`DCCODefermentResult`. When the deferment exceeds the
        permitted window, ``within_permitted_window`` is False and the
        provision fields are zero — the exposure must instead be treated
        under restructuring / downgrade norms.

    Raises:
        ValueError: On negative inputs.
    """
    if funded_outstanding < 0.0:
        raise ValueError("funded_outstanding must be non-negative")
    if deferral_years < 0.0:
        raise ValueError("deferral_years must be non-negative")
    if base_provision_rate < 0.0:
        raise ValueError("base_provision_rate must be non-negative")

    quarters = math.ceil(round(deferral_years * 4.0, 9))
    within = deferral_years <= dcco_max_deferral_years(is_infrastructure)

    if not within:
        logger.warning(
            "DCCO deferral %.2fy exceeds the permitted %.0fy window; "
            "standard classification is lost",
            deferral_years, dcco_max_deferral_years(is_infrastructure),
        )
        return DCCODefermentResult(
            quarters_deferred=quarters,
            within_permitted_window=False,
            additional_provision_rate=0.0,
            total_provision_rate=0.0,
            additional_provision=0.0,
            total_provision=0.0,
        )

    quarterly_rate = (
        RBI_DCCO_INFRA_QUARTERLY_RATE
        if is_infrastructure
        else RBI_DCCO_NON_INFRA_QUARTERLY_RATE
    )
    addon_rate = quarterly_rate * quarters
    total_rate = base_provision_rate + addon_rate

    return DCCODefermentResult(
        quarters_deferred=quarters,
        within_permitted_window=True,
        additional_provision_rate=round(addon_rate, 8),
        total_provision_rate=round(total_rate, 8),
        additional_provision=round(addon_rate * funded_outstanding, 6),
        total_provision=round(total_rate * funded_outstanding, 6),
    )
