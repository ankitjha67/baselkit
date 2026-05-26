"""
Self-validating assertions for RBI ECL Master Direction 2026 parameters.

This module locks in the published numerical values from RBI/DOR/2026-27/398
(DOR.STR.REC.No.6/21.06.011/2026-27, April 27, 2026) so that any future
edit to a parameter triggers a test failure. Acts as a regulatory-source-
of-truth guard.

Reference: RBI Master Direction on Asset Classification, Provisioning and
Income Recognition (Commercial Banks), April 27, 2026, in force from
April 1, 2027.
"""

from __future__ import annotations

from datetime import date

from creditriskengine.core.types import IFRS9Stage
from creditriskengine.ecl.ind_as109.pd_lgd_floors import (
    RBI_LGD_BACKSTOP_SECURED,
    RBI_LGD_BACKSTOP_UNSECURED,
    RBI_LGD_ELIGIBLE_COLLATERAL,
    RBI_PD_FLOOR,
)
from creditriskengine.ecl.ind_as109.provision_floors_2026 import (
    RBI_DCCO_INFRA_QUARTERLY_RATE,
    RBI_DCCO_NON_INFRA_QUARTERLY_RATE,
    RBI_ECL_FLOOR_STAGE_1_2,
    rbi_ecl_floor_2026,
)
from creditriskengine.ecl.ind_as109.transition import (
    CAPITAL_ADD_BACK_SCHEDULE,
    RBI_ECL_EFFECTIVE_DATE,
    RBI_EIR_MIGRATION_DEADLINE,
)
from creditriskengine.ecl.ind_as109.types import RBIExposureCategory

# ---------------------------------------------------------------------------
# Source-of-truth published values (from RBI/DOR/2026-27/398)
# ---------------------------------------------------------------------------

PUBLISHED_PD_FLOOR: float = 0.0003
"""Paragraph 96."""

PUBLISHED_LGD_SECURED: float = 0.65
PUBLISHED_LGD_UNSECURED: float = 0.70
PUBLISHED_LGD_ELIGIBLE_COLLATERAL: float = 0.30
"""Paragraphs 97-98."""

PUBLISHED_DCCO_INFRA_QUARTERLY: float = 0.00375
PUBLISHED_DCCO_NON_INFRA_QUARTERLY: float = 0.005625
"""Paragraph 82(4) Note 1."""

PUBLISHED_EFFECTIVE_DATE: date = date(2027, 4, 1)
PUBLISHED_EIR_MIGRATION_DEADLINE: date = date(2030, 3, 31)
"""Paragraphs 2, 21, 114."""

PUBLISHED_CAPITAL_ADD_BACK: dict[int, float] = {
    2028: 0.80,
    2029: 0.60,
    2030: 0.40,
    2031: 0.20,
}
"""Paragraph 108: 4/5 -> 3/5 -> 2/5 -> 1/5."""

PUBLISHED_STAGE_1_2_FLOORS: dict[RBIExposureCategory, tuple[float, float]] = {
    RBIExposureCategory.SECURED_RETAIL: (0.0040, 0.05),
    RBIExposureCategory.CORPORATE: (0.0040, 0.05),
    RBIExposureCategory.SMALL_MICRO_ENTERPRISE: (0.0025, 0.05),
    RBIExposureCategory.MEDIUM_ENTERPRISE: (0.0040, 0.05),
    RBIExposureCategory.FARM_CREDIT_AGRICULTURAL: (0.0025, 0.05),
    RBIExposureCategory.BANKS_NBFCS_REGULATED_FIS: (0.0040, 0.05),
    RBIExposureCategory.LOANS_AGAINST_DEPOSITS_LIC_KVP: (0.0040, 0.0040),
    RBIExposureCategory.GOLD_LOANS: (0.0040, 0.015),
    RBIExposureCategory.STATE_GOVT_GUARANTEED: (0.0040, 0.025),
    RBIExposureCategory.UNSECURED_RETAIL: (0.01, 0.05),
    RBIExposureCategory.HOUSING_LOANS_INDIVIDUALS: (0.0025, 0.015),
    RBIExposureCategory.CRE_ADC_150: (0.0125, 0.05),
    RBIExposureCategory.CRE_RH_ADC: (0.01, 0.05),
    RBIExposureCategory.OTHER_RESIDENTIAL_RE: (0.0040, 0.015),
    RBIExposureCategory.OTHER_COMMERCIAL_RE: (0.0040, 0.025),
    RBIExposureCategory.PROJECT_FINANCE_PRE_OPERATIONAL: (0.01, 0.05),
    RBIExposureCategory.PROJECT_FINANCE_OPERATIONAL: (0.0040, 0.05),
    RBIExposureCategory.CENTRAL_GOVT_GUARANTEED: (0.0025, 0.0025),
    RBIExposureCategory.NATURAL_CALAMITY_RESTRUCTURED: (0.05, 0.10),
    RBIExposureCategory.OTHER: (0.0040, 0.05),
}
"""Paragraph 82(1)-(4)."""

# Stage 3 spot checks per Paragraph 82(5)
PUBLISHED_STAGE3_SAMPLES: list[tuple[RBIExposureCategory, float, bool, float]] = [
    # (category, years_in_stage3, is_secured, expected_floor_rate)
    # Set A
    (RBIExposureCategory.CORPORATE, 0.5, True, 0.25),
    (RBIExposureCategory.CORPORATE, 0.5, False, 0.40),  # corrected unsecured 0-1y
    (RBIExposureCategory.CORPORATE, 1.5, True, 0.40),
    (RBIExposureCategory.CORPORATE, 4.5, True, 1.00),
    # Set B (housing, deposits, gold, CGTMSE)
    (RBIExposureCategory.HOUSING_LOANS_INDIVIDUALS, 0.5, True, 0.10),
    (RBIExposureCategory.GOLD_LOANS, 0.5, True, 0.10),
    # Set C (unsecured retail)
    (RBIExposureCategory.UNSECURED_RETAIL, 0.5, False, 0.25),
    (RBIExposureCategory.UNSECURED_RETAIL, 1.5, False, 1.00),
    # Set F (other RE-secured)
    (RBIExposureCategory.OTHER_RESIDENTIAL_RE, 0.5, True, 0.15),
    (RBIExposureCategory.OTHER_COMMERCIAL_RE, 2.5, True, 0.40),
]


class RBIParameterMismatch(AssertionError):  # noqa: N818
    """Raised when a live RBI parameter diverges from published value."""


def _assert_close(actual: float, expected: float, label: str) -> None:
    if abs(actual - expected) > 1e-9:
        raise RBIParameterMismatch(
            f"{label}: live value {actual} != published {expected}"
        )


def assert_rbi_2026_parameters_match_published() -> None:
    """Verify every live constant matches its published RBI value.

    Run this at module-import time, in CI, or as a smoke test before
    each ECL calculation cycle. If any future code change accidentally
    perturbs a published parameter, this raises ``RBIParameterMismatch``.

    Reference:
        RBI/DOR/2026-27/398, all paragraphs covered.
    """
    # PD / LGD floors
    _assert_close(RBI_PD_FLOOR, PUBLISHED_PD_FLOOR, "PD floor (Para 96)")
    _assert_close(
        RBI_LGD_BACKSTOP_SECURED, PUBLISHED_LGD_SECURED,
        "LGD secured (Para 97)",
    )
    _assert_close(
        RBI_LGD_BACKSTOP_UNSECURED, PUBLISHED_LGD_UNSECURED,
        "LGD unsecured (Para 97)",
    )
    _assert_close(
        RBI_LGD_ELIGIBLE_COLLATERAL, PUBLISHED_LGD_ELIGIBLE_COLLATERAL,
        "LGD eligible collateral (Para 98)",
    )

    # DCCO
    _assert_close(
        RBI_DCCO_INFRA_QUARTERLY_RATE, PUBLISHED_DCCO_INFRA_QUARTERLY,
        "DCCO infra (Para 82(4) Note 1)",
    )
    _assert_close(
        RBI_DCCO_NON_INFRA_QUARTERLY_RATE, PUBLISHED_DCCO_NON_INFRA_QUARTERLY,
        "DCCO non-infra (Para 82(4) Note 1)",
    )

    # Dates
    if RBI_ECL_EFFECTIVE_DATE != PUBLISHED_EFFECTIVE_DATE:
        raise RBIParameterMismatch(
            f"Effective date {RBI_ECL_EFFECTIVE_DATE} != "
            f"published {PUBLISHED_EFFECTIVE_DATE}"
        )
    if RBI_EIR_MIGRATION_DEADLINE != PUBLISHED_EIR_MIGRATION_DEADLINE:
        raise RBIParameterMismatch(
            f"EIR deadline {RBI_EIR_MIGRATION_DEADLINE} != "
            f"published {PUBLISHED_EIR_MIGRATION_DEADLINE}"
        )

    # Capital add-back
    if CAPITAL_ADD_BACK_SCHEDULE != PUBLISHED_CAPITAL_ADD_BACK:
        raise RBIParameterMismatch(
            f"Capital add-back schedule {CAPITAL_ADD_BACK_SCHEDULE} != "
            f"published {PUBLISHED_CAPITAL_ADD_BACK}"
        )

    # Stage 1/2 floors — every category
    for category, (pub_s1, pub_s2) in PUBLISHED_STAGE_1_2_FLOORS.items():
        live_s1, live_s2 = RBI_ECL_FLOOR_STAGE_1_2[category]
        _assert_close(live_s1, pub_s1, f"{category.value} Stage 1 (Para 82)")
        _assert_close(live_s2, pub_s2, f"{category.value} Stage 2 (Para 82)")

    # Stage 3 spot checks
    ead = 100_000.0
    for category, years, is_secured, expected_rate in PUBLISHED_STAGE3_SAMPLES:
        actual_floor = rbi_ecl_floor_2026(
            ead, IFRS9Stage.STAGE_3, category,
            is_secured=is_secured, years_in_stage3=years,
        )
        expected_floor = ead * expected_rate
        if abs(actual_floor - expected_floor) > 0.01:
            raise RBIParameterMismatch(
                f"Stage 3 floor for {category.value} "
                f"(years={years}, secured={is_secured}): "
                f"{actual_floor:.2f} != published {expected_floor:.2f}"
            )


def regulatory_self_check() -> dict[str, str]:
    """Return a structured self-check report.

    Calls :func:`assert_rbi_2026_parameters_match_published` and returns
    a status dict suitable for inclusion in audit packs and validation
    reports.

    Returns:
        Dict with ``status`` ("ok" or "mismatch"), ``message``, and
        ``reference``.
    """
    try:
        assert_rbi_2026_parameters_match_published()
        return {
            "status": "ok",
            "message": (
                "All live RBI ECL 2026 parameters match published values "
                "from RBI/DOR/2026-27/398."
            ),
            "reference": (
                "RBI/DOR/2026-27/398; DOR.STR.REC.No.6/21.06.011/2026-27, "
                "April 27, 2026"
            ),
        }
    except RBIParameterMismatch as exc:
        return {
            "status": "mismatch",
            "message": str(exc),
            "reference": (
                "RBI/DOR/2026-27/398; DOR.STR.REC.No.6/21.06.011/2026-27, "
                "April 27, 2026"
            ),
        }
