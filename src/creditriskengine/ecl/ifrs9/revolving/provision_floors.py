"""Multi-jurisdiction provision floors for revolving credit ECL.

Different regulators impose varying minimum provision levels on top
of IFRS 9's principle-based framework.  This module applies the
applicable floor based on jurisdiction, stage, and product type.

References:
    - CBUAE Circular 3/2024 (1.5% CRWA floor, graduated Stage 3)
    - RBI Draft Directions October 2025 (1%/5% unsecured retail)
    - MAS Notice 612 (1% MRLA for D-SIBs)
    - APRA APS 112 (40% CCF implicit floor)
    - Basel III (general 1% Tier 2 provision)
"""

from __future__ import annotations

from dataclasses import dataclass

from creditriskengine.core.types import IFRS9Stage, Jurisdiction


@dataclass(frozen=True)
class ProvisionFloor:
    """Minimum provision requirement for a jurisdiction/stage combination.

    Attributes:
        jurisdiction: Applicable jurisdiction.
        stage: IFRS 9 stage (or None for cross-stage floors).
        floor_rate: Minimum provision as a fraction of the reference
            amount (EAD or CRWA depending on floor_basis).
        floor_basis: What the floor_rate applies to (``"ead"`` or
            ``"crwa"`` for credit risk weighted assets).
        description: Regulatory reference for the floor.
    """

    jurisdiction: Jurisdiction
    stage: IFRS9Stage | None
    floor_rate: float
    floor_basis: str
    description: str


_PROVISION_FLOORS: list[ProvisionFloor] = [
    # CBUAE: 1.5% of CRWA for Stage 1 + Stage 2 combined
    ProvisionFloor(
        jurisdiction=Jurisdiction.UAE,
        stage=None,
        floor_rate=0.015,
        floor_basis="crwa",
        description="CBUAE Circular 3/2024: 1.5% CRWA floor (S1+S2)",
    ),
    # RBI: Stage 1 floor for unsecured retail (credit cards)
    ProvisionFloor(
        jurisdiction=Jurisdiction.INDIA,
        stage=IFRS9Stage.STAGE_1,
        floor_rate=0.01,
        floor_basis="ead",
        description="RBI Draft Oct 2025: 1% Stage 1 unsecured retail",
    ),
    # RBI: Stage 2 floor for unsecured retail (credit cards)
    ProvisionFloor(
        jurisdiction=Jurisdiction.INDIA,
        stage=IFRS9Stage.STAGE_2,
        floor_rate=0.05,
        floor_basis="ead",
        description="RBI Draft Oct 2025: 5% Stage 2 unsecured retail",
    ),
    # MAS: 1% MRLA for locally incorporated D-SIBs
    ProvisionFloor(
        jurisdiction=Jurisdiction.SINGAPORE,
        stage=None,
        floor_rate=0.01,
        floor_basis="ead",
        description="MAS Notice 612: 1% MRLA for D-SIBs",
    ),
    # SAMA: ~1% general provision floor
    ProvisionFloor(
        jurisdiction=Jurisdiction.SAUDI_ARABIA,
        stage=None,
        floor_rate=0.01,
        floor_basis="crwa",
        description="SAMA: ~1% CRWA general provision",
    ),
]


def get_provision_floors(
    jurisdiction: Jurisdiction,
    stage: IFRS9Stage | None = None,
) -> list[ProvisionFloor]:
    """Return applicable provision floors for a jurisdiction and stage.

    Args:
        jurisdiction: Reporting jurisdiction.
        stage: IFRS 9 stage.  If None, returns all floors for the
            jurisdiction.

    Returns:
        List of applicable :class:`ProvisionFloor` objects.
    """
    results: list[ProvisionFloor] = []
    for floor in _PROVISION_FLOORS:
        if floor.jurisdiction != jurisdiction:
            continue
        if stage is not None and floor.stage is not None and floor.stage != stage:
            continue
        results.append(floor)
    return results


def apply_provision_floor(
    ecl: float,
    ead: float,
    jurisdiction: Jurisdiction,
    stage: IFRS9Stage,
    crwa: float | None = None,
) -> float:
    """Apply the binding provision floor to a computed ECL.

    Returns the higher of the model ECL and the regulatory minimum.

    Args:
        ecl: Model-computed ECL amount.
        ead: Exposure at default.
        jurisdiction: Reporting jurisdiction.
        stage: IFRS 9 stage of the exposure.
        crwa: Credit risk weighted assets (required for CRWA-based
            floors such as CBUAE and SAMA).

    Returns:
        ECL after applying the binding floor.
    """
    floors = get_provision_floors(jurisdiction, stage)
    floor_ecl = ecl

    for floor in floors:
        if floor.floor_basis == "ead":
            minimum = floor.floor_rate * ead
        elif floor.floor_basis == "crwa" and crwa is not None:
            minimum = floor.floor_rate * crwa
        else:
            continue
        floor_ecl = max(floor_ecl, minimum)

    return floor_ecl
