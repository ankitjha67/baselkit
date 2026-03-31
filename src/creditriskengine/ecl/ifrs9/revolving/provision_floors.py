"""Multi-jurisdiction provision floors for revolving credit ECL.

Loads from ``regulatory/provision_floors.yml`` at module import.
Users can call :func:`load_provision_floors` with a custom YAML path
to override defaults at runtime.

References:
    - CBUAE Circular 3/2024 (1.5% CRWA floor, graduated Stage 3)
    - RBI Draft Directions October 2025 (1%/5% unsecured retail)
    - MAS Notice 612 (1% MRLA for D-SIBs)
    - SAMA general provision (~1% CRWA)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from creditriskengine.core.types import IFRS9Stage, Jurisdiction

logger = logging.getLogger(__name__)

_DEFAULT_YAML = (
    Path(__file__).resolve().parents[3]
    / "regulatory"
    / "provision_floors.yml"
)

# Jurisdiction string → enum mapping
_JURISDICTION_MAP: dict[str, Jurisdiction] = {j.value: j for j in Jurisdiction}


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


def _parse_floor(data: dict[str, Any]) -> ProvisionFloor | None:
    """Parse a single provision floor entry from YAML."""
    jur_str = data.get("jurisdiction", "")
    jur = _JURISDICTION_MAP.get(jur_str)
    if jur is None:
        logger.warning("Unknown jurisdiction in provision floor: %r", jur_str)
        return None

    stage_val = data.get("stage")
    stage: IFRS9Stage | None = None
    if stage_val is not None:
        stage = IFRS9Stage(int(stage_val))

    return ProvisionFloor(
        jurisdiction=jur,
        stage=stage,
        floor_rate=float(data.get("floor_rate", 0.0)),
        floor_basis=str(data.get("floor_basis", "ead")),
        description=str(data.get("description", "")),
    )


def load_provision_floors(
    yaml_path: Path | None = None,
) -> list[ProvisionFloor]:
    """Load provision floors from a YAML file.

    Args:
        yaml_path: Path to YAML file.  Defaults to the bundled
            ``regulatory/provision_floors.yml``.

    Returns:
        List of :class:`ProvisionFloor` objects.
    """
    path = yaml_path or _DEFAULT_YAML
    if not path.exists():
        logger.warning(
            "Provision floors YAML not found at %s; returning empty list",
            path,
        )
        return []

    with open(path) as f:
        raw = yaml.safe_load(f)

    floors_data: list[dict[str, Any]] = raw.get("floors", [])
    result: list[ProvisionFloor] = []
    for entry in floors_data:
        floor = _parse_floor(entry)
        if floor is not None:
            result.append(floor)
    return result


# Module-level singleton loaded from bundled YAML
_PROVISION_FLOORS: list[ProvisionFloor] = load_provision_floors()


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
        if (
            stage is not None
            and floor.stage is not None
            and floor.stage != stage
        ):
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
