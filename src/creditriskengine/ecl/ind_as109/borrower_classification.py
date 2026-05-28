"""
RBI ECL Master Direction 2026 — Borrower-level Stage 3 classification.

Reference: RBI/DOR/2026-27/398 Paragraph 76 and Paragraph 8(9).

Stage 3 classification is applied at the borrower level (contagion):
if any one exposure to a borrower is classified as Stage 3, all
exposures to that borrower — including non-funded exposures — must
also be classified as Stage 3.

Stage 2 classification remains at the facility level (no contagion).
"""

from __future__ import annotations

import logging
from typing import Any

from creditriskengine.core.types import IFRS9Stage

logger = logging.getLogger(__name__)


def apply_borrower_level_staging(
    facilities: list[dict[str, Any]],
    borrower_id_key: str = "counterparty_id",
    stage_key: str = "stage",
) -> list[dict[str, Any]]:
    """Apply RBI borrower-level Stage 3 contagion rule.

    Per Paragraph 76: if any exposure to a borrower is classified as
    Stage 3, all exposures to that borrower are elevated to Stage 3,
    including non-funded exposures. Stage 2 classification remains at
    facility level.

    The input list is not mutated; new dicts are returned with the
    elevated stage where applicable.

    Args:
        facilities: List of facility dicts. Each dict must contain
            ``borrower_id_key`` and ``stage_key``.
        borrower_id_key: Key identifying the borrower (default
            ``"counterparty_id"``).
        stage_key: Key identifying the IFRS 9 stage (default ``"stage"``).

    Returns:
        New list of facility dicts with Stage 3 contagion applied.

    Reference:
        RBI/DOR/2026-27/398 Paragraph 76, Paragraph 8(9).
    """
    # Identify borrowers with any Stage 3 facility
    stage3_borrowers: set[Any] = set()
    for facility in facilities:
        if facility.get(stage_key) == IFRS9Stage.STAGE_3:
            borrower_id = facility.get(borrower_id_key)
            if borrower_id is not None:
                stage3_borrowers.add(borrower_id)

    # Build the output list with contagion applied
    result: list[dict[str, Any]] = []
    elevation_count = 0
    for facility in facilities:
        new_facility = dict(facility)
        borrower_id = facility.get(borrower_id_key)
        current_stage = facility.get(stage_key)
        if (
            borrower_id in stage3_borrowers
            and current_stage != IFRS9Stage.STAGE_3
        ):
            new_facility[stage_key] = IFRS9Stage.STAGE_3
            elevation_count += 1
        result.append(new_facility)

    if elevation_count > 0:
        logger.debug(
            "Borrower-level contagion elevated %d facilities to Stage 3 "
            "across %d borrower(s)",
            elevation_count, len(stage3_borrowers),
        )

    return result
