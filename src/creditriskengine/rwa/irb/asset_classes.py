"""
IRB asset class classification — BCBS d424, CRE30.4-30.15.

Provides functions for classifying exposures into the appropriate IRB
asset class and sub-class, and for identifying specialised lending
exposures that may require supervisory slotting.
"""

import logging

from creditriskengine.core.exposure import Exposure
from creditriskengine.core.types import (
    IRBAssetClass,
    IRBCorporateSubClass,
    IRBRetailSubClass,
    SAExposureClass,
)

logger = logging.getLogger(__name__)

__all__ = [
    "classify_irb_asset_class",
    "get_retail_subclass",
    "is_specialised_lending",
    "requires_slotting",
]

# SA -> IRB asset class inference mapping (best-effort)
_SA_TO_IRB_MAP: dict[SAExposureClass, IRBAssetClass] = {
    SAExposureClass.SOVEREIGN: IRBAssetClass.SOVEREIGN,
    SAExposureClass.PSE: IRBAssetClass.SOVEREIGN,
    SAExposureClass.MDB: IRBAssetClass.SOVEREIGN,
    SAExposureClass.BANK: IRBAssetClass.BANK,
    SAExposureClass.SECURITIES_FIRM: IRBAssetClass.BANK,
    SAExposureClass.CORPORATE: IRBAssetClass.CORPORATE,
    SAExposureClass.CORPORATE_SME: IRBAssetClass.CORPORATE,
    SAExposureClass.SUBORDINATED_DEBT: IRBAssetClass.CORPORATE,
    SAExposureClass.EQUITY: IRBAssetClass.EQUITY,
    SAExposureClass.RETAIL: IRBAssetClass.RETAIL,
    SAExposureClass.RETAIL_REGULATORY: IRBAssetClass.RETAIL,
    SAExposureClass.RESIDENTIAL_MORTGAGE: IRBAssetClass.RETAIL,
    SAExposureClass.COMMERCIAL_REAL_ESTATE: IRBAssetClass.CORPORATE,
    SAExposureClass.LAND_ADC: IRBAssetClass.CORPORATE,
}


def classify_irb_asset_class(exposure: Exposure) -> IRBAssetClass:
    """Classify an exposure into an IRB asset class per BCBS CRE30.4.

    The five IRB asset classes are (CRE30.4):
        1. Corporate
        2. Sovereign
        3. Bank
        4. Retail
        5. Equity

    If ``exposure.irb_asset_class`` is already set, it is returned directly.
    Otherwise, inference is attempted from ``exposure.sa_exposure_class``.

    Args:
        exposure: Fully populated :class:`Exposure` instance.

    Returns:
        The :class:`IRBAssetClass` for the exposure.

    Raises:
        ValueError: If the asset class cannot be determined.
    """
    # Prefer explicit IRB classification
    if exposure.irb_asset_class is not None:
        logger.debug(
            "Exposure %s: explicit IRB asset class = %s",
            exposure.exposure_id,
            exposure.irb_asset_class,
        )
        return exposure.irb_asset_class

    # Infer from SA exposure class
    if exposure.sa_exposure_class is not None:
        mapped = _SA_TO_IRB_MAP.get(exposure.sa_exposure_class)
        if mapped is not None:
            logger.debug(
                "Exposure %s: inferred IRB class %s from SA class %s",
                exposure.exposure_id,
                mapped,
                exposure.sa_exposure_class,
            )
            return mapped

    raise ValueError(
        f"Cannot determine IRB asset class for exposure "
        f"'{exposure.exposure_id}': irb_asset_class is None and "
        f"sa_exposure_class={exposure.sa_exposure_class!r} is unmapped."
    )


def is_specialised_lending(exposure: Exposure) -> bool:
    """Check if a corporate exposure is specialised lending per CRE30.7.

    Specialised lending comprises project finance, object finance,
    commodities finance, income-producing real estate, and high-volatility
    commercial real estate (BCBS CRE30.7).

    An exposure is classified as specialised lending if:
    - It has ``irb_corporate_subclass == SPECIALISED_LENDING``, OR
    - It is a corporate exposure backed by income-producing real estate
      where repayment is materially dependent on the property's cash flows
      (CRE30.7(4)).

    Args:
        exposure: Fully populated :class:`Exposure` instance.

    Returns:
        ``True`` if the exposure is specialised lending.
    """
    if exposure.irb_corporate_subclass == IRBCorporateSubClass.SPECIALISED_LENDING:
        return True

    # IPRE inference: corporate exposure materially dependent on property
    # cash flows (CRE30.7(4))
    if (
        exposure.irb_asset_class == IRBAssetClass.CORPORATE
        and exposure.is_income_producing
        and exposure.is_materially_dependent_on_cashflows
    ):
        logger.debug(
            "Exposure %s: inferred as specialised lending (IPRE) from "
            "income-producing + cash-flow-dependent flags",
            exposure.exposure_id,
        )
        return True

    return False


def get_retail_subclass(exposure: Exposure) -> IRBRetailSubClass:
    """Determine retail sub-class per BCBS CRE30.11-30.15.

    Sub-classes:
        - Residential mortgage (CRE30.11): secured by residential property.
        - QRRE (CRE30.12-30.13): revolving, unsecured, to individuals,
          with exposure <= EUR 100k.
        - SME retail (CRE30.14): to SMEs treated as retail (exposure < EUR 1M).
        - Other retail (CRE30.15): all remaining retail exposures.

    If ``exposure.irb_retail_subclass`` is set, it is returned directly.

    Args:
        exposure: Fully populated :class:`Exposure` instance.

    Returns:
        The :class:`IRBRetailSubClass` for the exposure.

    Raises:
        ValueError: If the exposure is not retail or sub-class cannot be
            determined.
    """
    # Prefer explicit classification
    if exposure.irb_retail_subclass is not None:
        return exposure.irb_retail_subclass

    # Inference from SA exposure class and exposure attributes
    if exposure.sa_exposure_class == SAExposureClass.RESIDENTIAL_MORTGAGE:
        return IRBRetailSubClass.RESIDENTIAL_MORTGAGE

    # SME retail: corporate SME treated as retail (CRE30.14)
    if (
        exposure.sa_exposure_class == SAExposureClass.CORPORATE_SME
        and exposure.ead < 1_000_000
    ):
        return IRBRetailSubClass.SME_RETAIL

    # Default to other retail
    logger.debug(
        "Exposure %s: defaulting retail sub-class to OTHER_RETAIL",
        exposure.exposure_id,
    )
    return IRBRetailSubClass.OTHER_RETAIL


def requires_slotting(exposure: Exposure) -> bool:
    """Check if the exposure must use supervisory slotting (CRE30.7, CRE34).

    Supervisory slotting applies to specialised lending exposures where
    the bank does not meet the requirements for estimating PD under the
    IRB approach. In practice this is indicated by:
    - The exposure being classified as specialised lending, AND
    - The bank not having estimated a PD (``exposure.pd is None``).

    Args:
        exposure: Fully populated :class:`Exposure` instance.

    Returns:
        ``True`` if the exposure must use supervisory slotting.
    """
    if not is_specialised_lending(exposure):
        return False

    # If the bank has not estimated PD, slotting is required
    if exposure.pd is None:
        logger.debug(
            "Exposure %s: requires supervisory slotting (no PD estimate)",
            exposure.exposure_id,
        )
        return True

    return False
