"""
Advanced IRB (A-IRB) calculator — BCBS d424, CRE31-32.

Under A-IRB, banks provide their own estimates of PD, LGD, and EAD,
subject to regulatory floors. Effective maturity is bank-estimated,
floored at 1 year and capped at 5 years (CRE32.44-46).

Key regulatory references:
    - LGD floors: CRE32.23-32.25
    - EAD / CCF floor: CRE32.33 (50% for off-balance-sheet)
    - PD floor: CRE32.13 (0.03%)
    - Maturity: CRE32.44-46 (1-5 years)
"""

import logging

from creditriskengine.core.exposure import Exposure
from creditriskengine.core.types import (
    CollateralType,
    CreditRiskApproach,
    IRBAssetClass,
    IRBRetailSubClass,
)
from creditriskengine.rwa.base import BaseRWACalculator, RWAResult
from creditriskengine.rwa.irb.correlation import get_asset_correlation
from creditriskengine.rwa.irb.formulas import (
    PD_FLOOR,
    irb_capital_requirement_k,
    maturity_adjustment,
)
from creditriskengine.rwa.irb.maturity import (
    effective_maturity_airb,
    needs_maturity_adjustment,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AdvancedIRBCalculator",
    "apply_ccf_floor",
    "apply_lgd_floor",
]

# ============================================================
# LGD FLOORS — BCBS CRE32.23-32.25
# ============================================================

# Corporate LGD floors by collateral type (CRE32.23)
_LGD_FLOOR_CORPORATE_UNSECURED: float = 0.25
_LGD_FLOOR_CORPORATE_SECURED: dict[CollateralType, float] = {
    CollateralType.CASH: 0.0,
    CollateralType.GOLD: 0.0,
    CollateralType.DEBT_SECURITIES: 0.0,
    CollateralType.EQUITIES: 0.0,
    CollateralType.MUTUAL_FUNDS: 0.0,
    CollateralType.RECEIVABLES: 0.10,
    CollateralType.RESIDENTIAL_REAL_ESTATE: 0.10,
    CollateralType.COMMERCIAL_REAL_ESTATE: 0.10,
    CollateralType.OTHER_PHYSICAL: 0.15,
}

# Retail LGD floors (CRE32.25)
_LGD_FLOOR_RETAIL_UNSECURED: float = 0.25
_LGD_FLOOR_RETAIL_QRRE: float = 0.50
_LGD_FLOOR_RETAIL_MORTGAGE: float = 0.05

# CCF floor for off-balance-sheet exposures (CRE32.33)
_CCF_FLOOR: float = 0.50


def apply_lgd_floor(
    lgd: float,
    asset_class: str,
    retail_subclass: IRBRetailSubClass | None = None,
    collateral_type: CollateralType | None = None,
) -> float:
    """Apply the regulatory LGD floor per BCBS CRE32.23-32.25.

    Floors ensure that bank-estimated LGDs do not fall below minimum
    thresholds set by the Committee.

    Corporate / Sovereign / Bank (CRE32.23):
        - Unsecured: 25%
        - Secured by financial collateral: 0%
        - Secured by receivables: 10%
        - Secured by RRE or CRE: 10%
        - Secured by other physical: 15%

    Retail (CRE32.25):
        - Unsecured: 25%
        - QRRE: 50%
        - Residential mortgage: 5%

    Args:
        lgd: Bank-estimated LGD as a fraction (e.g. 0.30 for 30%).
        asset_class: IRB asset class string (e.g. ``'corporate'``).
        retail_subclass: Retail sub-class, required if *asset_class*
            maps to retail.
        collateral_type: Type of primary collateral, if any.

    Returns:
        LGD floored at the applicable regulatory minimum.
    """
    floor: float

    if asset_class in ("corporate", "sovereign", "bank"):
        if collateral_type is not None:
            floor = _LGD_FLOOR_CORPORATE_SECURED.get(
                collateral_type, _LGD_FLOOR_CORPORATE_UNSECURED
            )
        else:
            floor = _LGD_FLOOR_CORPORATE_UNSECURED
    elif asset_class in ("residential_mortgage",) or (
        retail_subclass == IRBRetailSubClass.RESIDENTIAL_MORTGAGE
    ):
        floor = _LGD_FLOOR_RETAIL_MORTGAGE
    elif asset_class == "qrre" or retail_subclass == IRBRetailSubClass.QRRE:
        floor = _LGD_FLOOR_RETAIL_QRRE
    elif asset_class == "other_retail" or retail_subclass in (
        IRBRetailSubClass.OTHER_RETAIL,
        IRBRetailSubClass.SME_RETAIL,
    ):
        floor = _LGD_FLOOR_RETAIL_UNSECURED
    else:
        # Conservative fallback
        floor = _LGD_FLOOR_CORPORATE_UNSECURED

    floored_lgd = max(lgd, floor)
    if floored_lgd > lgd:
        logger.debug(
            "LGD floor applied: bank_lgd=%.4f -> floored=%.4f "
            "(asset_class=%s, collateral=%s)",
            lgd,
            floored_lgd,
            asset_class,
            collateral_type,
        )
    return floored_lgd


def apply_ccf_floor(ccf: float) -> float:
    """Apply the 50% CCF floor for off-balance-sheet items (CRE32.33).

    Under A-IRB, bank-estimated CCFs are subject to a floor of 50% for
    undrawn commitments, except for unconditionally cancellable facilities.

    Args:
        ccf: Bank-estimated CCF as a fraction (e.g. 0.30 for 30%).

    Returns:
        CCF floored at 50%.
    """
    floored = max(ccf, _CCF_FLOOR)
    if floored > ccf:
        logger.debug("CCF floor applied: bank_ccf=%.4f -> floored=%.4f", ccf, floored)
    return floored


class AdvancedIRBCalculator(BaseRWACalculator):
    """Advanced IRB RWA calculator.

    Implements the A-IRB approach where:
    - PD: bank-estimated, floored at 0.03% (CRE32.13)
    - LGD: bank-estimated, subject to floors (CRE32.23-25)
    - EAD: bank-estimated, CCF floored at 50% (CRE32.33)
    - Maturity: bank-estimated, 1y floor / 5y cap (CRE32.44-46)
    """

    def calculate(self, exposure: Exposure) -> RWAResult:
        """Calculate A-IRB RWA for a single exposure.

        Pipeline:
            1. Validate required inputs (PD, LGD)
            2. Floor PD at 0.03% (CRE32.13)
            3. Apply LGD floor (CRE32.23-25)
            4. Compute EAD with CCF floor (CRE32.33)
            5. Determine asset correlation (CRE31.5-31.10)
            6. Compute capital requirement K (CRE31.4)
            7. Apply maturity adjustment if wholesale (CRE31.7)
            8. RW = K * 12.5 * 100
            9. RWA = EAD * RW / 100

        Args:
            exposure: Fully populated :class:`Exposure` instance with
                bank-estimated PD, LGD, and optionally EAD/CCF.

        Returns:
            :class:`RWAResult` with A-IRB risk weight and capital.

        Raises:
            ValueError: If required risk parameters are missing.
        """
        self._validate_inputs(exposure)

        # Step 1-2: PD floor
        assert exposure.pd is not None, "PD must not be None"
        pd = max(exposure.pd, PD_FLOOR)

        # Defaulted exposures
        if exposure.is_defaulted or pd >= 1.0:
            return self._defaulted_result(exposure)

        # Determine asset class string for correlation routing
        asset_class = self._resolve_asset_class(exposure)

        # Step 3: LGD floor
        primary_collateral = (
            exposure.collaterals[0].collateral_type
            if exposure.collaterals
            else None
        )
        assert exposure.lgd is not None, "LGD must not be None"
        lgd = apply_lgd_floor(
            lgd=exposure.lgd,
            asset_class=asset_class,
            retail_subclass=exposure.irb_retail_subclass,
            collateral_type=primary_collateral,
        )

        # Step 4: EAD with CCF floor for off-balance-sheet
        ead = self._compute_ead(exposure)

        # Step 5: Asset correlation
        r = get_asset_correlation(
            asset_class, pd, exposure.turnover_eur_millions
        )

        # Step 6: Capital requirement K
        k = irb_capital_requirement_k(pd, lgd, r)

        # Step 7: Maturity adjustment
        m: float = 2.5  # default
        ma: float = 1.0
        if needs_maturity_adjustment(asset_class):
            m = effective_maturity_airb(
                exposure.maturity_years if exposure.maturity_years is not None else 2.5
            )
            ma = maturity_adjustment(pd, m)
            k *= ma

        # Step 8-9: Risk weight and RWA
        rw_decimal = k * 12.5
        rw_pct = rw_decimal * 100.0
        rwa = ead * rw_decimal
        capital_req = rwa * 0.08

        logger.debug(
            "A-IRB: exp=%s pd=%.4f lgd=%.4f R=%.4f K=%.6f MA=%.4f "
            "RW=%.2f%% EAD=%.2f RWA=%.2f",
            exposure.exposure_id,
            pd,
            lgd,
            r,
            k,
            ma,
            rw_pct,
            ead,
            rwa,
        )

        return RWAResult(
            exposure_id=exposure.exposure_id,
            risk_weight=rw_pct,
            rwa=rwa,
            ead=ead,
            capital_requirement=capital_req,
            approach=CreditRiskApproach.AIRB,
            asset_class=asset_class,
            details={
                "pd": pd,
                "lgd": lgd,
                "lgd_bank_estimate": exposure.lgd if exposure.lgd is not None else 0.0,
                "correlation": r,
                "k": k,
                "maturity": m,
                "maturity_adjustment": ma,
            },
        )

    @staticmethod
    def _validate_inputs(exposure: Exposure) -> None:
        """Validate that required A-IRB parameters are present."""
        if exposure.pd is None:
            raise ValueError(
                f"Exposure '{exposure.exposure_id}': A-IRB requires "
                f"bank-estimated PD."
            )
        if exposure.lgd is None:
            raise ValueError(
                f"Exposure '{exposure.exposure_id}': A-IRB requires "
                f"bank-estimated LGD."
            )

    @staticmethod
    def _resolve_asset_class(exposure: Exposure) -> str:
        """Map exposure to an IRB risk-weight asset class string."""
        if exposure.irb_asset_class == IRBAssetClass.CORPORATE:
            return "corporate"
        if exposure.irb_asset_class == IRBAssetClass.SOVEREIGN:
            return "sovereign"
        if exposure.irb_asset_class == IRBAssetClass.BANK:
            return "bank"
        if exposure.irb_asset_class == IRBAssetClass.RETAIL:
            if exposure.irb_retail_subclass == IRBRetailSubClass.RESIDENTIAL_MORTGAGE:
                return "residential_mortgage"
            if exposure.irb_retail_subclass == IRBRetailSubClass.QRRE:
                return "qrre"
            return "other_retail"
        raise ValueError(
            f"Cannot determine IRB asset class for exposure "
            f"'{exposure.exposure_id}': irb_asset_class="
            f"{exposure.irb_asset_class!r}"
        )

    def _compute_ead(self, exposure: Exposure) -> float:
        """Compute EAD with CCF floor for off-balance-sheet items.

        If the bank has provided an EAD model estimate, use it. Otherwise
        compute as drawn + CCF * undrawn, where the CCF is floored at 50%
        (CRE32.33) for off-balance-sheet exposures.

        Args:
            exposure: The exposure with amount and commitment data.

        Returns:
            Exposure at Default amount.
        """
        if exposure.ead_model is not None:
            return max(exposure.ead_model, 0.0)

        if exposure.undrawn_commitment > 0:
            # Apply 50% CCF floor (CRE32.33)
            ccf = apply_ccf_floor(0.0)  # bank CCF not on model -> use floor
            ead = exposure.drawn_amount + ccf * exposure.undrawn_commitment
        else:
            ead = exposure.drawn_amount

        return max(ead, 0.0)

    @staticmethod
    def _defaulted_result(exposure: Exposure) -> RWAResult:
        """Return zero-RW result for defaulted exposures.

        For defaulted exposures, K = max(0, LGD - EL_BE) which is handled
        by the bank's provisioning. Risk weight is set to 0%.
        """
        return RWAResult(
            exposure_id=exposure.exposure_id,
            risk_weight=0.0,
            rwa=0.0,
            ead=exposure.ead,
            capital_requirement=0.0,
            approach=CreditRiskApproach.AIRB,
            asset_class=(
                str(exposure.irb_asset_class) if exposure.irb_asset_class else None
            ),
            details={"pd": 1.0, "lgd": 0.0, "defaulted": 1.0},
        )
