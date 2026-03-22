"""
Foundation IRB (F-IRB) calculator — BCBS d424, CRE31-32.

Under F-IRB, banks provide their own PD estimates but use supervisory
values for LGD and CCF (credit conversion factor). The effective
maturity is fixed at 2.5 years (CRE32.47).

Key regulatory references:
    - Supervisory LGD: CRE32.17-32.22
    - Supervisory CCF: CRE32.28-32.32
    - PD floor: CRE32.13 (0.03%)
    - Fixed maturity: CRE32.47 (2.5 years)
"""

import logging

from creditriskengine.core.exposure import Exposure
from creditriskengine.core.types import (
    CollateralType,
    CreditRiskApproach,
    IRBAssetClass,
)
from creditriskengine.rwa.base import BaseRWACalculator, RWAResult
from creditriskengine.rwa.irb.correlation import get_asset_correlation
from creditriskengine.rwa.irb.formulas import (
    PD_FLOOR,
    irb_capital_requirement_k,
    maturity_adjustment,
)
from creditriskengine.rwa.irb.maturity import effective_maturity_firb, needs_maturity_adjustment

logger = logging.getLogger(__name__)

__all__ = [
    "FoundationIRBCalculator",
    "get_supervisory_ccf",
    "get_supervisory_lgd",
]

# ============================================================
# SUPERVISORY LGD VALUES — BCBS CRE32.17-32.22
# ============================================================

# Base supervisory LGD for unsecured exposures (CRE32.17)
_LGD_SENIOR_UNSECURED: float = 0.45
_LGD_SUBORDINATED: float = 0.75

# LGD for secured portions, by collateral type (CRE32.18-32.22)
_LGD_SECURED: dict[CollateralType, float] = {
    CollateralType.CASH: 0.0,
    CollateralType.GOLD: 0.0,
    CollateralType.RECEIVABLES: 0.35,
    CollateralType.RESIDENTIAL_REAL_ESTATE: 0.35,
    CollateralType.COMMERCIAL_REAL_ESTATE: 0.35,
    CollateralType.OTHER_PHYSICAL: 0.40,
    CollateralType.DEBT_SECURITIES: 0.0,
    CollateralType.EQUITIES: 0.0,
    CollateralType.NETTING: 0.45,
    CollateralType.GUARANTEE: 0.45,
    CollateralType.CREDIT_DERIVATIVE: 0.45,
}

# ============================================================
# SUPERVISORY CCF VALUES — BCBS CRE32.28-32.32
# ============================================================

# CCF for off-balance-sheet items (CRE32.28-32.32)
# Mapped by broad category; full granularity depends on product type.
_CCF_COMMITMENTS_GT_1Y: float = 0.75  # CRE32.29: > 1 year maturity
_CCF_COMMITMENTS_LE_1Y: float = 0.75  # CRE32.29: <= 1 year maturity
_CCF_NIF_RUF: float = 0.75  # CRE32.30: note issuance / revolving underwriting
_CCF_TRADE_RELATED: float = 0.20  # CRE32.31: trade-related contingencies
_CCF_TRANSACTION_RELATED: float = 0.50  # CRE32.31: transaction-related
_CCF_DIRECT_SUBSTITUTES: float = 1.00  # CRE32.28: direct credit substitutes
_CCF_UNCONDITIONALLY_CANCELLABLE: float = 0.40  # CRE32.32


def get_supervisory_lgd(
    exposure: Exposure,
    is_subordinated: bool = False,
) -> float:
    """Determine the supervisory LGD for an F-IRB exposure.

    Logic (BCBS CRE32.17-32.22):
        - Subordinated exposures: 75%
        - Senior unsecured (no eligible collateral): 45%
        - Secured: blended LGD based on collateral coverage

    For secured exposures with eligible collateral, the effective LGD is
    computed as a weighted blend of the secured LGD (by collateral type)
    and the unsecured LGD for the uncovered portion.

    Args:
        exposure: The exposure for which to compute supervisory LGD.
        is_subordinated: ``True`` if the claim is subordinated (CRE32.17).

    Returns:
        Supervisory LGD as a fraction (e.g. 0.45 for 45%).
    """
    if is_subordinated:
        logger.debug(
            "Exposure %s: subordinated -> LGD=%.2f",
            exposure.exposure_id,
            _LGD_SUBORDINATED,
        )
        return _LGD_SUBORDINATED

    # No collateral -> senior unsecured
    if not exposure.collaterals:
        return _LGD_SENIOR_UNSECURED

    # Blended LGD for partially secured exposures
    total_collateral_value = sum(c.value for c in exposure.collaterals)
    if total_collateral_value <= 0 or exposure.ead <= 0:
        return _LGD_SENIOR_UNSECURED

    # Weighted average secured LGD across collateral types
    weighted_secured_lgd = 0.0
    for collateral in exposure.collaterals:
        collateral_lgd = _LGD_SECURED.get(
            collateral.collateral_type, _LGD_SENIOR_UNSECURED
        )
        weighted_secured_lgd += collateral.value * collateral_lgd

    coverage_ratio = min(total_collateral_value / exposure.ead, 1.0)
    secured_portion_lgd = weighted_secured_lgd / total_collateral_value
    blended_lgd = (
        coverage_ratio * secured_portion_lgd
        + (1.0 - coverage_ratio) * _LGD_SENIOR_UNSECURED
    )

    logger.debug(
        "Exposure %s: blended LGD=%.4f (coverage=%.2f%%)",
        exposure.exposure_id,
        blended_lgd,
        coverage_ratio * 100.0,
    )
    return blended_lgd


def get_supervisory_ccf(
    exposure: Exposure,
    is_unconditionally_cancellable: bool = False,
    is_trade_related: bool = False,
    is_transaction_related: bool = False,
    is_direct_credit_substitute: bool = False,
) -> float:
    """Determine the supervisory CCF for off-balance-sheet items.

    The CCF converts undrawn commitments to an EAD-equivalent amount
    (BCBS CRE32.28-32.32).

    Args:
        exposure: The exposure with undrawn commitment information.
        is_unconditionally_cancellable: Commitment unconditionally
            cancellable without notice (CRE32.32).
        is_trade_related: Short-term self-liquidating trade letters
            of credit (CRE32.31).
        is_transaction_related: Transaction-related contingencies such
            as performance bonds (CRE32.31).
        is_direct_credit_substitute: Direct credit substitutes such as
            guarantees, standby LCs (CRE32.28).

    Returns:
        Supervisory CCF as a fraction (e.g. 0.75 for 75%).
    """
    if exposure.undrawn_commitment <= 0:
        return 0.0

    if is_direct_credit_substitute:
        return _CCF_DIRECT_SUBSTITUTES

    if is_unconditionally_cancellable:
        return _CCF_UNCONDITIONALLY_CANCELLABLE

    if is_trade_related:
        return _CCF_TRADE_RELATED

    if is_transaction_related:
        return _CCF_TRANSACTION_RELATED

    # Default: committed facilities (CRE32.29)
    return _CCF_COMMITMENTS_GT_1Y


class FoundationIRBCalculator(BaseRWACalculator):
    """Foundation IRB RWA calculator.

    Implements the F-IRB approach where:
    - PD: bank-estimated (floored at 0.03%, CRE32.13)
    - LGD: supervisory values (CRE32.17-32.22)
    - CCF: supervisory values (CRE32.28-32.32)
    - Maturity: fixed 2.5 years (CRE32.47)
    """

    def calculate(self, exposure: Exposure) -> RWAResult:
        """Calculate F-IRB RWA for a single exposure.

        Pipeline:
            1. Floor PD at 0.03% (CRE32.13)
            2. Determine supervisory LGD
            3. Compute EAD (drawn + CCF * undrawn)
            4. Determine asset correlation
            5. Compute capital requirement K (CRE31.4)
            6. Apply maturity adjustment if wholesale (CRE31.7)
            7. RW = K * 12.5 * 100 (percentage)
            8. RWA = EAD * RW / 100

        Args:
            exposure: Fully populated :class:`Exposure` instance.

        Returns:
            :class:`RWAResult` with F-IRB risk weight and capital.

        Raises:
            ValueError: If required fields (PD, asset class) are missing.
        """
        if exposure.pd is None:
            raise ValueError(
                f"Exposure '{exposure.exposure_id}' has no PD estimate; "
                f"F-IRB requires bank-estimated PD."
            )

        # Step 1: PD floor (CRE32.13)
        pd = max(exposure.pd, PD_FLOOR)

        # Defaulted exposures
        if exposure.is_defaulted or pd >= 1.0:
            return self._defaulted_result(exposure)

        # Step 2: Supervisory LGD
        lgd = get_supervisory_lgd(exposure)

        # Step 3: EAD = drawn + CCF * undrawn
        ccf = get_supervisory_ccf(exposure)
        ead = exposure.drawn_amount + ccf * exposure.undrawn_commitment
        ead = max(ead, 0.0)

        # Determine asset class string for correlation routing
        asset_class = self._resolve_asset_class(exposure)

        # Step 4: Asset correlation
        r = get_asset_correlation(
            asset_class, pd, exposure.turnover_eur_millions
        )

        # Step 5: Capital requirement K
        k = irb_capital_requirement_k(pd, lgd, r)

        # Step 6: Maturity adjustment (CRE31.7, CRE32.47)
        m = effective_maturity_firb()
        ma = 1.0
        if needs_maturity_adjustment(asset_class):
            ma = maturity_adjustment(pd, m)
            k *= ma

        # Step 7-8: Risk weight and RWA
        rw_decimal = k * 12.5
        rw_pct = rw_decimal * 100.0
        rwa = ead * rw_decimal
        capital_req = rwa * 0.08

        logger.debug(
            "F-IRB: exp=%s pd=%.4f lgd=%.2f R=%.4f K=%.6f MA=%.4f "
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
            approach=CreditRiskApproach.FIRB,
            asset_class=asset_class,
            details={
                "pd": pd,
                "lgd": lgd,
                "correlation": r,
                "k": k,
                "maturity": m,
                "maturity_adjustment": ma,
                "ccf": ccf,
            },
        )

    @staticmethod
    def _resolve_asset_class(exposure: Exposure) -> str:
        """Map exposure to an IRB risk-weight asset class string.

        Returns the string expected by :func:`get_asset_correlation` and
        :func:`irb_risk_weight`.
        """
        if exposure.irb_asset_class == IRBAssetClass.CORPORATE:
            return "corporate"
        if exposure.irb_asset_class == IRBAssetClass.SOVEREIGN:
            return "sovereign"
        if exposure.irb_asset_class == IRBAssetClass.BANK:
            return "bank"
        if exposure.irb_asset_class == IRBAssetClass.RETAIL:
            # Route to retail sub-class
            if exposure.irb_retail_subclass is not None:
                return exposure.irb_retail_subclass.value
            return "other_retail"
        raise ValueError(
            f"Cannot determine IRB asset class for exposure "
            f"'{exposure.exposure_id}': irb_asset_class="
            f"{exposure.irb_asset_class!r}"
        )

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
            approach=CreditRiskApproach.FIRB,
            asset_class=str(exposure.irb_asset_class) if exposure.irb_asset_class else None,
            details={"pd": 1.0, "lgd": 0.0, "defaulted": 1.0},
        )
