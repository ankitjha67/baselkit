"""
Exposure and facility data models.

Data models represent the minimum set of attributes required for
regulatory capital calculation under both SA and IRB approaches.
"""

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from creditriskengine.core.types import (
    CollateralType,
    CreditQualityStep,
    CreditRiskApproach,
    IFRS9Stage,
    IRBAssetClass,
    IRBCorporateSubClass,
    IRBRetailSubClass,
    Jurisdiction,
    SAExposureClass,
)


class Collateral(BaseModel):
    """Collateral pledged against an exposure."""
    collateral_type: CollateralType
    value: float = Field(ge=0, description="Current market/appraised value")
    currency: str = Field(default="USD", max_length=3)
    haircut: Optional[float] = Field(
        default=None, ge=0, le=1, description="Supervisory or own-estimate haircut"
    )
    ltv: Optional[float] = Field(default=None, ge=0, description="Loan-to-value ratio at origination")


class Exposure(BaseModel):
    """
    Single credit exposure / facility for capital calculation.

    Contains all fields needed for SA, F-IRB, and A-IRB RWA
    computation, IFRS 9/CECL ECL calculation, and model validation.
    """
    # ---- Identifiers ----
    exposure_id: str
    counterparty_id: str

    # ---- Amounts ----
    ead: float = Field(ge=0, description="Exposure at Default amount")
    drawn_amount: float = Field(ge=0, description="Current drawn/outstanding balance")
    undrawn_commitment: float = Field(default=0, ge=0, description="Undrawn committed amount")

    # ---- Classification ----
    jurisdiction: Jurisdiction
    approach: CreditRiskApproach
    sa_exposure_class: Optional[SAExposureClass] = None
    irb_asset_class: Optional[IRBAssetClass] = None
    irb_corporate_subclass: Optional[IRBCorporateSubClass] = None
    irb_retail_subclass: Optional[IRBRetailSubClass] = None

    # ---- SA Parameters ----
    credit_quality_step: Optional[CreditQualityStep] = None
    is_investment_grade: Optional[bool] = None

    # ---- IRB Parameters ----
    pd: Optional[float] = Field(default=None, ge=0, le=1, description="Probability of Default")
    lgd: Optional[float] = Field(default=None, ge=0, le=1, description="Loss Given Default")
    ead_model: Optional[float] = Field(default=None, ge=0, description="EAD from internal model")
    maturity_years: Optional[float] = Field(
        default=None, ge=0, le=30, description="Effective residual maturity M in years"
    )
    turnover_eur_millions: Optional[float] = Field(
        default=None, ge=0, description="Annual turnover for SME firm-size adjustment"
    )

    # ---- Collateral and CRM ----
    collaterals: list[Collateral] = Field(default_factory=list)
    is_guaranteed: bool = False
    guarantor_risk_weight: Optional[float] = None

    # ---- Real Estate ----
    property_value: Optional[float] = Field(default=None, ge=0)
    ltv_ratio: Optional[float] = Field(default=None, ge=0)
    is_income_producing: bool = False
    is_materially_dependent_on_cashflows: bool = False

    # ---- IFRS 9 / ECL ----
    ifrs9_stage: Optional[IFRS9Stage] = None
    days_past_due: int = Field(default=0, ge=0)
    origination_date: Optional[date] = None
    maturity_date: Optional[date] = None
    origination_pd: Optional[float] = Field(
        default=None, ge=0, le=1, description="12-month PD at origination"
    )
    current_pd: Optional[float] = Field(default=None, ge=0, le=1, description="Current 12-month PD")
    is_credit_impaired: bool = False
    effective_interest_rate: Optional[float] = Field(
        default=None, ge=0, description="EIR for ECL discounting"
    )

    # ---- Flags ----
    is_defaulted: bool = False
    is_in_default_workout: bool = False
    currency: str = Field(default="USD", max_length=3)

    @field_validator("pd")
    @classmethod
    def pd_floor_check(cls, v: Optional[float]) -> Optional[float]:
        """Basel III PD floor is 0.03% (3 bps) for non-defaulted per CRE32.13."""
        return v
