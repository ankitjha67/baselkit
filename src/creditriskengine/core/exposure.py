"""
Exposure and facility data models.

Data models represent the minimum set of attributes required for
regulatory capital calculation under both SA and IRB approaches.
"""

from datetime import date

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
    haircut: float | None = Field(
        default=None, ge=0, le=1, description="Supervisory or own-estimate haircut"
    )
    ltv: float | None = Field(default=None, ge=0, description="Loan-to-value ratio at origination")


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
    sa_exposure_class: SAExposureClass | None = None
    irb_asset_class: IRBAssetClass | None = None
    irb_corporate_subclass: IRBCorporateSubClass | None = None
    irb_retail_subclass: IRBRetailSubClass | None = None

    # ---- SA Parameters ----
    credit_quality_step: CreditQualityStep | None = None
    is_investment_grade: bool | None = None

    # ---- IRB Parameters ----
    pd: float | None = Field(default=None, ge=0, le=1, description="Probability of Default")
    lgd: float | None = Field(default=None, ge=0, le=1, description="Loss Given Default")
    ead_model: float | None = Field(default=None, ge=0, description="EAD from internal model")
    maturity_years: float | None = Field(
        default=None, ge=0, le=30, description="Effective residual maturity M in years"
    )
    turnover_eur_millions: float | None = Field(
        default=None, ge=0, description="Annual turnover for SME firm-size adjustment"
    )

    # ---- Collateral and CRM ----
    collaterals: list[Collateral] = Field(default_factory=list)
    is_guaranteed: bool = False
    guarantor_risk_weight: float | None = None

    # ---- Real Estate ----
    property_value: float | None = Field(default=None, ge=0)
    ltv_ratio: float | None = Field(default=None, ge=0)
    is_income_producing: bool = False
    is_materially_dependent_on_cashflows: bool = False

    # ---- IFRS 9 / ECL ----
    ifrs9_stage: IFRS9Stage | None = None
    days_past_due: int = Field(default=0, ge=0)
    origination_date: date | None = None
    maturity_date: date | None = None
    origination_pd: float | None = Field(
        default=None, ge=0, le=1, description="12-month PD at origination"
    )
    current_pd: float | None = Field(default=None, ge=0, le=1, description="Current 12-month PD")
    is_credit_impaired: bool = False
    effective_interest_rate: float | None = Field(
        default=None, ge=0, description="EIR for ECL discounting"
    )

    # ---- Flags ----
    is_defaulted: bool = False
    is_in_default_workout: bool = False
    currency: str = Field(default="USD", max_length=3)

    @field_validator("pd")
    @classmethod
    def pd_floor_check(cls, v: float | None) -> float | None:
        """Validate PD is between 0 and 1 (inclusive).

        Note: The Basel III PD floor of 0.03% (CRE32.13) is enforced at the
        IRB calculation level (``rwa.irb.formulas``), not at the data model
        level, because SA and ECL workflows accept PDs below the IRB floor.
        """
        if v is not None and not 0 <= v <= 1:
            msg = f"PD must be between 0 and 1, got {v}"
            raise ValueError(msg)
        return v
