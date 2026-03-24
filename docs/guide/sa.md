# Standardized Approach

The SA assigns risk weights based on external ratings (CQS) and
exposure characteristics (LTV for real estate).

## Basic Usage

```python
from creditriskengine.rwa.standardized.credit_risk_sa import assign_sa_risk_weight
from creditriskengine.core.types import SAExposureClass, CreditQualityStep, Jurisdiction

# Corporate with rating
rw = assign_sa_risk_weight(SAExposureClass.CORPORATE, CreditQualityStep.CQS_2)
print(f"A-rated corporate: {rw}%")  # 50%

# Residential mortgage by LTV
rw = assign_sa_risk_weight(SAExposureClass.RESIDENTIAL_MORTGAGE, ltv=0.75)
print(f"RRE at 75% LTV: {rw}%")  # 35%

# UK PRA loan-splitting
from creditriskengine.rwa.standardized.credit_risk_sa import uk_pra_loan_splitting_rre
result = uk_pra_loan_splitting_rre(loan_amount=200_000, property_value=300_000)
print(f"Blended RW: {result['blended_rw']:.1f}%")
```

## SCRA for Banks

In jurisdictions not using external ratings:

```python
rw = assign_sa_risk_weight(SAExposureClass.BANK, scra_grade="A")  # 40%
rw = assign_sa_risk_weight(SAExposureClass.BANK, scra_grade="B")  # 75%
rw = assign_sa_risk_weight(SAExposureClass.BANK, scra_grade="C")  # 150%
```

::: creditriskengine.rwa.standardized.credit_risk_sa
