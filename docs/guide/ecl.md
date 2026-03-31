# ECL Calculation

Expected Credit Loss engines for IFRS 9, US CECL, and Ind AS 109 --
including revolving credit ECL with behavioral life, CCF models, and
drawn/undrawn decomposition.

## IFRS 9 ECL

```python
from creditriskengine.ecl.ifrs9.ecl_calc import ecl_12_month, ecl_lifetime

# Stage 1: 12-month ECL
ecl_1 = ecl_12_month(pd_12m=0.02, lgd=0.40, ead=1_000_000, eir=0.05)

# Stage 2: Lifetime ECL
ecl_2 = ecl_lifetime(
    pd_term_structure=[0.02, 0.025, 0.03, 0.035, 0.04],
    lgd=0.40,
    ead=1_000_000,
    eir=0.05,
)
```

## Multi-Scenario Weighting

```python
from creditriskengine.ecl.ifrs9.scenarios import weighted_ecl, Scenario

scenarios = [
    Scenario("base", 0.50, ecl_base),
    Scenario("downside", 0.30, ecl_down),
    Scenario("severe", 0.20, ecl_severe),
]
final_ecl = weighted_ecl(scenarios)
```

## Revolving Credit ECL

For revolving facilities (credit cards, overdrafts, HELOCs, corporate
revolvers), ECL must be measured over the behavioral life -- not the
contractual cancellation period -- per IFRS 9 paragraph 5.5.20.

The module produces a drawn/undrawn split per IFRS 7 B8E:
the drawn ECL is a loss allowance (contra-asset), the undrawn ECL
is a provision (liability).

```python
from creditriskengine.ecl.ifrs9.revolving import (
    calculate_revolving_ecl, determine_behavioral_life,
    RevolvingProductType,
)
from creditriskengine.core.types import IFRS9Stage
import numpy as np

# Determine behavioral life (IFRS 9 B5.5.40)
life = determine_behavioral_life(
    product_type=RevolvingProductType.CREDIT_CARD
)  # 36 months

# Monthly marginal PDs over the behavioral life
marginal_pds = np.full(life, 0.0025)  # ~3% annual PD

result = calculate_revolving_ecl(
    stage=IFRS9Stage.STAGE_2,
    drawn=6000.0, undrawn=4000.0, ccf=0.80,
    pd_12m=0.03, lgd=0.85, eir=0.015,
    marginal_pds=marginal_pds,
    behavioral_life_months=life,
)
print(f"Total ECL:  ${result.total_ecl:,.2f}")
print(f"  Drawn:    ${result.ecl_drawn:,.2f} (loss allowance)")
print(f"  Undrawn:  ${result.ecl_undrawn:,.2f} (provision)")
```

### Using Exposure objects

```python
from creditriskengine.core.exposure import Exposure
from creditriskengine.ecl.ifrs9.revolving import revolving_ecl_from_exposure

exposure = Exposure(
    exposure_id="CC-001", counterparty_id="CUST-001",
    ead=9200.0, drawn_amount=6000.0, undrawn_commitment=4000.0,
    jurisdiction="bcbs", approach="standardized",
    ifrs9_stage=IFRS9Stage.STAGE_1,
    current_pd=0.03, lgd=0.85, effective_interest_rate=0.18,
    is_revolving=True, credit_limit=10000.0,
    behavioral_life_months=36, ccf=0.80,
)
result = revolving_ecl_from_exposure(exposure)
```

### Multi-jurisdiction provision floors

```python
from creditriskengine.ecl.ifrs9.revolving import apply_provision_floor
from creditriskengine.core.types import Jurisdiction, IFRS9Stage

# RBI Stage 2 floor: 5% of EAD for unsecured retail
floored_ecl = apply_provision_floor(
    ecl=200.0, ead=10000.0,
    jurisdiction=Jurisdiction.INDIA, stage=IFRS9Stage.STAGE_2,
)  # Returns 500.0 (5% floor binds)
```

::: creditriskengine.ecl.ifrs9.ecl_calc
