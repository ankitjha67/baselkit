# Getting Started

## Installation

```bash
pip install creditriskengine
```

For development:

```bash
pip install creditriskengine[dev]
```

## Quick Examples

### IRB Risk Weight

```python
from creditriskengine.rwa.irb.formulas import irb_risk_weight

# Corporate exposure: PD=1%, LGD=45%, M=2.5 years
rw = irb_risk_weight(pd=0.01, lgd=0.45, asset_class="corporate", maturity=2.5)
print(f"Risk Weight: {rw:.2f}%")
# Risk Weight: 72.40%
```

### SA Risk Weight

```python
from creditriskengine.rwa.standardized.credit_risk_sa import (
    get_sovereign_risk_weight,
    get_corporate_risk_weight,
)
from creditriskengine.core.types import CreditQualityStep

# AAA-rated sovereign
rw = get_sovereign_risk_weight(CreditQualityStep.CQS_1)
print(f"Sovereign RW: {rw}%")  # 0%

# BBB-rated corporate
rw = get_corporate_risk_weight(CreditQualityStep.CQS_3)
print(f"Corporate RW: {rw}%")  # 75%
```

### Output Floor

```python
from datetime import date
from creditriskengine.core.types import Jurisdiction
from creditriskengine.rwa.output_floor import OutputFloorCalculator

calc = OutputFloorCalculator(Jurisdiction.EU, date(2026, 6, 30))
result = calc.calculate(irb_rwa=800.0, sa_rwa=1200.0)
print(f"Floored RWA: {result['floored_rwa']}")
print(f"Floor binding: {result['is_binding']}")
```

### IFRS 9 ECL

```python
import numpy as np
from creditriskengine.core.types import IFRS9Stage
from creditriskengine.ecl.ifrs9.ecl_calc import calculate_ecl

# Stage 1: 12-month ECL
ecl = calculate_ecl(IFRS9Stage.STAGE_1, pd_12m=0.02, lgd=0.45, ead=1_000_000)
print(f"12-month ECL: {ecl:,.0f}")  # 9,000

# Stage 2: Lifetime ECL
marginal_pds = np.array([0.02, 0.025, 0.03, 0.035, 0.04])
ecl = calculate_ecl(
    IFRS9Stage.STAGE_2,
    pd_12m=0.02, lgd=0.45, ead=1_000_000,
    marginal_pds=marginal_pds,
)
print(f"Lifetime ECL: {ecl:,.0f}")
```

### Revolving Credit ECL (Credit Cards, Overdrafts, HELOCs)

```python
from creditriskengine.ecl.ifrs9.revolving import (
    calculate_revolving_ecl, determine_behavioral_life,
    regulatory_ccf_sa, RevolvingProductType,
)
from creditriskengine.core.types import IFRS9Stage
import numpy as np

# Credit card: $10K limit, $6K drawn, $4K undrawn
life = determine_behavioral_life(
    product_type=RevolvingProductType.CREDIT_CARD
)  # 36 months per B5.5.40
marginal_pds = np.full(life, 0.0025)  # ~3% annual PD

result = calculate_revolving_ecl(
    stage=IFRS9Stage.STAGE_2,
    drawn=6000, undrawn=4000, ccf=0.80,
    pd_12m=0.03, lgd=0.85, eir=0.015,
    marginal_pds=marginal_pds, behavioral_life_months=life,
)
print(f"Total ECL:       ${result.total_ecl:,.2f}")
print(f"  Drawn (allow): ${result.ecl_drawn:,.2f}")
print(f"  Undrawn (prov):${result.ecl_undrawn:,.2f}")
```

### PD Scorecard

```python
import numpy as np
from creditriskengine.models.pd.scorecard import (
    score_to_pd,
    build_master_scale,
    assign_rating_grade,
    calibrate_pd_anchor_point,
)

# Convert model scores to PDs
scores = np.array([-3.0, -2.0, -1.0, 0.0, 1.0])
pds = score_to_pd(scores)

# Build a master scale
boundaries = [0.0003, 0.001, 0.005, 0.01, 0.05, 0.10, 0.20, 1.0]
labels = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
scale = build_master_scale(boundaries, labels)

# Assign rating grade
grade = assign_rating_grade(0.015, scale)
print(f"Grade: {grade}")  # BB

# Calibrate PDs to a central tendency of 2%
calibrated = calibrate_pd_anchor_point(0.02, pds)
```

### Model Validation

```python
import numpy as np
from creditriskengine.validation.discrimination import auroc, gini_coefficient
from creditriskengine.validation.stability import population_stability_index

# Discrimination
y_true = np.array([0, 0, 0, 1, 1, 1])
y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
print(f"AUROC: {auroc(y_true, y_score):.3f}")
print(f"Gini:  {gini_coefficient(y_true, y_score):.3f}")

# Stability
rng = np.random.default_rng(42)
expected = rng.normal(0, 1, 1000)
actual = rng.normal(0.2, 1, 1000)
psi = population_stability_index(actual, expected)
print(f"PSI: {psi:.4f}")  # < 0.10 = stable
```

### Loading Jurisdiction Config

```python
from creditriskengine.core.types import Jurisdiction
from creditriskengine.regulatory.loader import load_config

config = load_config(Jurisdiction.EU)
print(config["regulator"])       # European Commission / EBA
print(config["output_floor"])    # Phase-in schedule details
```

## Running Tests

```bash
pytest                         # Run all tests with coverage
pytest tests/ -q --no-cov     # Quick run without coverage
pytest tests/test_rwa/ -v     # Run only RWA tests
```

## Linting and Type Checking

```bash
ruff check src/creditriskengine/
mypy src/creditriskengine/ --ignore-missing-imports
```
