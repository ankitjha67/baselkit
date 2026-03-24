# Stress Testing

Multi-framework stress testing: EBA, BoE ACS, US CCAR/DFAST, and RBI.

## EBA Stress Test

```python
from creditriskengine.portfolio.stress_testing import EBAStressTest, MacroScenario
import numpy as np

scenario = MacroScenario(
    name="Adverse",
    horizon_years=3,
    variables={
        "gdp_growth": np.array([-0.04, -0.02, 0.01]),
        "house_price_index": np.array([-0.15, -0.10, -0.03]),
    },
)
eba = EBAStressTest(scenario)
result = eba.run(base_pds, base_lgds, base_eads)
```

## BoE ACS

```python
from creditriskengine.portfolio.stress_testing import BoEACSStressTest

boe = BoEACSStressTest(scenario, horizon_years=5)
result = boe.run(base_pds, base_lgds, base_eads, initial_cet1_ratio=0.12)
print(f"CET1 breach: {result['cet1_hurdle_breach']}")
```

::: creditriskengine.portfolio.stress_testing
