# ECL Calculation

Expected Credit Loss engines for IFRS 9, US CECL, and Ind AS 109.

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

::: creditriskengine.ecl.ifrs9.ecl_calc
