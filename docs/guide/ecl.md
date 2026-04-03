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

## Management Overlays (Post-Model Adjustments)

Management overlays let you apply expert-judgment adjustments on top of
modeled ECL, with full governance metadata for auditors and validators.

```python
from creditriskengine.ecl.ifrs9.overlays import (
    ManagementOverlay, OverlayType, apply_overlays,
    overlay_impact_summary, validate_overlay,
)
from datetime import datetime, UTC, timedelta

overlay = ManagementOverlay(
    name="Emerging market stress",
    overlay_type=OverlayType.EMERGING_RISK,
    adjustment_rate=0.10,  # +10% on model ECL
    rationale="Geopolitical risk not captured in PD models",
    regulatory_basis="IFRS 9.B5.5.52",
    approved_by="Credit Risk Committee",
    approval_date=datetime.now(UTC),
    expiry_date=datetime.now(UTC) + timedelta(days=90),
    portfolio_scope="EM sovereign and bank exposures",
)

# Check governance completeness
warnings = validate_overlay(overlay)  # [] = passes all checks

# Apply to model ECL
result = apply_overlays(model_ecl=5_000_000, overlays=[overlay])
summary = overlay_impact_summary(result)
print(f"Overlay impact: {summary['adjustment_pct']}%")
```

## Scenario Governance & Sensitivity

Track scenario weight approvals and analyse ECL sensitivity to weight
perturbations.

```python
from creditriskengine.ecl.ifrs9.scenarios import (
    Scenario, ScenarioSetMetadata,
    validate_scenario_governance, scenario_sensitivity_analysis,
)
from datetime import datetime, UTC, timedelta

scenarios = [
    Scenario("base", 0.50, 500_000),
    Scenario("downside", 0.30, 900_000),
    Scenario("severe", 0.20, 1_500_000),
]

meta = ScenarioSetMetadata(
    scenarios=scenarios,
    approved_by="Model Risk Committee",
    approval_date=datetime(2025, 1, 15, tzinfo=UTC),
    next_review_date=datetime(2025, 4, 15, tzinfo=UTC),
    methodology="Expert panel + GDP consensus forecasts",
    data_sources="IMF WEO Oct 2024, Bloomberg consensus",
)

# Validate governance
warnings = validate_scenario_governance(meta)

# Sensitivity: what happens if we shift each weight by +10pp?
sens = scenario_sensitivity_analysis(scenarios, shift_size=0.10)
print(f"Most sensitive: {sens.max_sensitivity_scenario} ({sens.max_sensitivity_pct:.1f}%)")
```

## Forward-Looking Information (Satellite Models)

Multi-variable satellite models with logistic link and mean-reversion.

```python
from creditriskengine.ecl.ifrs9.forward_looking import (
    SatelliteModelConfig, satellite_model_predict,
    apply_fli_with_reversion,
)
import numpy as np

# Configure satellite model: GDP and unemployment drive PD
config = SatelliteModelConfig(
    variable_names=["gdp_growth", "unemployment"],
    coefficients=[-2.0, 3.0],  # GDP decreases PD, unemployment increases it
    intercept=1.0,
    link="logistic",  # Bounded output in (0, 2)
)

# Forecast 5 years of macro variables
forecasts = {
    "gdp_growth": np.array([-0.02, 0.00, 0.01, 0.02, 0.02]),
    "unemployment": np.array([0.08, 0.07, 0.06, 0.05, 0.045]),
}

factors = satellite_model_predict(config, forecasts)

# Apply with mean-reversion beyond 3-year forecast horizon
base_pds = np.full(5, 0.02)
long_run_pd = 0.02
adjusted = apply_fli_with_reversion(
    base_pds, factors, long_run_pd,
    forecast_horizon=3, reversion_periods=2,
)
```

## Ind AS 109 with RBI IRAC Norms

Full RBI asset classification with provisioning floors.

```python
from creditriskengine.ecl.ind_as109 import (
    classify_irac, irac_to_ifrs9_stage, rbi_minimum_provision,
    calculate_ecl_ind_as, IRACAssetClass,
)
import numpy as np

# Classify per RBI IRAC norms
irac = classify_irac(days_past_due=95, months_as_npa=3)
# IRACAssetClass.SUBSTANDARD

stage = irac_to_ifrs9_stage(irac)  # IFRS9Stage.STAGE_3

# RBI minimum provision (15% secured substandard)
floor = rbi_minimum_provision(ead=1_000_000, irac_class=irac)
# 150,000

# ECL with RBI floor (higher of model ECL and IRAC minimum)
ecl = calculate_ecl_ind_as(
    stage=stage, pd_12m=0.10, lgd=0.45, ead=1_000_000,
    marginal_pds=np.array([0.10, 0.08]),
    irac_class=irac, is_secured=True,
)
```

::: creditriskengine.ecl.ifrs9.ecl_calc
