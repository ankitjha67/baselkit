# CreditRiskEngine

Production-grade open-source credit risk analytics library.

**The scikit-learn of credit risk.**

## Features

- **RWA Calculation** -- Basel III/IV Standardized Approach and IRB (F-IRB / A-IRB) with output floor phase-in
- **ECL Engines** -- IFRS 9, US CECL (ASC 326), and Ind AS 109 with staging, SICR, lifetime PD, and scenario weighting
- **PD / LGD / EAD Modeling** -- Scorecard development, calibration (anchor-point & Bayesian), TTC-to-PIT conversion, and term structures
- **Model Validation** -- Discrimination (AUROC, Gini, KS, IV), calibration (binomial, Hosmer-Lemeshow, traffic-light), stability (PSI, CSI, migration)
- **Portfolio Risk** -- Vasicek ASRF, Gaussian copula Monte Carlo, parametric VaR, economic capital, and stress testing
- **Concentration Risk** -- Single-name, sector-level, and granularity adjustment analytics
- **Multi-Jurisdiction** -- EU CRR3, UK PRA, US Basel III Endgame, India RBI, Singapore MAS, Hong Kong HKMA, Japan JFSA, Australia APRA, Canada OSFI, Saudi Arabia SAMA, and BCBS baseline
- **Regulatory Reporting** -- COREP credit risk summaries, Pillar 3 disclosures, and model inventory entries

## Installation

```bash
pip install creditriskengine
```

### From source

```bash
git clone https://github.com/ankitjha67/baselkit.git
cd baselkit
pip install -e ".[dev]"
```

Requires Python 3.11+.

## Quick Start

### IRB Risk Weight

```python
from creditriskengine.rwa.irb.formulas import irb_risk_weight

# Corporate exposure: PD=1%, LGD=45%, maturity=2.5y
rw = irb_risk_weight(pd=0.01, lgd=0.45, asset_class="corporate", maturity=2.5)
print(f"Risk Weight: {rw:.2f}%")
```

### Standardized Approach

```python
from creditriskengine.rwa.standardized.credit_risk_sa import assign_sa_risk_weight
from creditriskengine.core.types import SAExposureClass, CreditQualityStep, Jurisdiction

rw = assign_sa_risk_weight(
    exposure_class=SAExposureClass.CORPORATES,
    cqs=CreditQualityStep.CQS2,
    jurisdiction=Jurisdiction.EU,
)
print(f"SA Risk Weight: {rw:.0f}%")
```

### IFRS 9 ECL

```python
from creditriskengine.ecl.ifrs9.ecl_calc import calculate_ecl
from creditriskengine.core.types import IFRS9Stage

ecl = calculate_ecl(
    stage=IFRS9Stage.STAGE_1,
    pd_12m=0.02,
    lgd=0.40,
    ead=1_000_000,
    eir=0.05,
)
print(f"12-month ECL: {ecl:,.2f}")
```

### PD Scorecard

```python
from creditriskengine.models.pd.scorecard import score_to_pd, assign_rating_grade, build_master_scale
import numpy as np

scores = np.array([350, 500, 650, 800])
pds = score_to_pd(scores)

master_scale = build_master_scale(n_grades=10, min_pd=0.0003, max_pd=0.20)
grades = [assign_rating_grade(pd, master_scale) for pd in pds]
```

### Model Validation

```python
from creditriskengine.validation.discrimination import auroc, gini_coefficient, ks_statistic
import numpy as np

y_true = np.array([0, 0, 1, 0, 1, 1, 0, 0, 1, 0])
y_score = np.array([0.1, 0.2, 0.7, 0.3, 0.8, 0.6, 0.2, 0.15, 0.9, 0.05])

print(f"AUROC: {auroc(y_true, y_score):.4f}")
print(f"Gini:  {gini_coefficient(y_true, y_score):.4f}")
print(f"KS:    {ks_statistic(y_true, y_score):.4f}")
```

## Project Structure

```
src/creditriskengine/
    core/           # Exposure model, portfolio container, enums, config
    regulatory/     # Jurisdiction YAML configs and loader
    rwa/
        standardized/   # SA risk weight tables (CRE20)
        irb/            # IRB formulas (CRE31) -- correlation, K, RW
        output_floor.py # Output floor phase-in by jurisdiction
    ecl/
        ifrs9/      # Staging, SICR, lifetime PD, scenarios, TTC-to-PIT, ECL calc
        cecl/       # PD*LGD, loss-rate, vintage, DCF, qualitative factors
        ind_as109/  # India-specific wrapper over IFRS 9
    models/
        pd/             # Scorecard, calibration, master scale, Vasicek PD
        lgd/            # Workout LGD, downturn LGD, term structure, floors
        ead/            # CCF estimation, supervisory CCF, EAD term structure
        concentration/  # Single-name, sector, granularity adjustment
    portfolio/      # Copula simulation, VaR, economic capital, stress testing, Vasicek ASRF
    validation/     # Discrimination, calibration, stability, backtesting, benchmarking
    reporting/      # COREP, Pillar 3, model inventory
```

## Testing

```bash
# Run full test suite
pytest

# Quick run without coverage
pytest -q --no-cov

# Specific module
pytest tests/test_rwa/ -v
```

387 tests covering all modules. Type-checked with `mypy --strict` and linted with `ruff`.

## Dependencies

| Package | Purpose |
|---------|---------|
| NumPy | Numerical computation |
| SciPy | Statistical distributions and optimization |
| pandas | Data manipulation |
| Pydantic | Data validation and exposure model |
| PyYAML | Regulatory configuration loading |
| scikit-learn | AUC and model validation metrics |
| statsmodels | Statistical tests for calibration |

## License

Apache 2.0
