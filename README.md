# CreditRiskEngine

[![PyPI](https://img.shields.io/pypi/v/creditriskengine)](https://pypi.org/project/creditriskengine/)
[![Python](https://img.shields.io/pypi/pyversions/creditriskengine)](https://pypi.org/project/creditriskengine/)
[![CI](https://github.com/ankitjha67/baselkit/actions/workflows/ci.yml/badge.svg)](https://github.com/ankitjha67/baselkit/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)]()
[![License](https://img.shields.io/pypi/l/creditriskengine)](https://github.com/ankitjha67/baselkit/blob/main/LICENSE)
[![FINOS - Incubating](https://cdn.jsdelivr.net/gh/finos/contrib-toolbox@master/images/badge-incubating.svg)](https://community.finos.org/)

Production-grade open-source credit risk analytics library.

**The scikit-learn of credit risk.**

## Features

- **RWA Calculation** -- Basel III/IV Standardized Approach and IRB (F-IRB / A-IRB) with output floor phase-in, double default (CRE32), and equity IRB (CRE33)
- **ECL Engines** -- IFRS 9, US CECL (ASC 326), and Ind AS 109 with staging, SICR, lifetime PD, and scenario weighting
- **PD / LGD / EAD Modeling** -- Scorecard development, calibration (anchor-point & Bayesian), TTC-to-PIT conversion, term structures, Merton structural model, Altman Z-score, and transition matrix estimation
- **Model Validation** -- Discrimination (AUROC, Gini, KS, IV), calibration (binomial, Hosmer-Lemeshow, traffic-light), stability (PSI, CSI, migration)
- **Portfolio Risk** -- Vasicek ASRF, Gaussian copula Monte Carlo, parametric VaR, economic capital, and stress testing (including reverse stress)
- **Concentration Risk** -- Single-name, sector-level, and granularity adjustment analytics
- **Capital Adequacy** -- Capital buffers (CConB, CCyB, G-SIB/D-SIB), leverage ratio (CRE80), and MDA framework
- **CVA Risk** -- BA-CVA (CVA25) and SA-CVA delta risk charge (CVA26) with supervisory parameters
- **Market Risk** -- FRTB credit spread SbM (MAR21), Default Risk Charge (MAR22), and RRAO (MAR23)
- **Securitisation** -- SEC-SA, SEC-ERBA, and SEC-IRBA per CRE40-45
- **Operational Risk** -- Standardised Measurement Approach (SMA) per OPE25
- **Credit Risk Mitigation** -- Comprehensive and simple approaches, haircut framework per CRE22
- **Multi-Jurisdiction** -- EU CRR3, UK PRA, US Basel III Endgame, India RBI, Singapore MAS, Hong Kong HKMA, Japan JFSA, Australia APRA, Canada OSFI, Saudi Arabia SAMA, and BCBS baseline
- **Regulatory Reporting** -- COREP, Pillar 3 disclosure templates (CR1/CR3/CR4/CR6), FR Y-14 (CCAR), FR 2052a (Complex Institution Liquidity Monitoring), and model inventory
- **Stress Testing** -- EBA, BoE ACS, US CCAR/DFAST, RBI frameworks, and reverse stress testing

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
    exposure_class=SAExposureClass.CORPORATE,
    cqs=CreditQualityStep.CQS_2,
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
from creditriskengine.models.pd.scorecard import scorecard_to_pd, assign_rating_grade, build_master_scale
import numpy as np

scores = np.array([537, 587, 640, 706])
pds = scorecard_to_pd(scores)  # Convert scorecard points to PD

master_scale = build_master_scale([0.0003, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 1.0])
grades = [assign_rating_grade(pd, master_scale) for pd in pds]
# grades: ['Grade_7', 'Grade_5', 'Grade_2', 'Grade_1']
print(f"PDs: {pds}")
print(f"Grades: {grades}")
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

### FR 2052a Liquidity Report

```python
from creditriskengine.reporting.fr2052a import (
    InflowAssetRecord, OutflowDepositRecord,
    build_submission, validate_submission, generate_summary,
    AssetCategory, CounterpartyType, InsuredType,
    MaturityBucket, ReporterCategory,
)

# Build records for each schedule
records = [
    InflowAssetRecord(
        reporting_entity="MegaBank", as_of_date="2024-06-30",
        product_id=1,  # Unencumbered Assets
        maturity_bucket=MaturityBucket.OPEN,
        maturity_amount=5000.0,
        collateral_class=AssetCategory.A_1_Q,  # US Treasury
        market_value=5000.0, treasury_control=True,
    ),
    OutflowDepositRecord(
        reporting_entity="MegaBank", as_of_date="2024-06-30",
        product_id=1,  # Transactional Accounts
        maturity_bucket=MaturityBucket.OPEN,
        maturity_amount=3000.0,
        counterparty=CounterpartyType.RETAIL,
        insured=InsuredType.FDIC,
    ),
]

# Validate and generate summary
result = validate_submission(records, reporting_entity="MegaBank")
print(f"Valid: {result.is_valid}")

submission = build_submission(
    "MegaBank", "2024-06-30", ReporterCategory.CATEGORY_I, records
)
summary = generate_summary(submission)
print(f"Net liquidity: {summary['net_liquidity_position']:,.0f}M")
# Net liquidity: 2,000M
```

## Project Structure

```
src/creditriskengine/
    core/               # Exposure model, portfolio container, enums, audit trail, logging
    regulatory/         # Jurisdiction YAML configs (17 jurisdictions) and loader
    rwa/
        standardized/   # SA risk weight tables (CRE20)
        irb/            # IRB formulas (CRE31) -- correlation, K, RW
        output_floor.py # Output floor phase-in by jurisdiction
        capital_buffers.py # CConB, CCyB, G-SIB/D-SIB, MDA (RBC40)
        leverage_ratio.py # Basel III leverage ratio (CRE80)
        cva.py          # BA-CVA (CVA25) and SA-CVA (CVA26) capital charges
        market_risk.py  # FRTB SbM, DRC, RRAO (MAR21-23)
        securitisation.py # SEC-SA, SEC-ERBA, SEC-IRBA (CRE40-45)
        operational_risk.py # Standardised Measurement Approach (OPE25)
        crm.py          # Credit risk mitigation, haircuts (CRE22)
    ecl/
        ifrs9/          # Staging, SICR, lifetime PD, scenarios, TTC-to-PIT, ECL calc
        cecl/           # PD*LGD, loss-rate, vintage, DCF, qualitative factors
        ind_as109/      # India-specific wrapper over IFRS 9
    models/
        pd/             # Scorecard, calibration, master scale, Vasicek PD, Merton, Z-score, transition matrices
        lgd/            # Workout LGD, downturn LGD, term structure, floors, cure rate
        ead/            # CCF estimation, supervisory CCF, EAD term structure
        concentration/  # Single-name, sector, granularity adjustment
    portfolio/          # Copula simulation, VaR, economic capital, stress testing, Vasicek ASRF
    validation/         # Discrimination, calibration, stability, backtesting, benchmarking
    reporting/          # COREP, Pillar 3 (CR1/CR3/CR4/CR6), FR Y-14 (CCAR), model inventory
        fr2052a/        # FR 2052a Complex Institution Liquidity Monitoring (OMB 7100-0361)
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

1,786 tests across all modules with **100% line coverage**. Type-checked with `mypy --strict` and linted with `ruff`.

## Performance

Benchmarked on a single core (run `python benchmarks/bench_portfolio.py`):

| Operation | Throughput |
|-----------|-----------|
| IRB risk weight (single) | ~100k calc/sec |
| IRB portfolio (10k exposures) | ~10k exp/sec |
| SA risk weight (10k exposures) | ~100k exp/sec |
| IFRS 9 ECL (10k calculations) | ~100k calc/sec |
| Stress test (10k × 3yr) | < 0.01s |
| Monte Carlo (10k × 10k sims) | < 2s |

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| NumPy | ≥1.26, <3.0 | Numerical computation |
| SciPy | ≥1.12, <2.0 | Statistical distributions (norm CDF/PPF) |
| pandas | ≥2.2, <3.0 | Data manipulation and audit trail export |
| Pydantic | ≥2.6, <3.0 | Data validation and exposure model |
| PyYAML | ≥6.0.1, <7.0 | Regulatory configuration loading |
| scikit-learn | ≥1.4, <2.0 | AUC, logistic regression, model validation |
| statsmodels | ≥0.14, <1.0 | Statistical tests for calibration |
| Jinja2 | ≥3.1, <4.0 | Regulatory report templating |

## Documentation

Build and serve the docs locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

## Governance

- **[Regulatory Mapping](docs/regulatory_mapping.md)**: Every function traced to its Basel/IFRS paragraph
- **[Regulatory Disclaimers](docs/regulatory_disclaimers.md)**: Important caveats for production use
- **[Config Versioning](docs/regulatory_versioning.md)**: How regulatory config changes are managed
- **Audit Trail**: `AuditTrail` class records every calculation with inputs, outputs, timestamps, and regulatory references
- **Input Validation**: Schema validation for YAML configs and sanitization for exposure inputs

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines. Quick start:

```bash
git clone https://github.com/ankitjha67/baselkit.git
cd baselkit
pip install -e ".[dev]"
pytest                   # Run tests
ruff check src/ tests/   # Lint
mypy src/                # Type check
```

## License

Apache 2.0 -- see [LICENSE](LICENSE) for details.

> **Disclaimer**: This library is provided for educational and analytical
> purposes. It has not been reviewed or endorsed by any regulatory authority.
> See [Regulatory Disclaimers](docs/regulatory_disclaimers.md) for details.
