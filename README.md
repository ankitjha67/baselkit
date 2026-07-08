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
- **ECL Engines** -- IFRS 9, US CECL (ASC 326), and Ind AS 109 with staging, SICR, lifetime PD, scenario weighting, management overlays (post-model adjustments), and revolving credit ECL (credit cards, overdrafts, HELOCs, corporate revolvers) with behavioral life, multi-approach CCF, drawn/undrawn split, and multi-jurisdiction provision floors
- **ECL Governance Layer** -- Management overlay framework (7 overlay types with approval/expiry/rationale tracking per EBA/GL/2020/06), scenario governance with sensitivity analysis (IFRS 9.B5.5.41-43), multi-variable satellite models with logistic/log link functions and mean-reversion (IFRS 9.B5.5.50), LGD macro overlays, CECL Q-factor governance with per-category caps (OCC 2020-49), and overlay audit trail with JSON export
- **PD / LGD / EAD Modeling** -- Scorecard development, calibration (anchor-point & Bayesian), TTC-to-PIT conversion, term structures, Merton structural model, Altman Z-score, transition matrix estimation, Cox proportional-hazards survival analysis (Kaplan-Meier, Nelson-Aalen, Weibull), Pluto-Tasche low-default-portfolio PD, CDS-implied PD (Q→P conversion), and behavioural scoring with early-warning triggers
- **Advanced LGD** -- Workout/downturn LGD, cure rate, recovery-curve modeling (Weibull/lognormal/gamma) with discounted workout LGD, and beta-distribution LGD with downturn quantiles
- **Retail Loss Forecasting** -- Delinquency-bucket Markov roll-rate matrices and multi-period charge-off projection
- **Model Validation** -- Discrimination (AUROC, Gini, KS, IV), calibration (binomial, Hosmer-Lemeshow, traffic-light), stability (PSI, CSI, migration)
- **Portfolio Risk** -- Vasicek ASRF, Gaussian copula Monte Carlo, parametric VaR, economic capital, and stress testing (including reverse stress)
- **Concentration Risk** -- Single-name and sector-level (HHI) analytics, plus the Martin-Wilde/Gordy granularity adjustment for the ASRF model (single-factor Vasicek, second-order idiosyncratic add-on at the 99.9% quantile)
- **Counterparty Credit Risk** -- Full SA-CCR EAD engine (CRE52): trade-level adjusted notionals with supervisory duration, supervisory deltas (incl. Black-Scholes option and CDO-tranche deltas), maturity factors, asset-class add-ons with the correct hedging-set aggregation (IR maturity buckets, FX pairs, single-factor systematic credit/equity/commodity), the PFE multiplier, and unmargined/margined replacement cost. Plus EPE/EEPE/PFE exposure profiles from simulated paths, netting-set aggregation, IMM EAD (alpha=1.4), and wrong-way risk (general alpha adjustment + specific WWR flagging) per CRE52/53
- **Risk-Based Pricing & Capital Allocation** -- RAROC, Economic Value Added (EVA), break-even spread, all-in risk-based loan rate, and portfolio capital allocation (marginal, Euler/VaR, Expected-Shortfall contributions per Tasche 2008)
- **Capital Adequacy** -- Capital buffers (CConB, CCyB, G-SIB/D-SIB), leverage ratio (CRE80), and MDA framework
- **TLAC (FSB)** -- Total Loss-Absorbing Capacity for G-SIBs: available TLAC net of buffer CET1, the higher-of 18%-RWA / 6.75%-leverage minimums (16%/6% conformance period), binding-constraint identification, and shortfall reporting
- **MREL (BRRD2 / SRMR2)** -- EU resolution requirement calibrated as Loss Absorption + Recapitalisation Amount (Pillar 1 + P2R + market-confidence charge) against both TREA and the leverage exposure (TEM), with the G-SII TLAC floors, binding-constraint identification, and shortfall reporting
- **Large Exposures (BCBS LEX)** -- Pre-risk-weight exposure-value measurement (on/off-balance with CCF, derivative EAD, SFTs, net of eligible CRM), connected-counterparty grouping, the 25%-of-Tier-1 limit (15% G-SIB-to-G-SIB), 10% reporting threshold, and a portfolio breach/headroom report
- **Liquidity Ratios (BCBS LCR / NSFR)** -- Liquidity Coverage Ratio with tiered HQLA haircuts and the Level 2 (40%) / Level 2B (15%) caps plus the 75% inflow cap, and the Net Stable Funding Ratio with the full ASF/RSF factor tables — each with compliance flags against the 100% minimum
- **CVA Risk** -- BA-CVA (CVA25) and SA-CVA delta risk charge (CVA26) with supervisory parameters
- **Market Risk** -- FRTB Standardised Approach: credit spread SbM (MAR21), Default Risk Charge (MAR22), and RRAO (MAR23)
- **FRTB Internal Models Approach** -- Expected Shortfall at 97.5% (MAR33.4), liquidity-horizon scaling (10/20/40/60/120 days), stressed-ES capital charge, P&L Attribution Test (Spearman + KS traffic light), full bucketed Default Risk Charge (MAR22: obligor JTD netting, default risk-weight table, book-wide hedge-benefit ratio, per-bucket aggregation) at 99.9%, and Non-Modellable Risk Factor stress charge
- **IRRBB** -- Economic Value of Equity sensitivity to the six BCBS d368 shock scenarios, Net Interest Income sensitivity, and the Supervisory Outlier Test (15% Tier 1 EVE / 2.5% NII per EBA RTS/2022/09, CRR3 Art. 84)
- **Operational Resilience** -- EU DORA ICT incident classification (Reg 2022/2554, RTS 2024/1772), impact tolerances for Important Business Services (BCBS d516, PRA SS1/21), and third-party (ICT provider) concentration via HHI
- **Securitisation** -- SEC-SA, SEC-ERBA, and SEC-IRBA per CRE40-45
- **Operational Risk** -- Standardised Measurement Approach (SMA) per OPE25
- **Credit Risk Mitigation** -- Comprehensive and simple approaches, haircut framework per CRE22
- **Settlement Risk (CRE70)** -- DvP failed-trade capital charge by business-days-unsettled multiplier (8/50/75/100%) and the non-DvP free-delivery treatment (counterparty RW, then 1250% at 5+ days)
- **Equity Investments in Funds (CRE60)** -- Look-Through, Mandate-Based, and Fall-Back (1250%) approaches with the fund average-risk-weight × leverage calculation, capped at 1250%
- **SFT Haircut Floors (CRE56)** -- Minimum collateral haircut floors by collateral type and residual maturity (corporate/securitised debt, main-index equity, other assets), single-transaction recognition test, and the netting-agreement portfolio floor test
- **Multi-Jurisdiction** -- EU CRR3, UK PRA, US Basel III Endgame, India RBI (full IRAC norms plus the **RBI ECL Master Direction 2026** (RBI/DOR/2026-27/398) — 20-category provisioning floor table, Stage 3 duration-dependent floors, PD 0.03% / LGD 65%-70%-30% backstops, borrower-level Stage 3 contagion, DCCO project finance provisioning, capital add-back phase-in, effective April 1, 2027), Singapore MAS, Hong Kong HKMA, Japan JFSA, Australia APRA, Canada OSFI, Saudi Arabia SAMA, and BCBS baseline
- **Emerging-Market Asset Classification** -- China NFRA five-tier risk classification (DPD + ECL-ratio triggers, CAS 22 staging, Feb 2023 Measures) and Indonesia OJK five-tier collectability with minimum provisioning (1/5/15/50/100% per POJK 40/2019, net of eligible collateral)
- **ESG Ratings Integration** -- Vendor-agnostic adapter normalising MSCI (AAA-CCC), Sustainalytics (0-40+ risk score), and S&P Global ESG ratings to a common [0,1] risk score, with a bounded PD multiplier overlay per EBA (2023) guidance
- **ESG Risk Management (EBA/GL/2025/01)** -- Likelihood × impact, exposure-weighted ESG materiality assessment with the EBA time-horizon → method mapping (short → exposure-based, medium → sector/portfolio, long → scenario-based), plus CRD Art. 76(2)/87a transition-plan monitoring against intermediate net-zero / GAR-uplift targets (applicable from 11 January 2026)
- **Regulatory Reporting** -- COREP, Pillar 3 disclosure templates (CR1/CR3/CR4/CR6), FR Y-14 (CCAR), FR 2052a (Complex Institution Liquidity Monitoring), and model inventory
- **Stress Testing** -- EBA, BoE ACS, US CCAR/DFAST, RBI frameworks, and reverse stress testing
- **Climate & ESG Risk** -- NGFS Phase V scenario library (6 scenarios), physical risk PD/LGD adjustments (flood, wildfire, drought, sea-level rise, storm, extreme heat), transition risk PD multiplier with sector-specific elasticities and CBAM flagging, PCAF financed emissions (attribution factor, data quality score 1-5), EU Green Asset Ratio (GAR/BTAR), and BCBS SCO60 crypto-asset capital (Group 1a/1b/2a/2b, 1250% RW, Tier 1 limit monitoring)
- **Climate Scenario Analysis (EBA/GL/2025/02)** -- Portfolio-level climate ECL projection under NGFS scenarios: per-exposure stressed PD (transition carbon-cost multiplier × physical-hazard multiplier) and stressed LGD (physical collateral haircut), with horizon-interpolated carbon prices and a transition-vs-physical decomposition of the aggregate ECL uplift
- **AI/ML Model Governance** -- Algorithmic fairness (disparate impact 4/5 rule, demographic parity, equal opportunity per EU AI Act / CFPB / MAS FEAT), drift detection (PSI with regulatory thresholds per SR 11-7 / PRA SS1/23), and model risk classification

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

### Revolving Credit ECL (Credit Card)

```python
from creditriskengine.ecl.ifrs9.revolving import (
    calculate_revolving_ecl, regulatory_ccf_sa,
    RevolvingProductType, determine_behavioral_life,
)
from creditriskengine.core.types import IFRS9Stage
import numpy as np

# Credit card: $10K limit, $6K drawn, $4K undrawn
life = determine_behavioral_life(product_type=RevolvingProductType.CREDIT_CARD)
ccf = 0.80  # Behavioral CCF (regulatory SA = 10%, but IFRS 9 uses PIT)
marginal_pds = np.full(life, 0.0025)  # ~3% annual PD

result = calculate_revolving_ecl(
    stage=IFRS9Stage.STAGE_2, drawn=6000, undrawn=4000, ccf=ccf,
    pd_12m=0.03, lgd=0.85, eir=0.015,
    marginal_pds=marginal_pds, behavioral_life_months=life,
)
print(f"Total ECL: ${result.total_ecl:,.2f}")
print(f"  Drawn (loss allowance): ${result.ecl_drawn:,.2f}")
print(f"  Undrawn (provision):    ${result.ecl_undrawn:,.2f}")
```

### ECL Governance: Management Overlays & Scenario Sensitivity

```python
from creditriskengine.ecl.ifrs9.overlays import (
    ManagementOverlay, OverlayType, apply_overlays, validate_overlay,
)
from creditriskengine.ecl.ifrs9.scenarios import (
    Scenario, ScenarioSetMetadata, weighted_ecl,
    scenario_sensitivity_analysis, validate_scenario_governance,
)
from datetime import datetime, UTC, timedelta

# --- Management Overlays (post-model adjustments) ---
overlay = ManagementOverlay(
    name="CRE sector stress",
    overlay_type=OverlayType.SECTOR_SPECIFIC,
    adjustment_rate=0.15,  # +15% uplift on model ECL
    rationale="Commercial real estate valuations declining in Q3",
    regulatory_basis="IFRS 9.B5.5.52",
    approved_by="Credit Risk Committee",
    approval_date=datetime.now(UTC),
    expiry_date=datetime.now(UTC) + timedelta(days=90),
    portfolio_scope="UK CRE Stage 2 exposures",
)

# Validate governance completeness (auditor-ready)
warnings = validate_overlay(overlay)  # [] = fully compliant

# Apply overlay to model ECL
result = apply_overlays(model_ecl=1_000_000, overlays=[overlay])
print(f"Model ECL: {result.model_ecl:,.0f}")
print(f"After overlay: {result.overlay_ecl:,.0f} (+{result.total_adjustment:,.0f})")

# --- Scenario Sensitivity Analysis ---
scenarios = [
    Scenario("base", 0.50, 500_000),
    Scenario("downside", 0.30, 900_000),
    Scenario("severe", 0.20, 1_500_000),
]
ecl = weighted_ecl(scenarios)  # Probability-weighted ECL

sensitivity = scenario_sensitivity_analysis(scenarios, shift_size=0.10)
print(f"Most sensitive to: {sensitivity.max_sensitivity_scenario}")
print(f"  ECL changes {sensitivity.max_sensitivity_pct:.1f}% per 10pp weight shift")
```

### Ind AS 109 with RBI IRAC Norms

```python
from creditriskengine.ecl.ind_as109 import (
    classify_irac, irac_to_ifrs9_stage, rbi_minimum_provision,
    calculate_ecl_ind_as, IRACAssetClass,
)
from creditriskengine.core.types import IFRS9Stage
import numpy as np

# Classify per RBI IRAC norms
irac = classify_irac(days_past_due=95, months_as_npa=3)
print(f"IRAC class: {irac.value}")  # "substandard"
stage = irac_to_ifrs9_stage(irac)   # Stage 3

# RBI minimum provision (15% for secured substandard)
rbi_floor = rbi_minimum_provision(ead=1_000_000, irac_class=irac, is_secured=True)
print(f"RBI floor: {rbi_floor:,.0f}")  # 150,000

# ECL with RBI provisioning floor (higher of model ECL and RBI floor)
ecl = calculate_ecl_ind_as(
    stage=stage, pd_12m=0.10, lgd=0.45, ead=1_000_000,
    marginal_pds=np.array([0.10, 0.08]),
    irac_class=irac, is_secured=True,
)
print(f"ECL (with RBI floor): {ecl:,.0f}")
```

### RBI ECL Master Direction 2026 (effective April 1, 2027)

```python
from datetime import date
from creditriskengine.ecl.ind_as109 import (
    RBIExposureCategory, calculate_ecl_ind_as_2026,
    apply_borrower_level_staging, assess_sicr_rbi,
    capital_add_back_factor, is_ecl_framework_effective,
    dcco_additional_provision,
)
from creditriskengine.core.types import IFRS9Stage
import numpy as np

# 1. ECL with all RBI 2026 floors applied
ecl = calculate_ecl_ind_as_2026(
    stage=IFRS9Stage.STAGE_2,
    pd_12m=0.05, lgd=0.30, ead=1_000_000,
    marginal_pds=np.array([0.05, 0.04, 0.03]),
    category=RBIExposureCategory.UNSECURED_RETAIL,
    is_secured=False,
)
# PD floored at 0.03%, LGD floored at 70% (unsecured), regulatory
# floor 5% of EAD = 50,000 binds if model ECL falls below.

# 2. Borrower-level Stage 3 contagion (Paragraph 76)
facilities = [
    {"counterparty_id": "B1", "facility_id": "F1", "stage": IFRS9Stage.STAGE_3},
    {"counterparty_id": "B1", "facility_id": "F2", "stage": IFRS9Stage.STAGE_1},
]
contagion = apply_borrower_level_staging(facilities)
# Both facilities now Stage 3

# 3. Revolving SICR backstop (60 days over limit, Paragraph 33)
sicr = assess_sicr_rbi(is_revolving=True, days_over_limit=75)
# True

# 4. DCCO additional provisioning for project finance (Paragraph 82(4))
extra = dcco_additional_provision(ead=10_000_000, quarters_of_deferment=4,
                                   is_infrastructure=True)
# 4 * 0.375% * 10M = 150,000

# 5. Transition phase-in: capital add-back factor (Paragraph 108)
add_back = capital_add_back_factor(reporting_fy=2028)
# 0.80 (4/5 add-back in FY 2027-28)

# 6. Effective-date dispatch
print(is_ecl_framework_effective(date(2027, 4, 1)))  # True
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
            revolving/  # Revolving credit ECL: behavioral life, CCF, drawn/undrawn split, provision floors
            overlays.py # Management overlay / PMA framework (7 types, governance, audit)
            forward_looking.py # Satellite models, mean-reversion, LGD macro overlay
            scenarios.py # Probability-weighted ECL, governance metadata, sensitivity analysis
        cecl/           # PD*LGD, loss-rate, vintage, DCF, Q-factors with governance caps
        ind_as109/      # Ind AS 109 with full RBI IRAC norms (SMA/NPA classification, provisioning)
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

2,797 tests across all modules with **100% line coverage**. Type-checked with `mypy --strict` and linted with `ruff`.

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

- **[Regulatory Mapping](docs/regulatory_mapping.md)**: Every function traced to its Basel/IFRS paragraph (185+ mappings)
- **[Regulatory Disclaimers](docs/regulatory_disclaimers.md)**: Important caveats for production use
- **[Config Versioning](docs/regulatory_versioning.md)**: How regulatory config changes are managed
- **Audit Trail**: `AuditTrail` class records every calculation with inputs, outputs, timestamps, and regulatory references. Overlay audit records track post-model adjustment lifecycle events (applied, reviewed, revoked, expired)
- **Management Overlay Governance**: Structured PMA framework with approval chain, expiry dates, rationale documentation, and governance validation per EBA/GL/2020/06 and PRA Dear CFO letter (Jul 2020)
- **Scenario Governance**: Scenario weight approval metadata, review cadence tracking, methodology documentation, and sensitivity analysis per IFRS 9.B5.5.41-43
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

## Community Projects

- [Basel Risk Dashboard](https://adipandey956.github.io/Basel-risk-dashboard/) by [@adipandey956](https://github.com/adipandey956) — ICAAP stress testing + jurisdiction RWA dashboard with IRB scenarios across BCBS, EU CRR3, UK PRA, and RBI

## License

Apache 2.0 -- see [LICENSE](LICENSE) for details.

> **Disclaimer**: This library is provided for educational and analytical
> purposes. It has not been reviewed or endorsed by any regulatory authority.
> See [Regulatory Disclaimers](docs/regulatory_disclaimers.md) for details.
