# Architecture

CreditRiskEngine is organized into six major subsystems, each mapping to
a distinct regulatory or analytical domain.

## Package Layout

```
creditriskengine/
├── core/               # Data models, types, config, exceptions
│   ├── types.py        # Enums: Jurisdiction, IRBAssetClass, SAExposureClass, ...
│   ├── exposure.py     # Pydantic Exposure + Collateral models
│   ├── portfolio.py    # Portfolio container with filtering
│   └── config.py       # Jurisdiction config loader
│
├── rwa/                # Risk-Weighted Assets
│   ├── standardized/
│   │   └── credit_risk_sa.py   # CRE20 SA risk weights (all exposure classes)
│   ├── irb/
│   │   └── formulas.py         # CRE31 IRB: correlations, K, maturity adj, RW
│   └── output_floor.py         # RBC25 output floor with multi-jurisdiction phase-in
│
├── ecl/                # Expected Credit Loss engines
│   ├── ifrs9/          # IFRS 9 impairment
│   │   ├── ecl_calc.py     # 12-month and lifetime ECL
│   │   ├── staging.py      # Three-stage assignment
│   │   ├── sicr.py         # Significant Increase in Credit Risk
│   │   ├── lifetime_pd.py  # Cumulative/marginal PD term structures
│   │   ├── ttc_to_pit.py   # TTC-to-PIT conversion (Vasicek Z-factor)
│   │   ├── forward_looking.py  # Macro overlay adjustments
│   │   └── scenarios.py    # Probability-weighted scenarios
│   ├── cecl/           # US CECL (ASC 326)
│   │   ├── cecl_calc.py    # PD/LGD and loss-rate methods
│   │   ├── methods.py      # WARM, vintage, DCF
│   │   └── qualitative.py  # Q-factor adjustments
│   └── ind_as109/      # Indian Accounting Standard 109
│       └── ind_as_ecl.py   # Wraps IFRS 9 with RBI-specific defaults
│
├── models/             # PD / LGD / EAD model development
│   ├── pd/
│   │   └── scorecard.py    # Logistic scorecard, master scale, PD calibration
│   ├── lgd/
│   │   └── lgd_model.py    # Workout LGD, downturn LGD, LGD floors
│   ├── ead/
│   │   └── ead_model.py    # EAD calculation, CCF estimation, supervisory CCFs
│   └── concentration/
│       └── concentration.py # Single-name HHI, sector concentration, GA
│
├── validation/         # Model validation toolkit
│   ├── discrimination.py   # AUROC, Gini, KS, CAP, IV, Somers' D
│   ├── calibration.py      # Binomial, HL, Spiegelhalter, traffic light, Jeffreys
│   ├── stability.py        # PSI, CSI, HHI, migration matrix stability
│   ├── backtesting.py      # PD backtest summary
│   ├── benchmarking.py     # Model-vs-benchmark comparison
│   └── reporting.py        # Validation summary generator
│
├── portfolio/          # Portfolio credit risk models
│   ├── vasicek.py          # Vasicek ASRF: conditional DR, loss quantile, distribution
│   ├── copula.py           # Single/multi-factor Gaussian copula Monte Carlo
│   ├── economic_capital.py # EC via single-factor simulation
│   ├── var.py              # Parametric credit VaR, marginal VaR
│   └── stress_testing.py   # PD/LGD stress, RWA impact
│
├── regulatory/         # Jurisdiction-specific YAML configs (17 jurisdictions)
│   ├── loader.py           # YAML config loader
│   ├── bcbs/               # BCBS d424
│   ├── eu/                 # EU CRR3
│   ├── uk/                 # UK PRA PS1/26
│   ├── us/                 # US Basel III Endgame
│   ├── india/              # RBI
│   └── ...                 # + 12 more jurisdictions
│
└── reporting/          # Regulatory reporting
    └── reports.py          # COREP summary, Pillar 3 disclosure, model inventory
```

## Design Principles

1. **Regulatory traceability** — Every formula cites its BCBS/CRE/RBC
   paragraph reference. Risk weight tables map 1:1 to the Basel standard.

2. **Jurisdiction-aware** — Functions accept a `Jurisdiction` enum. SA risk
   weights, output floor schedules, and default definitions vary by
   jurisdiction automatically.

3. **Composable** — Each module is independently usable. You can call
   `irb_risk_weight()` without touching ECL, or compute PSI without
   importing portfolio models.

4. **Typed and validated** — Pydantic models for exposures and portfolios.
   Strict mypy, ruff linting, and comprehensive tests enforce correctness.

## Data Flow

```
Exposure (Pydantic model)
    │
    ├─► SA Risk Weight   ─► SA RWA
    │                          │
    ├─► IRB Risk Weight  ─► IRB RWA ──┐
    │                                  ├─► Output Floor ─► Floored RWA
    │                                  │
    ├─► ECL Calculation  ─► ECL Provision
    │
    └─► Validation       ─► AUROC, Gini, PSI, Calibration tests
```

## Key Regulatory References

| Module | BCBS Reference |
|--------|---------------|
| SA risk weights | CRE20 (Tables 1-10) |
| IRB formulas | CRE31.4-31.10 |
| PD floor | CRE32.13 (3 bps) |
| Supervisory LGD | CRE32.22-32.24 |
| LGD floors (A-IRB) | CRE32.25 |
| CCF (F-IRB) | CRE32.29-32.32 |
| Output floor | RBC25.2-25.4 |
| IFRS 9 staging | IFRS 9.5.5.1-5.5.20 |
| CECL | ASC 326-20 |
