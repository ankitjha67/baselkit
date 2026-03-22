# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-03-22

### Added

**Core**
- `Exposure` and `Collateral` Pydantic data models with field validation
- `Portfolio` container with filtering (by approach, default status)
- 17-value `Jurisdiction` enum (BCBS, EU, UK, US, India, + 12 more)
- Full enum set: `IRBAssetClass`, `SAExposureClass`, `CreditQualityStep`,
  `IFRS9Stage`, `CollateralType`, `CRMApproach`, `DefaultDefinition`
- Custom exception hierarchy (`ConfigurationError`, `JurisdictionError`, etc.)

**RWA Calculation**
- Standardized Approach (CRE20): sovereign, bank (ECRA/SCRA), corporate,
  retail, RRE/CRE LTV-based tables, ADC, equity, defaulted exposure classes
- Jurisdiction overrides: UK PRA 65% unrated IG, EU SME supporting factor,
  India RBI residential mortgage thresholds
- IRB formulas (CRE31): asset correlations for all 5 classes, SME firm-size
  adjustment, maturity adjustment, capital requirement K, risk weight dispatcher
- Output floor (RBC25): phase-in schedules for 10 jurisdictions including
  EU transitional 25% cap (CRR3 Art. 92a(3))

**ECL Engines**
- IFRS 9: 12-month and lifetime ECL, 3-stage assignment with DPD backstop,
  SICR assessment (relative/absolute PD change), lifetime PD term structures
  (from annual PDs or transition matrices), TTC-to-PIT conversion (Z-factor),
  macro overlay adjustments, probability-weighted scenario ECL
- CECL (ASC 326): PD/LGD lifetime method, loss-rate method, WARM, vintage
  analysis, DCF, qualitative Q-factor adjustments (8 interagency categories)
- Ind AS 109: RBI 90 DPD NPA threshold, delegates to IFRS 9 calculation

**PD/LGD/EAD Modeling**
- PD scorecard: logistic score, score-to-PD, PD-to-scorecard-points,
  master scale construction, rating grade assignment
- PD calibration: anchor-point method, Bayesian calibration,
  Vasicek conditional (stressed) PD
- LGD model: workout LGD with discounting, downturn LGD (additive/haircut/
  regulatory methods), LGD term structure, A-IRB LGD floors (CRE32.25)
- EAD model: EAD calculation, realized CCF estimation, supervisory CCFs
  (CRE32.29-32), A-IRB CCF floor, EAD term structure with amortization

**Concentration Risk**
- Single-name concentration (HHI, top-N shares)
- Sector concentration via HHI
- Gordy (2003) Granularity Adjustment for Pillar 2

**Model Validation**
- Discrimination: AUROC (Mann-Whitney), Gini, KS, CAP curve, IV, Somers' D
- Calibration: binomial test, Hosmer-Lemeshow, Spiegelhalter, Basel traffic
  light (green/yellow/red), Jeffreys Bayesian test, Brier score
- Stability: PSI, CSI, HHI, migration matrix stability (Frobenius norm)
- Backtesting: PD backtest summary statistics
- Benchmarking: model-vs-benchmark metric comparison
- Reporting: validation summary with traffic-light assessment

**Portfolio Risk**
- Vasicek ASRF: conditional default rate, loss quantile, EL/UL/EC,
  full loss distribution PDF
- Gaussian copula: single-factor and multi-factor Monte Carlo with
  antithetic variates, credit VaR, expected shortfall
- Economic capital via single-factor simulation
- Parametric credit VaR and marginal VaR contribution
- Stress testing: PD/LGD stress application, RWA impact calculation

**Regulatory Reporting**
- COREP credit risk summary (8% capital requirement, floor binding)
- Pillar 3 credit risk disclosure template
- Model inventory entry with RAG assessment

**Multi-Jurisdiction**
- 17 jurisdiction YAML configs with risk weights, PD/LGD floors, output
  floor schedules, default definitions, capital requirements
- Jurisdictions: BCBS, EU, UK, US, India, Singapore, Hong Kong, Japan,
  Australia, Canada, China, South Korea, UAE, Saudi Arabia, South Africa,
  Brazil, Malaysia
- YAML config loader with jurisdiction-to-file mapping

**Infrastructure**
- GitHub Actions CI: Python 3.11/3.12, ruff, mypy, pytest with coverage
- GitHub Actions release workflow
- Example portfolio and run configuration files
- Documentation: architecture, getting started, regulatory mapping

### Tests
- 384+ tests covering all modules
- Test coverage: 89%+
- 0 ruff lint errors, 0 mypy type errors
