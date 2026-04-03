# Changelog

All notable changes to this project will be documented in this file.

## [0.6.0] - 2026-04-03

### Added

- **ECL Governance Layer** -- Management overlays, scenario governance, and
  overlay audit trail for the governance layer on top of model output:
  - `ecl/ifrs9/overlays.py`: Management overlay (PMA) framework with 7
    overlay types (model limitation, emerging risk, data gap, economic
    uncertainty, sector-specific, regulatory, temporary event), effectiveness
    dating, rate-based and absolute adjustments, governance validation
    (rationale, approval, expiry), and Pillar 3 disclosure summary per
    IFRS 9.B5.5.52, EBA/GL/2020/06, PRA Dear CFO letter (Jul 2020)
  - `ecl/ifrs9/scenarios.py`: ScenarioSetMetadata for approval chain and
    review cadence, validate_scenario_governance() per EBA/GL/2017/06,
    scenario_sensitivity_analysis() quantifying ECL sensitivity to weight
    perturbations per IFRS 9.B5.5.41-43
  - `core/audit.py`: OverlayAuditRecord for immutable overlay lifecycle
    tracking (applied/reviewed/revoked/expired), record_overlay(),
    get_overlay_records(), overlay_summary(), JSON export with overlay section

- **Advanced Forward-Looking Information** -- Multi-variable satellite
  model framework replacing the basic linear approach:
  - SatelliteModelConfig with configurable link functions (linear, logistic,
    log) per ECB Guide to Internal Models Ch. 7
  - Forecast-horizon mean-reversion per IFRS 9.B5.5.50 with configurable
    reversion ramp
  - LGD macro overlay driven by collateral index (HPI) changes per
    EBA/GL/2017/16 Art. 181
  - FLI impact summary for IFRS 7.35G disclosure

- **Full RBI IRAC Norms for Ind AS 109** -- Comprehensive India-specific
  implementation replacing the thin wrapper:
  - IRACAssetClass enum: Standard, SMA-0/1/2, Substandard,
    Doubtful-1/2/3, Loss per IRAC Norms para 2.1-2.5
  - classify_irac() with agricultural sector DPD thresholds (60 DPD
    short crop, 90 DPD long crop)
  - RBI minimum provisioning percentages per IRAC para 4.2-4.5
    (0.25%-100% by classification and sector)
  - irac_to_ifrs9_stage() mapping per RBI/2019-20/170
  - restructured_account_stage() per IRAC para 12-14
  - calculate_ecl_ind_as() now applies RBI provisioning floor
    (higher of model ECL and IRAC minimum)

- **CECL Q-Factor Governance** -- Governance framework for qualitative
  factors per OCC Bulletin 2020-49:
  - Approval metadata (approved_by, approval_date, expiry_date, rationale)
  - Per-category caps with warnings (DEFAULT_CATEGORY_CAPS_BPS)
  - apply_q_factors_with_caps() enforcing governance guardrails
  - validate_q_factors() checking 8-category coverage against
    interagency guidance
  - q_factor_summary() for governance reporting

### Changed

- `forward_looking.py`: Expanded from 62 to 390 lines with satellite
  model, mean-reversion, and LGD overlay
- `ind_as109/ind_as_ecl.py`: Expanded from 98 to 310 lines with full
  IRAC norms
- `cecl/qualitative.py`: Expanded from 71 to 230 lines with governance
  framework
- `core/__init__.py`: Exports AuditTrail, CalculationRecord,
  OverlayAuditRecord
- `ecl/ifrs9/__init__.py`: Exports overlay and scenario governance symbols
- `ecl/ind_as109/__init__.py`: Exports IRAC classification symbols
- `regulatory_mapping.md`: 33 new regulatory references (185+ total)
- `core/audit.py`: export_json() now includes overlay records section

### Tests
- Test count: 1,960 → 2,068 (+108 new tests)
- New test files: test_overlays.py, test_scenario_governance.py,
  test_maturity.py (previously untested module)
- Expanded: test_ind_as109.py (55 → 190 lines), test_ttc_pit_fli.py
  (115 → 200 lines)
- 100% line coverage, 0 mypy errors, 0 ruff errors

## [0.5.0] - 2026-03-31

### Added

- **Revolving Credit ECL** -- Full IFRS 9 ECL engine for revolving facilities
  (credit cards, overdrafts, HELOCs, corporate revolvers, working capital,
  margin lending) with:
  - Behavioral life determination per IFRS 9 B5.5.40 three-factor framework
    (historical life, time-to-default, CRM actions -- shortest wins)
  - Multi-approach CCF models: Regulatory SA (Basel III/CRR3 10% UCC, APRA
    40%), F-IRB supervisory, behavioral LEQ, EADF, PIT macro adjustment
  - Drawn/undrawn ECL decomposition per IFRS 7 B8E (loss allowance vs.
    provision liability)
  - Revolving EAD term structures with repayment/redraw dynamics
  - Probability-weighted multi-scenario ECL
  - Multi-jurisdiction provision floors loaded from YAML (CBUAE 1.5% CRWA,
    RBI 1%/5% unsecured retail, MAS 1% MRLA, SAMA 1% CRWA)
  - Product configs loaded from YAML (`regulatory/revolving_products.yml`)
  - Convenience function `revolving_ecl_from_exposure()` accepting Exposure
    objects directly
  - Unified CCF architecture: `ead_model.get_sa_ccf()` is the single source
    of truth; revolving module delegates via product-to-facility mapping
  - Exposure model extended with `is_revolving`, `credit_limit`,
    `behavioral_life_months`, `ccf` fields

- **FR 2052a Complex Institution Liquidity Monitoring** -- Federal Reserve
  FR 2052a reporting framework (OMB 7100-0361) with:
  - 23 enumeration types (counterparties, asset categories, maturity buckets)
  - 13 Pydantic schema models (one per FR 2052a schedule table)
  - Complete catalog of 137 products across all 13 tables
  - Record-level and submission-level validation engine
  - Report generator with inflow/outflow aggregation and 30-day profiles

- **FINOS-compatible governance files** -- DCO, NOTICE, MAINTAINERS.md,
  SECURITY.md, CODE_OF_CONDUCT, issue templates, DCO CI check

- **PyPI release workflow** -- Trusted publishing via GitHub Actions
  (`release.yml`), triggered on GitHub Release events

### Changed

- `ead_model.py`: Added `get_sa_ccf()` and `get_airb_ccf_floor()` with
  jurisdiction-aware lookups (APRA overrides, CRR3 transitional)
- `core/exposure.py`: Extended with revolving credit fields
- ECL `__init__.py` files: Export revolving module symbols
- Tests: Parametrized CCF and provision floor tests via `pytest.mark.parametrize`
- Test count: 1,786 → 1,960

## [0.3.0] - 2026-03-24

### Fixed
- Version bump to 0.3.0 to match PyPI release
- Removed broken GitHub Pages docs link (site not yet deployed)
- Fixed CI: ruff ANN rules now exclude test files, mypy configured to ignore missing third-party stubs
- Removed unused `type: ignore` comment in `core/portfolio.py`

## [0.2.0] - 2026-03-24

### Added

**Spec Compliance (8 Gap Fixes)**
- QRRE transactor 0.75× RW scalar (BCBS CRE31.9 footnote 15)
- UK PRA loan-splitting for residential RE (`uk_pra_loan_splitting_rre()`)
- BoE ACS stress testing class with 5-year horizon and CET1 hurdle tracking
- 13 spec-aligned re-export modules for file granularity compliance
- SA Bank SCRA grade routing fix through `assign_sa_risk_weight()` dispatcher
- `is_short_term` parameter wired through SA bank risk weight dispatcher

**New Modules**
- Credit Risk Mitigation engine (`rwa/crm.py`): comprehensive approach,
  simple approach, guarantee substitution, supervisory haircuts, maturity
  mismatch (CRE22)
- Operational Risk SMA (`rwa/operational_risk.py`): Business Indicator
  Component, Internal Loss Multiplier (OPE25)
- Market Risk FRTB integration (`rwa/market_risk.py`): SbM credit spread,
  Default Risk Charge, Residual Risk Add-on (MAR)
- Audit trail (`core/audit.py`): `CalculationRecord`, `AuditTrail` with
  JSON export and DataFrame output
- YAML schema validation (`regulatory/schema.py`): config validation,
  input sanitization, range checks
- Structured logging (`core/logging_config.py`): JSON formatter for
  production log aggregation

**Documentation**
- MkDocs site with Material theme (`mkdocs.yml`)
- API reference for all modules
- User guides: IRB, SA, ECL, stress testing, model validation
- Regulatory disclaimers with interpretation choices table
- Regulatory config versioning strategy
- Performance benchmarks (`benchmarks/bench_portfolio.py`)

**Packaging**
- `py.typed` PEP 561 marker for typed package
- Upper-bounded dependency versions (numpy<3.0, scipy<2.0, etc.)
- Project URLs (homepage, docs, issues, changelog)
- Python 3.13 classifier added

### Changed
- EU CRR3 YAML: full 7-bucket LTV tables for residential and commercial RE,
  detailed AIRB restrictions with revenue threshold, infrastructure supporting
  factor
- EU CRR3 YAML: LGD supervisory values corrected (secured: 0.20→0.10,
  other: 0.25→0.15) per CRE32.22-24
- EU CRR3 YAML: PD floor corrected from 5 bps to 3 bps per CRE32.13
- UK PRA YAML: LGD supervisory values corrected, loan-splitting config added

### Tests
- 1,200+ tests with 100% line coverage
- Regulatory back-testing against Basel Committee worked examples
- Integration tests with 10,000+ exposure portfolios
- Cross-validation against known vendor reference ranges
- 0 ruff lint errors, 0 mypy type errors

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
