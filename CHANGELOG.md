# Changelog

All notable changes to this project will be documented in this file.

## [0.28.0] - 2026-06-29

### Added — RBI Project Finance & Gold Loan Directions 2025

- **`ecl/ind_as109/project_finance.py`** — DCCO deferment framework per
  the RBI (Project Finance) Directions, 2025 (effective 1 Oct 2025):
  `dcco_deferment_provision` computes the cumulative additional provision
  (0.375%/quarter infrastructure, 0.5625%/quarter non-infrastructure,
  partial quarters rounded up) on top of the construction-phase standard
  rate, and enforces the permitted DCCO extension window (3 years
  infrastructure / 2 years non-infrastructure) — beyond it the exposure
  loses standard classification.
- **`ecl/ind_as109/gold_loans.py`** — tiered LTV ceilings per the RBI
  (Lending Against Gold and Silver Collateral) Directions, 2025
  (compliance 1 Apr 2026): 85% up to INR 2.5 lakh, 80% to 5 lakh, 75%
  above; `assess_gold_loan_ltv` returns compliance and the maximum
  permissible loan for the collateral.

## [0.27.0] - 2026-06-29

### Added — EU CRR3 SA credit-risk drivers (Arts. 465(3), 501, 501a, 123a)

Four high-frequency CRR3 Standardised Approach features, verified against
Regulation (EU) 2024/1623:

- **Unrated-corporate transitional (Art. 465(3))**: unrated EU corporates
  with institution-estimated PD <= 0.5% receive a 65% risk weight until
  31 December 2032 (100% from 2033). `get_corporate_risk_weight` gains
  `pd` and `reporting_date` parameters with the date gate.
- **Tiered SME supporting factor (Art. 501)**: new
  `eu_sme_supporting_factor(total_exposure_eur)` applies 0.7619 on the
  portion of the obligor's total exposure up to EUR 2.5m and 0.85 on the
  excess (previously a flat 0.7619 regardless of size).
- **Infrastructure supporting factor (Art. 501a)**: qualifying
  infrastructure exposures receive a 0.75 multiplier (caller asserts the
  Art. 501a(1) eligibility criteria); the factor was in the CRR3 YAML but
  never applied in code. Correctly not applied for the UK.
- **Currency-mismatch multiplier (Art. 123a / BCBS CRE20.92)**: new
  `currency_mismatch_multiplier` — unhedged retail and residential-RE
  exposures to individuals with a loan/income currency mismatch get
  RW x 1.5 capped at 150%; wired into `get_retail_risk_weight` and
  `get_residential_re_risk_weight` (including the India RBI branch).

`crr3.yml` updated with the new constants; the stale "3 bps" PD-floor
comment corrected.

## [0.26.0] - 2026-06-29

### Fixed — APAC jurisdiction-config correctness (2023-2025 updates)

Data-correctness fixes surfaced by the multi-jurisdiction research pass:

- **China `nfra.yml`**: five-tier DPD buckets corrected to the NFRA Measures
  2023 thresholds — substandard **>90**, doubtful **>270**, loss **>360**
  days (were the repealed 90-360 / 360-720 / >720; `ecl/emerging/china.py`
  was already correct — this removes a config/code contradiction).
- **Hong Kong `hkma.yml`**: HK countercyclical buffer **1.0% → 0.5%**
  (HKMA, effective 1 January 2025).
- **Singapore `mas_637.yml`**: output-floor final 72.5% now reached on
  **1 January 2029** (revised MAS Notice 637, effective 1 Jul 2024) — one
  year earlier than the Basel/EU 2030 endpoint.

Added `test_regulatory/test_config_corrections.py` regression tests.

## [0.25.0] - 2026-06-29

### Fixed — IRB PD input floor raised to the Basel III finalisation value

The IRB PD input floor was hardcoded at the Basel II value of **0.03%
(3 bps)**; the Basel III finalisation (BCBS CRE32.13) and CRR3 Art. 160/161
raise it to **0.05% (5 bps)** for corporate and retail exposures, with a
higher **0.10% (10 bps)** floor for qualifying revolving retail (QRRE).
The old value silently under-floored every low-PD IRB risk weight.

- `rwa/irb/formulas.py`: `PD_FLOOR` 0.0003 → 0.0005; new `PD_FLOOR_QRRE`
  (0.0010) and a `pd_input_floor(asset_class)` helper; `irb_risk_weight`
  now applies the QRRE floor for `asset_class="qrre"`. `foundation.py` and
  `advanced.py` inherit the corrected floor.
- `models/pd/margin_of_conservatism.py`: PD clip floor 0.0003 → 0.0005.
- `regulatory/eu/crr3.yml`: `pd_floor_bps` 3 → 5.

(The RBI ECL 0.03% backstop is a separate, jurisdiction-specific floor and
is unchanged.)

## [0.24.0] - 2026-06-29

### Added — SFT minimum haircut floors (CRE56)

New `rwa/sft_haircut_floors.py` implementing the minimum haircut floors for
non-centrally-cleared securities financing transactions.

- `minimum_haircut_floor(collateral_type, residual_maturity)` — the CRE56.2
  floor table (corporate/other debt 0.5-4%, securitisations 1-7% by
  maturity, main-index equity 6%, other assets 10%; cash and government
  debt out of scope).
- `sft_haircut` / `assess_sft_floor` — a single SFT's collateral haircut
  and whether it meets its floor (collateral is not recognised below the
  floor, per CRE56.4).
- `portfolio_floor_compliant` — the netting-agreement portfolio floor test
  (CRE56.5): portfolio haircut vs the exposure-weighted average floor.

## [0.23.0] - 2026-06-29

### Added — MREL (BRRD2 / SRMR2)

New `rwa/mrel.py` implementing the EU resolution Minimum Requirement for
own funds and Eligible Liabilities — the EU analogue of TLAC.

- `mrel_trea_requirement` / `mrel_tem_requirement` — the Loss Absorption +
  Recapitalisation Amount calibration (Pillar 1 + P2R, plus a market-
  confidence charge on the TREA recapitalisation amount), floored at the
  18% / 6.75% TLAC minimums for G-SII resolution entities.
- `assess_mrel` — compares eligible MREL against the higher of the TREA and
  TEM (leverage) requirements, reporting both ratios, the binding
  constraint, shortfalls, and a compliance flag (`MRELResult`).

## [0.22.0] - 2026-06-29

### Added — Equity investments in funds (CRE60)

New `rwa/equity_in_funds.py` implementing the CRE60 hierarchy:

- `fund_average_risk_weight` / `fund_leverage` — the fund's average risk
  weight (underlying RWA / total assets) and leverage (assets / equity).
- `look_through_rwa` / `mandate_based_rwa` — RWA = min(avg_RW × leverage,
  1250%) × investment, for the Look-Through and Mandate-Based approaches.
- `fall_back_rwa` — the 1250% Fall-Back Approach.

Each returns a `FundRWAResult` with the effective risk weight, RWA, and a
cap-binding flag.

## [0.21.0] - 2026-06-29

### Added — Settlement / failed-trade risk capital (CRE70)

New `rwa/settlement_risk.py`.

- `dvp_settlement_multiplier` / `dvp_settlement_capital` — DvP unsettled-
  transaction capital charge on the positive current exposure, scaled by
  the CRE70.5 business-days multiplier (8% at 5-15 days, 50% at 16-30,
  75% at 31-45, 100% at 46+), with the RWA equivalent.
- `non_dvp_risk_weight` — free-delivery treatment: counterparty risk
  weight until 4 business days after the second leg, then 1250% (CRE70.7).

## [0.20.0] - 2026-06-29

### Added — Total Loss-Absorbing Capacity (TLAC)

New `rwa/tlac.py` implementing the FSB TLAC standard for G-SIBs.

- `available_tlac(...)` — regulatory capital + eligible TLAC debt, less the
  CET1 used to meet the combined buffer requirement (buffers sit on top of
  TLAC, no double-counting).
- `tlac_ratios(...)` — assesses TLAC against the higher of the 18%-of-RWA
  and 6.75%-of-leverage-exposure minimums (16% / 6% during the 2019-2021
  conformance period), reporting both ratios, the binding constraint, the
  RWA and leverage shortfalls, and a compliance flag (`TLACResult`).

## [0.19.0] - 2026-06-29

### Added — Basel III liquidity ratios (LCR & NSFR)

New `liquidity/` package implementing the two Basel III liquidity metrics.

- **`liquidity/lcr.py`** — Liquidity Coverage Ratio (BCBS d238):
  `stock_of_hqla` applies the 15%/50% Level 2A/2B haircuts and the
  closed-form Level 2 (40% of HQLA) and Level 2B (15% of HQLA) caps;
  `net_cash_outflows` applies the 75% inflow cap;
  `liquidity_coverage_ratio` returns the ratio and compliance flag.
- **`liquidity/nsfr.py`** — Net Stable Funding Ratio (BCBS d295):
  ASF and RSF factor tables (`ASFCategory`/`RSFCategory`),
  `available_stable_funding` / `required_stable_funding`, and
  `net_stable_funding_ratio` with the 100% compliance check.

## [0.18.0] - 2026-06-29

### Added — Large Exposures framework (BCBS LEX)

New `rwa/large_exposures.py` implementing the supervisory framework for
measuring and controlling large exposures (BCBS d283 / LEX).

- `exposure_value(...)` — pre-risk-weight exposure value: on-balance +
  CCF × off-balance + derivative EAD + SFT exposure, net of eligible CRM.
- `aggregate_connected_group(...)` — sums exposures across connected
  counterparties (control / economic interdependence) per LEX10.
- `assess_large_exposure(...)` — tests an exposure against the 25%-of-
  Tier-1 limit (15% for G-SIB-to-G-SIB), the 10% reporting threshold, and
  reports the ratio, breach flag and headroom (`LargeExposureResult`).
- `large_exposures_report(...)` — portfolio-level LEX30 report listing
  reportable exposures (sorted) and limit breaches (`LargeExposureReport`).

## [0.17.0] - 2026-06-29

### Added — Climate scenario analysis (EBA/GL/2025/02)

New `climate/scenario_analysis.py` orchestrates the existing NGFS,
transition-risk and physical-risk building blocks into a portfolio-level
projection of climate-adjusted ECL.

- `project_climate_ecl(exposures, scenario, horizon_year)` — for each
  exposure computes a stressed PD (baseline PD × transition carbon-cost
  multiplier × physical-hazard multiplier) and stressed LGD (baseline LGD
  + physical collateral haircut), then aggregates the baseline-vs-stressed
  ECL uplift with a transition/physical decomposition. Returns
  `ClimateScenarioResult` with per-exposure detail.
- `scenario_carbon_price(scenario, horizon_year)` — horizon carbon price,
  linearly interpolated between the NGFS 2030 and 2050 anchors.
- `ClimateExposure` input dataclass with validation.

## [0.16.1] - 2026-06-29

### Tests — 100% line coverage restored

Added targeted unit tests across the codebase to bring line coverage from
98% back to 100% (2,667 tests total). Coverage was driven up across CECL
Q-factor governance, the full SA-CCR engine, the DRC engine, SEC-ERBA /
securitisation, the granularity adjustment, the revolving-credit ECL
sub-engine, the RBI ECL 2026 modules, and the advanced PD/LGD models.

Two genuinely-unreachable defensive branches (a tail-mask guard after a
quantile in `pricing/allocation.py` and the equivalent in `frtb_ima.py`'s
expected-shortfall) are marked `# pragma: no cover` rather than tested,
since a quantile always has at least one observation at or beyond it.

No API or behavioural changes.

## [0.16.0] - 2026-06-29

### Changed — Replace remaining simplified calculations with full implementations

A sweep to remove the last simplified/heuristic calculations in favour of
the full regulatory or theoretical formulas.

- **`rwa/frtb_ima.py` — full Default Risk Charge (MAR22)**: the previous
  single-bucket DRC is replaced by ``default_risk_charge`` implementing the
  complete MAR22.18-22.33 methodology — obligor-level JTD netting, the
  MAR22.24 default risk-weight table, the book-wide hedge-benefit ratio
  (WtS), and per-bucket aggregation with no cross-bucket diversification.
  Adds ``DRCPosition`` and ``drc_default_risk_weight``;
  ``default_risk_charge_ima`` now delegates to the full engine.
- **`rwa/securitisation.py` — full SEC-ERBA (CRE43)**: the simplified
  ``RW * (1 + 0.4*(MT-1))`` maturity factor is replaced by the CRE43
  Table 2 two-column (1-year / 5-year) risk weights with proper linear
  maturity interpolation (CRE43.5), the non-senior tranche-thickness
  adjustment ``RW * (1 - min(D-A, 0.5))`` (CRE43.6), and the 15 % floor.
- **`models/concentration/concentration.py` — rigorous granularity
  adjustment**: replaces the ad-hoc ``0.5 * HHI * (1-rho)`` heuristic with
  the Martin-Wilde (2002) / Gordy (2003) second-order granularity
  adjustment for the single-factor Vasicek ASRF model, evaluated at the
  99.9 % quantile.

## [0.15.0] - 2026-06-29

### Added — Full SA-CCR engine (CRE52)

Replaces the thin ``alpha * (RC + PFE)`` leverage-ratio wrapper with a
complete trade-level SA-CCR EAD engine in ``ccr/sa_ccr.py``:

- ``sa_ccr_ead(trades, net_mtm, collateral, ...)`` computes
  ``EAD = alpha * (RC + multiplier * AddOn)`` from first principles.
- Adjusted notionals with the supervisory duration (CRE52.34); supervisory
  deltas for linear trades, options (Black-Scholes with supervisory vol)
  and CDO tranches (CRE52.34-52.40); maturity factors for unmargined and
  margined sets (CRE52.48/52.50).
- Asset-class add-ons with the correct hedging-set aggregation: interest-
  rate per-currency maturity buckets (CRE52.52), FX currency pairs, and the
  single-factor systematic/idiosyncratic structure for credit, equity and
  commodity (CRE52.55-52.70).
- ``pfe_multiplier`` (CRE52.23) and unmargined/margined ``replacement_cost``
  (CRE52.10/52.18). Validated against a hand-computed single-swap example.

### Fixed — SEC-IRBA / SEC-SA supervisory formula (CRE44)

The securitisation SSFA was previously simplified and produced incorrect
risk weights. Now corrected and validated against the BCBS d374 worked
examples (Tranche A 30-100% -> 28.78 %; Tranche B 5-30% -> 1056.94 %):

- SEC-IRBA supervisory parameter ``p`` now uses the full CRE44.13
  five-coefficient table ``p = max(0.3, A + B/N + C*KIRB + D*LGD + E*MT)``,
  keyed on exposure type (retail/wholesale), seniority and granularity
  (N >= 25), replacing the previous fixed-``p``-with-ad-hoc-adjustment.
- ``KSSFA`` corrected to the CRE44.14 form with KIRB-adjusted bounds
  ``u = D - KIRB``, ``l = max(A - KIRB, 0)``.
- Risk-weight assembly corrected to the CRE44.15 three-region form
  (1250 % below KA, ``12.5 * KSSFA`` above, exposure-weighted when the
  tranche straddles KA), removing an erroneous tranche-thickness division.
- SEC-SA now applies the CRE42.2 delinquency adjustment
  ``KA = (1 - W) * KSA + W * 0.5`` via a ``delinquency_ratio`` argument.
- ``SecuritisationPool`` gains an ``is_retail`` flag.

## [0.14.0] - 2026-06-29

### Added — ESG risk management per EBA/GL/2025/01

Implements the two quantitative centrepieces of the EBA Guidelines on
the management of ESG risks (EBA/GL/2025/01, applicable from 11 January
2026), sitting above the existing climate (NGFS, PCAF, GAR) and
ESG-ratings modules.

- **`esg/risk_management.py`**:
  - `assess_esg_materiality(drivers, total_exposure, horizon,
    materiality_threshold=0.10)` — exposure-weighted likelihood × impact
    materiality score in [0, 1] with a material / not-material verdict and
    a per-factor breakdown (environmental physical/transition, social,
    governance). Returns `MaterialityResult`.
  - `recommended_method(horizon)` — encodes the EBA Title 4 mapping of
    assessment method to time horizon: short → exposure-based, medium →
    sector/portfolio-alignment, long → scenario-based.
  - `transition_plan_alignment(current, base, target, base_year,
    target_year, current_year)` — monitors a portfolio metric (e.g.
    financed-emissions intensity, or a GAR-uplift target) against its
    intermediate target on a straight-line pathway, per CRD Art. 76(2)
    and Art. 87a. Infers reduction vs growth direction, reports the
    expected-by-now value, gap, on-track flag, alignment %, and required
    annual change. Returns `TransitionPlanStatus`.

## [0.13.0] - 2026-06-29

### Added — Contractual EAD amortisation schedules

Auto-derive the IFRS 9 EAD term structure from a loan's contractual
repayment terms, so lifetime ECL no longer requires a hand-supplied EAD
profile for non-revolving (amortising) exposures.

- **`models/ead/ead_model.py`**:
  - `amortising_balance_schedule(principal, annual_rate, n_periods,
    periods_per_year=1, balloon_fraction=0.0)` — end-of-period
    outstanding-balance path for an annuity (equal-instalment) loan.
    Handles straight-line (0% rate), monthly/quarterly compounding,
    pure bullet (`balloon_fraction=1.0`), and partial balloon
    structures; the amortising portion never drops below the balloon
    until the final maturity repayment.
  - `ead_term_structure_from_schedule(principal, annual_rate,
    n_periods, undrawn_commitment=0.0, ccf=0.0, periods_per_year=1,
    balloon_fraction=0.0)` — combines the contractual drawn-balance
    path with `CCF × undrawn` to yield a period-by-period EAD curve
    that feeds directly into `ecl_lifetime`.

This makes the EAD curve fully automated from loan terms rather than
assuming `EAD = current outstanding balance` flat across the lifetime
horizon.

## [0.7.0] - 2026-05-27

### Bug fixes (full-repo audit)

A comprehensive audit of the entire codebase identified and corrected
the following defects:

- **`rwa/irb/advanced.py`**: A-IRB EAD calculation hardcoded the CCF
  input to 0.0 before applying the 50% floor, ignoring any bank-
  estimated `exposure.ccf`. The fix reads `exposure.ccf` (defaulting
  to 0.0 when None) and applies the floor per CRR3 Art. 166(8b).
  Material correction for all A-IRB exposures with undrawn commitments.
- **`rwa/standardized/cre.py`**: CRE NOT-cashflow-dependent table had
  a fixed 80% RW for the 60-80% LTV bucket. Per BCBS d424 CRE20.88,
  LTV > 60% should use the counterparty risk weight directly. Table
  reduced from 3 to 2 buckets, second bucket uses sentinel -1.0.

### Added — Robustness tooling

- **`parameter_assertions.py`**: Source-of-truth guard that locks all
  RBI ECL 2026 parameters to published values. Any future edit that
  perturbs a constant trips `RBIParameterMismatch`. Includes
  `regulatory_self_check()` for audit packs.
- **`parallel_run.py`**: Side-by-side comparator for legacy IRACP vs
  ECL 2026 frameworks. Returns delta, percentage change, binding-floor
  diagnostic, and CET1 transitional add-back amount per Paragraph 108.
  Includes `portfolio_parallel_run_summary()` for board reporting.

### Added

- **RBI ECL Master Direction 2026 (RBI/DOR/2026-27/398)** -- Full implementation
  of the Reserve Bank of India's final ECL directions for commercial banks,
  effective April 1, 2027. Replaces the legacy IRAC provisioning framework
  with a stage-aware, exposure-category-keyed model:

  - `ecl/ind_as109/types.py` -- `RBIExposureCategory` (20 categories) and
    `RBICollateralCategory` enums per Paragraph 82.
  - `ecl/ind_as109/provision_floors_2026.py` -- Stage 1 / Stage 2 floor
    table for all 20 categories per Paragraphs 82(1)-(4) (ranging from
    0.25% to 1.25% Stage 1, 0.25% to 10% Stage 2). Four Stage 3 duration-
    dependent schedules: standard (25/40/55/75/100), deposits-gold-state
    (10/20/30/40/100), unsecured retail (25 / 100), housing-residential
    RE (10/20/30/40/100). Plus `rbi_ecl_floor_2026()`,
    `classify_rbi_exposure_category()`, and DCCO project-finance
    additional provisioning (0.375% / 0.5625% per quarter).
  - `ecl/ind_as109/pd_lgd_floors.py` -- PD floor 0.03% (Paragraph 96),
    LGD backstops 65% secured / 70% unsecured / 30% eligible collateral
    (Paragraphs 97-98) with `apply_rbi_pd_floor()` and
    `apply_rbi_lgd_backstop()`.
  - `ecl/ind_as109/borrower_classification.py` -- Borrower-level Stage 3
    contagion per Paragraphs 8(9), 76. `apply_borrower_level_staging()`
    elevates all facilities of a borrower to Stage 3 if any one is
    Stage 3 (Stage 2 remains facility-level).
  - `ecl/ind_as109/transition.py` -- April 1, 2027 effective date,
    March 31, 2030 EIR migration deadline, capital add-back phase-in
    schedule (4/5 -> 3/5 -> 2/5 -> 1/5) per Paragraph 108. Includes
    `is_ecl_framework_effective()`, `capital_add_back_factor()`,
    `eir_required()`.
  - `ecl/ind_as109/dlg.py` -- Default Loss Guarantee adjustment per
    Paragraph 88. `ecl_with_dlg()` and `DLGAdjustment` dataclass with
    capacity tracking after invocation.
  - `ecl/ind_as109/collateral_valuation.py` -- Collateral revaluation
    compliance per Paragraph 55: Stage 3 exposures >= INR 7.5 crore must
    be revalued every 2 years; stock collateral annually.
  - `ind_as_ecl.py` extended with `assess_sicr_rbi()` (30 DPD backstop
    plus 60-day revolving overlimit per Paragraph 33),
    `determine_upgrade_eligibility()` (Paragraphs 77-79 upgrade paths),
    `calculate_ecl_ind_as_2026()` (end-to-end pipeline applying PD/LGD
    floors, model ECL, DLG, and regulatory floor), and
    `calculate_ecl_ind_as_auto()` (date-based dispatch between legacy
    IRAC and 2026 frameworks).

- **Updated `regulatory/india/rbi.yml`** -- New `ecl_master_direction_2026`
  section with PD/LGD floors, SICR thresholds, transition timeline,
  governance requirements, DCCO rates, full 20-category Stage 1/2
  floor table, and all 4 Stage 3 duration schedules.

### Backward Compatibility

The legacy IRAC framework (`IRACAssetClass`, `classify_irac()`,
`rbi_minimum_provision()`, `calculate_ecl_ind_as()`) is preserved
unchanged. Use `calculate_ecl_ind_as_auto(reporting_date, ...)` for
automatic framework selection based on the reporting date.

### Tests

- 140 new tests in `tests/test_ecl/test_ind_as_2026.py` covering all
  12 implementation areas: category enum and classifier, Stage 1/2
  floors (all 20 categories), Stage 3 duration-dependent schedules,
  PD/LGD floors, SICR rules (DPD + revolving overlimit), borrower-
  level contagion, DCCO provisioning, upgradation criteria,
  transition timeline, DLG adjustment, collateral revaluation,
  end-to-end ECL calculation, and date-based auto-dispatch.
- Total test count: 2,068 -> 2,241 (with audit-grade gap closures).
  mypy --strict clean, ruff clean.

### Audit-grade corrections (post-initial release)

Following an independent audit-grade review against the official RBI
Master Direction text, three factual corrections were applied and nine
additional gaps were closed:

- **Stage 3 Set A unsecured 0-1yr corrected**: 25% -> 40% per Paragraph
  82(5) Set A (the "25/40%" notation denotes secured / unsecured).
- **Other RE-secured Stage 3 schedule**: Added separate Set F schedule
  (15/25/40/55/100% secured) for OTHER_RESIDENTIAL_RE and
  OTHER_COMMERCIAL_RE per Paragraph 82(5) Set F (previously incorrectly
  mapped to housing schedule).
- **RBI reference corrected**: RBI/2026-27/34 -> RBI/DOR/2026-27/398;
  DOR.STR.REC.No.6/21.06.011/2026-27 across all files.
- **Wilful defaulter +5% surcharge**: New `is_wilful_defaulter` parameter
  on the floor function per Paragraph 101(4).
- **Sovereign / SLR carve-out**: New `is_sovereign_slr` parameter returns
  zero ECL per Paragraphs 37-38.
- **IRACP standard-asset provisioning** (`ecl/ind_as109/iracp.py`):
  `StandardAssetSector` enum with 12 sectors and rates per Master Circular
  DOR.STR.REC.9/21.04.048/2025-26 plus Project Finance Directions 2025.
- **Resolution Framework 1.0/2.0 add-ons**: `resolution_framework_addon()`
  applying 10% on restructured residual debt + 5% on slippage per
  DOR.No.BP.BC/3/21.04.048/2020-21 and DOR.STR.REC.12/21.04.048/2021-22.
- **Out-of-order CC/OD (3 conditions)**: `is_out_of_order()` per
  DOR.STR.REC.68/21.04.048/2021-22.
- **NBFC Ind AS 109 prudential backstop** (`ecl/ind_as109/nbfc_backstop.py`):
  `apply_nbfc_backstop()` returning max(ECL, IRACP) plus Impairment Reserve
  per DOR(NBFC).CC.PD.No.109/22.10.106/2019-20.
- **SBR NPA glide-path**: `npa_dpd_threshold()` returning 180 -> 150 -> 120
  -> 90 DPD by `as_of_date` per DOR.CRE.REC.No.60/03.10.001/2021-22.
- **NBFC-UL differential standard-asset rates** per DOR.STR.REC.40/
  21.04.048/2022-23.
- **NBFC Layer enum** (Base/Middle/Upper/Top) per SBR Master Direction
  October 19, 2023.

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
