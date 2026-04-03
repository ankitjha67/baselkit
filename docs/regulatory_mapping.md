# Regulatory Mapping

This document maps CreditRiskEngine modules and functions to their
authoritative regulatory sources.

## Basel III / BCBS d424

| BCBS Reference | Topic | Module | Function(s) |
|---|---|---|---|
| CRE20.4-20.7 | Sovereign SA RW | `rwa/standardized/credit_risk_sa.py` | `get_sovereign_risk_weight()` |
| CRE20.15-20.18 | Bank ECRA RW | `rwa/standardized/credit_risk_sa.py` | `get_bank_risk_weight()` |
| CRE20.28-20.32 | Corporate SA RW | `rwa/standardized/credit_risk_sa.py` | `get_corporate_risk_weight()` |
| CRE20.54-20.62 | Retail SA RW | `rwa/standardized/credit_risk_sa.py` | `get_retail_risk_weight()` |
| CRE20.71-20.86 | RRE SA RW (LTV) | `rwa/standardized/credit_risk_sa.py` | `get_residential_re_risk_weight()` |
| CRE20.87-20.95 | CRE SA RW (LTV) | `rwa/standardized/credit_risk_sa.py` | `get_commercial_re_risk_weight()` |
| CRE31.4 | IRB capital K | `rwa/irb/formulas.py` | `irb_capital_requirement()` |
| CRE31.5 | Corporate/Sov/Bank ρ | `rwa/irb/formulas.py` | `correlation_corporate_sovereign_bank()` |
| CRE31.6 | SME firm-size adj | `rwa/irb/formulas.py` | `correlation_corporate_sovereign_bank()` |
| CRE31.7 | Maturity adjustment | `rwa/irb/formulas.py` | `maturity_adjustment()` |
| CRE31.8 | Residential mortgage ρ | `rwa/irb/formulas.py` | `correlation_residential_mortgage()` |
| CRE31.9 | QRRE ρ | `rwa/irb/formulas.py` | `correlation_qrre()` |
| CRE31.10 | Other retail ρ | `rwa/irb/formulas.py` | `correlation_other_retail()` |
| CRE32.13 | PD floor (3 bps) | `rwa/irb/formulas.py` | `PD_FLOOR = 0.0003` |
| CRE32.22-24 | Supervisory LGD (F-IRB) | `models/lgd/lgd_model.py` | `SUPERVISORY_LGD_*` constants |
| CRE32.25 | LGD floors (A-IRB) | `models/lgd/lgd_model.py` | `apply_lgd_floor()` |
| CRE32.29-32 | Supervisory CCF (F-IRB) | `models/ead/ead_model.py` | `SUPERVISORY_CCFS`, `get_supervisory_ccf()` |
| CRE32.33 | CCF floor (A-IRB) | `models/ead/ead_model.py` | `apply_ccf_floor()` |
| CRE20 Table 2 | SA CCFs (Basel III/CRR3) | `models/ead/ead_model.py` | `SA_CCFS`, `get_sa_ccf()` |
| CRE32.33 / CRR3 Art. 166(8b) | A-IRB CCF input floor (50% of SA) | `models/ead/ead_model.py` | `get_airb_ccf_floor()` |
| CRE20.19-20.21 | Bank SCRA RW | `rwa/standardized/credit_risk_sa.py` | `get_bank_risk_weight(scra_grade=)` |
| CRE22.35-22.39 | CRM simple approach | `rwa/crm.py` | `simple_approach()` |
| CRE22.40-22.56 | CRM supervisory haircuts | `rwa/crm.py` | `supervisory_haircut()` |
| CRE22.57-22.77 | CRM comprehensive approach | `rwa/crm.py` | `comprehensive_approach()` |
| CRE22.78-22.93 | CRM guarantees | `rwa/crm.py` | `guarantee_substitution()` |
| CRE22.33 | CRM maturity mismatch | `rwa/crm.py` | `maturity_mismatch_adjustment()` |
| CRE31.9 fn 15 | QRRE transactor scalar | `rwa/irb/formulas.py` | `irb_risk_weight(is_qrre_transactor=)` |
| OPE25.8-25.13 | Operational risk SMA | `rwa/operational_risk.py` | `sma_capital()`, `calculate_bic()` |
| MAR (FRTB) | Market risk SbM/DRC | `rwa/market_risk.py` | `calculate_sa_market_risk()` |
| RBC25.2-25.4 | Output floor | `rwa/output_floor.py` | `OutputFloorCalculator` |

## IFRS 9

| IFRS Reference | Topic | Module | Function(s) |
|---|---|---|---|
| IFRS 9.5.5.1-9.5.5.5 | Stage assignment | `ecl/ifrs9/staging.py` | `assign_stage()` |
| IFRS 9.5.5.9-9.5.5.12 | SICR assessment | `ecl/ifrs9/sicr.py` | `assess_sicr()` |
| IFRS 9.5.5.15-9.5.5.16 | 12-month ECL (Stage 1) | `ecl/ifrs9/ecl_calc.py` | `ecl_12_month()` |
| IFRS 9.5.5.3 | Lifetime ECL (Stage 2/3) | `ecl/ifrs9/ecl_calc.py` | `ecl_lifetime()` |
| IFRS 9.5.5.17(c) | Forward-looking info | `ecl/ifrs9/forward_looking.py` | `macro_adjustment_factor()` |
| IFRS 9.B5.5.49-54 | Scenario weighting | `ecl/ifrs9/scenarios.py` | `weighted_ecl()` |
| IFRS 9.5.4.1 | POCI treatment | `ecl/ifrs9/staging.py` | `assign_stage()` (POCI branch) |
| IFRS 9.5.5.20 | Revolving credit ECL exception | `ecl/ifrs9/revolving/ecl_revolving.py` | `calculate_revolving_ecl()` |
| IFRS 9.B5.5.39-40 | Behavioral life (3-factor) | `ecl/ifrs9/revolving/behavioral_life.py` | `determine_behavioral_life()` |
| IFRS 9.B5.5.31 | Drawdown expectations (CCF) | `ecl/ifrs9/revolving/ccf.py` | `behavioral_ccf()`, `eadf_ccf()`, `ccf_pit_adjustment()` |
| IFRS 7.B8E | Drawn/undrawn ECL split | `ecl/ifrs9/revolving/ecl_revolving.py` | `RevolvingECLResult.ecl_drawn`, `.ecl_undrawn` |
| IFRS 9.5.5.17 | Scenario-weighted ECL | `ecl/ifrs9/revolving/ecl_revolving.py` | `revolving_ecl_scenario_weighted()` |

## Management Overlays & Governance Layer

| Reference | Topic | Module | Function(s) |
|---|---|---|---|
| IFRS 9.B5.5.1 | Reasonable and supportable information | `ecl/ifrs9/overlays.py` | `ManagementOverlay` (rationale field) |
| IFRS 9.B5.5.52 | Adjustments for current conditions and forecasts | `ecl/ifrs9/overlays.py` | `apply_overlays()` |
| IFRS 9.5.5.17(c) | Forward-looking information requirement | `ecl/ifrs9/overlays.py` | `OverlayType.ECONOMIC_UNCERTAINTY` |
| ECB "Letter to banks on IFRS 9" (Dec 2020) | COVID-19 overlay guidance | `ecl/ifrs9/overlays.py` | `OverlayType.TEMPORARY_EVENT` |
| EBA/GL/2020/06 para 25-28 | Overlay governance and disclosure | `ecl/ifrs9/overlays.py` | `validate_overlay()` |
| EBA/GL/2017/06 para 74-75 | Credit risk management and ECL accounting | `ecl/ifrs9/overlays.py`, `ecl/ifrs9/scenarios.py` | `validate_overlay()`, `validate_scenario_governance()` |
| PRA Dear CFO letter (Jul 2020) | Overlay documentation expectations | `ecl/ifrs9/overlays.py` | `validate_overlay()` (expiry check) |
| IFRS 7.35F-35L | Disclosure of ECL measurement | `ecl/ifrs9/overlays.py` | `overlay_impact_summary()` |
| IFRS 9.B5.5.41-B5.5.43 | Range of possible outcomes / scenario design | `ecl/ifrs9/scenarios.py` | `ScenarioSetMetadata`, `validate_scenario_governance()` |
| IFRS 9.B5.5.43 | Evaluation of range of scenarios | `ecl/ifrs9/scenarios.py` | `scenario_sensitivity_analysis()` |
| BCBS "COVID-19 measures" (Apr 2020) | Overlay expectations | `ecl/ifrs9/overlays.py` | `OverlayType.TEMPORARY_EVENT` |
| PRA SS1/23 | Model risk management expectations | `ecl/ifrs9/scenarios.py` | `validate_scenario_governance()` |
| OCC 2011-12 / SR 11-7 | Model risk management — post-model adjustments | `core/audit.py` | `AuditTrail.record_overlay()`, `OverlayAuditRecord` |

## ASC 326 (CECL)

| ASC Reference | Topic | Module | Function(s) |
|---|---|---|---|
| ASC 326-20-30-2 | Lifetime ECL from day 1 | `ecl/cecl/cecl_calc.py` | `cecl_ecl_pd_lgd()` |
| ASC 326-20-30-5 | Loss-rate method | `ecl/cecl/cecl_calc.py` | `cecl_ecl_loss_rate()` |
| ASC 326-20-30-9 | Q-factor adjustments | `ecl/cecl/qualitative.py` | `apply_q_factors()` |
| Interagency guidance | WARM method | `ecl/cecl/methods.py` | `warm_method()` |
| Interagency guidance | Vintage analysis | `ecl/cecl/methods.py` | `vintage_analysis()` |

## Ind AS 109

| Reference | Topic | Module | Function(s) |
|---|---|---|---|
| RBI Master Circular | 90 DPD NPA threshold | `ecl/ind_as109/ind_as_ecl.py` | `RBI_DEFAULT_DPD_THRESHOLD = 90` |
| Ind AS 109 (= IFRS 9) | Stage assignment | `ecl/ind_as109/ind_as_ecl.py` | `assign_stage_ind_as()` |

## Revolving Credit CCF -- Regulatory SA by Jurisdiction

| Jurisdiction | Regulator | UCC CCF | Committed CCF | Module | Function |
|---|---|---|---|---|---|
| BCBS | Basel Committee | 10% | 40% | `ecl/ifrs9/revolving/ccf.py` | `regulatory_ccf_sa()` |
| EU | EBA (CRR3) | 10% (0% transitional to 2029) | 40% | `ecl/ifrs9/revolving/ccf.py` | `regulatory_ccf_sa(use_crr3_transitional=)` |
| Australia | APRA (APS 112) | **40%** | 40% | `ecl/ifrs9/revolving/ccf.py` | `regulatory_ccf_sa(jurisdiction=AUSTRALIA)` |

References: BCBS d424 Table 2, CRR3 Art. 495d, APRA APS 112.

## Revolving Credit Provision Floors

| Jurisdiction | Stage | Floor | Basis | Module | Reference |
|---|---|---|---|---|---|
| UAE (CBUAE) | S1+S2 combined | 1.5% | CRWA | `ecl/ifrs9/revolving/provision_floors.py` | Circular 3/2024 |
| India (RBI) | Stage 1 | 1.0% | EAD | `ecl/ifrs9/revolving/provision_floors.py` | Draft Directions Oct 2025 |
| India (RBI) | Stage 2 | 5.0% | EAD | `ecl/ifrs9/revolving/provision_floors.py` | Draft Directions Oct 2025 |
| Singapore (MAS) | Cross-stage | 1.0% | EAD | `ecl/ifrs9/revolving/provision_floors.py` | MAS Notice 612 |
| Saudi Arabia (SAMA) | Cross-stage | 1.0% | CRWA | `ecl/ifrs9/revolving/provision_floors.py` | General provision |

Floor configs loaded from `regulatory/provision_floors.yml`.

## FR 2052a -- Complex Institution Liquidity Monitoring

| FR 2052a Reference | Topic | Module | Function(s) |
|---|---|---|---|
| General Instructions | Submission container | `reporting/fr2052a/report.py` | `FR2052aSubmission`, `build_submission()` |
| Field Definitions (pp. 16-32) | 23 enum types | `reporting/fr2052a/types.py` | `CounterpartyType`, `AssetCategory`, `MaturityBucket`, etc. |
| Product Definitions (pp. 33-79) | 137 products | `reporting/fr2052a/products.py` | `ALL_PRODUCTS`, `get_product()` |
| Appendix I | 13 table schemas | `reporting/fr2052a/schemas.py` | `InflowAssetRecord`, `OutflowDepositRecord`, etc. |
| Appendix II-a | Sub-product requirements | `reporting/fr2052a/products.py` | `FR2052aProduct.sub_products` |
| Appendix II-b | Counterparty requirements | `reporting/fr2052a/products.py` | `FR2052aProduct.counterparty_required` |
| Appendix II-c | Collateral class requirements | `reporting/fr2052a/products.py` | `FR2052aProduct.collateral_required` |
| Appendix II-d | Forward start exclusions | `reporting/fr2052a/products.py` | `FR2052aProduct.forward_start_excluded` |
| Appendix III | Asset category table (91 codes) | `reporting/fr2052a/types.py` | `AssetCategory`, `HQLA_LEVEL_1/2A/2B` |
| Appendix IV-a | Maturity bucket values (76) | `reporting/fr2052a/types.py` | `MaturityBucket` |
| 12 CFR 249 (Reg WW) | LRM Standards | `reporting/fr2052a/validation.py` | `validate_record()`, `validate_submission()` |

## Jurisdiction Configs

17 YAML config files in `regulatory/` provide jurisdiction-specific parameters:

| Jurisdiction | Regulator | Config File | Key Differences |
|---|---|---|---|
| BCBS | Basel Committee | `bcbs/bcbs_d424.yml` | Reference standard |
| EU | EBA | `eu/crr3.yml` | SME supporting factor, 2-year delayed floor |
| UK | PRA | `uk/pra_basel31.yml` | Further delayed floor (2027+), 65% unrated IG |
| US | Fed/OCC/FDIC | `us/us_endgame.yml` | SA-only for most banks |
| India | RBI | `india/rbi.yml` | 80% output floor, 90 DPD all categories |
| Singapore | MAS | `singapore/mas_637.yml` | BCBS-aligned schedule |
| Hong Kong | HKMA | `hongkong/hkma.yml` | BCBS-aligned schedule |
| Japan | JFSA | `japan/jfsa.yml` | March fiscal year |
| Australia | APRA | `australia/apra.yml` | 72.5% floor immediately, no F-IRB |
| Canada | OSFI | `canada/osfi.yml` | 72.5% floor from Q2 2024 |
| China | NFRA | `china/nfra.yml` | Five-tier asset classification |
| South Korea | FSS | `southkorea/fss.yml` | Jeonse loan treatment |
| UAE | CBUAE | `uae/cbuae.yml` | Higher minimum capital (CET1 7%) |
| Saudi Arabia | SAMA | `saudi/sama.yml` | Islamic finance mapping |
| South Africa | SARB | `southafrica/sarb.yml` | 4% leverage ratio |
| Brazil | BCB | `brazil/bcb.yml` | 9-tier risk classification |
| Malaysia | BNM | `malaysia/bnm.yml` | Dual conventional/Islamic reporting |

## Model Validation References

| Reference | Topic | Module |
|---|---|---|
| BCBS WP14 (2005) | PD backtesting | `validation/backtesting.py` |
| OCC 2011-12 / SR 11-7 | Model risk management | `validation/stability.py` |
| ECB Guide to Internal Models | PSI/CSI monitoring | `validation/stability.py` |
| EBA GL/2017/16 | PD/LGD estimation | `models/pd/scorecard.py`, `models/lgd/lgd_model.py` |
| Gordy (2003) | Granularity Adjustment | `models/concentration/concentration.py` |
| Merton (1974) / Vasicek (2002) | ASRF model | `portfolio/vasicek.py`, `ecl/ifrs9/ttc_to_pit.py` |
| Araten & Jacobs (2001) | LEQ / behavioral CCF | `ecl/ifrs9/revolving/ccf.py` |
| Tong et al. (2016) | Credit card CCF distribution | `ecl/ifrs9/revolving/product_config.py` |
| PwC "In Depth" (Nov 2017) | Revolving ECL FAQs (B5.5.40) | `ecl/ifrs9/revolving/behavioral_life.py` |
| IASB Feb 2017 staff paper | Shortest B5.5.40 factor rule | `ecl/ifrs9/revolving/behavioral_life.py` |
| EBA/CP/2025/10 | CCF estimation guidelines | `ecl/ifrs9/revolving/ccf.py` |
