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
