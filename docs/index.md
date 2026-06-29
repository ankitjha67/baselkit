# CreditRiskEngine

**Production-grade open-source credit risk analytics.**

CreditRiskEngine provides a complete, auditable implementation of Basel III/IV
capital calculations, IFRS 9/CECL expected credit loss engines, and model
validation toolkits — covering the full credit risk lifecycle from PD estimation
through regulatory reporting.

## Key Features

| Domain | Capabilities |
|--------|-------------|
| **RWA Calculation** | SA (CRE20), F-IRB, A-IRB (CRE31-32), output floor (RBC25), CRM (CRE22) |
| **ECL Engines** | IFRS 9 (staging, SICR, lifetime PD, **revolving credit with drawn/undrawn split**), CECL (ASC 326), Ind AS 109 |
| **PD/LGD/EAD** | Scorecard, calibration, master scale, workout/downturn LGD, CCF, contractual EAD amortisation schedules |
| **Model Validation** | AUROC, Gini, KS, PSI, binomial test, Hosmer-Lemeshow, traffic light |
| **AI/ML Governance** | Fairness metrics, PSI/population drift monitoring |
| **Portfolio Risk** | Vasicek ASRF, copula Monte Carlo, Credit VaR, stress testing |
| **Operational Risk** | Basel III SMA (OPE25), operational resilience (DORA) |
| **Market Risk** | FRTB IMA — expected shortfall (97.5%), PLAT, DRC, NMRF (MAR) |
| **Counterparty Risk** | EPE/EEPE/PFE exposure, wrong-way risk |
| **IRRBB** | EVE/NII sensitivity, supervisory outlier test |
| **Risk-Based Pricing** | RAROC/EVA, Euler/ES capital allocation |
| **Climate & ESG** | NGFS scenarios, PCAF financed emissions, GAR, ESG ratings adapter, EBA/GL/2025/01 materiality assessment + transition-plan monitoring |
| **Jurisdictions** | 17 supported: BCBS, EU, UK, US, India, Singapore, + 11 more |

## Quick Start

```python
from creditriskengine.rwa.irb.formulas import irb_risk_weight

# Corporate: PD=1%, LGD=45%, maturity=2.5y
rw = irb_risk_weight(pd=0.01, lgd=0.45, asset_class="corporate", maturity=2.5)
print(f"Risk Weight: {rw:.2f}%")  # ~75%
```

See [Getting Started](getting_started.md) for installation and detailed examples.

!!! warning "Regulatory Disclaimer"
    This library is provided for **educational and analytical purposes**.
    See [Regulatory Disclaimers](regulatory_disclaimers.md) before use in
    production capital calculations.
