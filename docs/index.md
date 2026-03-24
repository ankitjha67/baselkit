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
| **ECL Engines** | IFRS 9 (staging, SICR, lifetime PD), CECL (ASC 326), Ind AS 109 |
| **PD/LGD/EAD** | Scorecard, calibration, master scale, workout/downturn LGD, CCF |
| **Model Validation** | AUROC, Gini, KS, PSI, binomial test, Hosmer-Lemeshow, traffic light |
| **Portfolio Risk** | Vasicek ASRF, copula Monte Carlo, Credit VaR, stress testing |
| **Operational Risk** | Basel III SMA (OPE25) |
| **Market Risk** | FRTB integration point (MAR) |
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
