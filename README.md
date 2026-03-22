# CreditRiskEngine

Production-grade open-source credit risk analytics library.

**The scikit-learn of credit risk.**

## Features

- **RWA Calculation**: Basel III/CRR3 Standardized and IRB approaches
- **ECL Engines**: IFRS 9, CECL (ASC 326), Ind AS 109
- **PD/LGD/EAD Modeling**: Scorecard development, calibration, validation
- **Model Validation**: Discrimination, calibration, stability tests
- **Portfolio Risk**: Vasicek ASRF, Gaussian copula, stress testing
- **Multi-Jurisdiction**: EU CRR3, UK PRA, US Endgame, RBI, MAS, HKMA, JFSA, APRA, OSFI, and more

## Installation

```bash
pip install creditriskengine
```

## Quick Start

```python
from creditriskengine.rwa.irb.formulas import irb_risk_weight

# Corporate exposure: PD=1%, LGD=45%, M=2.5 years
rw = irb_risk_weight(pd=0.01, lgd=0.45, asset_class="corporate", maturity=2.5)
print(f"Risk Weight: {rw:.2f}%")
```

## License

Apache 2.0
