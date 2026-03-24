# IRB Risk Weights

The Internal Ratings-Based (IRB) approach allows banks to use internal
estimates of PD, LGD, and EAD for capital calculations.

## Basic Usage

```python
from creditriskengine.rwa.irb.formulas import irb_risk_weight

# Corporate exposure
rw = irb_risk_weight(pd=0.01, lgd=0.45, asset_class="corporate", maturity=2.5)
print(f"Corporate RW: {rw:.2f}%")  # ~75%

# SME with firm-size adjustment
rw_sme = irb_risk_weight(
    pd=0.01, lgd=0.45, asset_class="corporate",
    maturity=2.5, turnover_eur_millions=15.0
)
print(f"SME RW: {rw_sme:.2f}%")  # Lower than general corporate

# Residential mortgage
rw_rre = irb_risk_weight(pd=0.005, lgd=0.15, asset_class="residential_mortgage")

# QRRE with transactor scalar
rw_qrre = irb_risk_weight(
    pd=0.02, lgd=0.80, asset_class="qrre", is_qrre_transactor=True
)
```

## Asset Classes

| Asset Class | Correlation | Maturity Adj | Reference |
|-------------|-----------|------------|-----------|
| `corporate` | R ∈ [0.12, 0.24] | Yes | CRE31.5 |
| `sovereign` | Same as corporate | Yes | CRE31.5 |
| `bank` | Same as corporate | Yes | CRE31.5 |
| `residential_mortgage` | R = 0.15 (fixed) | No | CRE31.8 |
| `qrre` | R = 0.04 (fixed) | No | CRE31.9 |
| `other_retail` | R ∈ [0.03, 0.16] | No | CRE31.10 |

## API Reference

::: creditriskengine.rwa.irb.formulas
