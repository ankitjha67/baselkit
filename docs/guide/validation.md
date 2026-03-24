# Model Validation

Comprehensive toolkit for PD model validation per EBA GL/2017/16.

## Discrimination

```python
from creditriskengine.validation.discrimination import auroc, gini_coefficient, ks_statistic

auc = auroc(y_true, y_score)
gini = gini_coefficient(y_true, y_score)
ks = ks_statistic(y_true, y_score)
```

## Calibration

```python
from creditriskengine.validation.calibration import binomial_test, hosmer_lemeshow_test

result = binomial_test(n_defaults=15, n_observations=1000, predicted_pd=0.02)
```

## Stability

```python
from creditriskengine.validation.stability import population_stability_index

psi = population_stability_index(base_distribution, current_distribution)
```

::: creditriskengine.validation
