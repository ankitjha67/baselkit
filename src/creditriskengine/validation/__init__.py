"""Model validation and backtesting framework.

Provides discrimination, calibration, stability, and backtesting
metrics for credit risk model validation per SR 11-7, EBA GL/2017/16.
"""

from creditriskengine.validation.backtesting import pd_backtest_full, pd_backtest_summary
from creditriskengine.validation.calibration import (
    binomial_test,
    brier_score,
    hosmer_lemeshow_test,
    traffic_light_test,
)
from creditriskengine.validation.discrimination import auroc, gini_coefficient, ks_statistic
from creditriskengine.validation.stability import (
    characteristic_stability_index,
    population_stability_index,
)

__all__ = [
    "auroc",
    "gini_coefficient",
    "ks_statistic",
    "binomial_test",
    "hosmer_lemeshow_test",
    "traffic_light_test",
    "brier_score",
    "population_stability_index",
    "characteristic_stability_index",
    "pd_backtest_summary",
    "pd_backtest_full",
]
