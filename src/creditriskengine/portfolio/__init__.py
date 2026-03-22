"""Portfolio credit risk analytics.

Provides Vasicek ASRF model, copula-based Monte Carlo simulation,
economic capital, Credit VaR, and stress testing frameworks.
"""

from creditriskengine.portfolio.copula import simulate_single_factor
from creditriskengine.portfolio.economic_capital import ec_single_factor
from creditriskengine.portfolio.stress_testing import (
    BoEACSStressTest,
    CCARScenario,
    EBAStressTest,
    MacroScenario,
    apply_pd_stress,
)
from creditriskengine.portfolio.var import (
    expected_shortfall,
    historical_simulation_var,
    parametric_credit_var,
)
from creditriskengine.portfolio.vasicek import (
    economic_capital_asrf,
    vasicek_loss_quantile,
)

__all__ = [
    "vasicek_loss_quantile",
    "economic_capital_asrf",
    "simulate_single_factor",
    "ec_single_factor",
    "parametric_credit_var",
    "historical_simulation_var",
    "expected_shortfall",
    "MacroScenario",
    "EBAStressTest",
    "BoEACSStressTest",
    "CCARScenario",
    "apply_pd_stress",
]
