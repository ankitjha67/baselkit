"""
Macro stress testing framework.

Supports EBA, BoE ACS, US CCAR/DFAST, and RBI methodologies.
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class MacroScenario:
    """A macroeconomic stress scenario."""

    def __init__(
        self,
        name: str,
        horizon_years: int = 3,
        variables: dict[str, np.ndarray] | None = None,
        severity: str = "adverse",
    ) -> None:
        self.name = name
        self.horizon_years = horizon_years
        self.variables = variables or {}
        self.severity = severity


def apply_pd_stress(
    base_pds: np.ndarray,
    stress_multiplier: float,
    pd_cap: float = 1.0,
) -> np.ndarray:
    """Apply stress multiplier to PDs.

    Args:
        base_pds: Baseline PD array.
        stress_multiplier: Multiplicative stress factor.
        pd_cap: Maximum PD cap.

    Returns:
        Stressed PD array.
    """
    stressed = base_pds * stress_multiplier
    return np.minimum(stressed, pd_cap)


def apply_lgd_stress(
    base_lgds: np.ndarray,
    stress_add_on: float,
    lgd_cap: float = 1.0,
) -> np.ndarray:
    """Apply additive stress to LGDs.

    Args:
        base_lgds: Baseline LGD array.
        stress_add_on: Additive stress increase.
        lgd_cap: Maximum LGD cap.

    Returns:
        Stressed LGD array.
    """
    stressed = base_lgds + stress_add_on
    return np.clip(stressed, 0.0, lgd_cap)


def stress_test_rwa_impact(
    base_rwa: float,
    stressed_rwa: float,
) -> dict[str, float]:
    """Calculate RWA impact from stress test.

    Args:
        base_rwa: Baseline RWA.
        stressed_rwa: Stressed RWA.

    Returns:
        Dict with absolute and relative impact.
    """
    delta = stressed_rwa - base_rwa
    pct = delta / base_rwa if base_rwa > 0 else 0.0

    return {
        "base_rwa": base_rwa,
        "stressed_rwa": stressed_rwa,
        "delta_rwa": delta,
        "pct_change": pct,
    }


# ============================================================
# Predefined Scenario Library
# ============================================================


def scenario_library() -> dict[str, MacroScenario]:
    """Predefined macroeconomic stress scenarios.

    Provides a set of standard scenarios ranging from baseline to severely
    adverse, consistent with severity gradations used in EBA, CCAR, and
    RBI stress testing frameworks.

    Returns:
        Dict mapping scenario name to MacroScenario object.
        Scenarios provided:
            - 'baseline': Steady-state growth (GDP +2%, unemployment 4-4.5%)
            - 'mild_downturn': GDP -0.5% to +1.5%, unemployment 6-6.5%
            - 'moderate_recession': GDP -3% to +1%, unemployment rising 3pp
            - 'severe_recession': GDP -4% to +1%, unemployment 9-11%
            - 'stagflation': GDP -2%, inflation +4pp, rates +300bps
            - 'sovereign_crisis': GDP -5%, spreads +500bps, FX -20%
    """
    scenarios: dict[str, MacroScenario] = {
        "baseline": MacroScenario(
            name="Baseline",
            horizon_years=3,
            variables={
                "gdp_growth": np.array([0.02, 0.02, 0.02]),
                "unemployment": np.array([0.045, 0.04, 0.04]),
                "house_price_index": np.array([0.03, 0.03, 0.03]),
            },
            severity="baseline",
        ),
        "mild_downturn": MacroScenario(
            name="Mild Downturn",
            horizon_years=3,
            variables={
                "gdp_growth": np.array([-0.005, 0.005, 0.015]),
                "unemployment": np.array([0.06, 0.065, 0.06]),
                "house_price_index": np.array([-0.03, -0.01, 0.02]),
                "interest_rate_change_bps": np.array([0, -25, -25]),
            },
            severity="mild",
        ),
        "moderate_recession": MacroScenario(
            name="Moderate Recession",
            horizon_years=3,
            variables={
                "gdp_growth": np.array([-0.03, -0.01, 0.01]),
                "unemployment": np.array([0.07, 0.085, 0.08]),
                "house_price_index": np.array([-0.10, -0.05, 0.0]),
                "interest_rate_change_bps": np.array([0, -50, -50]),
            },
            severity="adverse",
        ),
        "severe_recession": MacroScenario(
            name="Severe Recession",
            horizon_years=3,
            variables={
                "gdp_growth": np.array([-0.04, -0.02, 0.01]),
                "unemployment": np.array([0.09, 0.11, 0.10]),
                "house_price_index": np.array([-0.15, -0.10, -0.03]),
                "equity_market_change": np.array([-0.35, -0.10, 0.05]),
            },
            severity="severely_adverse",
        ),
        "stagflation": MacroScenario(
            name="Stagflation",
            horizon_years=3,
            variables={
                "gdp_growth": np.array([-0.02, -0.01, 0.005]),
                "unemployment": np.array([0.07, 0.08, 0.075]),
                "inflation_change_pp": np.array([4.0, 3.0, 1.0]),
                "interest_rate_change_bps": np.array([300, 100, -50]),
            },
            severity="adverse",
        ),
        "sovereign_crisis": MacroScenario(
            name="Sovereign Crisis",
            horizon_years=3,
            variables={
                "gdp_growth": np.array([-0.05, -0.025, 0.0]),
                "unemployment": np.array([0.10, 0.115, 0.10]),
                "sovereign_spread_change_bps": np.array([500, 300, 100]),
                "fx_depreciation": np.array([-0.20, -0.10, -0.03]),
                "house_price_index": np.array([-0.15, -0.10, -0.03]),
            },
            severity="severely_adverse",
        ),
    }

    logger.debug("Scenario library loaded: %d scenarios.", len(scenarios))
    return scenarios


# ============================================================
# Multi-Period Projection
# ============================================================


def multi_period_projection(
    base_pds: np.ndarray,
    base_lgds: np.ndarray,
    base_eads: np.ndarray,
    pd_multipliers: np.ndarray,
    lgd_add_ons: np.ndarray,
    amortisation_rates: np.ndarray | None = None,
) -> dict[str, Any]:
    """Project credit risk parameters over multiple periods with time-step simulation.

    Applies period-specific stress factors to compute stressed PD, LGD, EAD,
    and expected loss projections. Optionally accounts for portfolio
    amortisation (run-off) across periods.

    Useful for through-the-cycle projections in IFRS 9 ECL, CCAR, and EBA
    contexts. When amortisation_rates is None, a static balance sheet is
    assumed (EBA convention).

    Reference:
        - EBA Methodological Note (static balance sheet)
        - IFRS 9 B5.5.13 (lifetime ECL projection)

    Args:
        base_pds: Baseline PDs (n_exposures,).
        base_lgds: Baseline LGDs (n_exposures,).
        base_eads: Baseline EADs (n_exposures,).
        pd_multipliers: PD stress multipliers per period (n_periods,).
        lgd_add_ons: LGD additive stress per period (n_periods,).
        amortisation_rates: Optional per-period amortisation rate (n_periods,).
            Each entry is the fraction of EAD that amortises away per period.
            Defaults to zero (static balance sheet).

    Returns:
        Dict with:
            - 'stressed_pds': (n_periods, n_exposures) stressed PD matrix
            - 'stressed_lgds': (n_periods, n_exposures) stressed LGD matrix
            - 'expected_losses': (n_periods, n_exposures) per-exposure EL
            - 'period_el': (n_periods,) total EL per period
            - 'period_eads': (n_periods,) total outstanding EAD per period
            - 'cumulative_el': float cumulative expected loss across all periods

    Raises:
        ValueError: If array lengths are inconsistent.
    """
    base_pds = np.asarray(base_pds, dtype=np.float64)
    base_lgds = np.asarray(base_lgds, dtype=np.float64)
    base_eads = np.asarray(base_eads, dtype=np.float64)
    pd_multipliers = np.asarray(pd_multipliers, dtype=np.float64)
    lgd_add_ons = np.asarray(lgd_add_ons, dtype=np.float64)

    n_periods = len(pd_multipliers)
    n_exposures = len(base_pds)

    if len(lgd_add_ons) != n_periods:
        raise ValueError("lgd_add_ons length must match pd_multipliers length.")

    if amortisation_rates is not None:
        amortisation_rates = np.asarray(amortisation_rates, dtype=np.float64)
        if len(amortisation_rates) != n_periods:
            raise ValueError("amortisation_rates length must match n_periods.")
    else:
        amortisation_rates = np.zeros(n_periods)

    stressed_pds = np.zeros((n_periods, n_exposures))
    stressed_lgds = np.zeros((n_periods, n_exposures))
    expected_losses = np.zeros((n_periods, n_exposures))
    period_el = np.zeros(n_periods)
    period_eads = np.zeros(n_periods)

    current_eads = base_eads.copy()

    for t in range(n_periods):
        stressed_pds[t] = np.minimum(base_pds * pd_multipliers[t], 1.0)
        stressed_lgds[t] = np.clip(base_lgds + lgd_add_ons[t], 0.0, 1.0)
        expected_losses[t] = stressed_pds[t] * stressed_lgds[t] * current_eads
        period_el[t] = float(expected_losses[t].sum())
        period_eads[t] = float(current_eads.sum())

        # Apply amortisation and default write-off for next period
        default_writeoff = stressed_pds[t] * current_eads
        amort = amortisation_rates[t] * current_eads
        current_eads = np.maximum(current_eads - default_writeoff - amort, 0.0)

    cumulative_el = float(period_el.sum())

    logger.debug(
        "Multi-period projection: %d periods, %d exposures, total EL=%.2f",
        n_periods, n_exposures, cumulative_el,
    )

    return {
        "stressed_pds": stressed_pds,
        "stressed_lgds": stressed_lgds,
        "expected_losses": expected_losses,
        "period_el": period_el,
        "period_eads": period_eads,
        "cumulative_el": cumulative_el,
    }


# ============================================================
# EBA Stress Test Framework
# ============================================================


class EBAStressTest:
    """EBA stress test framework -- constrained bottom-up approach.

    Implements the methodology used in the EU-wide stress testing exercise
    coordinated by the European Banking Authority.

    Reference:
        - EBA Methodological Note (latest: 2023/2025 exercise)
        - EBA GL/2018/04 on institutions' stress testing

    Key features:
        - Static balance sheet assumption (portfolio composition frozen).
        - 3-year projection horizon (baseline and adverse).
        - Constrained bottom-up: banks use own models, EBA provides macro
          scenario and prescriptive constraints on key parameters.
        - PD/LGD shifts derived from macro scenario translation.
        - Regulatory PD floor applied (CRR Art. 160).

    Args:
        scenario: Macro scenario with at least 3 years of projections.
        horizon_years: Projection horizon (default 3, EBA standard).
        static_balance_sheet: Whether to enforce static balance sheet.
        pd_floor: Regulatory PD floor (CRR Art. 160: 0.03% for corporate).
    """

    def __init__(
        self,
        scenario: MacroScenario,
        horizon_years: int = 3,
        static_balance_sheet: bool = True,
        pd_floor: float = 0.0003,
    ) -> None:
        if horizon_years < 3:
            raise ValueError("EBA stress test requires a minimum 3-year horizon.")
        self.scenario = scenario
        self.horizon_years = horizon_years
        self.static_balance_sheet = static_balance_sheet
        self.pd_floor = pd_floor
        logger.info(
            "EBAStressTest initialised: scenario='%s', horizon=%d years, "
            "static_bs=%s, pd_floor=%.4f",
            scenario.name,
            horizon_years,
            static_balance_sheet,
            pd_floor,
        )

    def translate_macro_to_pd_stress(
        self,
        base_pds: np.ndarray,
        gdp_sensitivity: float = 2.0,
    ) -> np.ndarray:
        """Translate macro scenario to PD stress multipliers.

        Simple linear translation: multiplier = 1 - sensitivity * gdp_growth_deviation.

        Args:
            base_pds: Baseline PDs (unused, for interface consistency).
            gdp_sensitivity: Sensitivity of PD to GDP growth deviation.

        Returns:
            PD multipliers per period (shape: horizon_years,).
        """
        gdp = self.scenario.variables.get("gdp_growth", np.zeros(self.horizon_years))
        baseline_gdp = 0.02  # Assumed baseline GDP growth
        multipliers = 1.0 - gdp_sensitivity * (gdp - baseline_gdp)
        return np.maximum(multipliers, 1.0)  # Floor at 1.0 (no benefit from growth)

    def translate_macro_to_lgd_stress(self) -> np.ndarray:
        """Translate macro scenario to LGD add-ons.

        Driven by house price index changes for secured lending.
        Negative HPI changes increase LGD.

        Returns:
            LGD add-ons per period (shape: horizon_years,).
        """
        hpi = self.scenario.variables.get(
            "house_price_index", np.zeros(self.horizon_years),
        )
        # Negative HPI changes increase LGD
        return np.maximum(-hpi * 0.5, 0.0)

    def run(
        self,
        base_pds: np.ndarray,
        base_lgds: np.ndarray,
        base_eads: np.ndarray,
    ) -> dict[str, Any]:
        """Run full EBA stress test projection.

        Translates the macro scenario into PD multipliers and LGD add-ons,
        then runs a multi-period projection under the static balance sheet
        assumption. PDs are floored at the regulatory minimum.

        Args:
            base_pds: Baseline PDs (n_exposures,).
            base_lgds: Baseline LGDs (n_exposures,).
            base_eads: Baseline EADs (n_exposures,).

        Returns:
            Dict with stressed PDs, LGDs, expected losses per period,
            cumulative EL, and scenario metadata.
        """
        base_pds = np.asarray(base_pds, dtype=np.float64)
        # Apply PD floor
        base_pds = np.maximum(base_pds, self.pd_floor)

        pd_mult = self.translate_macro_to_pd_stress(base_pds)
        lgd_add = self.translate_macro_to_lgd_stress()

        result = multi_period_projection(
            base_pds, base_lgds, base_eads, pd_mult, lgd_add
        )

        # Compute baseline EL for comparison
        baseline_el = float(
            np.sum(base_pds * np.asarray(base_lgds) * np.asarray(base_eads))
        )
        result["scenario"] = self.scenario.name
        result["severity"] = self.scenario.severity
        result["horizon_years"] = self.horizon_years
        result["static_balance_sheet"] = self.static_balance_sheet
        result["baseline_el"] = baseline_el
        result["delta_el"] = result["cumulative_el"] - baseline_el * self.horizon_years

        logger.info(
            "EBA stress test complete: scenario='%s', baseline_EL=%.2f, "
            "cumulative_stressed_EL=%.2f, delta=%.2f",
            self.scenario.name,
            baseline_el,
            result["cumulative_el"],
            result["delta_el"],
        )

        return result


# ============================================================
# US CCAR/DFAST Framework
# ============================================================


class CCARScenario:
    """US CCAR/DFAST stress testing with 9-quarter projection horizon.

    Implements the Fed's Comprehensive Capital Analysis and Review framework.

    Reference:
        - Federal Reserve: SR 15-18, SR 15-19 (CCAR/DFAST instructions)
        - 12 CFR 252 Subpart E (stress testing requirements)

    Key features:
        - 9-quarter projection horizon (Q1 through Q9).
        - Baseline, adverse, and severely adverse scenarios.
        - Pre-Provision Net Revenue (PPNR) hook for income projection.
        - Capital adequacy assessment at each quarter.

    Args:
        scenario: MacroScenario used for the stress test.
        horizon_quarters: Number of projection quarters (default 9).
        ppnr_quarterly: Optional pre-provision net revenue per quarter (9,).
            If not provided, PPNR is assumed to be zero each quarter.
    """

    def __init__(
        self,
        scenario: MacroScenario,
        horizon_quarters: int = 9,
        ppnr_quarterly: np.ndarray | None = None,
    ) -> None:
        self.scenario = scenario
        self.horizon_quarters = horizon_quarters
        self.ppnr_quarterly = (
            np.asarray(ppnr_quarterly, dtype=np.float64)
            if ppnr_quarterly is not None
            else np.zeros(self.horizon_quarters)
        )
        if len(self.ppnr_quarterly) != self.horizon_quarters:
            raise ValueError(
                f"ppnr_quarterly must have exactly {self.horizon_quarters} elements."
            )
        logger.info(
            "CCARScenario initialised: scenario='%s', quarters=%d, "
            "cumulative_ppnr=%.2f",
            scenario.name,
            self.horizon_quarters,
            float(np.sum(self.ppnr_quarterly)),
        )

    def project_quarterly_losses(
        self,
        base_pds: np.ndarray,
        base_lgds: np.ndarray,
        base_eads: np.ndarray,
        pd_quarterly_multipliers: np.ndarray | None = None,
        lgd_add_ons_quarterly: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Project quarterly credit losses over the CCAR horizon.

        Quarterly PD is derived from annual PD:
            PD_q = 1 - (1 - PD_annual)^(1/4)
        The stress multiplier is then applied to the quarterly PD.

        Args:
            base_pds: Annual PDs (n_exposures,).
            base_lgds: Baseline LGDs (n_exposures,).
            base_eads: Baseline EADs (n_exposures,).
            pd_quarterly_multipliers: Optional quarterly PD stress factors
                (horizon_quarters,). Defaults to 1.0 each quarter.
            lgd_add_ons_quarterly: Optional LGD add-ons per quarter
                (horizon_quarters,). Defaults to 0.0.

        Returns:
            Dict with quarterly_losses matrix, per-quarter totals,
            cumulative loss trajectory, and total loss.
        """
        base_pds = np.asarray(base_pds, dtype=np.float64)
        base_lgds = np.asarray(base_lgds, dtype=np.float64)
        base_eads = np.asarray(base_eads, dtype=np.float64)

        if pd_quarterly_multipliers is None:
            pd_quarterly_multipliers = np.ones(self.horizon_quarters)
        else:
            pd_quarterly_multipliers = np.asarray(pd_quarterly_multipliers, dtype=np.float64)

        if lgd_add_ons_quarterly is None:
            lgd_add_ons_quarterly = np.zeros(self.horizon_quarters)
        else:
            lgd_add_ons_quarterly = np.asarray(lgd_add_ons_quarterly, dtype=np.float64)

        # Convert annual PD to quarterly: PD_q = 1 - (1 - PD_annual)^(1/4)
        quarterly_pds = 1.0 - np.power(np.maximum(1.0 - base_pds, 0.0), 0.25)

        n_q = self.horizon_quarters
        losses = np.zeros((n_q, len(base_pds)))

        for q in range(n_q):
            mult = (
                pd_quarterly_multipliers[q]
                if q < len(pd_quarterly_multipliers)
                else 1.0
            )
            stressed_q_pd = np.minimum(quarterly_pds * mult, 1.0)
            stressed_lgd = np.clip(base_lgds + lgd_add_ons_quarterly[q], 0.0, 1.0)
            losses[q] = stressed_q_pd * stressed_lgd * base_eads

        quarterly_totals = losses.sum(axis=1)

        return {
            "quarterly_losses": losses,
            "quarterly_totals": quarterly_totals.tolist(),
            "cumulative_loss": np.cumsum(quarterly_totals).tolist(),
            "total_loss": float(losses.sum()),
        }

    def run(
        self,
        base_pds: np.ndarray,
        base_lgds: np.ndarray,
        base_eads: np.ndarray,
        pd_quarterly_multipliers: np.ndarray | None = None,
        lgd_add_ons_quarterly: np.ndarray | None = None,
        initial_capital: float = 0.0,
    ) -> dict[str, Any]:
        """Execute the full CCAR stress scenario with capital trajectory.

        Combines credit loss projection with PPNR to compute net income
        and a quarter-by-quarter capital adequacy trajectory.

        Args:
            base_pds: Annual baseline PDs (n_exposures,).
            base_lgds: Baseline LGDs (n_exposures,).
            base_eads: EAD array (n_exposures,).
            pd_quarterly_multipliers: PD stress multipliers per quarter.
            lgd_add_ons_quarterly: Optional LGD add-ons per quarter.
            initial_capital: Starting capital buffer for capital trajectory.

        Returns:
            Dict with quarterly losses, PPNR, net income, capital trajectory,
            minimum capital point, and summary statistics.
        """
        loss_result = self.project_quarterly_losses(
            base_pds,
            base_lgds,
            base_eads,
            pd_quarterly_multipliers,
            lgd_add_ons_quarterly,
        )

        quarterly_totals = np.array(loss_result["quarterly_totals"])
        net_income = self.ppnr_quarterly - quarterly_totals

        capital_trajectory = np.empty(self.horizon_quarters, dtype=np.float64)
        capital = initial_capital
        for q in range(self.horizon_quarters):
            capital += net_income[q]
            capital_trajectory[q] = capital

        min_capital = float(np.min(capital_trajectory))
        min_capital_quarter = int(np.argmin(capital_trajectory)) + 1

        logger.info(
            "CCAR run complete: cumulative_loss=%.2f, min_capital=%.2f at Q%d",
            loss_result["total_loss"],
            min_capital,
            min_capital_quarter,
        )

        return {
            "scenario": self.scenario.name,
            "horizon_quarters": self.horizon_quarters,
            "quarterly_losses": loss_result["quarterly_totals"],
            "ppnr_quarterly": self.ppnr_quarterly.tolist(),
            "net_income_quarterly": net_income.tolist(),
            "capital_trajectory": capital_trajectory.tolist(),
            "cumulative_loss": loss_result["cumulative_loss"],
            "total_loss": loss_result["total_loss"],
            "cumulative_ppnr": float(np.sum(self.ppnr_quarterly)),
            "min_capital": min_capital,
            "min_capital_quarter": min_capital_quarter,
            "initial_capital": initial_capital,
            "final_capital": float(capital_trajectory[-1]),
        }


# ============================================================
# RBI Stress Testing
# ============================================================


class RBIStressTest:
    """RBI (Reserve Bank of India) stress testing with sensitivity analysis.

    Implements the RBI's stress testing framework as outlined in the
    RBI Master Circular on Stress Testing (DBOD.No.BP.BC.94/21.06.001)
    and the Financial Stability Report methodology.

    Key features:
        - Severity-calibrated credit quality deterioration (NPA migration).
        - Interest rate sensitivity analysis (EVE and NII impact).
        - Liquidity sensitivity analysis (LCR impact from deposit outflows).
        - Single-factor shock isolation for each risk driver.

    Args:
        severity: Stress severity level ('mild', 'moderate', or 'severe').
        baseline_metrics: Optional dict with baseline values for sensitivity
            analysis. Keys: 'npa_ratio', 'car', 'net_interest_income',
            'total_advances'. If not provided, sensitivity methods that
            require these will raise ValueError.
    """

    def __init__(
        self,
        severity: str = "moderate",
        baseline_metrics: dict[str, float] | None = None,
    ) -> None:
        self.severity = severity
        self.baseline_metrics = baseline_metrics or {}
        self._severity_map: dict[str, dict[str, float]] = {
            "mild": {"pd_mult": 1.5, "lgd_add": 0.05, "npa_shift_pct": 0.02},
            "moderate": {"pd_mult": 2.0, "lgd_add": 0.10, "npa_shift_pct": 0.05},
            "severe": {"pd_mult": 3.0, "lgd_add": 0.15, "npa_shift_pct": 0.10},
        }
        logger.info(
            "RBIStressTest initialised: severity='%s', baseline_metrics=%s",
            severity,
            list(self.baseline_metrics.keys()) if self.baseline_metrics else "none",
        )

    def _require_baseline(self, *keys: str) -> None:
        """Validate that required baseline metrics are present."""
        missing = set(keys) - set(self.baseline_metrics.keys())
        if missing:
            raise ValueError(
                f"Missing required baseline_metrics for this analysis: {missing}. "
                "Pass them in the RBIStressTest constructor."
            )

    def credit_quality_stress(
        self,
        base_pds: np.ndarray,
        base_lgds: np.ndarray,
        base_eads: np.ndarray,
    ) -> dict[str, float]:
        """Apply credit quality deterioration stress.

        Simulates NPA migration per RBI's macro stress testing framework.
        PDs are multiplied by a severity-dependent factor and LGDs receive
        an additive stress.

        Args:
            base_pds: Baseline PDs (n_exposures,).
            base_lgds: Baseline LGDs (n_exposures,).
            base_eads: Baseline EADs (n_exposures,).

        Returns:
            Dict with base EL, stressed EL, incremental provisions, and
            severity label.
        """
        base_pds = np.asarray(base_pds, dtype=np.float64)
        base_lgds = np.asarray(base_lgds, dtype=np.float64)
        base_eads = np.asarray(base_eads, dtype=np.float64)

        params = self._severity_map.get(self.severity, self._severity_map["moderate"])
        stressed_pds = np.minimum(base_pds * params["pd_mult"], 1.0)
        stressed_lgds = np.clip(base_lgds + params["lgd_add"], 0.0, 1.0)

        base_el = float((base_pds * base_lgds * base_eads).sum())
        stressed_el = float((stressed_pds * stressed_lgds * base_eads).sum())

        logger.debug(
            "Credit quality stress (%s): base_EL=%.2f, stressed_EL=%.2f",
            self.severity,
            base_el,
            stressed_el,
        )

        return {
            "base_el": base_el,
            "stressed_el": stressed_el,
            "incremental_provisions": stressed_el - base_el,
            "severity": self.severity,
            "pd_multiplier": params["pd_mult"],
            "lgd_add_on": params["lgd_add"],
        }

    def interest_rate_sensitivity(
        self,
        rate_shock_bps: float,
        duration_gap: float,
        total_assets: float,
    ) -> dict[str, float]:
        """Interest rate sensitivity analysis.

        Estimates the impact of a parallel shift in interest rates on
        net interest income (NII) and economic value of equity (EVE).

        Impact on EVE: -Duration_gap x Delta_Rate x Total_Assets

        Requires baseline_metrics with 'net_interest_income' and
        'total_advances'.

        Args:
            rate_shock_bps: Interest rate shock in basis points (e.g. +200).
            duration_gap: Duration gap (years) between assets and liabilities.
            total_assets: Total asset value.

        Returns:
            Dict with EVE impact, NII impact, and stressed CAR estimate.
        """
        self._require_baseline("net_interest_income", "total_advances", "car")

        rate_shock = rate_shock_bps / 10_000.0

        # EVE impact
        eve_impact = -duration_gap * rate_shock * total_assets

        # NII impact: approximate as rate_shock x rate-sensitive advances
        # (simplified: assume 60% of advances are rate-sensitive)
        rate_sensitive_advances = self.baseline_metrics["total_advances"] * 0.6
        nii_impact = rate_shock * rate_sensitive_advances
        stressed_nii = self.baseline_metrics["net_interest_income"] + nii_impact

        # CAR impact (simplified: EVE change relative to RWA proxy)
        rwa_proxy = total_assets * 0.75
        car_impact_pp = (eve_impact / rwa_proxy) * 100 if rwa_proxy > 0 else 0.0
        stressed_car = self.baseline_metrics["car"] + car_impact_pp

        logger.debug(
            "IR sensitivity: shock=%+dbps, EVE_impact=%.2f, NII_impact=%.2f, "
            "CAR %.2f -> %.2f",
            rate_shock_bps,
            eve_impact,
            nii_impact,
            self.baseline_metrics["car"],
            stressed_car,
        )

        return {
            "rate_shock_bps": rate_shock_bps,
            "eve_impact": eve_impact,
            "nii_impact": nii_impact,
            "baseline_nii": self.baseline_metrics["net_interest_income"],
            "stressed_nii": stressed_nii,
            "baseline_car": self.baseline_metrics["car"],
            "stressed_car": stressed_car,
            "car_change_pp": car_impact_pp,
        }

    def credit_quality_sensitivity(
        self,
        npa_increase_pct: float,
        provision_coverage_ratio: float = 0.70,
    ) -> dict[str, float]:
        """Credit quality sensitivity analysis via NPA ratio shift.

        Models the impact of an increase in non-performing assets on
        provisioning requirements and capital adequacy.

        Requires baseline_metrics with 'npa_ratio', 'car', 'total_advances'.

        Args:
            npa_increase_pct: Percentage point increase in NPA ratio
                (e.g. 2.0 means NPA ratio rises by 2 pp).
            provision_coverage_ratio: Provisioning coverage ratio for
                incremental NPAs (default 0.70).

        Returns:
            Dict with stressed NPA ratio, incremental provisions, and CAR impact.
        """
        self._require_baseline("npa_ratio", "car", "total_advances")

        baseline_npa = self.baseline_metrics["npa_ratio"]
        stressed_npa = baseline_npa + npa_increase_pct
        total_advances = self.baseline_metrics["total_advances"]

        incremental_npa_amount = (npa_increase_pct / 100.0) * total_advances
        incremental_provisions = incremental_npa_amount * provision_coverage_ratio

        # CAR impact: provisions reduce capital, RWA unchanged
        rwa_proxy = total_advances * 0.75  # average risk weight 75%
        car_reduction = (
            (incremental_provisions / rwa_proxy) * 100 if rwa_proxy > 0 else 0.0
        )
        stressed_car = self.baseline_metrics["car"] - car_reduction

        logger.debug(
            "Credit quality sensitivity: NPA +%.1fpp, provisions=%.2f, "
            "CAR %.2f -> %.2f",
            npa_increase_pct,
            incremental_provisions,
            self.baseline_metrics["car"],
            stressed_car,
        )

        return {
            "baseline_npa_ratio": baseline_npa,
            "stressed_npa_ratio": stressed_npa,
            "npa_increase_pct": npa_increase_pct,
            "incremental_npa_amount": incremental_npa_amount,
            "incremental_provisions": incremental_provisions,
            "provision_coverage_ratio": provision_coverage_ratio,
            "baseline_car": self.baseline_metrics["car"],
            "stressed_car": stressed_car,
            "car_reduction_pp": car_reduction,
        }

    def liquidity_sensitivity(
        self,
        deposit_outflow_pct: float,
        hqla: float,
        total_deposits: float,
        net_cash_outflows_30d: float,
    ) -> dict[str, float]:
        """Liquidity sensitivity analysis.

        Estimates the impact of deposit outflows on the Liquidity Coverage
        Ratio (LCR) as per Basel III / RBI guidelines.

        LCR = HQLA / Net cash outflows over 30 days

        Args:
            deposit_outflow_pct: Assumed deposit run-off percentage.
            hqla: High-quality liquid assets.
            total_deposits: Total deposit base.
            net_cash_outflows_30d: Baseline 30-day net cash outflows.

        Returns:
            Dict with baseline and stressed LCR, deposit outflow amount,
            and whether the RBI minimum LCR (100%) is breached.
        """
        if net_cash_outflows_30d <= 0:
            raise ValueError("net_cash_outflows_30d must be positive.")

        deposit_outflow = (deposit_outflow_pct / 100.0) * total_deposits
        stressed_outflows = net_cash_outflows_30d + deposit_outflow

        baseline_lcr = (hqla / net_cash_outflows_30d) * 100.0
        stressed_lcr = (
            (hqla / stressed_outflows) * 100.0 if stressed_outflows > 0 else 0.0
        )

        # RBI minimum LCR requirement: 100%
        lcr_breach = stressed_lcr < 100.0

        logger.debug(
            "Liquidity sensitivity: deposit_outflow=%.1f%%, LCR %.1f%% -> %.1f%%",
            deposit_outflow_pct,
            baseline_lcr,
            stressed_lcr,
        )

        return {
            "deposit_outflow_pct": deposit_outflow_pct,
            "deposit_outflow_amount": deposit_outflow,
            "baseline_lcr_pct": baseline_lcr,
            "stressed_lcr_pct": stressed_lcr,
            "lcr_breach": lcr_breach,
            "rbi_min_lcr_pct": 100.0,
        }
