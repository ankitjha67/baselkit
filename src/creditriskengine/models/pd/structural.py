"""Merton structural model for probability of default estimation.

Implements the Merton (1974) structural credit risk model, which treats
a firm's equity as a call option on its assets. Default occurs when asset
value falls below the face value of debt at maturity.

References:
    - Merton, R.C. (1974): On the Pricing of Corporate Debt
    - BCBS d350: IRB approach — structural model benchmarks
    - Crosbie & Bohn (2003): Modeling Default Risk (KMV/Moody's)
"""

import logging
import math

from scipy.stats import norm

logger = logging.getLogger(__name__)


def distance_to_default(
    asset_value: float,
    debt_face_value: float,
    asset_volatility: float,
    risk_free_rate: float,
    time_horizon: float = 1.0,
) -> float:
    """Compute the Merton distance to default (DD).

    DD = (ln(V/D) + (r - sigma^2/2) * T) / (sigma * sqrt(T))

    where V is asset value, D is debt face value, r is risk-free rate,
    sigma is asset volatility, and T is time horizon.

    Args:
        asset_value: Current asset value (V).
        debt_face_value: Face value of debt (D).
        asset_volatility: Annualised asset volatility (sigma).
        risk_free_rate: Continuous risk-free rate (r).
        time_horizon: Time to maturity in years (T).

    Returns:
        Distance to default (DD).

    Raises:
        ValueError: If asset_value, debt_face_value, asset_volatility,
            or time_horizon are non-positive.
    """
    if asset_value <= 0:
        raise ValueError("asset_value must be positive.")
    if debt_face_value <= 0:
        raise ValueError("debt_face_value must be positive.")
    if asset_volatility <= 0:
        raise ValueError("asset_volatility must be positive.")
    if time_horizon <= 0:
        raise ValueError("time_horizon must be positive.")

    numerator = (
        math.log(asset_value / debt_face_value)
        + (risk_free_rate - 0.5 * asset_volatility**2) * time_horizon
    )
    denominator = asset_volatility * math.sqrt(time_horizon)

    dd = numerator / denominator

    logger.debug(
        "Distance to default: V=%.2f, D=%.2f, sigma=%.4f, r=%.4f, "
        "T=%.2f -> DD=%.4f",
        asset_value,
        debt_face_value,
        asset_volatility,
        risk_free_rate,
        time_horizon,
        dd,
    )

    return dd


def merton_default_probability(
    asset_value: float,
    debt_face_value: float,
    asset_volatility: float,
    risk_free_rate: float,
    time_horizon: float = 1.0,
) -> float:
    """Compute the Merton model probability of default.

    PD = Phi(-DD) where DD is the distance to default and Phi is the
    standard normal CDF.

    Args:
        asset_value: Current asset value (V).
        debt_face_value: Face value of debt (D).
        asset_volatility: Annualised asset volatility (sigma).
        risk_free_rate: Continuous risk-free rate (r).
        time_horizon: Time to maturity in years (T).

    Returns:
        Probability of default under the Merton model.

    Raises:
        ValueError: If inputs are non-positive where required.
    """
    dd = distance_to_default(
        asset_value, debt_face_value, asset_volatility,
        risk_free_rate, time_horizon,
    )
    pd = float(norm.cdf(-dd))

    logger.debug(
        "Merton PD: DD=%.4f -> PD=%.6f",
        dd,
        pd,
    )

    return pd


def implied_asset_value(
    equity_value: float,
    debt_face_value: float,
    asset_volatility: float,
    risk_free_rate: float,
    time_horizon: float = 1.0,
    max_iterations: int = 200,
    tolerance: float = 1e-8,
) -> float:
    """Solve for the implied asset value using Newton's method.

    Given observed equity value, solves the Black-Scholes-Merton
    equity equation for the unobserved asset value:

        Equity = V * N(d1) - D * exp(-r*T) * N(d2)

    where:
        d1 = (ln(V/D) + (r + sigma^2/2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

    Newton's method iterates:
        V_{n+1} = V_n - f(V_n) / f'(V_n)

    where f(V) = V*N(d1) - D*exp(-rT)*N(d2) - E
    and f'(V) = N(d1) (by the BSM delta property).

    Args:
        equity_value: Observed market equity value (E).
        debt_face_value: Face value of debt (D).
        asset_volatility: Annualised asset volatility (sigma).
        risk_free_rate: Continuous risk-free rate (r).
        time_horizon: Time to maturity in years (T).
        max_iterations: Maximum Newton iterations (default 200).
        tolerance: Convergence tolerance (default 1e-8).

    Returns:
        Implied asset value (V).

    Raises:
        ValueError: If equity_value or debt_face_value are non-positive.
        RuntimeError: If Newton's method does not converge.
    """
    if equity_value <= 0:
        raise ValueError("equity_value must be positive.")
    if debt_face_value <= 0:
        raise ValueError("debt_face_value must be positive.")
    if asset_volatility <= 0:
        raise ValueError("asset_volatility must be positive.")
    if time_horizon <= 0:
        raise ValueError("time_horizon must be positive.")

    sqrt_t = math.sqrt(time_horizon)
    d_discounted = debt_face_value * math.exp(-risk_free_rate * time_horizon)

    # Initial guess: V = E + D
    v = equity_value + debt_face_value

    for iteration in range(1, max_iterations + 1):
        d1 = (
            math.log(v / debt_face_value)
            + (risk_free_rate + 0.5 * asset_volatility**2) * time_horizon
        ) / (asset_volatility * sqrt_t)
        d2 = d1 - asset_volatility * sqrt_t

        equity_model = v * norm.cdf(d1) - d_discounted * norm.cdf(d2)
        f_val = equity_model - equity_value

        # Derivative: dEquity/dV = N(d1)
        f_prime = norm.cdf(d1)
        if f_prime < 1e-15:
            raise RuntimeError(
                f"Newton's method failed: derivative near zero at iteration "
                f"{iteration}, V={v:.4f}."
            )

        v_new = v - f_val / f_prime

        # Ensure V stays positive
        v_new = max(v_new, 1e-10)

        if abs(v_new - v) < tolerance:
            logger.debug(
                "Implied asset value converged: V=%.4f after %d iterations.",
                v_new,
                iteration,
            )
            return v_new

        v = v_new

    raise RuntimeError(
        f"Newton's method did not converge after {max_iterations} iterations. "
        f"Last V={v:.4f}, residual={f_val:.6e}."
    )
