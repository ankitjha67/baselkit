"""Standardised Approach for Counterparty Credit Risk (SA-CCR), BCBS CRE52.

Reference:
    - BCBS CRE52 — The standardised approach for measuring counterparty
      credit risk exposures.
    - BCBS d279 (March 2014) — original SA-CCR standard and worked examples.

Computes the full counterparty EAD from trade-level data:

    EAD = alpha * (RC + multiplier * AddOn_aggregate)

with alpha = 1.4. Unlike a thin ``alpha * (RC + PFE)`` wrapper, this module
derives every component from first principles per CRE52:

    * Adjusted notional (CRE52.30-52.33), incl. the supervisory duration
      for interest-rate and credit derivatives.
    * Supervisory delta (CRE52.34-52.40), incl. Black-Scholes option deltas
      and CDO-tranche deltas.
    * Maturity factor (CRE52.48, CRE52.50) for unmargined and margined sets.
    * Asset-class add-ons with the correct hedging-set aggregation:
      interest-rate maturity buckets (CRE52.45-52.52), FX currency pairs
      (CRE52.53), and the single-factor systematic/idiosyncratic structure
      for credit, equity and commodity (CRE52.55-52.70).
    * The PFE multiplier recognising negative MtM / over-collateralisation
      (CRE52.23).
    * Replacement cost for unmargined and margined netting sets
      (CRE52.10, CRE52.18).

Supervisory parameters are taken from the CRE52.72 table.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import StrEnum

logger = logging.getLogger(__name__)

ALPHA: float = 1.4  # CRE52.1


class AssetClass(StrEnum):
    """SA-CCR asset classes (CRE52.20)."""

    INTEREST_RATE = "interest_rate"
    FX = "fx"
    CREDIT = "credit"
    EQUITY = "equity"
    COMMODITY = "commodity"


class OptionType(StrEnum):
    """Option directionality for the supervisory delta (CRE52.36-52.38)."""

    NONE = "none"  # linear instrument
    BOUGHT_CALL = "bought_call"
    SOLD_CALL = "sold_call"
    BOUGHT_PUT = "bought_put"
    SOLD_PUT = "sold_put"


# --- CRE52.72 supervisory parameters -------------------------------------

# Supervisory factors (as decimals).
_SF_IR: float = 0.005
_SF_FX: float = 0.04
_SF_CREDIT_SINGLE: dict[str, float] = {
    "AAA": 0.0038,
    "AA": 0.0038,
    "A": 0.0042,
    "BBB": 0.0054,
    "BB": 0.0106,
    "B": 0.0160,
    "CCC": 0.0600,
}
_SF_CREDIT_INDEX: dict[str, float] = {"IG": 0.0038, "SG": 0.0106}
_SF_EQUITY_SINGLE: float = 0.32
_SF_EQUITY_INDEX: float = 0.20
_SF_COMMODITY: dict[str, float] = {
    "electricity": 0.40,
    "oil_gas": 0.18,
    "metals": 0.18,
    "agricultural": 0.18,
    "other": 0.18,
}

# Supervisory correlations (single-factor model), CRE52.72.
_RHO_CREDIT_SINGLE: float = 0.50
_RHO_CREDIT_INDEX: float = 0.80
_RHO_EQUITY_SINGLE: float = 0.50
_RHO_EQUITY_INDEX: float = 0.80
_RHO_COMMODITY: float = 0.40

# Supervisory option volatilities, CRE52.72.
_VOL: dict[AssetClass, float] = {
    AssetClass.INTEREST_RATE: 0.50,
    AssetClass.FX: 0.15,
    AssetClass.CREDIT: 1.00,
    AssetClass.EQUITY: 1.20,
    AssetClass.COMMODITY: 0.70,
}


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via the error function (no SciPy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass(frozen=True)
class SACCRTrade:
    """A single derivative trade within a netting set.

    Attributes:
        asset_class: SA-CCR asset class.
        notional: Trade notional in the reporting currency (positive).
        start: Start of the period referenced (years from today, S). For
            an already-started trade this is 0.
        end: End of the period referenced (years from today, E), i.e. the
            remaining maturity for a vanilla swap.
        direction: +1 if long the primary risk factor, -1 if short. Used
            for linear instruments; ignored when an option type is set.
        hedging_set: Hedging-set key — currency for IR, currency pair for
            FX, commodity type for commodity. Defaults to a single set.
        reference: Reference entity (credit) or name (equity) for the
            systematic aggregation. Defaults to ``hedging_set``.
        credit_rating: Rating bucket for credit single names (e.g. "BBB")
            or "IG"/"SG" for credit indices. Required for credit trades.
        is_index: True for index trades (credit/equity), affecting the
            supervisory factor and correlation.
        option_type: Option directionality; NONE for linear instruments.
        strike: Option strike price K (required for options).
        underlying_price: Option underlying price P (required for options).
        option_expiry: Time to latest exercise T in years (required for
            options).
        is_tranche: True for a CDO tranche (uses the CRE52.40 delta).
        attachment: CDO tranche attachment point A (tranche only).
        detachment: CDO tranche detachment point D (tranche only).
        margined_mpor: Margin period of risk in years for margined sets;
            None for unmargined trades.
    """

    asset_class: AssetClass
    notional: float
    start: float
    end: float
    direction: int = 1
    hedging_set: str = "default"
    reference: str = ""
    credit_rating: str = ""
    is_index: bool = False
    option_type: OptionType = OptionType.NONE
    strike: float = 0.0
    underlying_price: float = 0.0
    option_expiry: float = 0.0
    is_tranche: bool = False
    attachment: float = 0.0
    detachment: float = 0.0
    margined_mpor: float | None = None

    def __post_init__(self) -> None:
        if self.notional < 0.0:
            raise ValueError("notional must be non-negative; use direction for sign")
        if self.end < self.start:
            raise ValueError("end must be >= start")
        if self.direction not in (-1, 1):
            raise ValueError("direction must be +1 or -1")
        if self.asset_class == AssetClass.CREDIT and not self.credit_rating:
            raise ValueError("credit trades require a credit_rating bucket")
        if self.option_type != OptionType.NONE and self.option_expiry <= 0.0:
            raise ValueError("options require a positive option_expiry")


def supervisory_duration(start: float, end: float) -> float:
    """Supervisory duration SD for IR and credit trades (CRE52.34).

        SD = (exp(-0.05 * S) - exp(-0.05 * E)) / 0.05

    Floored at 10 business days (10/250 years) per CRE52.34.
    """
    s = max(start, 0.0)
    e = max(end, s + 10.0 / 250.0)
    return (math.exp(-0.05 * s) - math.exp(-0.05 * e)) / 0.05


def adjusted_notional(trade: SACCRTrade) -> float:
    """Trade-level adjusted notional d (CRE52.30-52.33).

    For interest-rate and credit derivatives the notional is scaled by the
    supervisory duration. For FX, equity and commodity it is the notional
    (in the reporting currency / current price * units) as supplied.
    """
    if trade.asset_class in (AssetClass.INTEREST_RATE, AssetClass.CREDIT):
        return trade.notional * supervisory_duration(trade.start, trade.end)
    return trade.notional


def supervisory_delta(trade: SACCRTrade) -> float:
    """Supervisory delta (CRE52.34-52.40).

    Linear trades: +/- 1 by direction. Options: Black-Scholes delta using
    the supervisory volatility. CDO tranches: the CRE52.40 tranche delta.
    """
    # CDO tranche (CRE52.40): sign by direction (purchased vs sold protection).
    if trade.is_tranche:
        a, d = trade.attachment, trade.detachment
        if not 0.0 <= a < d <= 1.0:
            raise ValueError("tranche requires 0 <= attachment < detachment <= 1")
        mag = 15.0 / ((1.0 + 14.0 * a) * (1.0 + 14.0 * d))
        return float(trade.direction) * mag

    if trade.option_type == OptionType.NONE:
        return float(trade.direction)

    # Option delta via Black-Scholes with supervisory vol (CRE52.36-52.38).
    sigma = _VOL[trade.asset_class]
    p = trade.underlying_price
    k = trade.strike
    t = trade.option_expiry
    if p <= 0.0 or k <= 0.0:
        raise ValueError("option underlying_price and strike must be positive")
    d1 = (math.log(p / k) + 0.5 * sigma * sigma * t) / (sigma * math.sqrt(t))

    if trade.option_type == OptionType.BOUGHT_CALL:
        return _norm_cdf(d1)
    if trade.option_type == OptionType.SOLD_CALL:
        return -_norm_cdf(d1)
    if trade.option_type == OptionType.BOUGHT_PUT:
        return -_norm_cdf(-d1)
    # SOLD_PUT
    return _norm_cdf(-d1)


def maturity_factor(trade: SACCRTrade) -> float:
    """Maturity factor MF (CRE52.48 unmargined, CRE52.50 margined)."""
    if trade.margined_mpor is None:
        # Unmargined: sqrt(min(M, 1yr) / 1yr), M floored at 10 business days.
        m = max(trade.end - trade.start, 10.0 / 250.0)
        return math.sqrt(min(m, 1.0))
    # Margined: 1.5 * sqrt(MPOR / 1yr).
    mpor = max(trade.margined_mpor, 10.0 / 250.0)
    return 1.5 * math.sqrt(mpor)


def _effective_notional(trade: SACCRTrade) -> float:
    """delta * d * MF — the signed effective notional of a trade."""
    return supervisory_delta(trade) * adjusted_notional(trade) * maturity_factor(trade)


def _supervisory_factor(trade: SACCRTrade) -> float:
    """Resolve the CRE52.72 supervisory factor for a trade."""
    ac = trade.asset_class
    if ac == AssetClass.INTEREST_RATE:
        return _SF_IR
    if ac == AssetClass.FX:
        return _SF_FX
    if ac == AssetClass.CREDIT:
        if trade.is_index:
            return _SF_CREDIT_INDEX.get(trade.credit_rating.upper(), _SF_CREDIT_INDEX["SG"])
        return _SF_CREDIT_SINGLE.get(trade.credit_rating.upper(), _SF_CREDIT_SINGLE["B"])
    if ac == AssetClass.EQUITY:
        return _SF_EQUITY_INDEX if trade.is_index else _SF_EQUITY_SINGLE
    # Commodity
    return _SF_COMMODITY.get(trade.hedging_set, _SF_COMMODITY["other"])


def _ir_bucket(trade: SACCRTrade) -> int:
    """Interest-rate maturity bucket index (CRE52.51): 0=<1y,1=1-5y,2=>5y."""
    m = trade.end
    if m < 1.0:
        return 0
    if m <= 5.0:
        return 1
    return 2


def _addon_interest_rate(trades: list[SACCRTrade]) -> float:
    """IR add-on with per-currency maturity-bucket aggregation (CRE52.52)."""
    by_currency: dict[str, list[float]] = {}
    for t in trades:
        buckets = by_currency.setdefault(t.hedging_set, [0.0, 0.0, 0.0])
        buckets[_ir_bucket(t)] += _effective_notional(t)

    total = 0.0
    for buckets in by_currency.values():
        d1, d2, d3 = buckets
        eff = math.sqrt(
            d1 * d1 + d2 * d2 + d3 * d3
            + 1.4 * d1 * d2 + 1.4 * d2 * d3 + 0.6 * d1 * d3
        )
        total += _SF_IR * eff
    return total


def _addon_fx(trades: list[SACCRTrade]) -> float:
    """FX add-on: full offsetting within each currency-pair hedging set."""
    by_pair: dict[str, float] = {}
    for t in trades:
        by_pair[t.hedging_set] = by_pair.get(t.hedging_set, 0.0) + _effective_notional(t)
    return sum(_SF_FX * abs(eff) for eff in by_pair.values())


def _addon_systematic(
    trades: list[SACCRTrade],
    rho_single: float,
    rho_index: float,
) -> float:
    """Single-factor systematic add-on for credit / equity (CRE52.55-52.65).

        AddOn = sqrt[ (sum_k rho_k * AddOn_k)^2 + sum_k (1 - rho_k^2) * AddOn_k^2 ]

    where AddOn_k is the entity-level add-on (full offsetting within an
    entity) and rho_k is the entity's supervisory correlation.
    """
    # Entity-level effective notionals and supervisory factors.
    entity_eff: dict[str, float] = {}
    entity_sf: dict[str, float] = {}
    entity_rho: dict[str, float] = {}
    for t in trades:
        key = t.reference or t.hedging_set
        entity_eff[key] = entity_eff.get(key, 0.0) + _effective_notional(t)
        entity_sf[key] = _supervisory_factor(t)
        entity_rho[key] = rho_index if t.is_index else rho_single

    systematic = 0.0
    idiosyncratic = 0.0
    for key, eff in entity_eff.items():
        addon_k = entity_sf[key] * eff
        rho = entity_rho[key]
        systematic += rho * addon_k
        idiosyncratic += (1.0 - rho * rho) * addon_k * addon_k
    return math.sqrt(systematic * systematic + idiosyncratic)


def _addon_commodity(trades: list[SACCRTrade]) -> float:
    """Commodity add-on with per-type hedging sets (CRE52.66-52.70)."""
    # Effective notional per commodity type (full offsetting within type).
    type_eff: dict[str, float] = {}
    type_sf: dict[str, float] = {}
    for t in trades:
        type_eff[t.hedging_set] = type_eff.get(t.hedging_set, 0.0) + _effective_notional(t)
        type_sf[t.hedging_set] = _supervisory_factor(t)

    systematic = 0.0
    idiosyncratic = 0.0
    for key, eff in type_eff.items():
        addon_k = type_sf[key] * eff
        systematic += _RHO_COMMODITY * addon_k
        idiosyncratic += (1.0 - _RHO_COMMODITY * _RHO_COMMODITY) * addon_k * addon_k
    return math.sqrt(systematic * systematic + idiosyncratic)


def aggregate_addon(trades: list[SACCRTrade]) -> float:
    """Aggregate add-on across all asset classes (CRE52.21).

    Per-asset-class add-ons are summed (no cross-asset offsetting).
    """
    by_class: dict[AssetClass, list[SACCRTrade]] = {}
    for t in trades:
        by_class.setdefault(t.asset_class, []).append(t)

    total = 0.0
    for ac, group in by_class.items():
        if ac == AssetClass.INTEREST_RATE:
            total += _addon_interest_rate(group)
        elif ac == AssetClass.FX:
            total += _addon_fx(group)
        elif ac == AssetClass.CREDIT:
            total += _addon_systematic(group, _RHO_CREDIT_SINGLE, _RHO_CREDIT_INDEX)
        elif ac == AssetClass.EQUITY:
            total += _addon_systematic(group, _RHO_EQUITY_SINGLE, _RHO_EQUITY_INDEX)
        else:
            total += _addon_commodity(group)
    return total


def pfe_multiplier(net_mtm: float, collateral: float, aggregate_addon_value: float) -> float:
    """PFE multiplier recognising excess collateral / negative MtM (CRE52.23).

        multiplier = min{1, floor + (1 - floor) *
                          exp[ (V - C) / (2 * (1 - floor) * AddOn) ]}

    with floor = 0.05. Returns 1.0 when V - C >= 0 or the add-on is zero.
    """
    floor = 0.05
    v_minus_c = net_mtm - collateral
    if v_minus_c >= 0.0 or aggregate_addon_value <= 0.0:
        return 1.0
    exponent = v_minus_c / (2.0 * (1.0 - floor) * aggregate_addon_value)
    return min(1.0, floor + (1.0 - floor) * math.exp(exponent))


def replacement_cost(
    net_mtm: float,
    collateral: float = 0.0,
    *,
    margined: bool = False,
    threshold: float = 0.0,
    mta: float = 0.0,
    nica: float = 0.0,
) -> float:
    """Replacement cost RC (CRE52.10 unmargined, CRE52.18 margined).

    Unmargined: ``RC = max(V - C, 0)``.
    Margined:   ``RC = max(V - C, TH + MTA - NICA, 0)``.

    Args:
        net_mtm: Net mark-to-market value of the netting set (V).
        collateral: Net collateral held (C), incl. variation margin.
        margined: True if the netting set is subject to a margin agreement.
        threshold: Margin threshold TH (margined only).
        mta: Minimum transfer amount MTA (margined only).
        nica: Net independent collateral amount NICA (margined only).

    Returns:
        Replacement cost.
    """
    unmargined = max(net_mtm - collateral, 0.0)
    if not margined:
        return unmargined
    return max(net_mtm - collateral, threshold + mta - nica, 0.0)


@dataclass(frozen=True)
class SACCRResult:
    """SA-CCR exposure result.

    Attributes:
        ead: Exposure at default = alpha * (RC + multiplier * AddOn).
        replacement_cost: RC component.
        aggregate_addon: Aggregate PFE add-on across asset classes.
        multiplier: PFE multiplier applied to the add-on.
        pfe: Potential future exposure = multiplier * AddOn.
        alpha: Alpha multiplier used (1.4).
    """

    ead: float
    replacement_cost: float
    aggregate_addon: float
    multiplier: float
    pfe: float
    alpha: float = ALPHA


def sa_ccr_ead(
    trades: list[SACCRTrade],
    net_mtm: float = 0.0,
    collateral: float = 0.0,
    *,
    margined: bool = False,
    threshold: float = 0.0,
    mta: float = 0.0,
    nica: float = 0.0,
    alpha: float = ALPHA,
) -> SACCRResult:
    """Compute the SA-CCR EAD for a netting set (CRE52).

        EAD = alpha * (RC + multiplier * AddOn_aggregate)

    Args:
        trades: Trades in the netting set.
        net_mtm: Net mark-to-market value V of the netting set.
        collateral: Net collateral held C.
        margined: Whether a margin agreement applies (affects RC and MF).
        threshold: Margin threshold TH (margined).
        mta: Minimum transfer amount MTA (margined).
        nica: Net independent collateral amount NICA (margined).
        alpha: Alpha multiplier (default 1.4).

    Returns:
        A :class:`SACCRResult` with the EAD and its components.

    Raises:
        ValueError: If alpha is not positive or no trades are supplied.
    """
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if not trades:
        raise ValueError("at least one trade is required")

    addon = aggregate_addon(trades)
    rc = replacement_cost(
        net_mtm, collateral,
        margined=margined, threshold=threshold, mta=mta, nica=nica,
    )
    mult = pfe_multiplier(net_mtm, collateral, addon)
    pfe = mult * addon
    ead = alpha * (rc + pfe)

    logger.debug(
        "SA-CCR: RC=%.4f AddOn=%.4f mult=%.4f PFE=%.4f alpha=%.2f EAD=%.4f",
        rc, addon, mult, pfe, alpha, ead,
    )
    return SACCRResult(
        ead=round(ead, 6),
        replacement_cost=round(rc, 6),
        aggregate_addon=round(addon, 6),
        multiplier=round(mult, 6),
        pfe=round(pfe, 6),
        alpha=alpha,
    )
