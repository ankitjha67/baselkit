"""Tests for the full SA-CCR engine (BCBS CRE52)."""

import math

import pytest

from creditriskengine.ccr.sa_ccr import (
    AssetClass,
    OptionType,
    SACCRTrade,
    adjusted_notional,
    aggregate_addon,
    maturity_factor,
    pfe_multiplier,
    replacement_cost,
    sa_ccr_ead,
    supervisory_delta,
    supervisory_duration,
)


class TestSupervisoryDuration:
    def test_known_value_10y(self) -> None:
        # SD = (1 - exp(-0.5)) / 0.05 = (1 - 0.606531) / 0.05 = 7.86939
        assert supervisory_duration(0.0, 10.0) == pytest.approx(7.86939, abs=1e-4)

    def test_floored_at_ten_business_days(self) -> None:
        # start == end: floored so SD is small but positive
        sd = supervisory_duration(5.0, 5.0)
        assert sd > 0.0


class TestSupervisoryDelta:
    def test_linear_long(self) -> None:
        t = SACCRTrade(AssetClass.INTEREST_RATE, 100.0, 0.0, 5.0, direction=1)
        assert supervisory_delta(t) == pytest.approx(1.0)

    def test_linear_short(self) -> None:
        t = SACCRTrade(AssetClass.INTEREST_RATE, 100.0, 0.0, 5.0, direction=-1)
        assert supervisory_delta(t) == pytest.approx(-1.0)

    def test_bought_call_atm(self) -> None:
        # ATM call, P=K=100, sigma=0.5 (IR), T=1 -> d1=0.25 -> N(0.25)=0.5987
        t = SACCRTrade(
            AssetClass.INTEREST_RATE, 100.0, 0.0, 1.0,
            option_type=OptionType.BOUGHT_CALL,
            strike=100.0, underlying_price=100.0, option_expiry=1.0,
        )
        assert supervisory_delta(t) == pytest.approx(0.59871, abs=1e-4)

    def test_sold_put_sign(self) -> None:
        t = SACCRTrade(
            AssetClass.EQUITY, 100.0, 0.0, 1.0,
            option_type=OptionType.SOLD_PUT,
            strike=100.0, underlying_price=100.0, option_expiry=1.0,
        )
        # Sold put delta is positive
        assert supervisory_delta(t) > 0.0

    def test_tranche_delta(self) -> None:
        # CRE52.40: 0-3% tranche -> 15 / ((1+0)*(1+14*0.03)) = 15/1.42 = 10.563
        t = SACCRTrade(
            AssetClass.CREDIT, 100.0, 0.0, 5.0, credit_rating="IG",
            is_tranche=True, attachment=0.0, detachment=0.03,
        )
        assert supervisory_delta(t) == pytest.approx(15.0 / 1.42, abs=1e-3)

    def test_invalid_tranche(self) -> None:
        t = SACCRTrade(
            AssetClass.CREDIT, 100.0, 0.0, 5.0, credit_rating="IG",
            is_tranche=True, attachment=0.5, detachment=0.3,
        )
        with pytest.raises(ValueError, match="tranche requires"):
            supervisory_delta(t)


class TestMaturityFactor:
    def test_unmargined_long_maturity(self) -> None:
        t = SACCRTrade(AssetClass.INTEREST_RATE, 100.0, 0.0, 10.0)
        assert maturity_factor(t) == pytest.approx(1.0)

    def test_unmargined_short_maturity(self) -> None:
        # M = 0.25 -> sqrt(0.25) = 0.5
        t = SACCRTrade(AssetClass.INTEREST_RATE, 100.0, 0.0, 0.25)
        assert maturity_factor(t) == pytest.approx(0.5)

    def test_margined(self) -> None:
        # MPOR = 10 business days = 0.04y -> 1.5*sqrt(0.04) = 0.3
        t = SACCRTrade(AssetClass.INTEREST_RATE, 100.0, 0.0, 5.0, margined_mpor=10.0 / 250.0)
        assert maturity_factor(t) == pytest.approx(1.5 * math.sqrt(0.04), abs=1e-6)


class TestSingleIRSwap:
    """Hand-verifiable single 10y USD IR swap, unmargined, V=0."""

    def _trade(self) -> SACCRTrade:
        return SACCRTrade(
            AssetClass.INTEREST_RATE, notional=100.0, start=0.0, end=10.0,
            direction=1, hedging_set="USD",
        )

    def test_adjusted_notional(self) -> None:
        # d = 100 * SD(0,10) = 100 * 7.86939 = 786.939
        assert adjusted_notional(self._trade()) == pytest.approx(786.939, abs=1e-2)

    def test_aggregate_addon(self) -> None:
        # eff notional in >5y bucket = 786.939; EffNotional = 786.939
        # AddOn = 0.005 * 786.939 = 3.93469
        assert aggregate_addon([self._trade()]) == pytest.approx(3.93469, abs=1e-3)

    def test_ead(self) -> None:
        # V=0 -> RC=0, multiplier=1, PFE=3.93469, EAD=1.4*3.93469=5.50857
        res = sa_ccr_ead([self._trade()], net_mtm=0.0)
        assert res.replacement_cost == pytest.approx(0.0)
        assert res.multiplier == pytest.approx(1.0)
        assert res.ead == pytest.approx(5.50857, abs=1e-3)


class TestReplacementCost:
    def test_unmargined_positive_mtm(self) -> None:
        assert replacement_cost(50.0, 20.0) == pytest.approx(30.0)

    def test_unmargined_floored_at_zero(self) -> None:
        assert replacement_cost(-10.0, 0.0) == pytest.approx(0.0)

    def test_margined_uses_threshold(self) -> None:
        # max(V-C, TH+MTA-NICA, 0) = max(5-0, 10+1-0, 0) = 11
        rc = replacement_cost(5.0, 0.0, margined=True, threshold=10.0, mta=1.0)
        assert rc == pytest.approx(11.0)


class TestPFEMultiplier:
    def test_unity_when_in_the_money(self) -> None:
        assert pfe_multiplier(100.0, 0.0, 50.0) == pytest.approx(1.0)

    def test_below_one_when_out_of_money(self) -> None:
        # V-C = -100, AddOn=50 -> multiplier < 1
        m = pfe_multiplier(-100.0, 0.0, 50.0)
        assert 0.05 <= m < 1.0

    def test_floor_not_breached(self) -> None:
        m = pfe_multiplier(-1e9, 0.0, 1.0)
        assert m == pytest.approx(0.05, abs=1e-6)


class TestNettingAndOffsetting:
    def test_ir_offsetting_within_currency(self) -> None:
        # A payer and receiver swap of equal size in the same bucket offset.
        long = SACCRTrade(AssetClass.INTEREST_RATE, 100.0, 0.0, 10.0, 1, "USD")
        short = SACCRTrade(AssetClass.INTEREST_RATE, 100.0, 0.0, 10.0, -1, "USD")
        assert aggregate_addon([long, short]) == pytest.approx(0.0, abs=1e-9)

    def test_fx_pairs_do_not_offset(self) -> None:
        # Different currency pairs are separate hedging sets -> add.
        a = SACCRTrade(AssetClass.FX, 100.0, 0.0, 1.0, 1, "EURUSD")
        b = SACCRTrade(AssetClass.FX, 100.0, 0.0, 1.0, -1, "GBPUSD")
        addon = aggregate_addon([a, b])
        assert addon == pytest.approx(0.04 * 100.0 + 0.04 * 100.0, abs=1e-6)

    def test_cross_asset_classes_sum(self) -> None:
        ir = SACCRTrade(AssetClass.INTEREST_RATE, 100.0, 0.0, 10.0, 1, "USD")
        fx = SACCRTrade(AssetClass.FX, 100.0, 0.0, 1.0, 1, "EURUSD")
        total = aggregate_addon([ir, fx])
        assert total == pytest.approx(
            aggregate_addon([ir]) + aggregate_addon([fx]), abs=1e-9
        )

    def test_credit_systematic_aggregation(self) -> None:
        # Two different BBB single names: partial offset via 50% correlation.
        a = SACCRTrade(AssetClass.CREDIT, 100.0, 0.0, 5.0, 1, reference="X", credit_rating="BBB")
        b = SACCRTrade(AssetClass.CREDIT, 100.0, 0.0, 5.0, 1, reference="Y", credit_rating="BBB")
        combined = aggregate_addon([a, b])
        sum_individual = aggregate_addon([a]) + aggregate_addon([b])
        # Diversified combined add-on is below the naive sum.
        assert combined < sum_individual
        assert combined > aggregate_addon([a])


class TestValidation:
    def test_empty_trades_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one trade"):
            sa_ccr_ead([])

    def test_bad_alpha(self) -> None:
        t = SACCRTrade(AssetClass.FX, 100.0, 0.0, 1.0, 1, "EURUSD")
        with pytest.raises(ValueError, match="alpha must be positive"):
            sa_ccr_ead([t], alpha=0.0)

    def test_credit_requires_rating(self) -> None:
        with pytest.raises(ValueError, match="credit_rating"):
            SACCRTrade(AssetClass.CREDIT, 100.0, 0.0, 5.0)

    def test_negative_notional(self) -> None:
        with pytest.raises(ValueError, match="notional"):
            SACCRTrade(AssetClass.FX, -1.0, 0.0, 1.0)
