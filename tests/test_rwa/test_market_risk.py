"""Tests for FRTB market risk integration — BCBS d424, MAR."""

import math

import pytest

from creditriskengine.rwa.market_risk import (
    RRAO_EXOTIC_RATE,
    RRAO_OTHER_RATE,
    FRTBApproach,
    calculate_drc,
    calculate_rrao,
    calculate_sa_market_risk,
    calculate_sbm_credit_spread,
    total_market_risk_capital,
)


# ============================================================
# FRTBApproach enum
# ============================================================


class TestFRTBApproach:
    """Tests for the FRTBApproach enum."""

    def test_standardised(self) -> None:
        assert FRTBApproach.STANDARDISED.value == "standardised"

    def test_ima(self) -> None:
        assert FRTBApproach.IMA.value == "internal_models_approach"

    def test_members(self) -> None:
        names = {m.name for m in FRTBApproach}
        assert names == {"STANDARDISED", "IMA"}


# ============================================================
# SbM credit spread risk
# ============================================================


class TestSbMCreditSpread:
    """Tests for calculate_sbm_credit_spread()."""

    def test_single_factor(self) -> None:
        """Single factor: K = |WS| = |s * rw|."""
        k = calculate_sbm_credit_spread(
            sensitivities=[100.0],
            risk_weights=[0.05],
            correlations=[[1.0]],
        )
        assert k == pytest.approx(5.0)

    def test_two_factors_perfect_correlation(self) -> None:
        """With rho=1, K = |WS1 + WS2|."""
        k = calculate_sbm_credit_spread(
            sensitivities=[100.0, 200.0],
            risk_weights=[0.05, 0.10],
            correlations=[[1.0, 1.0], [1.0, 1.0]],
        )
        ws1, ws2 = 100 * 0.05, 200 * 0.10
        expected = ws1 + ws2
        assert k == pytest.approx(expected)

    def test_two_factors_zero_correlation(self) -> None:
        """With rho=0, K = sqrt(WS1^2 + WS2^2)."""
        k = calculate_sbm_credit_spread(
            sensitivities=[100.0, 200.0],
            risk_weights=[0.05, 0.10],
            correlations=[[1.0, 0.0], [0.0, 1.0]],
        )
        ws1, ws2 = 100 * 0.05, 200 * 0.10
        expected = math.sqrt(ws1 ** 2 + ws2 ** 2)
        assert k == pytest.approx(expected)

    def test_two_factors_negative_correlation(self) -> None:
        """Negative correlation reduces capital."""
        k_zero = calculate_sbm_credit_spread(
            sensitivities=[100.0, 100.0],
            risk_weights=[0.10, 0.10],
            correlations=[[1.0, 0.0], [0.0, 1.0]],
        )
        k_neg = calculate_sbm_credit_spread(
            sensitivities=[100.0, 100.0],
            risk_weights=[0.10, 0.10],
            correlations=[[1.0, -0.5], [-0.5, 1.0]],
        )
        assert k_neg < k_zero

    def test_negative_total_floors_to_zero(self) -> None:
        """Strong anti-correlation can floor total at zero."""
        k = calculate_sbm_credit_spread(
            sensitivities=[100.0, 100.0],
            risk_weights=[0.10, 0.10],
            correlations=[[1.0, -1.5], [-1.5, 1.0]],  # extreme
        )
        assert k == 0.0

    def test_three_factors(self) -> None:
        s = [100.0, 200.0, 150.0]
        rw = [0.05, 0.10, 0.08]
        corr = [
            [1.0, 0.3, 0.2],
            [0.3, 1.0, 0.5],
            [0.2, 0.5, 1.0],
        ]
        ws = [si * ri for si, ri in zip(s, rw)]
        total = sum(w ** 2 for w in ws)
        for i in range(3):
            for j in range(i + 1, 3):
                total += 2 * corr[i][j] * ws[i] * ws[j]
        expected = math.sqrt(max(total, 0.0))
        k = calculate_sbm_credit_spread(s, rw, corr)
        assert k == pytest.approx(expected)

    # --- Validation ---

    def test_empty_sensitivities_raises(self) -> None:
        with pytest.raises(ValueError, match="not be empty"):
            calculate_sbm_credit_spread([], [], [])

    def test_mismatched_risk_weights_raises(self) -> None:
        with pytest.raises(ValueError, match="risk_weights length"):
            calculate_sbm_credit_spread([1.0], [0.1, 0.2], [[1.0]])

    def test_mismatched_correlation_rows_raises(self) -> None:
        with pytest.raises(ValueError, match="correlations must be"):
            calculate_sbm_credit_spread([1.0], [0.1], [[1.0], [1.0]])

    def test_mismatched_correlation_cols_raises(self) -> None:
        with pytest.raises(ValueError, match="correlations row"):
            calculate_sbm_credit_spread(
                [1.0, 2.0], [0.1, 0.2], [[1.0, 0.5], [0.5]],
            )

    def test_negative_sensitivities(self) -> None:
        """Negative sensitivities are valid (short positions)."""
        k = calculate_sbm_credit_spread(
            sensitivities=[-100.0],
            risk_weights=[0.05],
            correlations=[[1.0]],
        )
        assert k == pytest.approx(5.0)


# ============================================================
# Default Risk Charge
# ============================================================


class TestDRC:
    """Tests for calculate_drc()."""

    def test_single_position(self) -> None:
        drc = calculate_drc(
            lgds=[0.6], notionals=[1_000_000.0], risk_weights=[0.03],
        )
        assert drc == pytest.approx(0.6 * 1_000_000.0 * 0.03)

    def test_multiple_positions(self) -> None:
        drc = calculate_drc(
            lgds=[0.6, 0.45],
            notionals=[1_000_000.0, 2_000_000.0],
            risk_weights=[0.03, 0.05],
        )
        expected = 0.6 * 1e6 * 0.03 + 0.45 * 2e6 * 0.05
        assert drc == pytest.approx(expected)

    def test_zero_lgd(self) -> None:
        drc = calculate_drc([0.0], [1_000_000.0], [0.03])
        assert drc == pytest.approx(0.0)

    def test_zero_notional(self) -> None:
        drc = calculate_drc([0.6], [0.0], [0.03])
        assert drc == pytest.approx(0.0)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="not be empty"):
            calculate_drc([], [], [])

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            calculate_drc([0.6], [1e6, 2e6], [0.03])

    def test_mismatched_rw_length_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            calculate_drc([0.6], [1e6], [0.03, 0.05])


# ============================================================
# RRAO
# ============================================================


class TestRRAO:
    """Tests for calculate_rrao()."""

    def test_exotic_only(self) -> None:
        rrao = calculate_rrao(exotic_gross_notional=10_000_000.0)
        assert rrao == pytest.approx(RRAO_EXOTIC_RATE * 10_000_000.0)

    def test_other_only(self) -> None:
        rrao = calculate_rrao(other_gross_notional=10_000_000.0)
        assert rrao == pytest.approx(RRAO_OTHER_RATE * 10_000_000.0)

    def test_both(self) -> None:
        rrao = calculate_rrao(
            exotic_gross_notional=5_000_000.0,
            other_gross_notional=20_000_000.0,
        )
        expected = 0.01 * 5e6 + 0.001 * 20e6
        assert rrao == pytest.approx(expected)

    def test_zero(self) -> None:
        assert calculate_rrao() == 0.0

    def test_negative_exotic_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            calculate_rrao(exotic_gross_notional=-1.0)

    def test_negative_other_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            calculate_rrao(other_gross_notional=-1.0)

    def test_rates(self) -> None:
        assert RRAO_EXOTIC_RATE == 0.01
        assert RRAO_OTHER_RATE == 0.001


# ============================================================
# Total market risk capital
# ============================================================


class TestTotalMarketRiskCapital:
    """Tests for total_market_risk_capital()."""

    def test_sum(self) -> None:
        result = total_market_risk_capital(100.0, 50.0, 10.0)
        assert result["total"] == pytest.approx(160.0)

    def test_keys(self) -> None:
        result = total_market_risk_capital(1.0, 2.0, 3.0)
        assert set(result.keys()) == {"sbm", "drc", "rrao", "total"}

    def test_components_preserved(self) -> None:
        result = total_market_risk_capital(11.0, 22.0, 33.0)
        assert result["sbm"] == pytest.approx(11.0)
        assert result["drc"] == pytest.approx(22.0)
        assert result["rrao"] == pytest.approx(33.0)

    def test_all_zero(self) -> None:
        result = total_market_risk_capital(0.0, 0.0, 0.0)
        assert result["total"] == 0.0


# ============================================================
# SA market risk — integrated
# ============================================================


class TestCalculateSAMarketRisk:
    """Tests for calculate_sa_market_risk()."""

    def test_sbm_only(self) -> None:
        result = calculate_sa_market_risk(
            credit_spread_sensitivities=[100.0],
            risk_weights=[0.05],
            correlations=[[1.0]],
        )
        assert result["sbm"] == pytest.approx(5.0)
        assert result["drc"] == 0.0
        assert result["rrao"] == 0.0
        assert result["total"] == pytest.approx(5.0)

    def test_with_drc(self) -> None:
        result = calculate_sa_market_risk(
            credit_spread_sensitivities=[100.0],
            risk_weights=[0.05],
            correlations=[[1.0]],
            drc_lgds=[0.6],
            drc_notionals=[1_000_000.0],
            drc_risk_weights=[0.03],
        )
        assert result["drc"] == pytest.approx(0.6 * 1e6 * 0.03)

    def test_with_rrao(self) -> None:
        result = calculate_sa_market_risk(
            credit_spread_sensitivities=[100.0],
            risk_weights=[0.05],
            correlations=[[1.0]],
            exotic_gross_notional=1_000_000.0,
            other_gross_notional=5_000_000.0,
        )
        expected_rrao = 0.01 * 1e6 + 0.001 * 5e6
        assert result["rrao"] == pytest.approx(expected_rrao)

    def test_full_combination(self) -> None:
        result = calculate_sa_market_risk(
            credit_spread_sensitivities=[100.0, 200.0],
            risk_weights=[0.05, 0.10],
            correlations=[[1.0, 0.5], [0.5, 1.0]],
            drc_lgds=[0.6],
            drc_notionals=[1_000_000.0],
            drc_risk_weights=[0.03],
            exotic_gross_notional=500_000.0,
            other_gross_notional=2_000_000.0,
        )
        assert result["total"] == pytest.approx(
            result["sbm"] + result["drc"] + result["rrao"]
        )

    def test_partial_drc_none_ignored(self) -> None:
        """If only some DRC args provided, DRC should be 0."""
        result = calculate_sa_market_risk(
            credit_spread_sensitivities=[100.0],
            risk_weights=[0.05],
            correlations=[[1.0]],
            drc_lgds=[0.6],
            drc_notionals=None,
            drc_risk_weights=None,
        )
        assert result["drc"] == 0.0
