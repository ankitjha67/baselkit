"""Tests for Vasicek ASRF portfolio model."""

import numpy as np
import pytest

from creditriskengine.portfolio.vasicek import (
    economic_capital_asrf,
    expected_loss,
    unexpected_loss_asrf,
    vasicek_conditional_default_rate,
    vasicek_loss_quantile,
    vasicek_portfolio_loss_distribution,
)


class TestVasicekConditionalDefaultRate:
    def test_neutral_factor(self) -> None:
        # Z=0 should give approximately PD (for small rho)
        cdr = vasicek_conditional_default_rate(0.02, 0.01, 0.0)
        assert cdr == pytest.approx(0.02, abs=0.005)

    def test_adverse_factor_increases_default(self) -> None:
        cdr_neutral = vasicek_conditional_default_rate(0.02, 0.20, 0.0)
        cdr_adverse = vasicek_conditional_default_rate(0.02, 0.20, 2.33)
        assert cdr_adverse > cdr_neutral

    def test_pd_zero(self) -> None:
        assert vasicek_conditional_default_rate(0.0, 0.20, 1.0) == 0.0

    def test_pd_one(self) -> None:
        assert vasicek_conditional_default_rate(1.0, 0.20, -1.0) == 1.0


class TestVasicekLossQuantile:
    def test_exceeds_expected_loss(self) -> None:
        var = vasicek_loss_quantile(0.02, 0.20, 0.45)
        el = 0.02 * 0.45
        assert var > el

    def test_higher_confidence_higher_loss(self) -> None:
        var_99 = vasicek_loss_quantile(0.02, 0.20, 0.45, 0.99)
        var_999 = vasicek_loss_quantile(0.02, 0.20, 0.45, 0.999)
        assert var_999 > var_99

    def test_higher_correlation_higher_tail(self) -> None:
        var_low_rho = vasicek_loss_quantile(0.02, 0.10, 0.45)
        var_high_rho = vasicek_loss_quantile(0.02, 0.30, 0.45)
        assert var_high_rho > var_low_rho


class TestVasicekConditionalDefaultRateRhoValidation:
    """Cover line 48: invalid rho raises ValueError."""

    def test_rho_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="rho must be in"):
            vasicek_conditional_default_rate(0.02, 0.0, 1.0)

    def test_rho_one_raises(self) -> None:
        with pytest.raises(ValueError, match="rho must be in"):
            vasicek_conditional_default_rate(0.02, 1.0, 1.0)

    def test_rho_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="rho must be in"):
            vasicek_conditional_default_rate(0.02, -0.5, 1.0)


class TestVasicekLossQuantileRhoValidation:
    """Cover line 78: invalid rho raises ValueError."""

    def test_rho_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="rho must be in"):
            vasicek_loss_quantile(0.02, 0.0, 0.45)

    def test_rho_one_raises(self) -> None:
        with pytest.raises(ValueError, match="rho must be in"):
            vasicek_loss_quantile(0.02, 1.0, 0.45)


class TestExpectedLoss:
    def test_basic(self) -> None:
        assert expected_loss(0.02, 0.45) == pytest.approx(0.009)

    def test_zero_pd(self) -> None:
        assert expected_loss(0.0, 0.45) == 0.0


class TestUnexpectedLoss:
    def test_positive(self) -> None:
        ul = unexpected_loss_asrf(0.02, 0.20, 0.45)
        assert ul > 0.0

    def test_ul_less_than_var(self) -> None:
        ul = unexpected_loss_asrf(0.02, 0.20, 0.45)
        var = vasicek_loss_quantile(0.02, 0.20, 0.45)
        assert ul < var


class TestEconomicCapital:
    def test_basic(self) -> None:
        result = economic_capital_asrf(0.02, 0.20, 0.45, 1_000_000.0)
        assert result["expected_loss"] == pytest.approx(9000.0, rel=1e-4)
        assert result["var"] > result["expected_loss"]
        assert result["unexpected_loss"] > 0.0
        assert result["economic_capital"] == result["unexpected_loss"]

    def test_keys(self) -> None:
        result = economic_capital_asrf(0.02, 0.20, 0.45, 1000.0)
        expected_keys = {
            "expected_loss", "var", "unexpected_loss",
            "economic_capital", "el_rate", "var_rate", "ul_rate",
        }
        assert set(result.keys()) == expected_keys


class TestPortfolioLossDistribution:
    def test_shape(self) -> None:
        losses, density = vasicek_portfolio_loss_distribution(0.02, 0.20, 0.45, 500)
        assert len(losses) == 500
        assert len(density) == 500

    def test_losses_non_negative(self) -> None:
        losses, _ = vasicek_portfolio_loss_distribution(0.02, 0.20, 0.45)
        assert np.all(losses >= 0.0)

    def test_density_non_negative(self) -> None:
        _, density = vasicek_portfolio_loss_distribution(0.02, 0.20, 0.45)
        assert np.all(density >= 0.0)
