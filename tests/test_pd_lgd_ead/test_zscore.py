"""Tests for Altman Z-Score models."""

import pytest

from creditriskengine.models.pd.zscore import (
    altman_z_score,
    altman_z_score_emerging,
    altman_z_score_private,
    z_score_zone,
)


class TestAltmanZScore:
    """Original Altman Z-Score for public manufacturing firms."""

    def test_healthy_firm(self) -> None:
        z = altman_z_score(
            working_capital=500.0,
            total_assets=2000.0,
            retained_earnings=400.0,
            ebit=300.0,
            market_equity=1500.0,
            total_liabilities=500.0,
            sales=3000.0,
        )
        # X1=0.25, X2=0.20, X3=0.15, X4=3.0, X5=1.5
        # Z = 1.2*0.25 + 1.4*0.20 + 3.3*0.15 + 0.6*3.0 + 1.0*1.5
        #   = 0.30 + 0.28 + 0.495 + 1.80 + 1.50 = 4.375
        assert z == pytest.approx(4.375)

    def test_distressed_firm(self) -> None:
        z = altman_z_score(
            working_capital=-100.0,
            total_assets=1000.0,
            retained_earnings=-200.0,
            ebit=-50.0,
            market_equity=100.0,
            total_liabilities=900.0,
            sales=500.0,
        )
        assert z < 1.81  # Distress zone

    def test_zero_total_assets_raises(self) -> None:
        with pytest.raises(ValueError, match="total_assets must be positive"):
            altman_z_score(0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0)

    def test_zero_liabilities_raises(self) -> None:
        with pytest.raises(ValueError, match="total_liabilities must be positive"):
            altman_z_score(0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class TestAltmanZScorePrivate:
    """Z'-Score for private firms."""

    def test_basic_computation(self) -> None:
        z = altman_z_score_private(
            working_capital=200.0,
            total_assets=1000.0,
            retained_earnings=150.0,
            ebit=100.0,
            book_equity=400.0,
            total_liabilities=600.0,
            sales=1200.0,
        )
        # X1=0.20, X2=0.15, X3=0.10, X4=0.6667, X5=1.20
        # Z' = 0.717*0.20 + 0.847*0.15 + 3.107*0.10 + 0.420*0.6667 + 0.998*1.20
        x1, x2, x3, x4, x5 = 0.20, 0.15, 0.10, 400.0 / 600.0, 1.20
        expected = 0.717 * x1 + 0.847 * x2 + 3.107 * x3 + 0.420 * x4 + 0.998 * x5
        assert z == pytest.approx(expected)

    def test_zero_total_assets_raises(self) -> None:
        with pytest.raises(ValueError, match="total_assets must be positive"):
            altman_z_score_private(0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0)

    def test_zero_liabilities_raises(self) -> None:
        with pytest.raises(ValueError, match="total_liabilities must be positive"):
            altman_z_score_private(0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class TestAltmanZScoreEmerging:
    """Z''-Score for emerging-market corporates."""

    def test_basic_computation(self) -> None:
        z = altman_z_score_emerging(
            working_capital=300.0,
            total_assets=1000.0,
            retained_earnings=200.0,
            ebit=150.0,
            book_equity=500.0,
            total_liabilities=500.0,
        )
        # X1=0.30, X2=0.20, X3=0.15, X4=1.0
        # Z'' = 6.56*0.30 + 3.26*0.20 + 6.72*0.15 + 1.05*1.0
        expected = 6.56 * 0.30 + 3.26 * 0.20 + 6.72 * 0.15 + 1.05 * 1.0
        assert z == pytest.approx(expected)

    def test_no_sales_term(self) -> None:
        """Z'' model has no sales term -- verify it doesn't appear."""
        z1 = altman_z_score_emerging(100.0, 1000.0, 50.0, 80.0, 400.0, 600.0)
        z2 = altman_z_score_emerging(100.0, 1000.0, 50.0, 80.0, 400.0, 600.0)
        assert z1 == pytest.approx(z2)

    def test_zero_total_assets_raises(self) -> None:
        with pytest.raises(ValueError, match="total_assets must be positive"):
            altman_z_score_emerging(0.0, 0.0, 0.0, 0.0, 0.0, 100.0)

    def test_zero_liabilities_raises(self) -> None:
        with pytest.raises(ValueError, match="total_liabilities must be positive"):
            altman_z_score_emerging(0.0, 1000.0, 0.0, 0.0, 0.0, 0.0)


class TestZScoreZone:
    """Zone classification based on Z-Score thresholds."""

    def test_original_safe(self) -> None:
        assert z_score_zone(3.5, model="original") == "safe"

    def test_original_grey(self) -> None:
        assert z_score_zone(2.5, model="original") == "grey"

    def test_original_distress(self) -> None:
        assert z_score_zone(1.0, model="original") == "distress"

    def test_original_boundary_safe(self) -> None:
        # Z > 2.99 is safe; Z = 2.99 is grey
        assert z_score_zone(2.99, model="original") == "grey"
        assert z_score_zone(3.0, model="original") == "safe"

    def test_original_boundary_distress(self) -> None:
        # Z < 1.81 is distress; Z = 1.81 is grey
        assert z_score_zone(1.81, model="original") == "grey"
        assert z_score_zone(1.80, model="original") == "distress"

    def test_private_safe(self) -> None:
        assert z_score_zone(3.0, model="private") == "safe"

    def test_private_grey(self) -> None:
        assert z_score_zone(2.0, model="private") == "grey"

    def test_private_distress(self) -> None:
        assert z_score_zone(1.0, model="private") == "distress"

    def test_emerging_safe(self) -> None:
        assert z_score_zone(3.0, model="emerging") == "safe"

    def test_emerging_grey(self) -> None:
        assert z_score_zone(2.0, model="emerging") == "grey"

    def test_emerging_distress(self) -> None:
        assert z_score_zone(0.5, model="emerging") == "distress"

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown model"):
            z_score_zone(2.5, model="nonexistent")
