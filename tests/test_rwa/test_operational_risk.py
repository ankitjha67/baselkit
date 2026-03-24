"""Tests for operational risk SMA — BCBS d424, OPE25."""

import math

import pytest

from creditriskengine.rwa.operational_risk import (
    _ALPHA_1,
    _ALPHA_2,
    _ALPHA_3,
    _BIC_AT_1BN,
    _BIC_AT_30BN,
    _BUCKET_1_LIMIT,
    _BUCKET_2_LIMIT,
    _LC_MULTIPLIER,
    calculate_bi,
    calculate_bic,
    calculate_ilm,
    sma_capital,
)


# ============================================================
# Business Indicator (BI)
# ============================================================


class TestCalculateBI:
    """Tests for calculate_bi()."""

    def test_simple_sum(self) -> None:
        assert calculate_bi(100.0, 200.0, 300.0) == 600.0

    def test_zeros(self) -> None:
        assert calculate_bi(0.0, 0.0, 0.0) == 0.0

    def test_single_component(self) -> None:
        assert calculate_bi(500.0, 0.0, 0.0) == 500.0
        assert calculate_bi(0.0, 500.0, 0.0) == 500.0
        assert calculate_bi(0.0, 0.0, 500.0) == 500.0

    @pytest.mark.parametrize(
        "ildc, sc, fc",
        [
            (-1, 0, 0),
            (0, -1, 0),
            (0, 0, -1),
            (-10, -20, -30),
        ],
    )
    def test_negative_components_raise(self, ildc: float, sc: float, fc: float) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            calculate_bi(ildc, sc, fc)

    def test_large_values(self) -> None:
        bi = calculate_bi(10e9, 20e9, 5e9)
        assert bi == pytest.approx(35e9)


# ============================================================
# Business Indicator Component (BIC)
# ============================================================


class TestCalculateBIC:
    """Tests for calculate_bic()."""

    def test_zero_bi(self) -> None:
        assert calculate_bic(0.0) == 0.0

    def test_negative_bi_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            calculate_bic(-1.0)

    # --- Bucket 1: BI <= 1 bn ---

    @pytest.mark.parametrize(
        "bi, expected",
        [
            (0.0, 0.0),
            (500_000_000.0, _ALPHA_1 * 500_000_000.0),
            (_BUCKET_1_LIMIT, _BIC_AT_1BN),
        ],
    )
    def test_bucket_1(self, bi: float, expected: float) -> None:
        assert calculate_bic(bi) == pytest.approx(expected)

    # --- Bucket 2: 1 bn < BI <= 30 bn ---

    @pytest.mark.parametrize(
        "bi",
        [
            _BUCKET_1_LIMIT + 1.0,
            5_000_000_000.0,
            15_000_000_000.0,
            _BUCKET_2_LIMIT,
        ],
    )
    def test_bucket_2(self, bi: float) -> None:
        expected = _BIC_AT_1BN + _ALPHA_2 * (bi - _BUCKET_1_LIMIT)
        assert calculate_bic(bi) == pytest.approx(expected)

    def test_bucket_2_boundary_exact(self) -> None:
        """BIC at BI = 30 bn must equal the precomputed boundary."""
        assert calculate_bic(_BUCKET_2_LIMIT) == pytest.approx(_BIC_AT_30BN)

    # --- Bucket 3: BI > 30 bn ---

    @pytest.mark.parametrize(
        "bi",
        [
            _BUCKET_2_LIMIT + 1.0,
            50_000_000_000.0,
            100_000_000_000.0,
        ],
    )
    def test_bucket_3(self, bi: float) -> None:
        expected = _BIC_AT_30BN + _ALPHA_3 * (bi - _BUCKET_2_LIMIT)
        assert calculate_bic(bi) == pytest.approx(expected)

    # --- Continuity at boundaries ---

    def test_continuity_at_1bn(self) -> None:
        below = calculate_bic(_BUCKET_1_LIMIT)
        above = calculate_bic(_BUCKET_1_LIMIT + 0.01)
        assert above > below
        assert above == pytest.approx(below, abs=1.0)  # nearly continuous

    def test_continuity_at_30bn(self) -> None:
        below = calculate_bic(_BUCKET_2_LIMIT)
        above = calculate_bic(_BUCKET_2_LIMIT + 0.01)
        assert above > below
        assert above == pytest.approx(below, abs=1.0)

    # --- Monotonicity ---

    def test_monotonically_increasing(self) -> None:
        values = [0, 500e6, 1e9, 5e9, 30e9, 50e9, 100e9]
        bics = [calculate_bic(v) for v in values]
        for i in range(1, len(bics)):
            assert bics[i] > bics[i - 1]

    # --- Slope increases across buckets ---

    def test_marginal_rate_increases(self) -> None:
        """Higher buckets have steeper marginal coefficients."""
        assert _ALPHA_1 < _ALPHA_2 < _ALPHA_3


# ============================================================
# Internal Loss Multiplier (ILM)
# ============================================================


class TestCalculateILM:
    """Tests for calculate_ilm()."""

    def test_zero_bic_returns_one(self) -> None:
        assert calculate_ilm(0.0, 0.0) == 1.0
        assert calculate_ilm(100.0, 0.0) == 1.0

    def test_negative_lc_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            calculate_ilm(-1.0, 100.0)

    def test_negative_bic_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            calculate_ilm(100.0, -1.0)

    def test_zero_lc(self) -> None:
        """With no losses, ILM = ln(e - 1) ~ 0.5413."""
        bic = 120_000_000.0
        ilm = calculate_ilm(0.0, bic)
        expected = math.log(math.exp(1) - 1)
        assert ilm == pytest.approx(expected)

    def test_lc_equals_bic(self) -> None:
        """When LC = BIC, ratio = 1, ILM = ln(e - 1 + 1) = 1.0."""
        bic = 500_000_000.0
        ilm = calculate_ilm(bic, bic)
        assert ilm == pytest.approx(1.0)

    def test_lc_greater_than_bic(self) -> None:
        """When LC > BIC, ILM > 1."""
        bic = 100_000_000.0
        lc = 500_000_000.0
        ilm = calculate_ilm(lc, bic)
        assert ilm > 1.0

    def test_lc_less_than_bic(self) -> None:
        """When LC < BIC, ILM < 1."""
        bic = 500_000_000.0
        lc = 100_000_000.0
        ilm = calculate_ilm(lc, bic)
        assert ilm < 1.0

    def test_formula_explicit(self) -> None:
        lc = 200_000_000.0
        bic = 400_000_000.0
        expected = math.log(math.exp(1) - 1 + (lc / bic) ** 0.8)
        assert calculate_ilm(lc, bic) == pytest.approx(expected)

    def test_ilm_increases_with_lc(self) -> None:
        bic = 300_000_000.0
        ilm_low = calculate_ilm(50_000_000.0, bic)
        ilm_high = calculate_ilm(500_000_000.0, bic)
        assert ilm_high > ilm_low


# ============================================================
# SMA capital
# ============================================================


class TestSMACapital:
    """Tests for sma_capital()."""

    def test_bucket_1_no_ilm(self) -> None:
        bi = 500_000_000.0
        result = sma_capital(bi, use_ilm=False)
        expected_bic = _ALPHA_1 * bi
        assert result["bi"] == bi
        assert result["bic"] == pytest.approx(expected_bic)
        assert result["ilm"] == 1.0
        assert result["lc"] is None
        assert result["capital"] == pytest.approx(expected_bic)

    def test_bucket_2_no_ilm(self) -> None:
        bi = 10_000_000_000.0
        result = sma_capital(bi, use_ilm=False)
        expected_bic = _BIC_AT_1BN + _ALPHA_2 * (bi - _BUCKET_1_LIMIT)
        assert result["capital"] == pytest.approx(expected_bic)

    def test_bucket_3_no_ilm(self) -> None:
        bi = 50_000_000_000.0
        result = sma_capital(bi, use_ilm=False)
        expected_bic = _BIC_AT_30BN + _ALPHA_3 * (bi - _BUCKET_2_LIMIT)
        assert result["capital"] == pytest.approx(expected_bic)

    def test_with_ilm(self) -> None:
        bi = 5_000_000_000.0
        avg_loss = 100_000_000.0
        result = sma_capital(bi, average_annual_loss=avg_loss, use_ilm=True)
        bic = calculate_bic(bi)
        lc = _LC_MULTIPLIER * avg_loss
        ilm = calculate_ilm(lc, bic)
        assert result["lc"] == pytest.approx(lc)
        assert result["ilm"] == pytest.approx(ilm)
        assert result["capital"] == pytest.approx(bic * ilm)

    def test_ilm_true_but_no_loss_data(self) -> None:
        """use_ilm=True but average_annual_loss=None -> ILM defaults to 1."""
        bi = 5_000_000_000.0
        result = sma_capital(bi, average_annual_loss=None, use_ilm=True)
        assert result["ilm"] == 1.0
        assert result["lc"] is None

    def test_ilm_false_ignores_loss_data(self) -> None:
        """use_ilm=False should ignore loss data and set ILM=1."""
        bi = 5_000_000_000.0
        result = sma_capital(bi, average_annual_loss=200_000_000.0, use_ilm=False)
        assert result["ilm"] == 1.0
        assert result["lc"] is None

    def test_zero_bi(self) -> None:
        result = sma_capital(0.0)
        assert result["bic"] == 0.0
        assert result["capital"] == 0.0

    def test_negative_loss_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            sma_capital(1_000_000_000.0, average_annual_loss=-1.0, use_ilm=True)

    def test_high_losses_increase_capital(self) -> None:
        bi = 10_000_000_000.0
        low = sma_capital(bi, average_annual_loss=10_000_000.0, use_ilm=True)
        high = sma_capital(bi, average_annual_loss=500_000_000.0, use_ilm=True)
        assert high["capital"] > low["capital"]

    def test_zero_loss_with_ilm(self) -> None:
        """Zero average loss with ILM enabled -> ILM = ln(e-1) < 1."""
        bi = 5_000_000_000.0
        result = sma_capital(bi, average_annual_loss=0.0, use_ilm=True)
        assert result["ilm"] < 1.0
        assert result["capital"] < result["bic"]

    def test_return_keys(self) -> None:
        result = sma_capital(1_000_000_000.0)
        assert set(result.keys()) == {"bi", "bic", "lc", "ilm", "capital"}

    def test_lc_multiplier_is_15(self) -> None:
        """Verify LC = 15 x average annual loss."""
        bi = 5_000_000_000.0
        avg = 100_000_000.0
        result = sma_capital(bi, average_annual_loss=avg, use_ilm=True)
        assert result["lc"] == pytest.approx(15 * avg)
