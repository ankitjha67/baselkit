"""Tests for CECL (ASC 326) calculations."""

import numpy as np
import pytest

from creditriskengine.ecl.cecl.cecl_calc import cecl_loss_rate, cecl_pd_lgd
from creditriskengine.ecl.cecl.methods import dcf_method, vintage_analysis, warm_method
from creditriskengine.ecl.cecl.qualitative import (
    QualitativeFactor,
    apply_q_factors,
    apply_q_factors_with_caps,
    q_factor_summary,
    total_q_factor_adjustment,
    validate_q_factors,
)


class TestCECLPDLGD:
    def test_basic(self) -> None:
        pds = np.array([0.02, 0.02, 0.02])
        ecl = cecl_pd_lgd(pds, lgds=0.45, eads=1000.0)
        # 3 * 0.02 * 0.45 * 1000 = 27.0 (no discounting)
        assert ecl == pytest.approx(27.0)

    def test_with_discounting(self) -> None:
        pds = np.array([0.02, 0.02])
        ecl_no_disc = cecl_pd_lgd(pds, 0.45, 1000.0, discount_rate=0.0)
        ecl_disc = cecl_pd_lgd(pds, 0.45, 1000.0, discount_rate=0.05)
        assert ecl_disc < ecl_no_disc


class TestCECLLossRate:
    def test_basic(self) -> None:
        ecl = cecl_loss_rate(ead=1000.0, historical_loss_rate=0.01, remaining_life_years=3.0)
        assert ecl == pytest.approx(30.0)

    def test_with_adjustments(self) -> None:
        ecl = cecl_loss_rate(
            ead=1000.0,
            historical_loss_rate=0.01,
            qualitative_adjustment=0.005,
            forecast_adjustment=0.002,
            remaining_life_years=2.0,
        )
        assert ecl == pytest.approx(1000.0 * 0.017 * 2.0)

    def test_floor_at_zero(self) -> None:
        ecl = cecl_loss_rate(ead=1000.0, historical_loss_rate=0.01, qualitative_adjustment=-0.05)
        assert ecl == pytest.approx(0.0)


class TestWARMMethod:
    def test_basic(self) -> None:
        ecl = warm_method(ead=1000.0, historical_loss_rate=0.01, remaining_life_years=3.0)
        assert ecl == pytest.approx(30.0)

    def test_with_q_factor(self) -> None:
        ecl = warm_method(
            ead=1000.0, historical_loss_rate=0.01,
            remaining_life_years=3.0, qualitative_factor=0.5,
        )
        assert ecl == pytest.approx(1000.0 * 0.015 * 3.0)


class TestVintageAnalysis:
    def test_basic(self) -> None:
        # 2 vintages, 3 ages; vintage 0 is at age 1, vintage 1 at age 0
        matrix = np.array([
            [0.01, 0.03, 0.0],
            [0.02, 0.0, 0.0],
        ])
        balances = np.array([100.0, 200.0])
        ecl = vintage_analysis(matrix, balances)
        # Ultimate loss (max of last column) = max(0.0, 0.0) = 0 → fallback to max of all last col
        # Actually: np.max(matrix[:, -1]) = 0.0
        assert ecl == pytest.approx(0.0)

    def test_with_ultimate_losses(self) -> None:
        matrix = np.array([
            [0.01, 0.03, 0.05],
            [0.02, 0.0, 0.0],
        ])
        balances = np.array([100.0, 200.0])
        ecl = vintage_analysis(matrix, balances)
        # Ultimate = max of col[-1] = 0.05
        # Vintage 0: at age 2, cum=0.05, remaining=0.0
        # Vintage 1: at age 0, cum=0.02, remaining=0.03
        # Total = 0 + 200*0.03 = 6.0
        assert ecl == pytest.approx(6.0)


class TestDCFMethod:
    def test_basic(self) -> None:
        contractual = np.array([100.0, 100.0, 100.0])
        expected = np.array([95.0, 90.0, 85.0])
        ecl = dcf_method(contractual, expected, discount_rate=0.05)
        # PV diff should be positive
        assert ecl > 0
        # Manual: PV(contractual) - PV(expected)
        dfs = 1.0 / (1.05 ** np.array([1, 2, 3]))
        expected_ecl = float(np.sum((contractual - expected) * dfs))
        assert ecl == pytest.approx(expected_ecl)

    def test_no_losses(self) -> None:
        cf = np.array([100.0, 100.0])
        assert dcf_method(cf, cf, discount_rate=0.05) == pytest.approx(0.0)


class TestQualitativeFactors:
    def test_total_adjustment(self) -> None:
        factors = [
            QualitativeFactor(name="econ", adjustment_bps=25),
            QualitativeFactor(name="portfolio", adjustment_bps=-10),
        ]
        assert total_q_factor_adjustment(factors) == pytest.approx(15 / 10_000)

    def test_apply_q_factors(self) -> None:
        factors = [QualitativeFactor(name="econ", adjustment_bps=50)]
        adjusted = apply_q_factors(0.01, factors)
        assert adjusted == pytest.approx(0.01 + 50 / 10_000)

    def test_apply_q_factors_floor(self) -> None:
        factors = [QualitativeFactor(name="negative", adjustment_bps=-200)]
        adjusted = apply_q_factors(0.01, factors, floor=0.005)
        assert adjusted == pytest.approx(0.005)


class TestQFactorCaps:
    def test_within_cap_no_warning(self) -> None:
        factors = [QualitativeFactor(name="e", adjustment_bps=50, category="economic_conditions")]
        adjusted, warnings = apply_q_factors_with_caps(0.01, factors)
        assert adjusted == pytest.approx(0.01 + 50 / 10_000)
        assert warnings == []

    def test_positive_cap_applied(self) -> None:
        # economic_conditions cap = 150 bps; raw 300 -> capped to 150.
        factors = [
            QualitativeFactor(name="a", adjustment_bps=200, category="economic_conditions"),
            QualitativeFactor(name="b", adjustment_bps=100, category="economic_conditions"),
        ]
        adjusted, warnings = apply_q_factors_with_caps(0.01, factors)
        assert adjusted == pytest.approx(0.01 + 150 / 10_000)
        assert len(warnings) == 1 and "capped" in warnings[0]

    def test_negative_cap_applied(self) -> None:
        factors = [QualitativeFactor(name="a", adjustment_bps=-300, category="staff_experience")]
        # staff_experience cap = 50 -> -50 bps.
        adjusted, warnings = apply_q_factors_with_caps(0.05, factors)
        assert adjusted == pytest.approx(0.05 - 50 / 10_000)
        assert len(warnings) == 1

    def test_inactive_factor_skipped(self) -> None:
        factors = [
            QualitativeFactor(name="a", adjustment_bps=100, category="economic_conditions"),
            QualitativeFactor(
                name="b", adjustment_bps=999, category="economic_conditions", is_active=False
            ),
        ]
        adjusted, warnings = apply_q_factors_with_caps(0.0, factors)
        assert adjusted == pytest.approx(100 / 10_000)
        assert warnings == []

    def test_custom_caps_and_floor(self) -> None:
        factors = [QualitativeFactor(name="a", adjustment_bps=-500, category="general")]
        adjusted, _ = apply_q_factors_with_caps(
            0.01, factors, category_caps_bps={"general": 30.0}, floor=0.002
        )
        # -30 bps -> 0.01 - 0.003 = 0.007 (above floor)
        assert adjusted == pytest.approx(0.007)

    def test_unknown_category_uses_200_fallback(self) -> None:
        # Non-empty caps without a "general" key -> 200 bps hard fallback.
        factors = [QualitativeFactor(name="a", adjustment_bps=500, category="weird")]
        adjusted, warnings = apply_q_factors_with_caps(
            0.0, factors, category_caps_bps={"economic_conditions": 50.0}
        )
        assert adjusted == pytest.approx(200 / 10_000)
        assert len(warnings) == 1


class TestValidateQFactors:
    def test_complete_factor_only_category_warning(self) -> None:
        from datetime import datetime

        factors = [
            QualitativeFactor(
                name="econ", adjustment_bps=25, category="economic_conditions",
                rationale="r", approved_by="ALLL", approval_date=datetime(2026, 1, 1),
            )
        ]
        warnings = validate_q_factors(factors)
        # Only the "categories not addressed" warning remains.
        assert any("Categories not addressed" in w for w in warnings)
        assert not any("missing" in w for w in warnings)

    def test_governance_gaps_flagged(self) -> None:
        factors = [QualitativeFactor(name="bad", adjustment_bps=0.0, category="weird")]
        warnings = validate_q_factors(factors)
        joined = " ".join(warnings)
        assert "missing rationale" in joined
        assert "missing approval authority" in joined
        assert "missing approval date" in joined
        assert "zero adjustment" in joined
        assert "non-standard category" in joined

    def test_inactive_factor_not_validated(self) -> None:
        factors = [QualitativeFactor(name="x", is_active=False)]
        warnings = validate_q_factors(factors)
        assert not any("Q-factor 'x'" in w for w in warnings)


class TestQFactorSummary:
    def test_summary_fields(self) -> None:
        factors = [
            QualitativeFactor(name="a", adjustment_bps=30, category="economic_conditions"),
            QualitativeFactor(name="b", adjustment_bps=20, category="portfolio_trends"),
            QualitativeFactor(name="c", adjustment_bps=99, is_active=False),
        ]
        summary = q_factor_summary(factors, base_loss_rate=0.01)
        assert summary["n_active_factors"] == 2
        assert summary["n_inactive_factors"] == 1
        assert summary["total_adjustment_bps"] == pytest.approx(50.0)
        assert summary["adjusted_loss_rate"] == pytest.approx(0.01 + 50 / 10_000)
        assert "economic_conditions" in str(summary["adjustment_by_category"])
