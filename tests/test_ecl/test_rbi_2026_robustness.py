"""Tests for the RBI ECL 2026 parameter self-check and parallel-run comparator."""

from __future__ import annotations

import numpy as np
import pytest

from creditriskengine.core.types import IFRS9Stage
from creditriskengine.ecl.ind_as109 import (
    IRACAssetClass,
    ParallelRunResult,
    RBIExposureCategory,
    RBIParameterMismatch,
    assert_rbi_2026_parameters_match_published,
    parallel_run,
    portfolio_parallel_run_summary,
    regulatory_self_check,
)

# ----------------------------------------------------------------------
# Parameter self-check
# ----------------------------------------------------------------------


class TestParameterSelfCheck:
    def test_passes_on_unmodified_library(self) -> None:
        # Should not raise; all live params match published values
        assert_rbi_2026_parameters_match_published()

    def test_regulatory_self_check_ok(self) -> None:
        report = regulatory_self_check()
        assert report["status"] == "ok"
        assert "RBI/DOR/2026-27/398" in report["reference"]

    def test_mismatch_exception_subclasses_assertion(self) -> None:
        # Sanity: RBIParameterMismatch should be an AssertionError subclass
        assert issubclass(RBIParameterMismatch, AssertionError)


class TestParameterMismatchRaises:
    """Trip each raise path by perturbing the live constants via monkeypatch.

    We never edit the source constants; we patch the names as referenced
    in the ``parameter_assertions`` module namespace so the published-value
    guard fails as designed.
    """

    def test_close_assert_raises_on_pd_floor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from creditriskengine.ecl.ind_as109 import parameter_assertions as pa
        monkeypatch.setattr(pa, "RBI_PD_FLOOR", 0.99)
        with pytest.raises(RBIParameterMismatch, match="PD floor"):
            assert_rbi_2026_parameters_match_published()

    def test_effective_date_mismatch_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from datetime import date

        from creditriskengine.ecl.ind_as109 import parameter_assertions as pa
        monkeypatch.setattr(pa, "RBI_ECL_EFFECTIVE_DATE", date(2099, 1, 1))
        with pytest.raises(RBIParameterMismatch, match="Effective date"):
            assert_rbi_2026_parameters_match_published()

    def test_eir_deadline_mismatch_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from datetime import date

        from creditriskengine.ecl.ind_as109 import parameter_assertions as pa
        monkeypatch.setattr(pa, "RBI_EIR_MIGRATION_DEADLINE", date(2099, 1, 1))
        with pytest.raises(RBIParameterMismatch, match="EIR deadline"):
            assert_rbi_2026_parameters_match_published()

    def test_capital_add_back_mismatch_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from creditriskengine.ecl.ind_as109 import parameter_assertions as pa
        monkeypatch.setattr(pa, "CAPITAL_ADD_BACK_SCHEDULE", {2028: 0.0})
        with pytest.raises(RBIParameterMismatch, match="Capital add-back"):
            assert_rbi_2026_parameters_match_published()

    def test_stage3_floor_mismatch_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from creditriskengine.ecl.ind_as109 import parameter_assertions as pa
        bad_samples = [(RBIExposureCategory.CORPORATE, 0.5, True, 0.99)]
        monkeypatch.setattr(pa, "PUBLISHED_STAGE3_SAMPLES", bad_samples)
        with pytest.raises(RBIParameterMismatch, match="Stage 3 floor"):
            assert_rbi_2026_parameters_match_published()

    def test_regulatory_self_check_mismatch_branch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from creditriskengine.ecl.ind_as109 import parameter_assertions as pa
        monkeypatch.setattr(pa, "RBI_PD_FLOOR", 0.99)
        report = regulatory_self_check()
        assert report["status"] == "mismatch"
        assert "PD floor" in report["message"]
        assert "RBI/DOR/2026-27/398" in report["reference"]


# ----------------------------------------------------------------------
# Parallel run
# ----------------------------------------------------------------------


class TestParallelRun:
    def test_basic_stage1_corporate(self) -> None:
        result = parallel_run(
            stage=IFRS9Stage.STAGE_1,
            pd_12m=0.01,
            lgd=0.45,
            ead=1_000_000,
            category=RBIExposureCategory.CORPORATE,
            is_secured=True,
        )
        assert isinstance(result, ParallelRunResult)
        # IRACP: pure model ECL with no IRAC class = 0.01 * 0.45 * 1M = 4500
        # 2026: LGD floored to 0.65 -> 6500, regulatory floor 0.40% = 4000
        # max(6500, 4000) = 6500
        assert result.iracp_provision == pytest.approx(4_500.0)
        assert result.ecl_2026_provision == pytest.approx(6_500.0)
        assert result.delta == pytest.approx(2_000.0)
        assert result.pct_change == pytest.approx(2_000.0 / 4_500.0)

    def test_iracp_legacy_floor_with_irac_class(self) -> None:
        # Stage 3 substandard secured: IRACP floor 15% of EAD = 150,000
        result = parallel_run(
            stage=IFRS9Stage.STAGE_3,
            pd_12m=0.10,
            lgd=0.45,
            ead=1_000_000,
            marginal_pds=np.array([0.10]),
            irac_class=IRACAssetClass.SUBSTANDARD,
            is_secured=True,
            category=RBIExposureCategory.CORPORATE,
            years_in_stage3=0.5,
        )
        # IRACP: max(model_ecl, 15% of 1M) = 150,000
        # 2026: PD floored, LGD floored 0.65, regulatory Stage 3 0-1y secured = 25%
        # max(model, 250,000) = 250,000
        assert result.iracp_provision == pytest.approx(150_000.0)
        assert result.ecl_2026_provision == pytest.approx(250_000.0)
        assert result.delta == pytest.approx(100_000.0)

    def test_binding_in_2026_flag(self) -> None:
        # Stage 2 unsecured retail with very low PD -> 5% floor binds
        result = parallel_run(
            stage=IFRS9Stage.STAGE_2,
            pd_12m=0.0001,
            lgd=0.20,
            ead=100_000,
            marginal_pds=np.array([0.0001, 0.0001]),
            category=RBIExposureCategory.UNSECURED_RETAIL,
            is_secured=False,
        )
        # ECL 2026 = 5% of 100,000 = 5,000 (floor binds)
        assert result.binding_in_2026 is True
        assert result.ecl_2026_provision == pytest.approx(5_000.0)

    def test_capital_add_back_phase_in(self) -> None:
        # Build a fixed delta and check the add-back per reporting FY
        kw = dict(
            stage=IFRS9Stage.STAGE_1,
            pd_12m=0.01,
            lgd=0.45,
            ead=1_000_000,
            category=RBIExposureCategory.CORPORATE,
            is_secured=True,
        )
        r28 = parallel_run(**kw, reporting_fy=2028)
        r29 = parallel_run(**kw, reporting_fy=2029)
        r30 = parallel_run(**kw, reporting_fy=2030)
        r31 = parallel_run(**kw, reporting_fy=2031)
        r32 = parallel_run(**kw, reporting_fy=2032)

        # Delta is constant; add-back declines 0.80 -> 0.60 -> 0.40 -> 0.20 -> 0.0
        delta = r28.delta
        assert r28.capital_add_back_amount == pytest.approx(delta * 0.80)
        assert r29.capital_add_back_amount == pytest.approx(delta * 0.60)
        assert r30.capital_add_back_amount == pytest.approx(delta * 0.40)
        assert r31.capital_add_back_amount == pytest.approx(delta * 0.20)
        assert r32.capital_add_back_amount == 0.0

    def test_sovereign_carve_out_in_parallel_run(self) -> None:
        result = parallel_run(
            stage=IFRS9Stage.STAGE_1,
            pd_12m=0.01,
            lgd=0.45,
            ead=10_000_000,
            category=RBIExposureCategory.CORPORATE,
            is_sovereign_slr=True,
        )
        assert result.ecl_2026_provision == 0.0

    def test_zero_iracp_no_division_error(self) -> None:
        # Edge case: IRACP = 0 should produce 0% change, not NaN
        result = parallel_run(
            stage=IFRS9Stage.STAGE_1,
            pd_12m=0.0,
            lgd=0.0,
            ead=1_000_000,
            category=RBIExposureCategory.CORPORATE,
            is_secured=True,
        )
        # Whichever floor binds, pct_change defined
        assert isinstance(result.pct_change, float)


# ----------------------------------------------------------------------
# Portfolio summary
# ----------------------------------------------------------------------


class TestPortfolioSummary:
    def test_empty_portfolio(self) -> None:
        s = portfolio_parallel_run_summary([])
        assert s["n_exposures"] == 0
        assert s["total_iracp"] == 0.0
        assert s["total_ecl_2026"] == 0.0
        assert s["total_capital_add_back"] == 0.0

    def test_aggregates_correctly(self) -> None:
        r1 = ParallelRunResult(
            iracp_provision=100, ecl_2026_provision=150, delta=50,
            pct_change=0.5, binding_in_2026=True, capital_add_back_amount=40,
        )
        r2 = ParallelRunResult(
            iracp_provision=200, ecl_2026_provision=180, delta=-20,
            pct_change=-0.1, binding_in_2026=False, capital_add_back_amount=0,
        )
        s = portfolio_parallel_run_summary([r1, r2])
        assert s["n_exposures"] == 2
        assert s["total_iracp"] == 300
        assert s["total_ecl_2026"] == 330
        assert s["total_delta"] == 30
        assert s["weighted_pct_change"] == pytest.approx(30 / 300)
        assert s["total_capital_add_back"] == 40
        assert s["n_binding_in_2026"] == 1
