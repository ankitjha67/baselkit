"""Tests for the Large Exposures framework (BCBS LEX)."""

import pytest

from creditriskengine.rwa.large_exposures import (
    LE_GSIB_TO_GSIB_LIMIT,
    LE_LIMIT,
    LE_REPORTING_THRESHOLD,
    aggregate_connected_group,
    assess_large_exposure,
    exposure_value,
    large_exposures_report,
)


class TestConstants:
    def test_limits(self) -> None:
        assert pytest.approx(0.10) == LE_REPORTING_THRESHOLD
        assert pytest.approx(0.25) == LE_LIMIT
        assert pytest.approx(0.15) == LE_GSIB_TO_GSIB_LIMIT


class TestExposureValue:
    def test_on_balance_only(self) -> None:
        assert exposure_value(1000.0) == pytest.approx(1000.0)

    def test_off_balance_with_ccf(self) -> None:
        # 800 on + 0.5 * 400 off = 1000
        assert exposure_value(800.0, off_balance_notional=400.0, ccf=0.5) == pytest.approx(1000.0)

    def test_all_components(self) -> None:
        v = exposure_value(
            on_balance=500.0, off_balance_notional=200.0, ccf=1.0,
            derivative_ead=100.0, sft_exposure=50.0, eligible_crm=150.0,
        )
        # 500 + 200 + 100 + 50 - 150 = 700
        assert v == pytest.approx(700.0)

    def test_crm_floored_at_zero(self) -> None:
        assert exposure_value(100.0, eligible_crm=500.0) == pytest.approx(0.0)

    def test_negative_amount_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            exposure_value(-1.0)

    def test_invalid_ccf(self) -> None:
        with pytest.raises(ValueError, match="ccf must be"):
            exposure_value(100.0, off_balance_notional=10.0, ccf=1.5)


class TestAggregateConnectedGroup:
    def test_sum(self) -> None:
        assert aggregate_connected_group([100.0, 200.0, 50.0]) == pytest.approx(350.0)

    def test_empty(self) -> None:
        assert aggregate_connected_group([]) == pytest.approx(0.0)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            aggregate_connected_group([100.0, -1.0])


class TestAssessLargeExposure:
    def test_within_limit_not_large(self) -> None:
        # 5% of Tier 1 -> not reportable, not breached.
        r = assess_large_exposure("C1", 50.0, tier1_capital=1000.0)
        assert r.ratio_pct == pytest.approx(5.0)
        assert r.is_large is False
        assert r.is_breach is False
        assert r.headroom == pytest.approx(200.0)  # 25%*1000 - 50

    def test_reportable_at_threshold(self) -> None:
        r = assess_large_exposure("C1", 100.0, tier1_capital=1000.0)
        assert r.is_large is True  # exactly 10%
        assert r.is_breach is False

    def test_breach(self) -> None:
        r = assess_large_exposure("C1", 300.0, tier1_capital=1000.0)
        assert r.is_breach is True  # 30% > 25%
        assert r.headroom == pytest.approx(-50.0)
        assert r.limit_pct == pytest.approx(25.0)

    def test_gsib_tighter_limit(self) -> None:
        # 20% exposure: within 25% normally, breaches the 15% G-SIB limit.
        normal = assess_large_exposure("C1", 200.0, tier1_capital=1000.0)
        gsib = assess_large_exposure("C1", 200.0, tier1_capital=1000.0, is_gsib_to_gsib=True)
        assert normal.is_breach is False
        assert gsib.is_breach is True
        assert gsib.limit_pct == pytest.approx(15.0)

    def test_invalid_tier1(self) -> None:
        with pytest.raises(ValueError, match="tier1_capital must be positive"):
            assess_large_exposure("C1", 100.0, tier1_capital=0.0)

    def test_negative_exposure(self) -> None:
        with pytest.raises(ValueError, match="exposure_value must be non-negative"):
            assess_large_exposure("C1", -1.0, tier1_capital=1000.0)

    def test_custom_reporting_threshold(self) -> None:
        # 8% with a 5% threshold -> reportable.
        r = assess_large_exposure("C1", 80.0, tier1_capital=1000.0, reporting_threshold=0.05)
        assert r.is_large is True


class TestLargeExposuresReport:
    def test_report_filters_and_sorts(self) -> None:
        exposures = [
            ("BigCo", 300.0),   # 30% -> large + breach
            ("MidCo", 120.0),   # 12% -> large
            ("SmallCo", 40.0),  # 4%  -> not large
        ]
        report = large_exposures_report(exposures, tier1_capital=1000.0)
        assert report.n_counterparties == 3
        # Only the two >= 10% appear, sorted by value descending.
        assert [r.counterparty_id for r in report.large_exposures] == ["BigCo", "MidCo"]
        assert len(report.breaches) == 1
        assert report.breaches[0].counterparty_id == "BigCo"
        assert report.total_large_exposure_value == pytest.approx(420.0)

    def test_gsib_counterparties_get_tight_limit(self) -> None:
        exposures = [("PeerGSIB", 200.0)]  # 20%
        report = large_exposures_report(
            exposures, tier1_capital=1000.0, gsib_counterparties=["PeerGSIB"]
        )
        assert len(report.breaches) == 1  # 20% > 15% G-SIB limit

    def test_no_large_exposures(self) -> None:
        report = large_exposures_report([("A", 10.0), ("B", 20.0)], tier1_capital=1000.0)
        assert report.large_exposures == ()
        assert report.breaches == ()
        assert report.total_large_exposure_value == pytest.approx(0.0)

    def test_invalid_tier1(self) -> None:
        with pytest.raises(ValueError, match="tier1_capital must be positive"):
            large_exposures_report([("A", 10.0)], tier1_capital=-5.0)
