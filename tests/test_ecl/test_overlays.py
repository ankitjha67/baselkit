"""Tests for IFRS 9 management overlays (post-model adjustments)."""

from datetime import UTC, datetime, timedelta

import pytest

from creditriskengine.ecl.ifrs9.overlays import (
    ManagementOverlay,
    OverlayType,
    apply_overlays,
    overlay_impact_summary,
    validate_overlay,
)


class TestApplyOverlays:
    def test_absolute_overlay(self) -> None:
        overlays = [
            ManagementOverlay(
                name="CRE stress",
                overlay_type=OverlayType.SECTOR_SPECIFIC,
                adjustment_amount=50.0,
            ),
        ]
        result = apply_overlays(100.0, overlays)
        assert result.model_ecl == pytest.approx(100.0)
        assert result.overlay_ecl == pytest.approx(150.0)
        assert result.total_adjustment == pytest.approx(50.0)
        assert len(result.applied_overlays) == 1

    def test_rate_overlay(self) -> None:
        overlays = [
            ManagementOverlay(
                name="Model limitation",
                overlay_type=OverlayType.MODEL_LIMITATION,
                adjustment_rate=0.10,
            ),
        ]
        result = apply_overlays(100.0, overlays)
        assert result.overlay_ecl == pytest.approx(110.0)

    def test_combined_rate_and_absolute(self) -> None:
        overlays = [
            ManagementOverlay(
                name="Rate uplift",
                overlay_type=OverlayType.ECONOMIC_UNCERTAINTY,
                adjustment_rate=0.20,
            ),
            ManagementOverlay(
                name="Sector add-on",
                overlay_type=OverlayType.SECTOR_SPECIFIC,
                adjustment_amount=10.0,
            ),
        ]
        result = apply_overlays(100.0, overlays)
        # Rate first: 100 * 1.2 = 120, then absolute: 120 + 10 = 130
        assert result.overlay_ecl == pytest.approx(130.0)
        assert len(result.applied_overlays) == 2

    def test_negative_overlay_with_floor(self) -> None:
        overlays = [
            ManagementOverlay(
                name="Favorable adjustment",
                overlay_type=OverlayType.EMERGING_RISK,
                adjustment_amount=-200.0,
            ),
        ]
        result = apply_overlays(100.0, overlays, floor=0.0)
        assert result.overlay_ecl == pytest.approx(0.0)

    def test_inactive_overlay_skipped(self) -> None:
        overlays = [
            ManagementOverlay(
                name="Inactive",
                overlay_type=OverlayType.REGULATORY,
                adjustment_amount=50.0,
                is_active=False,
            ),
        ]
        result = apply_overlays(100.0, overlays)
        assert result.overlay_ecl == pytest.approx(100.0)
        assert len(result.skipped_overlays) == 1
        assert len(result.applied_overlays) == 0

    def test_expired_overlay_skipped(self) -> None:
        past = datetime.now(UTC) - timedelta(days=1)
        overlays = [
            ManagementOverlay(
                name="Expired",
                overlay_type=OverlayType.TEMPORARY_EVENT,
                adjustment_amount=50.0,
                expiry_date=past,
            ),
        ]
        result = apply_overlays(100.0, overlays)
        assert result.overlay_ecl == pytest.approx(100.0)
        assert len(result.skipped_overlays) == 1

    def test_future_effective_date_skipped(self) -> None:
        future = datetime.now(UTC) + timedelta(days=30)
        overlays = [
            ManagementOverlay(
                name="Future",
                overlay_type=OverlayType.REGULATORY,
                adjustment_amount=50.0,
                effective_date=future,
            ),
        ]
        result = apply_overlays(100.0, overlays)
        assert result.overlay_ecl == pytest.approx(100.0)
        assert len(result.skipped_overlays) == 1

    def test_empty_overlays(self) -> None:
        result = apply_overlays(100.0, [])
        assert result.overlay_ecl == pytest.approx(100.0)
        assert result.total_adjustment == pytest.approx(0.0)

    def test_multiple_rate_overlays_multiplicative(self) -> None:
        overlays = [
            ManagementOverlay(
                name="A",
                overlay_type=OverlayType.MODEL_LIMITATION,
                adjustment_rate=0.10,
            ),
            ManagementOverlay(
                name="B",
                overlay_type=OverlayType.ECONOMIC_UNCERTAINTY,
                adjustment_rate=0.10,
            ),
        ]
        result = apply_overlays(100.0, overlays)
        # 100 * 1.1 * 1.1 = 121
        assert result.overlay_ecl == pytest.approx(121.0)

    def test_as_of_date_controls_effectiveness(self) -> None:
        start = datetime(2025, 6, 1, tzinfo=UTC)
        end = datetime(2025, 12, 31, tzinfo=UTC)
        overlays = [
            ManagementOverlay(
                name="H2 overlay",
                overlay_type=OverlayType.TEMPORARY_EVENT,
                adjustment_amount=50.0,
                effective_date=start,
                expiry_date=end,
            ),
        ]
        # Before effective → skipped
        before = datetime(2025, 5, 1, tzinfo=UTC)
        r1 = apply_overlays(100.0, overlays, as_of=before)
        assert r1.overlay_ecl == pytest.approx(100.0)

        # During effective → applied
        during = datetime(2025, 8, 1, tzinfo=UTC)
        r2 = apply_overlays(100.0, overlays, as_of=during)
        assert r2.overlay_ecl == pytest.approx(150.0)

        # After expiry → skipped
        after = datetime(2026, 1, 1, tzinfo=UTC)
        r3 = apply_overlays(100.0, overlays, as_of=after)
        assert r3.overlay_ecl == pytest.approx(100.0)


class TestOverlayImpactSummary:
    def test_summary_fields(self) -> None:
        overlays = [
            ManagementOverlay(
                name="Test",
                overlay_type=OverlayType.EMERGING_RISK,
                adjustment_amount=25.0,
                rationale="Geopolitical risk",
            ),
        ]
        result = apply_overlays(100.0, overlays)
        summary = overlay_impact_summary(result)
        assert summary["model_ecl"] == 100.0
        assert summary["overlay_ecl"] == 125.0
        assert summary["total_adjustment"] == 25.0
        assert summary["adjustment_pct"] == 25.0
        assert summary["overlays_applied"] == 1
        assert summary["overlays_skipped"] == 0

    def test_zero_model_ecl(self) -> None:
        overlays = [
            ManagementOverlay(
                name="Floor",
                overlay_type=OverlayType.REGULATORY,
                adjustment_amount=10.0,
            ),
        ]
        result = apply_overlays(0.0, overlays)
        summary = overlay_impact_summary(result)
        assert summary["adjustment_pct"] == 0.0


class TestValidateOverlay:
    def test_fully_compliant(self) -> None:
        ov = ManagementOverlay(
            name="Complete overlay",
            overlay_type=OverlayType.SECTOR_SPECIFIC,
            adjustment_amount=50.0,
            rationale="CRE sector downturn observed in Q3",
            regulatory_basis="IFRS 9.B5.5.52",
            approved_by="Credit Risk Committee",
            approval_date=datetime.now(UTC),
            expiry_date=datetime.now(UTC) + timedelta(days=90),
            portfolio_scope="UK CRE Stage 2",
        )
        warnings = validate_overlay(ov)
        assert warnings == []

    def test_missing_rationale(self) -> None:
        ov = ManagementOverlay(
            name="No rationale",
            overlay_type=OverlayType.EMERGING_RISK,
            adjustment_amount=50.0,
            approved_by="CRO",
            approval_date=datetime.now(UTC),
            expiry_date=datetime.now(UTC) + timedelta(days=90),
            portfolio_scope="All",
        )
        warnings = validate_overlay(ov)
        assert any("rationale" in w.lower() for w in warnings)

    def test_missing_approval(self) -> None:
        ov = ManagementOverlay(
            name="No approval",
            overlay_type=OverlayType.DATA_GAP,
            adjustment_amount=50.0,
            rationale="Data gap in new segment",
            portfolio_scope="Segment X",
        )
        warnings = validate_overlay(ov)
        assert any("approval authority" in w.lower() for w in warnings)
        assert any("approval date" in w.lower() for w in warnings)

    def test_no_expiry_warning(self) -> None:
        ov = ManagementOverlay(
            name="No expiry",
            overlay_type=OverlayType.MODEL_LIMITATION,
            adjustment_amount=50.0,
            rationale="Known model gap",
            approved_by="MRC",
            approval_date=datetime.now(UTC),
            portfolio_scope="All",
        )
        warnings = validate_overlay(ov)
        assert any("expiry" in w.lower() for w in warnings)

    def test_zero_impact_warning(self) -> None:
        ov = ManagementOverlay(
            name="Zero impact",
            overlay_type=OverlayType.REGULATORY,
            rationale="Placeholder",
            approved_by="CRO",
            approval_date=datetime.now(UTC),
            expiry_date=datetime.now(UTC) + timedelta(days=90),
            portfolio_scope="All",
        )
        warnings = validate_overlay(ov)
        assert any("zero impact" in w.lower() for w in warnings)

    def test_no_scope_warning(self) -> None:
        ov = ManagementOverlay(
            name="No scope",
            overlay_type=OverlayType.SECTOR_SPECIFIC,
            adjustment_amount=50.0,
            rationale="Sector stress",
            approved_by="CRO",
            approval_date=datetime.now(UTC),
            expiry_date=datetime.now(UTC) + timedelta(days=90),
        )
        warnings = validate_overlay(ov)
        assert any("scope" in w.lower() for w in warnings)


class TestOverlayType:
    def test_all_types_accessible(self) -> None:
        assert OverlayType.MODEL_LIMITATION.value == "model_limitation"
        assert OverlayType.EMERGING_RISK.value == "emerging_risk"
        assert OverlayType.DATA_GAP.value == "data_gap"
        assert OverlayType.ECONOMIC_UNCERTAINTY.value == "economic_uncertainty"
        assert OverlayType.SECTOR_SPECIFIC.value == "sector_specific"
        assert OverlayType.REGULATORY.value == "regulatory"
        assert OverlayType.TEMPORARY_EVENT.value == "temporary_event"
