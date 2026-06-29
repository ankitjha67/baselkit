"""Tests for operational resilience (DORA + impact tolerance + concentration)."""

from __future__ import annotations

import pytest

from creditriskengine.operational_resilience import (
    DORAIncidentClassification,
    ImpactTolerance,
    classify_ict_incident,
    is_major_incident,
    third_party_concentration,
    within_impact_tolerance,
)

# ============================================================================
# DORA incident classification
# ============================================================================


class TestDORAClassification:
    def test_minor_incident(self) -> None:
        result = classify_ict_incident(
            clients_affected=10,
            total_clients=1_000_000,
            downtime_hours=0.5,
            economic_impact_eur=5_000.0,
        )
        assert result.is_major is False
        assert result.triggered_criteria == []

    def test_major_by_downtime(self) -> None:
        result = classify_ict_incident(
            clients_affected=10,
            total_clients=1_000_000,
            downtime_hours=3.0,  # > 2h
            economic_impact_eur=5_000.0,
        )
        assert result.is_major is True
        assert "downtime" in result.triggered_criteria

    def test_major_by_clients_pct(self) -> None:
        result = classify_ict_incident(
            clients_affected=200_000,
            total_clients=1_000_000,  # 20% > 10%
            downtime_hours=0.5,
            economic_impact_eur=5_000.0,
        )
        assert result.is_major is True
        assert "clients_affected_pct" in result.triggered_criteria

    def test_major_by_economic_impact(self) -> None:
        result = classify_ict_incident(
            clients_affected=10,
            total_clients=1_000_000,
            downtime_hours=0.5,
            economic_impact_eur=200_000.0,  # > 100k
        )
        assert result.is_major is True
        assert "economic_impact" in result.triggered_criteria

    def test_major_by_data_loss(self) -> None:
        result = classify_ict_incident(
            clients_affected=10,
            total_clients=1_000_000,
            downtime_hours=0.5,
            economic_impact_eur=5_000.0,
            data_losses=True,
        )
        assert result.is_major is True
        assert "data_losses" in result.triggered_criteria

    def test_major_by_critical_services(self) -> None:
        result = classify_ict_incident(
            clients_affected=10,
            total_clients=1_000_000,
            downtime_hours=0.5,
            economic_impact_eur=5_000.0,
            critical_services_affected=True,
        )
        assert result.is_major is True
        assert "critical_services" in result.triggered_criteria

    def test_is_major_helper(self) -> None:
        result = classify_ict_incident(10, 1_000_000, 3.0, 5_000.0)
        assert is_major_incident(result)

    def test_multiple_criteria(self) -> None:
        result = classify_ict_incident(
            clients_affected=300_000,
            total_clients=1_000_000,
            downtime_hours=5.0,
            economic_impact_eur=500_000.0,
        )
        assert len(result.triggered_criteria) >= 3

    def test_classification_type(self) -> None:
        result = classify_ict_incident(10, 1_000_000, 0.5, 5_000.0)
        assert isinstance(result, DORAIncidentClassification)


# ============================================================================
# Impact tolerance
# ============================================================================


class TestImpactTolerance:
    def _tolerance(self) -> ImpactTolerance:
        return ImpactTolerance(
            service_name="Payments",
            max_tolerable_downtime_hours=4.0,
            max_tolerable_clients_affected=50_000,
        )

    def test_within_tolerance(self) -> None:
        assert within_impact_tolerance(self._tolerance(), 2.0, 10_000)

    def test_breach_downtime(self) -> None:
        assert not within_impact_tolerance(self._tolerance(), 6.0, 10_000)

    def test_breach_clients(self) -> None:
        assert not within_impact_tolerance(self._tolerance(), 2.0, 100_000)

    def test_at_boundary_within(self) -> None:
        assert within_impact_tolerance(self._tolerance(), 4.0, 50_000)


# ============================================================================
# Third-party concentration
# ============================================================================


class TestThirdPartyConcentration:
    def test_high_concentration(self) -> None:
        result = third_party_concentration({"AWS": 80.0, "Azure": 10.0, "GCP": 10.0})
        assert result["concentration_level"] == "high"
        assert result["largest_provider"] == "AWS"
        assert result["largest_share"] == pytest.approx(0.80)

    def test_low_concentration(self) -> None:
        # 10 equal providers → HHI = 10 * 0.1^2 = 0.10 boundary; use more for low
        low = third_party_concentration({str(i): 10.0 for i in range(20)})
        assert low["concentration_level"] == "low"

    def test_moderate_concentration(self) -> None:
        # 5 equal providers -> HHI = 5 * 0.2^2 = 0.20, in [0.15, 0.25)
        result = third_party_concentration({str(i): 10.0 for i in range(5)})
        assert result["hhi"] == pytest.approx(0.20)
        assert result["concentration_level"] == "moderate"

    def test_empty(self) -> None:
        result = third_party_concentration({})
        assert result["hhi"] == 0.0
        assert result["concentration_level"] == "low"

    def test_hhi_single_provider(self) -> None:
        result = third_party_concentration({"OnlyOne": 100.0})
        assert result["hhi"] == pytest.approx(1.0)
        assert result["concentration_level"] == "high"

    def test_zero_total(self) -> None:
        result = third_party_concentration({"A": 0.0, "B": 0.0})
        assert result["hhi"] == 0.0
