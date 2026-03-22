"""Tests for FR Y-14M/Q reporting schedules."""

import pytest

from creditriskengine.reporting.fr_y14 import (
    FRY14Schedule,
    generate_loss_schedule,
    generate_schedule_h1,
    generate_schedule_h2,
    schedule_to_dict,
)


def _sample_wholesale() -> list[dict[str, object]]:
    return [
        {
            "obligor_id": "OB-001",
            "obligor_name": "Acme Corp",
            "industry_code": "31-33",
            "committed_exposure": 5_000_000,
            "utilized_exposure": 3_000_000,
            "internal_rating": "BB+",
            "pd": 0.02,
            "lgd": 0.40,
            "ead": 4_000_000,
            "maturity_years": 3.0,
            "facility_type": "Term Loan",
        },
        {
            "obligor_id": "OB-002",
            "obligor_name": "Beta Inc",
            "committed_exposure": 2_000_000,
            "pd": 0.05,
            "lgd": 0.35,
            "ead": 2_000_000,
            "maturity_years": 2.0,
        },
    ]


def _sample_cre() -> list[dict[str, object]]:
    return [
        {
            "loan_id": "CRE-001",
            "property_type": "Office",
            "property_location": "NY",
            "committed_exposure": 10_000_000,
            "utilized_exposure": 8_000_000,
            "appraised_value": 15_000_000,
            "dscr": 1.35,
            "internal_rating": "BBB",
            "pd": 0.03,
            "lgd": 0.30,
            "ead": 9_000_000,
            "maturity_years": 5.0,
        },
    ]


class TestGenerateScheduleH1:
    """FR Y-14Q Schedule H.1 -- Wholesale Corporate."""

    def test_returns_schedule(self) -> None:
        sched = generate_schedule_h1(_sample_wholesale(), "2025-03-31", "Test BHC")
        assert isinstance(sched, FRY14Schedule)
        assert sched.schedule_id == "H.1"

    def test_row_count(self) -> None:
        sched = generate_schedule_h1(_sample_wholesale(), "2025-03-31")
        assert len(sched.rows) == 2

    def test_expected_loss_calculated(self) -> None:
        sched = generate_schedule_h1(_sample_wholesale(), "2025-03-31")
        row = sched.rows[0]
        # EL = PD * LGD * EAD = 0.02 * 0.40 * 4M = 32000
        assert row.expected_loss == pytest.approx(32_000, rel=1e-6)

    def test_metadata_totals(self) -> None:
        sched = generate_schedule_h1(_sample_wholesale(), "2025-03-31")
        assert sched.metadata["obligor_count"] == 2
        assert sched.metadata["total_ead"] == pytest.approx(6_000_000, rel=1e-6)

    def test_exposure_weighted_pd(self) -> None:
        sched = generate_schedule_h1(_sample_wholesale(), "2025-03-31")
        # (0.02*4M + 0.05*2M) / 6M = 0.18/6 = 0.03
        assert sched.metadata["exposure_weighted_pd"] == pytest.approx(
            (0.02 * 4e6 + 0.05 * 2e6) / 6e6, rel=1e-4
        )

    def test_empty_input(self) -> None:
        sched = generate_schedule_h1([], "2025-03-31")
        assert len(sched.rows) == 0

    def test_validate_exposure_missing_fields(self) -> None:
        """Exposure missing required fields logs warning and still creates row."""
        from creditriskengine.reporting.fr_y14 import _validate_exposure_fields

        exposure = {"obligor_id": "OB-BAD"}
        result = _validate_exposure_fields(
            exposure,
            ["obligor_id", "committed_exposure", "pd", "lgd"],
            "Schedule H.1",
        )
        assert result is False

    def test_validate_exposure_missing_fields_loan_id(self) -> None:
        """Exposure identified by loan_id when obligor_id absent."""
        from creditriskengine.reporting.fr_y14 import _validate_exposure_fields

        exposure = {"loan_id": "LN-001"}
        result = _validate_exposure_fields(
            exposure,
            ["committed_exposure", "pd"],
            "Schedule H.2",
        )
        assert result is False

    def test_validate_exposure_missing_fields_unknown(self) -> None:
        """Exposure with neither obligor_id nor loan_id uses 'unknown'."""
        from creditriskengine.reporting.fr_y14 import _validate_exposure_fields

        exposure: dict[str, object] = {}
        result = _validate_exposure_fields(
            exposure,
            ["committed_exposure"],
            "test",
        )
        assert result is False


class TestGenerateScheduleH2:
    """FR Y-14Q Schedule H.2 -- CRE."""

    def test_returns_schedule(self) -> None:
        sched = generate_schedule_h2(_sample_cre(), "2025-03-31", "Test BHC")
        assert isinstance(sched, FRY14Schedule)
        assert sched.schedule_id == "H.2"

    def test_ltv_auto_calculated(self) -> None:
        data = [
            {
                "loan_id": "CRE-002",
                "committed_exposure": 5_000_000,
                "utilized_exposure": 4_000_000,
                "appraised_value": 8_000_000,
                "pd": 0.02,
                "lgd": 0.30,
            }
        ]
        sched = generate_schedule_h2(data, "2025-03-31")
        # LTV = utilized / appraised = 4M/8M = 0.5
        assert sched.rows[0].ltv_ratio == pytest.approx(0.5, rel=1e-6)

    def test_property_type_breakdown(self) -> None:
        sched = generate_schedule_h2(_sample_cre(), "2025-03-31")
        assert "Office" in sched.metadata["property_type_breakdown"]

    def test_expected_loss(self) -> None:
        sched = generate_schedule_h2(_sample_cre(), "2025-03-31")
        # EL = 0.03 * 0.30 * 9M = 81000
        assert sched.rows[0].expected_loss == pytest.approx(81_000, rel=1e-6)


class TestGenerateLossSchedule:
    """FR Y-14Q Schedule B -- Projected Losses."""

    def test_9_quarter_projection(self) -> None:
        losses = {f"Q{(i % 4) + 1} 202{5 + i // 4}": 100_000.0 * (i + 1) for i in range(9)}
        sched = generate_loss_schedule(losses, "2025-03-31", "Test BHC")
        assert len(sched.rows) == 9

    def test_padding_when_fewer_quarters(self) -> None:
        losses = {"Q1 2025": 50_000, "Q2 2025": 60_000}
        sched = generate_loss_schedule(losses, "2025-03-31", horizon_quarters=9)
        assert len(sched.rows) == 9

    def test_cumulative_loss_rate_increases(self) -> None:
        losses = {f"Q{i+1} 2025": 10_000.0 for i in range(4)}
        losses.update({f"Q{i+1} 2026": 10_000.0 for i in range(4)})
        losses["Q1 2027"] = 10_000.0
        sched = generate_loss_schedule(losses, "2025-03-31")
        rates = [r.cumulative_loss_rate for r in sched.rows]
        # Cumulative loss rates should be non-decreasing
        for i in range(1, len(rates)):
            assert rates[i] >= rates[i - 1] - 1e-10

    def test_metadata_keys(self) -> None:
        losses = {"Q1 2025": 50_000}
        sched = generate_loss_schedule(losses, "2025-03-31")
        assert "total_net_charge_offs" in sched.metadata
        assert "peak_cumulative_loss_rate" in sched.metadata
        assert sched.metadata["scenario"] == "baseline"


class TestScheduleToDict:
    """Serialization."""

    def test_contains_required_keys(self) -> None:
        sched = generate_schedule_h1(_sample_wholesale(), "2025-03-31", "Bank X")
        d = schedule_to_dict(sched)
        assert "schedule_id" in d
        assert "rows" in d
        assert "row_count" in d
        assert d["row_count"] == 2

    def test_rows_are_dicts(self) -> None:
        sched = generate_schedule_h1(_sample_wholesale(), "2025-03-31")
        d = schedule_to_dict(sched)
        assert all(isinstance(r, dict) for r in d["rows"])

    def test_bhc_name_preserved(self) -> None:
        sched = generate_schedule_h1(_sample_wholesale(), "2025-03-31", "Mega BHC")
        d = schedule_to_dict(sched)
        assert d["bhc_name"] == "Mega BHC"
