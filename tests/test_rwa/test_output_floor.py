"""Tests for output floor mechanism — BCBS d424 RBC25."""

from datetime import date

from creditriskengine.core.types import Jurisdiction
from creditriskengine.rwa.output_floor import (
    OutputFloorCalculator,
    get_output_floor_pct,
)


class TestGetOutputFloorPct:
    def test_bcbs_2023(self) -> None:
        assert get_output_floor_pct(Jurisdiction.BCBS, date(2023, 6, 1)) == 0.50

    def test_bcbs_2028_full(self) -> None:
        assert get_output_floor_pct(Jurisdiction.BCBS, date(2028, 6, 1)) == 0.725

    def test_before_effective_returns_zero(self) -> None:
        assert get_output_floor_pct(Jurisdiction.EU, date(2024, 6, 1)) == 0.0

    def test_eu_2025_start(self) -> None:
        assert get_output_floor_pct(Jurisdiction.EU, date(2025, 1, 1)) == 0.50

    def test_uk_delayed_start(self) -> None:
        assert get_output_floor_pct(Jurisdiction.UK, date(2026, 6, 1)) == 0.0
        assert get_output_floor_pct(Jurisdiction.UK, date(2027, 6, 1)) == 0.50

    def test_india_conservative_80pct(self) -> None:
        assert get_output_floor_pct(Jurisdiction.INDIA, date(2023, 6, 1)) == 0.80

    def test_unknown_jurisdiction_falls_back_to_bcbs(self) -> None:
        # Malaysia not in schedule, should fall back to BCBS
        pct = get_output_floor_pct(Jurisdiction.MALAYSIA, date(2025, 6, 1))
        assert pct == 0.60  # BCBS 2025 rate


class TestOutputFloorCalculator:
    def test_floor_not_binding(self) -> None:
        calc = OutputFloorCalculator(Jurisdiction.BCBS, date(2023, 6, 1))
        result = calc.calculate(irb_rwa=1000.0, sa_rwa=1500.0)
        # 50% * 1500 = 750 < 1000
        assert result["floored_rwa"] == 1000.0
        assert result["is_binding"] is False
        assert result["add_on"] == 0.0

    def test_floor_binding(self) -> None:
        calc = OutputFloorCalculator(Jurisdiction.BCBS, date(2028, 6, 1))
        result = calc.calculate(irb_rwa=600.0, sa_rwa=1200.0)
        # 72.5% * 1200 = 870 > 600
        assert result["floored_rwa"] == 870.0
        assert result["is_binding"] is True
        assert result["add_on"] == 270.0

    def test_eu_transitional_cap(self) -> None:
        calc = OutputFloorCalculator(Jurisdiction.EU, date(2030, 6, 1))
        result = calc.calculate(irb_rwa=400.0, sa_rwa=1200.0)
        # 72.5% * 1200 = 870, add_on = 470
        # But cap = 25% * 400 = 100
        assert result["add_on"] == 100.0
        assert result["floored_rwa"] == 500.0

    def test_result_keys(self) -> None:
        calc = OutputFloorCalculator(Jurisdiction.BCBS, date(2025, 1, 1))
        result = calc.calculate(irb_rwa=800.0, sa_rwa=1000.0)
        expected_keys = {
            "floored_rwa", "floor_pct", "floor_rwa",
            "is_binding", "add_on", "irb_rwa", "sa_rwa",
        }
        assert set(result.keys()) == expected_keys
