"""Tests for abstract base classes for RWA calculators."""

import pytest

from creditriskengine.core.exposure import Exposure
from creditriskengine.core.types import CreditRiskApproach, Jurisdiction, SAExposureClass
from creditriskengine.rwa.base import BaseRWACalculator, RWAResult


def _make_exposure(**overrides: object) -> Exposure:
    """Helper to create a minimal Exposure."""
    defaults = {
        "exposure_id": "EXP001",
        "counterparty_id": "CP001",
        "ead": 1_000_000.0,
        "drawn_amount": 800_000.0,
        "jurisdiction": Jurisdiction.BCBS,
        "approach": CreditRiskApproach.SA,
        "sa_exposure_class": SAExposureClass.CORPORATE,
    }
    defaults.update(overrides)
    return Exposure(**defaults)


class DummyCalculator(BaseRWACalculator):
    """Concrete implementation for testing the abstract base."""

    def __init__(self, rw: float = 100.0) -> None:
        self._rw = rw

    def calculate(self, exposure: Exposure) -> RWAResult:
        rwa = exposure.ead * self._rw / 100.0
        return RWAResult(
            exposure_id=exposure.exposure_id,
            risk_weight=self._rw,
            rwa=rwa,
            ead=exposure.ead,
            capital_requirement=rwa * 0.08,
            approach="dummy",
        )


class FailingCalculator(BaseRWACalculator):
    """Calculator that always raises for testing error handling."""

    def calculate(self, exposure: Exposure) -> RWAResult:
        raise ValueError(f"Calculation failed for {exposure.exposure_id}")


class TestRWAResult:
    """Test the RWAResult dataclass."""

    def test_creation(self) -> None:
        result = RWAResult(
            exposure_id="EXP1",
            risk_weight=75.0,
            rwa=750_000.0,
            ead=1_000_000.0,
            capital_requirement=60_000.0,
            approach="sa",
        )
        assert result.exposure_id == "EXP1"
        assert result.risk_weight == 75.0
        assert result.rwa == 750_000.0
        assert result.capital_requirement == 60_000.0

    def test_default_details(self) -> None:
        result = RWAResult(
            exposure_id="EXP1",
            risk_weight=100.0,
            rwa=100.0,
            ead=100.0,
            capital_requirement=8.0,
            approach="sa",
        )
        assert result.details == {}
        assert result.asset_class is None

    def test_with_optional_fields(self) -> None:
        result = RWAResult(
            exposure_id="EXP1",
            risk_weight=50.0,
            rwa=500.0,
            ead=1000.0,
            capital_requirement=40.0,
            approach="foundation_irb",
            asset_class="corporate",
            details={"correlation": 0.20, "maturity_adj": 1.5},
        )
        assert result.asset_class == "corporate"
        assert result.details["correlation"] == pytest.approx(0.20)


class TestBaseRWACalculator:
    """Test the abstract base class methods."""

    def test_calculate_single(self) -> None:
        calc = DummyCalculator(rw=75.0)
        exp = _make_exposure()
        result = calc.calculate(exp)
        assert result.risk_weight == 75.0
        assert result.rwa == pytest.approx(1_000_000.0 * 0.75)
        assert result.capital_requirement == pytest.approx(1_000_000.0 * 0.75 * 0.08)

    def test_calculate_portfolio(self) -> None:
        calc = DummyCalculator(rw=100.0)
        exposures = [
            _make_exposure(exposure_id="E1", ead=1_000_000.0),
            _make_exposure(exposure_id="E2", ead=2_000_000.0),
            _make_exposure(exposure_id="E3", ead=500_000.0),
        ]
        results = calc.calculate_portfolio(exposures)
        assert len(results) == 3
        assert results[0].exposure_id == "E1"
        assert results[1].rwa == pytest.approx(2_000_000.0)
        assert results[2].ead == 500_000.0

    def test_calculate_portfolio_empty(self) -> None:
        calc = DummyCalculator()
        results = calc.calculate_portfolio([])
        assert results == []

    def test_total_rwa(self) -> None:
        calc = DummyCalculator(rw=50.0)
        exposures = [
            _make_exposure(exposure_id="E1", ead=1_000_000.0),
            _make_exposure(exposure_id="E2", ead=2_000_000.0),
        ]
        total = calc.total_rwa(exposures)
        assert total == pytest.approx(1_500_000.0)  # (1M + 2M) * 50%

    def test_total_rwa_empty(self) -> None:
        calc = DummyCalculator()
        assert calc.total_rwa([]) == pytest.approx(0.0)

    def test_calculate_portfolio_propagates_error(self) -> None:
        calc = FailingCalculator()
        exposures = [_make_exposure()]
        with pytest.raises(ValueError, match="Calculation failed"):
            calc.calculate_portfolio(exposures)

    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseRWACalculator()  # type: ignore[abstract]
