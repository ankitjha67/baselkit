"""Tests for core data models — Exposure, Portfolio, and config loader."""

import pytest

from creditriskengine.core.config import load_jurisdiction_config
from creditriskengine.core.exposure import Collateral, Exposure
from creditriskengine.core.portfolio import Portfolio
from creditriskengine.core.types import (
    CollateralType,
    CreditRiskApproach,
    Jurisdiction,
    SAExposureClass,
)


class TestExposure:
    def test_minimal_creation(self):
        e = Exposure(
            exposure_id="E001",
            counterparty_id="C001",
            ead=1000.0,
            drawn_amount=800.0,
            jurisdiction=Jurisdiction.BCBS,
            approach=CreditRiskApproach.SA,
        )
        assert e.ead == 1000.0
        assert e.is_defaulted is False

    def test_pd_field_allows_valid_range(self):
        e = Exposure(
            exposure_id="E001",
            counterparty_id="C001",
            ead=100.0,
            drawn_amount=100.0,
            jurisdiction=Jurisdiction.EU,
            approach=CreditRiskApproach.FIRB,
            pd=0.02,
        )
        assert e.pd == 0.02

    def test_pd_rejects_above_one(self):
        with pytest.raises(Exception):
            Exposure(
                exposure_id="E001",
                counterparty_id="C001",
                ead=100.0,
                drawn_amount=100.0,
                jurisdiction=Jurisdiction.EU,
                approach=CreditRiskApproach.FIRB,
                pd=1.5,
            )

    def test_pd_rejects_negative(self):
        with pytest.raises(Exception):
            Exposure(
                exposure_id="E001",
                counterparty_id="C001",
                ead=100.0,
                drawn_amount=100.0,
                jurisdiction=Jurisdiction.EU,
                approach=CreditRiskApproach.FIRB,
                pd=-0.01,
            )

    def test_ead_non_negative(self):
        with pytest.raises(Exception):
            Exposure(
                exposure_id="E001",
                counterparty_id="C001",
                ead=-100.0,
                drawn_amount=100.0,
                jurisdiction=Jurisdiction.BCBS,
                approach=CreditRiskApproach.SA,
            )


class TestCollateral:
    def test_basic(self):
        c = Collateral(collateral_type=CollateralType.CASH, value=100_000.0)
        assert c.collateral_type == CollateralType.CASH
        assert c.haircut is None


class TestPortfolio:
    def _make_exposure(self, eid: str, ead: float, defaulted: bool = False) -> Exposure:
        return Exposure(
            exposure_id=eid,
            counterparty_id=f"C_{eid}",
            ead=ead,
            drawn_amount=ead,
            jurisdiction=Jurisdiction.BCBS,
            approach=CreditRiskApproach.SA,
            sa_exposure_class=SAExposureClass.CORPORATE,
            is_defaulted=defaulted,
        )

    def test_add_and_len(self):
        p = Portfolio()
        p.add_exposure(self._make_exposure("E1", 100.0))
        p.add_exposure(self._make_exposure("E2", 200.0))
        assert len(p) == 2

    def test_total_ead(self):
        p = Portfolio()
        p.add_exposure(self._make_exposure("E1", 100.0))
        p.add_exposure(self._make_exposure("E2", 200.0))
        assert p.total_ead() == pytest.approx(300.0)

    def test_iteration(self):
        p = Portfolio()
        p.add_exposure(self._make_exposure("E1", 100.0))
        p.add_exposure(self._make_exposure("E2", 200.0))
        eids = [e.exposure_id for e in p]
        assert eids == ["E1", "E2"]

    def test_filter_defaulted(self):
        p = Portfolio()
        p.add_exposure(self._make_exposure("E1", 100.0, defaulted=False))
        p.add_exposure(self._make_exposure("E2", 200.0, defaulted=True))
        assert len(p.filter_defaulted()) == 1
        assert len(p.filter_non_defaulted()) == 1

    def test_filter_by_approach(self):
        p = Portfolio()
        p.add_exposure(self._make_exposure("E1", 100.0))
        result = p.filter_by_approach(CreditRiskApproach.SA)
        assert len(result) == 1


class TestCoreConfigDelegation:
    """Verify core.config delegates to regulatory.loader correctly."""

    def test_load_jurisdiction_config_returns_dict(self):
        config = load_jurisdiction_config(Jurisdiction.EU)
        assert isinstance(config, dict)
        assert "regulator" in config or "jurisdiction" in config

    def test_load_jurisdiction_config_matches_loader(self):
        from creditriskengine.regulatory.loader import load_config

        config_via_core = load_jurisdiction_config(Jurisdiction.UK)
        config_via_loader = load_config(Jurisdiction.UK)
        assert config_via_core == config_via_loader
