"""Tests for capital buffer calculations — BCBS d424 RBC40."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from creditriskengine.core.exceptions import ConfigurationError, ValidationError
from creditriskengine.rwa.capital_buffers import (
    capital_adequacy_summary,
    capital_conservation_buffer,
    combined_buffer_requirement,
    countercyclical_buffer,
    dsib_surcharge,
    gsib_surcharge,
    maximum_distributable_amount,
    minimum_capital_requirements,
)

# ---------------------------------------------------------------------------
# Fixtures — jurisdiction configs loaded from YAML
# ---------------------------------------------------------------------------

_REGULATORY_DIR = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "creditriskengine"
    / "regulatory"
)


def _load_yaml(rel_path: str) -> dict[str, Any]:
    path = _REGULATORY_DIR / rel_path
    with open(path) as f:
        return dict(yaml.safe_load(f))


@pytest.fixture()
def bcbs_config() -> dict[str, Any]:
    return _load_yaml("bcbs/bcbs_d424.yml")


@pytest.fixture()
def eu_config() -> dict[str, Any]:
    return _load_yaml("eu/crr3.yml")


@pytest.fixture()
def us_config() -> dict[str, Any]:
    return _load_yaml("us/us_endgame.yml")


# ---------------------------------------------------------------------------
# Minimal synthetic configs for targeted edge-case testing
# ---------------------------------------------------------------------------


def _minimal_config(
    cconb: float = 0.025,
    ccyb_min: float = 0.0,
    ccyb_max: float = 0.025,
    ccyb_fixed: float | None = None,
    dsib_pct: float | None = None,
    srb_enabled: bool = False,
    scb: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cap_req: dict[str, Any] = {
        "minimum_cet1_pct": 0.045,
        "minimum_tier1_pct": 0.06,
        "minimum_total_capital_pct": 0.08,
        "capital_conservation_buffer_pct": cconb,
        "countercyclical_buffer_range": {"min": ccyb_min, "max": ccyb_max},
    }
    if ccyb_fixed is not None:
        cap_req["countercyclical_buffer_pct"] = ccyb_fixed
    if dsib_pct is not None:
        cap_req["dsib_surcharge_pct"] = dsib_pct
    if srb_enabled:
        cap_req["systemic_risk_buffer"] = {"enabled": True, "max_pct": 0.05}
    if scb is not None:
        cap_req["stress_capital_buffer"] = scb
    return {"jurisdiction": "test", "capital_requirements": cap_req}


# ===================================================================
# capital_conservation_buffer
# ===================================================================


class TestCapitalConservationBuffer:
    def test_bcbs_standard_2_5pct(self, bcbs_config: dict[str, Any]) -> None:
        assert capital_conservation_buffer(bcbs_config) == 0.025

    def test_eu_standard_2_5pct(self, eu_config: dict[str, Any]) -> None:
        assert capital_conservation_buffer(eu_config) == 0.025

    def test_us_stress_capital_buffer(self, us_config: dict[str, Any]) -> None:
        # US replaces CConB with SCB; minimum SCB is 2.5%
        rate = capital_conservation_buffer(us_config)
        assert rate == 0.025

    def test_custom_scb_minimum(self) -> None:
        cfg = _minimal_config(
            scb={"enabled": True, "minimum_pct": 0.035, "replaces": "capital_conservation_buffer"}
        )
        assert capital_conservation_buffer(cfg) == 0.035

    def test_scb_present_but_does_not_replace(self) -> None:
        # SCB present but replaces something else — should use normal CConB
        cfg = _minimal_config(
            cconb=0.025,
            scb={"enabled": True, "minimum_pct": 0.05, "replaces": "something_else"},
        )
        assert capital_conservation_buffer(cfg) == 0.025

    def test_missing_capital_requirements_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="capital_requirements"):
            capital_conservation_buffer({"jurisdiction": "bad"})

    def test_default_when_key_absent(self) -> None:
        # capital_requirements exists but no capital_conservation_buffer_pct
        cfg: dict[str, Any] = {"capital_requirements": {}}
        assert capital_conservation_buffer(cfg) == 0.025


# ===================================================================
# countercyclical_buffer
# ===================================================================


class TestCountercyclicalBuffer:
    def test_bcbs_default_zero(self, bcbs_config: dict[str, Any]) -> None:
        # BCBS has no fixed rate; should default to 0.0
        assert countercyclical_buffer(bcbs_config) == 0.0

    def test_us_fixed_zero(self, us_config: dict[str, Any]) -> None:
        assert countercyclical_buffer(us_config) == 0.0

    def test_explicit_rate_within_range(self, bcbs_config: dict[str, Any]) -> None:
        rate = countercyclical_buffer(bcbs_config, ccyb_rate=0.015)
        assert rate == 0.015

    def test_explicit_rate_at_max(self, bcbs_config: dict[str, Any]) -> None:
        rate = countercyclical_buffer(bcbs_config, ccyb_rate=0.025)
        assert rate == 0.025

    def test_explicit_rate_at_min(self, bcbs_config: dict[str, Any]) -> None:
        rate = countercyclical_buffer(bcbs_config, ccyb_rate=0.0)
        assert rate == 0.0

    def test_explicit_rate_above_max_raises(self, bcbs_config: dict[str, Any]) -> None:
        with pytest.raises(ValidationError, match="outside permitted range"):
            countercyclical_buffer(bcbs_config, ccyb_rate=0.03)

    def test_explicit_rate_negative_raises(self) -> None:
        cfg = _minimal_config()
        with pytest.raises(ValidationError, match="outside permitted range"):
            countercyclical_buffer(cfg, ccyb_rate=-0.001)

    def test_fixed_rate_from_config(self) -> None:
        cfg = _minimal_config(ccyb_fixed=0.01)
        assert countercyclical_buffer(cfg) == 0.01

    def test_missing_capital_requirements_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="capital_requirements"):
            countercyclical_buffer({"jurisdiction": "bad"})


# ===================================================================
# gsib_surcharge
# ===================================================================


class TestGSIBSurcharge:
    def test_bucket_0_not_gsib(self) -> None:
        assert gsib_surcharge(0) == 0.0

    def test_bucket_1(self) -> None:
        assert gsib_surcharge(1) == 0.01

    def test_bucket_2(self) -> None:
        assert gsib_surcharge(2) == 0.015

    def test_bucket_3(self) -> None:
        assert gsib_surcharge(3) == 0.02

    def test_bucket_4(self) -> None:
        assert gsib_surcharge(4) == 0.025

    def test_bucket_5(self) -> None:
        assert gsib_surcharge(5) == 0.035

    def test_invalid_bucket_negative(self) -> None:
        with pytest.raises(ValidationError, match="Invalid G-SIB bucket"):
            gsib_surcharge(-1)

    def test_invalid_bucket_too_high(self) -> None:
        with pytest.raises(ValidationError, match="Invalid G-SIB bucket"):
            gsib_surcharge(6)


# ===================================================================
# dsib_surcharge
# ===================================================================


class TestDSIBSurcharge:
    def test_bcbs_no_dsib(self, bcbs_config: dict[str, Any]) -> None:
        assert dsib_surcharge(bcbs_config) == 0.0

    def test_eu_systemic_risk_buffer_enabled(self, eu_config: dict[str, Any]) -> None:
        # EU has systemic_risk_buffer enabled; rate is institution-specific → 0.0
        assert dsib_surcharge(eu_config) == 0.0

    def test_explicit_dsib_pct(self) -> None:
        cfg = _minimal_config(dsib_pct=0.015)
        assert dsib_surcharge(cfg) == 0.015

    def test_no_capital_requirements_returns_zero(self) -> None:
        assert dsib_surcharge({"jurisdiction": "empty"}) == 0.0

    def test_srb_enabled_returns_zero(self) -> None:
        cfg = _minimal_config(srb_enabled=True)
        assert dsib_surcharge(cfg) == 0.0


# ===================================================================
# combined_buffer_requirement
# ===================================================================


class TestCombinedBufferRequirement:
    def test_all_zero(self) -> None:
        assert combined_buffer_requirement(0.0, 0.0, 0.0, 0.0) == 0.0

    def test_standard_gsib_bank(self) -> None:
        # CConB 2.5% + CCyB 1% + G-SIB 1.5% + D-SIB 0%
        cbr = combined_buffer_requirement(0.025, 0.01, 0.015, 0.0)
        assert cbr == pytest.approx(0.05)

    def test_all_components(self) -> None:
        cbr = combined_buffer_requirement(0.025, 0.025, 0.035, 0.02)
        assert cbr == pytest.approx(0.105)

    def test_negative_component_raises(self) -> None:
        with pytest.raises(ValidationError, match="CConB.*negative"):
            combined_buffer_requirement(-0.01, 0.0, 0.0, 0.0)

    def test_negative_ccyb_raises(self) -> None:
        with pytest.raises(ValidationError, match="CCyB.*negative"):
            combined_buffer_requirement(0.025, -0.001, 0.0, 0.0)

    def test_negative_gsib_raises(self) -> None:
        with pytest.raises(ValidationError, match="G-SIB.*negative"):
            combined_buffer_requirement(0.025, 0.0, -0.01, 0.0)

    def test_negative_dsib_raises(self) -> None:
        with pytest.raises(ValidationError, match="D-SIB.*negative"):
            combined_buffer_requirement(0.025, 0.0, 0.0, -0.005)


# ===================================================================
# minimum_capital_requirements
# ===================================================================


class TestMinimumCapitalRequirements:
    def test_well_capitalised_bank(self) -> None:
        result = minimum_capital_requirements(0.14, 0.16, 0.18, 0.025)
        assert result["meets_minimum"] is True
        assert result["meets_buffer"] is True
        assert result["cet1_target"] == pytest.approx(0.07)
        assert result["tier1_target"] == pytest.approx(0.085)
        assert result["total_target"] == pytest.approx(0.105)

    def test_meets_minimum_but_not_buffer(self) -> None:
        # CET1=5%, Tier1=7%, Total=9%, buffer=2.5%
        result = minimum_capital_requirements(0.05, 0.07, 0.09, 0.025)
        assert result["meets_minimum"] is True
        assert result["meets_buffer"] is False

    def test_below_minimum(self) -> None:
        result = minimum_capital_requirements(0.03, 0.05, 0.07, 0.025)
        assert result["meets_minimum"] is False
        assert result["meets_buffer"] is False

    def test_exactly_at_minimum(self) -> None:
        result = minimum_capital_requirements(0.045, 0.06, 0.08, 0.0)
        assert result["meets_minimum"] is True
        assert result["meets_buffer"] is True

    def test_exactly_at_buffer_boundary(self) -> None:
        # CET1=7%, target=4.5%+2.5%=7% — use exact fractions to avoid fp drift
        cconb = 0.025
        cet1 = 0.045 + cconb   # exactly at boundary
        tier1 = 0.06 + cconb
        total = 0.08 + cconb
        result = minimum_capital_requirements(cet1, tier1, total, cconb)
        assert result["meets_buffer"] is True

    def test_binding_constraint_cet1(self) -> None:
        result = minimum_capital_requirements(0.06, 0.20, 0.25, 0.025)
        assert result["binding_constraint"] == "cet1"

    def test_binding_constraint_tier1(self) -> None:
        result = minimum_capital_requirements(0.20, 0.07, 0.25, 0.025)
        assert result["binding_constraint"] == "tier1"

    def test_binding_constraint_total(self) -> None:
        result = minimum_capital_requirements(0.20, 0.20, 0.09, 0.025)
        assert result["binding_constraint"] == "total"

    def test_surplus_calculation(self) -> None:
        result = minimum_capital_requirements(0.10, 0.12, 0.15, 0.025)
        assert result["cet1_surplus"] == pytest.approx(0.10 - 0.07)
        assert result["tier1_surplus"] == pytest.approx(0.12 - 0.085)
        assert result["total_surplus"] == pytest.approx(0.15 - 0.105)

    def test_negative_ratio_raises(self) -> None:
        with pytest.raises(ValidationError, match="negative"):
            minimum_capital_requirements(-0.01, 0.06, 0.08, 0.025)

    def test_negative_buffer_raises(self) -> None:
        with pytest.raises(ValidationError, match="negative"):
            minimum_capital_requirements(0.10, 0.12, 0.15, -0.01)

    def test_with_large_combined_buffer(self) -> None:
        # G-SIB bucket 5 scenario: CConB 2.5% + CCyB 2.5% + G-SIB 3.5% = 8.5%
        cbr = 0.085
        result = minimum_capital_requirements(0.15, 0.17, 0.20, cbr)
        assert result["cet1_target"] == pytest.approx(0.045 + 0.085)
        assert result["meets_buffer"] is True

    def test_result_keys(self) -> None:
        result = minimum_capital_requirements(0.10, 0.12, 0.15, 0.025)
        expected_keys = {
            "meets_minimum", "meets_buffer",
            "cet1_target", "tier1_target", "total_target",
            "cet1_surplus", "tier1_surplus", "total_surplus",
            "binding_constraint",
        }
        assert set(result.keys()) == expected_keys


# ===================================================================
# maximum_distributable_amount
# ===================================================================


class TestMaximumDistributableAmount:
    def test_above_buffer_no_restriction(self) -> None:
        result = maximum_distributable_amount(0.10, 0.045, 0.025, 1_000_000.0)
        assert result["in_buffer_zone"] is False
        assert result["max_payout_ratio"] == 1.0
        assert result["mda"] == 1_000_000.0

    def test_below_minimum_fully_restricted(self) -> None:
        result = maximum_distributable_amount(0.03, 0.045, 0.025, 1_000_000.0)
        assert result["in_buffer_zone"] is False
        assert result["quartile"] == 0
        assert result["max_payout_ratio"] == 0.0
        assert result["mda"] == 0.0
        assert result["buffer_used_pct"] == 1.0

    def test_quartile_1_top(self) -> None:
        # CET1 in top quartile (75-100% of buffer available)
        # Buffer: 0.045 to 0.070, quartile 1 = 0.06375 to 0.070
        result = maximum_distributable_amount(0.065, 0.045, 0.025, 500_000.0)
        assert result["in_buffer_zone"] is True
        assert result["quartile"] == 1
        assert result["max_payout_ratio"] == 1.0
        assert result["mda"] == 500_000.0

    def test_quartile_2(self) -> None:
        # 50-75% of buffer; CET1 = 0.045 + 0.025 * 0.6 = 0.060
        result = maximum_distributable_amount(0.060, 0.045, 0.025, 500_000.0)
        assert result["in_buffer_zone"] is True
        assert result["quartile"] == 2
        assert result["max_payout_ratio"] == 0.6
        assert result["mda"] == pytest.approx(300_000.0)

    def test_quartile_3(self) -> None:
        # 25-50% of buffer; CET1 = 0.045 + 0.025 * 0.35 = 0.05375
        result = maximum_distributable_amount(0.05375, 0.045, 0.025, 500_000.0)
        assert result["in_buffer_zone"] is True
        assert result["quartile"] == 3
        assert result["max_payout_ratio"] == 0.4
        assert result["mda"] == pytest.approx(200_000.0)

    def test_quartile_4_bottom(self) -> None:
        # 0-25% of buffer; CET1 = 0.045 + 0.025 * 0.1 = 0.0475
        result = maximum_distributable_amount(0.0475, 0.045, 0.025, 500_000.0)
        assert result["in_buffer_zone"] is True
        assert result["quartile"] == 4
        assert result["max_payout_ratio"] == 0.0
        assert result["mda"] == 0.0

    def test_at_minimum_exactly(self) -> None:
        # CET1 exactly at minimum (bottom of buffer zone)
        result = maximum_distributable_amount(0.045, 0.045, 0.025, 500_000.0)
        assert result["in_buffer_zone"] is True
        assert result["quartile"] == 4

    def test_at_buffer_ceiling(self) -> None:
        # CET1 exactly at ceiling — no restriction
        result = maximum_distributable_amount(0.070, 0.045, 0.025, 500_000.0)
        assert result["in_buffer_zone"] is False
        assert result["max_payout_ratio"] == 1.0

    def test_negative_net_income(self) -> None:
        # Negative income — MDA should be 0 even with full payout ratio
        result = maximum_distributable_amount(0.10, 0.045, 0.025, -200_000.0)
        assert result["mda"] == 0.0

    def test_zero_buffer_raises(self) -> None:
        with pytest.raises(ValidationError, match="positive"):
            maximum_distributable_amount(0.05, 0.045, 0.0, 100_000.0)

    def test_negative_buffer_raises(self) -> None:
        with pytest.raises(ValidationError, match="positive"):
            maximum_distributable_amount(0.05, 0.045, -0.01, 100_000.0)

    def test_buffer_used_pct(self) -> None:
        # CET1 = 0.055, min = 0.045, buffer = 0.025
        # position = (0.055 - 0.045) / 0.025 = 0.4 → used = 0.6
        result = maximum_distributable_amount(0.055, 0.045, 0.025, 100_000.0)
        assert result["buffer_used_pct"] == pytest.approx(0.6)

    def test_result_keys(self) -> None:
        result = maximum_distributable_amount(0.06, 0.045, 0.025, 100_000.0)
        expected_keys = {
            "in_buffer_zone", "quartile", "max_payout_ratio",
            "mda", "buffer_used_pct",
        }
        assert set(result.keys()) == expected_keys


# ===================================================================
# capital_adequacy_summary
# ===================================================================


class TestCapitalAdequacySummary:
    def test_bcbs_well_capitalised(self, bcbs_config: dict[str, Any]) -> None:
        summary = capital_adequacy_summary(
            cet1_ratio=0.14,
            tier1_ratio=0.16,
            total_ratio=0.20,
            jurisdiction_config=bcbs_config,
        )
        assert summary["jurisdiction"] == "BCBS"
        assert summary["cet1_ratio"] == 0.14
        assert summary["buffers"]["capital_conservation_buffer"] == 0.025
        assert summary["buffers"]["countercyclical_buffer"] == 0.0
        assert summary["buffers"]["gsib_surcharge"] == 0.0
        assert summary["combined_buffer"] == 0.025
        assert summary["requirements"]["meets_minimum"] is True
        assert summary["requirements"]["meets_buffer"] is True

    def test_eu_with_gsib_and_ccyb(self, eu_config: dict[str, Any]) -> None:
        summary = capital_adequacy_summary(
            cet1_ratio=0.12,
            tier1_ratio=0.14,
            total_ratio=0.18,
            jurisdiction_config=eu_config,
            gsib_bucket=2,
            ccyb_rate=0.01,
        )
        assert summary["jurisdiction"] == "EU"
        assert summary["buffers"]["gsib_surcharge"] == 0.015
        assert summary["buffers"]["countercyclical_buffer"] == 0.01
        assert summary["combined_buffer"] == pytest.approx(0.025 + 0.01 + 0.015)

    def test_us_with_scb(self, us_config: dict[str, Any]) -> None:
        summary = capital_adequacy_summary(
            cet1_ratio=0.10,
            tier1_ratio=0.12,
            total_ratio=0.15,
            jurisdiction_config=us_config,
        )
        assert summary["jurisdiction"] == "US"
        # SCB replaces CConB with minimum 2.5%
        assert summary["buffers"]["capital_conservation_buffer"] == 0.025

    def test_gsib_bucket_none_treated_as_non_gsib(
        self, bcbs_config: dict[str, Any]
    ) -> None:
        summary = capital_adequacy_summary(
            cet1_ratio=0.10,
            tier1_ratio=0.12,
            total_ratio=0.15,
            jurisdiction_config=bcbs_config,
            gsib_bucket=None,
        )
        assert summary["buffers"]["gsib_surcharge"] == 0.0

    def test_ccyb_rate_zero_default(self, bcbs_config: dict[str, Any]) -> None:
        summary = capital_adequacy_summary(
            cet1_ratio=0.10,
            tier1_ratio=0.12,
            total_ratio=0.15,
            jurisdiction_config=bcbs_config,
        )
        assert summary["buffers"]["countercyclical_buffer"] == 0.0

    def test_undercapitalised_bank(self, bcbs_config: dict[str, Any]) -> None:
        summary = capital_adequacy_summary(
            cet1_ratio=0.04,
            tier1_ratio=0.05,
            total_ratio=0.07,
            jurisdiction_config=bcbs_config,
        )
        assert summary["requirements"]["meets_minimum"] is False
        assert summary["requirements"]["meets_buffer"] is False

    def test_summary_result_keys(self, bcbs_config: dict[str, Any]) -> None:
        summary = capital_adequacy_summary(
            cet1_ratio=0.10,
            tier1_ratio=0.12,
            total_ratio=0.15,
            jurisdiction_config=bcbs_config,
        )
        expected_keys = {
            "jurisdiction", "cet1_ratio", "tier1_ratio", "total_ratio",
            "buffers", "combined_buffer", "requirements",
        }
        assert set(summary.keys()) == expected_keys

    def test_gsib_bucket_5_high_buffers(self, bcbs_config: dict[str, Any]) -> None:
        summary = capital_adequacy_summary(
            cet1_ratio=0.15,
            tier1_ratio=0.17,
            total_ratio=0.22,
            jurisdiction_config=bcbs_config,
            gsib_bucket=5,
            ccyb_rate=0.025,
        )
        # CConB 2.5% + CCyB 2.5% + G-SIB 3.5% = 8.5%
        assert summary["combined_buffer"] == pytest.approx(0.085)
        assert summary["requirements"]["cet1_target"] == pytest.approx(0.13)


# ===================================================================
# Cross-jurisdiction integration tests
# ===================================================================


class TestCrossJurisdiction:
    """Verify buffer calculations across all three reference jurisdictions."""

    def test_all_jurisdictions_have_cconb(
        self,
        bcbs_config: dict[str, Any],
        eu_config: dict[str, Any],
        us_config: dict[str, Any],
    ) -> None:
        for cfg in (bcbs_config, eu_config, us_config):
            rate = capital_conservation_buffer(cfg)
            assert rate >= 0.025

    def test_eu_systemic_risk_buffer_structure(
        self, eu_config: dict[str, Any]
    ) -> None:
        # EU config should have systemic_risk_buffer section
        cap_req = eu_config["capital_requirements"]
        assert "systemic_risk_buffer" in cap_req
        assert cap_req["systemic_risk_buffer"]["enabled"] is True

    def test_us_has_stress_capital_buffer(
        self, us_config: dict[str, Any]
    ) -> None:
        cap_req = us_config["capital_requirements"]
        assert "stress_capital_buffer" in cap_req
        assert cap_req["stress_capital_buffer"]["replaces"] == "capital_conservation_buffer"

    def test_combined_buffer_ordering(self) -> None:
        """Higher G-SIB buckets should always produce larger combined buffers."""
        base_cconb = 0.025
        base_ccyb = 0.01
        base_dsib = 0.0
        prev = 0.0
        for bucket in range(1, 6):
            gsib = gsib_surcharge(bucket)
            cbr = combined_buffer_requirement(base_cconb, base_ccyb, gsib, base_dsib)
            assert cbr > prev
            prev = cbr
