"""Regression tests for jurisdiction-config correctness fixes (2024-2026 updates)."""

from creditriskengine.core.types import Jurisdiction
from creditriskengine.regulatory.loader import load_config


class TestChinaFiveTierDPD:
    """NFRA Measures 2023: substandard >90, doubtful >270, loss >360 DPD."""

    def test_dpd_buckets_match_measures_2023(self) -> None:
        cfg = load_config(Jurisdiction.CHINA)
        tiers = cfg["default_definition"]["five_tier_classification"]
        assert ">90" in tiers["substandard"]
        assert ">270" in tiers["doubtful"]
        assert ">360" in tiers["loss"]
        # The repealed 90-360 / 360-720 / >720 buckets must be gone.
        assert "720" not in tiers["doubtful"] and "720" not in tiers["loss"]

    def test_config_consistent_with_code(self) -> None:
        # ecl/emerging/china.py encodes the same 90/270/360 thresholds.
        from pathlib import Path

        from creditriskengine.ecl.emerging import china

        text = Path(china.__file__).read_text()
        assert "270" in text and "360" in text


class TestHongKongCCyB:
    def test_ccyb_is_half_percent(self) -> None:
        # HKMA cut the HK CCyB from 1.0% to 0.5%, effective 1 Jan 2025.
        cfg = load_config(Jurisdiction.HONG_KONG)
        ccyb = cfg["capital_requirements"]["countercyclical_buffer"]
        assert ccyb["current_hk_rate_pct"] == 0.005


class TestSingaporeOutputFloor:
    def test_floor_reaches_final_in_2029(self) -> None:
        # Revised MAS Notice 637: output floor hits 72.5% on 1 Jan 2029.
        cfg = load_config(Jurisdiction.SINGAPORE)
        phase_in = cfg["output_floor"]["phase_in"]
        assert phase_in["2029-01-01"] == 0.725
        # No 2030 step (that was the stale Basel/EU endpoint).
        assert "2030-01-01" not in phase_in
