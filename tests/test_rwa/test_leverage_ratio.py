"""Tests for leverage ratio framework — BCBS d424 CRE80."""

import pytest

from creditriskengine.rwa.leverage_ratio import (
    MINIMUM_LEVERAGE_RATIO_PCT,
    OBS_CCF_FLOOR,
    SA_CCR_ALPHA,
    derivative_exposure_sa_ccr,
    leverage_ratio,
    leverage_ratio_summary,
    meets_leverage_requirement,
    off_balance_sheet_exposure,
    total_exposure_measure,
)


class TestLeverageRatio:
    def test_basic_ratio(self) -> None:
        # 60 / 1000 = 6%
        assert leverage_ratio(60.0, 1000.0) == pytest.approx(0.06)

    def test_minimum_3pct(self) -> None:
        # 30 / 1000 = exactly 3%
        assert leverage_ratio(30.0, 1000.0) == pytest.approx(0.03)

    def test_high_capital(self) -> None:
        assert leverage_ratio(100.0, 500.0) == pytest.approx(0.20)

    def test_zero_exposure_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            leverage_ratio(50.0, 0.0)

    def test_negative_exposure_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            leverage_ratio(50.0, -100.0)

    def test_zero_capital(self) -> None:
        assert leverage_ratio(0.0, 1000.0) == pytest.approx(0.0)


class TestTotalExposureMeasure:
    def test_simple_aggregation(self) -> None:
        tem = total_exposure_measure(
            on_balance_sheet=800.0,
            derivative_exposures=100.0,
            sft_exposures=50.0,
            off_balance_sheet_items=50.0,
        )
        assert tem == pytest.approx(1000.0)

    def test_with_ccf_obs(self) -> None:
        tem = total_exposure_measure(
            on_balance_sheet=800.0,
            derivative_exposures=100.0,
            sft_exposures=50.0,
            off_balance_sheet_items=200.0,
            ccf_obs=0.5,
        )
        # 800 + 100 + 50 + (200 * 0.5) = 1050
        assert tem == pytest.approx(1050.0)

    def test_default_ccf_obs_is_one(self) -> None:
        tem = total_exposure_measure(500.0, 200.0, 100.0, 300.0)
        assert tem == pytest.approx(1100.0)

    def test_all_zeros(self) -> None:
        assert total_exposure_measure(0.0, 0.0, 0.0, 0.0) == pytest.approx(0.0)


class TestDerivativeExposureSaCcr:
    def test_basic_sa_ccr(self) -> None:
        # alpha * (RC + PFE) = 1.4 * (100 + 50) = 210
        ead = derivative_exposure_sa_ccr(100.0, 50.0)
        assert ead == pytest.approx(210.0)

    def test_with_collateral(self) -> None:
        # RC_net = max(100 - 60, 0) = 40
        # EAD = 1.4 * (40 + 50) = 126
        ead = derivative_exposure_sa_ccr(100.0, 50.0, collateral_held=60.0)
        assert ead == pytest.approx(126.0)

    def test_collateral_exceeds_rc(self) -> None:
        # RC_net = max(100 - 200, 0) = 0
        # EAD = 1.4 * (0 + 50) = 70
        ead = derivative_exposure_sa_ccr(100.0, 50.0, collateral_held=200.0)
        assert ead == pytest.approx(70.0)

    def test_custom_alpha(self) -> None:
        ead = derivative_exposure_sa_ccr(100.0, 50.0, alpha=1.0)
        assert ead == pytest.approx(150.0)

    def test_default_alpha(self) -> None:
        assert SA_CCR_ALPHA == pytest.approx(1.4)

    def test_zero_rc_and_pfe(self) -> None:
        assert derivative_exposure_sa_ccr(0.0, 0.0) == pytest.approx(0.0)

    def test_invalid_alpha_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            derivative_exposure_sa_ccr(100.0, 50.0, alpha=0.0)

        with pytest.raises(ValueError, match="positive"):
            derivative_exposure_sa_ccr(100.0, 50.0, alpha=-1.0)


class TestOffBalanceSheetExposure:
    def test_ccf_above_floor(self) -> None:
        # CCF = 50% > 10% floor
        exp = off_balance_sheet_exposure(1000.0, 0.50)
        assert exp == pytest.approx(500.0)

    def test_ccf_at_floor(self) -> None:
        # CCF = 10% = floor
        exp = off_balance_sheet_exposure(1000.0, 0.10)
        assert exp == pytest.approx(100.0)

    def test_ccf_below_floor_gets_floored(self) -> None:
        # CCF = 0% < 10% floor, effective CCF = 10% per CRE80.38
        exp = off_balance_sheet_exposure(1000.0, 0.0)
        assert exp == pytest.approx(100.0)

    def test_ccf_5pct_floored_to_10pct(self) -> None:
        exp = off_balance_sheet_exposure(2000.0, 0.05)
        assert exp == pytest.approx(200.0)  # 10% * 2000

    def test_ccf_100pct(self) -> None:
        exp = off_balance_sheet_exposure(500.0, 1.0)
        assert exp == pytest.approx(500.0)

    def test_obs_ccf_floor_constant(self) -> None:
        assert OBS_CCF_FLOOR == pytest.approx(0.10)

    def test_zero_notional(self) -> None:
        assert off_balance_sheet_exposure(0.0, 0.50) == pytest.approx(0.0)

    def test_negative_notional_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            off_balance_sheet_exposure(-100.0, 0.50)


class TestMeetsLeverageRequirement:
    def test_above_minimum(self) -> None:
        assert meets_leverage_requirement(0.05) is True

    def test_at_minimum(self) -> None:
        assert meets_leverage_requirement(0.03) is True

    def test_below_minimum(self) -> None:
        assert meets_leverage_requirement(0.029) is False

    def test_default_minimum(self) -> None:
        assert MINIMUM_LEVERAGE_RATIO_PCT == pytest.approx(0.03)

    def test_with_gsib_buffer(self) -> None:
        # 3% min + 0.5% buffer = 3.5% required
        assert meets_leverage_requirement(0.035, gsib_buffer_pct=0.005) is True
        assert meets_leverage_requirement(0.034, gsib_buffer_pct=0.005) is False

    def test_custom_minimum(self) -> None:
        assert meets_leverage_requirement(0.04, minimum_pct=0.04) is True
        assert meets_leverage_requirement(0.039, minimum_pct=0.04) is False

    def test_gsib_plus_custom_minimum(self) -> None:
        # 4% + 1% buffer = 5%
        assert meets_leverage_requirement(0.05, minimum_pct=0.04, gsib_buffer_pct=0.01) is True
        assert meets_leverage_requirement(0.049, minimum_pct=0.04, gsib_buffer_pct=0.01) is False


class TestLeverageRatioSummary:
    def test_basic_summary(self) -> None:
        result = leverage_ratio_summary(
            tier1_capital=60.0,
            on_bs=800.0,
            derivatives=100.0,
            sfts=50.0,
            obs_items=50.0,
        )
        assert result["tier1_capital"] == pytest.approx(60.0)
        assert result["total_exposure_measure"] == pytest.approx(1000.0)
        assert result["leverage_ratio"] == pytest.approx(0.06)
        assert result["minimum_pct"] == pytest.approx(0.03)
        assert result["gsib_buffer_pct"] == pytest.approx(0.0)
        assert result["required_ratio"] == pytest.approx(0.03)
        assert result["meets_requirement"] is True

    def test_non_compliant(self) -> None:
        result = leverage_ratio_summary(
            tier1_capital=20.0,
            on_bs=800.0,
            derivatives=100.0,
            sfts=50.0,
            obs_items=50.0,
        )
        # 20 / 1000 = 2% < 3%
        assert result["leverage_ratio"] == pytest.approx(0.02)
        assert result["meets_requirement"] is False

    def test_with_gsib_surcharge(self) -> None:
        result = leverage_ratio_summary(
            tier1_capital=35.0,
            on_bs=900.0,
            derivatives=50.0,
            sfts=25.0,
            obs_items=25.0,
            jurisdiction_config={"gsib_surcharge_pct": 0.01},
        )
        # TEM = 1000, ratio = 3.5%
        # Required = 3% + 50% * 1% = 3.5%
        assert result["leverage_ratio"] == pytest.approx(0.035)
        assert result["gsib_buffer_pct"] == pytest.approx(0.005)
        assert result["required_ratio"] == pytest.approx(0.035)
        assert result["meets_requirement"] is True

    def test_gsib_surcharge_non_compliant(self) -> None:
        result = leverage_ratio_summary(
            tier1_capital=34.0,
            on_bs=900.0,
            derivatives=50.0,
            sfts=25.0,
            obs_items=25.0,
            jurisdiction_config={"gsib_surcharge_pct": 0.01},
        )
        # ratio = 3.4% < 3.5% required
        assert result["meets_requirement"] is False

    def test_custom_minimum_pct(self) -> None:
        result = leverage_ratio_summary(
            tier1_capital=40.0,
            on_bs=800.0,
            derivatives=100.0,
            sfts=50.0,
            obs_items=50.0,
            jurisdiction_config={"minimum_pct": 0.05},
        )
        # ratio = 4% < 5% required
        assert result["leverage_ratio"] == pytest.approx(0.04)
        assert result["minimum_pct"] == pytest.approx(0.05)
        assert result["meets_requirement"] is False

    def test_result_keys(self) -> None:
        result = leverage_ratio_summary(50.0, 800.0, 100.0, 50.0, 50.0)
        expected_keys = {
            "tier1_capital", "on_balance_sheet", "derivative_exposures",
            "sft_exposures", "off_balance_sheet", "total_exposure_measure",
            "leverage_ratio", "minimum_pct", "gsib_buffer_pct",
            "required_ratio", "meets_requirement",
        }
        assert set(result.keys()) == expected_keys

    def test_component_passthrough(self) -> None:
        result = leverage_ratio_summary(
            tier1_capital=50.0,
            on_bs=700.0,
            derivatives=150.0,
            sfts=80.0,
            obs_items=70.0,
        )
        assert result["on_balance_sheet"] == pytest.approx(700.0)
        assert result["derivative_exposures"] == pytest.approx(150.0)
        assert result["sft_exposures"] == pytest.approx(80.0)
        assert result["off_balance_sheet"] == pytest.approx(70.0)

    def test_no_jurisdiction_config_uses_defaults(self) -> None:
        result = leverage_ratio_summary(30.0, 500.0, 200.0, 150.0, 150.0)
        assert result["minimum_pct"] == pytest.approx(0.03)
        assert result["gsib_buffer_pct"] == pytest.approx(0.0)
