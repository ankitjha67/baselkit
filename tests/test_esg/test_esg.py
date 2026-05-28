"""Tests for the ESG ratings → PD adjustment adapter."""

from __future__ import annotations

import pytest

from creditriskengine.esg import (
    ESGProvider,
    esg_pd_multiplier,
    normalise_esg_score,
)

# ============================================================================
# Normalisation
# ============================================================================


class TestNormalise:
    def test_msci_best_and_worst(self) -> None:
        assert normalise_esg_score(ESGProvider.MSCI, "AAA") == 0.0
        assert normalise_esg_score(ESGProvider.MSCI, "CCC") == 1.0

    def test_msci_monotonic(self) -> None:
        aaa = normalise_esg_score(ESGProvider.MSCI, "AAA")
        bbb = normalise_esg_score(ESGProvider.MSCI, "BBB")
        ccc = normalise_esg_score(ESGProvider.MSCI, "CCC")
        assert aaa < bbb < ccc

    def test_msci_case_insensitive(self) -> None:
        assert normalise_esg_score(ESGProvider.MSCI, "aaa") == 0.0

    def test_msci_invalid_grade(self) -> None:
        with pytest.raises(ValueError, match="Unknown MSCI grade"):
            normalise_esg_score(ESGProvider.MSCI, "ZZZ")

    def test_msci_non_string_raises(self) -> None:
        with pytest.raises(ValueError, match="letter grade string"):
            normalise_esg_score(ESGProvider.MSCI, 5.0)

    def test_sustainalytics_higher_is_worse(self) -> None:
        low_risk = normalise_esg_score(ESGProvider.SUSTAINALYTICS, 5.0)
        high_risk = normalise_esg_score(ESGProvider.SUSTAINALYTICS, 40.0)
        assert low_risk < high_risk

    def test_sustainalytics_scaling(self) -> None:
        # 25 / 50 = 0.5
        assert normalise_esg_score(ESGProvider.SUSTAINALYTICS, 25.0) == pytest.approx(0.5)

    def test_sustainalytics_cap(self) -> None:
        # Above 50 caps at 1.0
        assert normalise_esg_score(ESGProvider.SUSTAINALYTICS, 80.0) == 1.0

    def test_sp_global_inverted(self) -> None:
        # Higher S&P score = better ESG = lower risk
        good = normalise_esg_score(ESGProvider.SP_GLOBAL, 90.0)
        bad = normalise_esg_score(ESGProvider.SP_GLOBAL, 10.0)
        assert good < bad
        assert good == pytest.approx(0.10)
        assert bad == pytest.approx(0.90)


# ============================================================================
# PD multiplier
# ============================================================================


class TestPDMultiplier:
    def test_neutral_score_is_one(self) -> None:
        assert esg_pd_multiplier(0.5) == pytest.approx(1.0)

    def test_best_esg_gives_relief(self) -> None:
        mult = esg_pd_multiplier(0.0, max_relief=0.10)
        assert mult == pytest.approx(0.90)

    def test_worst_esg_gives_uplift(self) -> None:
        mult = esg_pd_multiplier(1.0, max_uplift=0.30)
        assert mult == pytest.approx(1.30)

    def test_monotonic(self) -> None:
        m0 = esg_pd_multiplier(0.0)
        m5 = esg_pd_multiplier(0.5)
        m1 = esg_pd_multiplier(1.0)
        assert m0 < m5 < m1

    def test_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="must be in"):
            esg_pd_multiplier(1.5)
        with pytest.raises(ValueError, match="must be in"):
            esg_pd_multiplier(-0.1)

    def test_end_to_end_msci(self) -> None:
        # MSCI CCC (worst) → normalised 1.0 → uplift 1.30
        score = normalise_esg_score(ESGProvider.MSCI, "CCC")
        mult = esg_pd_multiplier(score)
        assert mult == pytest.approx(1.30)

    def test_end_to_end_msci_best(self) -> None:
        score = normalise_esg_score(ESGProvider.MSCI, "AAA")
        mult = esg_pd_multiplier(score)
        assert mult == pytest.approx(0.90)
