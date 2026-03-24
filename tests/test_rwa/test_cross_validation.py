"""Cross-validation tests against vendor reference ranges."""

import pytest

from creditriskengine.rwa.irb.formulas import (
    PD_FLOOR,
    asset_correlation_corporate,
    asset_correlation_other_retail,
    irb_risk_weight,
)

# ============================================================
# Monotonicity tests
# ============================================================

class TestMonotonicity:
    """Risk weights must be monotonically increasing with PD, LGD, and maturity."""

    PDS = [PD_FLOOR, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30]

    def test_rw_monotonic_in_pd_corporate(self):
        """Corporate RW increases with PD (LGD=45%, M=2.5)."""
        rws = [
            irb_risk_weight(pd=pd, lgd=0.45, asset_class="corporate", maturity=2.5)
            for pd in self.PDS
        ]
        for i in range(1, len(rws)):
            assert rws[i] > rws[i - 1], (
                f"Not monotonic at PD={self.PDS[i]}: "
                f"RW={rws[i]:.2f} <= prev={rws[i-1]:.2f}"
            )

    def test_rw_monotonic_in_pd_mortgage(self):
        """Residential mortgage RW increases with PD."""
        rws = [
            irb_risk_weight(pd=pd, lgd=0.35, asset_class="residential_mortgage")
            for pd in self.PDS
        ]
        for i in range(1, len(rws)):
            assert rws[i] > rws[i - 1]

    def test_rw_monotonic_in_pd_qrre(self):
        """QRRE RW increases with PD."""
        rws = [
            irb_risk_weight(pd=pd, lgd=0.75, asset_class="qrre")
            for pd in self.PDS
        ]
        for i in range(1, len(rws)):
            assert rws[i] > rws[i - 1]

    def test_rw_monotonic_in_pd_other_retail(self):
        """Other retail RW increases with PD."""
        rws = [
            irb_risk_weight(pd=pd, lgd=0.50, asset_class="other_retail")
            for pd in self.PDS
        ]
        for i in range(1, len(rws)):
            assert rws[i] > rws[i - 1]

    def test_rw_monotonic_in_lgd(self):
        """Corporate RW increases with LGD (PD=2%, M=2.5)."""
        lgds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 0.90]
        rws = [
            irb_risk_weight(pd=0.02, lgd=lgd, asset_class="corporate", maturity=2.5)
            for lgd in lgds
        ]
        for i in range(1, len(rws)):
            assert rws[i] > rws[i - 1], (
                f"Not monotonic at LGD={lgds[i]}: "
                f"RW={rws[i]:.2f} <= prev={rws[i-1]:.2f}"
            )

    def test_rw_monotonic_in_maturity(self):
        """Corporate RW increases with maturity (PD=2%, LGD=45%)."""
        maturities = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        rws = [
            irb_risk_weight(pd=0.02, lgd=0.45, asset_class="corporate", maturity=m)
            for m in maturities
        ]
        for i in range(1, len(rws)):
            assert rws[i] > rws[i - 1], (
                f"Not monotonic at M={maturities[i]}: "
                f"RW={rws[i]:.2f} <= prev={rws[i-1]:.2f}"
            )


# ============================================================
# Correlation bounds cross-validation
# ============================================================

class TestCorrelationBoundsCrossValidation:
    """Cross-validate correlation bounds across the full PD spectrum."""

    def test_corporate_correlation_bounds_full_range(self):
        """Corporate correlation stays in [0.12, 0.24] for all PDs in (0, 1)."""
        pds = [PD_FLOOR, 0.001, 0.005, 0.01, 0.05, 0.10, 0.20, 0.50, 0.99]
        for pd in pds:
            r = asset_correlation_corporate(pd)
            assert 0.12 <= r <= 0.24, f"Corporate R({pd})={r}"

    def test_other_retail_correlation_bounds_full_range(self):
        """Other retail correlation stays in [0.03, 0.16] for all PDs."""
        pds = [PD_FLOOR, 0.001, 0.005, 0.01, 0.05, 0.10, 0.20, 0.50, 0.99]
        for pd in pds:
            r = asset_correlation_other_retail(pd)
            assert 0.03 <= r <= 0.16, f"Other retail R({pd})={r}"

    def test_corporate_correlation_decreases_with_pd(self):
        """Corporate correlation decreases as PD increases (inverse relationship)."""
        pds = [PD_FLOOR, 0.001, 0.005, 0.01, 0.05, 0.10, 0.20, 0.50]
        rs = [asset_correlation_corporate(pd) for pd in pds]
        for i in range(1, len(rs)):
            assert rs[i] <= rs[i - 1], (
                f"Correlation not decreasing: R({pds[i]})={rs[i]:.4f} "
                f"> R({pds[i-1]})={rs[i-1]:.4f}"
            )


# ============================================================
# Known approximate RW ranges (QIS-aligned)
# ============================================================

class TestQISReferenceRanges:
    """Verify RW values fall within ranges consistent with published regulatory
    QIS (Quantitative Impact Study) data and Basel Committee worked examples.

    These ranges are intentionally wide to accommodate minor methodology
    differences while catching egregious formula errors.
    """

    def test_investment_grade_corporate(self):
        """IG corporate (PD=0.1%, LGD=45%, M=2.5): RW approx 30-55%."""
        rw = irb_risk_weight(pd=0.001, lgd=0.45, asset_class="corporate", maturity=2.5)
        assert 25.0 < rw < 60.0, f"IG corporate RW={rw:.1f}%"

    def test_sub_investment_grade_corporate(self):
        """Sub-IG corporate (PD=3%, LGD=45%, M=2.5): RW approx 90-140%."""
        rw = irb_risk_weight(pd=0.03, lgd=0.45, asset_class="corporate", maturity=2.5)
        assert 80.0 < rw < 160.0, f"Sub-IG corporate RW={rw:.1f}%"

    def test_high_yield_corporate(self):
        """High-yield corporate (PD=10%, LGD=45%, M=2.5): RW approx 130-200%."""
        rw = irb_risk_weight(pd=0.10, lgd=0.45, asset_class="corporate", maturity=2.5)
        assert 120.0 < rw < 220.0, f"HY corporate RW={rw:.1f}%"

    def test_sme_corporate_lower_rw(self):
        """SME corporate gets lower RW due to firm-size adjustment."""
        rw_large = irb_risk_weight(pd=0.02, lgd=0.45, asset_class="corporate", maturity=2.5)
        rw_sme = irb_risk_weight(
            pd=0.02, lgd=0.45, asset_class="corporate", maturity=2.5,
            turnover_eur_millions=10.0,
        )
        assert rw_sme < rw_large, "SME RW should be less than large corporate"
        # Reduction is typically 5-15%
        reduction_pct = (rw_large - rw_sme) / rw_large * 100.0
        assert 3.0 < reduction_pct < 25.0, f"SME reduction: {reduction_pct:.1f}%"

    def test_prime_mortgage(self):
        """Prime mortgage (PD=0.5%, LGD=15%): RW approx 5-20%."""
        rw = irb_risk_weight(pd=0.005, lgd=0.15, asset_class="residential_mortgage")
        assert 3.0 < rw < 25.0, f"Prime mortgage RW={rw:.1f}%"

    def test_subprime_mortgage(self):
        """Subprime mortgage (PD=5%, LGD=35%): RW approx 60-150%."""
        rw = irb_risk_weight(pd=0.05, lgd=0.35, asset_class="residential_mortgage")
        assert 50.0 < rw < 150.0, f"Subprime mortgage RW={rw:.1f}%"

    def test_credit_card_qrre(self):
        """Credit card QRRE (PD=3%, LGD=75%): RW approx 30-80%."""
        rw = irb_risk_weight(pd=0.03, lgd=0.75, asset_class="qrre")
        assert 25.0 < rw < 90.0, f"QRRE RW={rw:.1f}%"

    def test_qrre_transactor_discount(self):
        """QRRE transactor gets 75% of revolver RW."""
        rw_revolver = irb_risk_weight(pd=0.02, lgd=0.75, asset_class="qrre")
        rw_transactor = irb_risk_weight(
            pd=0.02, lgd=0.75, asset_class="qrre", is_qrre_transactor=True,
        )
        assert rw_transactor == pytest.approx(rw_revolver * 0.75, rel=1e-9)

    def test_sovereign_low_pd(self):
        """Sovereign (PD=0.03%, LGD=45%, M=2.5): RW approx 14-25%."""
        rw = irb_risk_weight(pd=PD_FLOOR, lgd=0.45, asset_class="sovereign", maturity=2.5)
        assert 10.0 < rw < 30.0, f"Sovereign low PD RW={rw:.1f}%"

    def test_bank_exposure(self):
        """Bank (PD=0.5%, LGD=45%, M=2.5): RW approx 40-75%."""
        rw = irb_risk_weight(pd=0.005, lgd=0.45, asset_class="bank", maturity=2.5)
        assert 30.0 < rw < 85.0, f"Bank RW={rw:.1f}%"

    def test_other_retail(self):
        """Other retail (PD=5%, LGD=50%): RW approx 30-70%."""
        rw = irb_risk_weight(pd=0.05, lgd=0.50, asset_class="other_retail")
        assert 20.0 < rw < 80.0, f"Other retail RW={rw:.1f}%"

    def test_short_maturity_reduces_rw(self):
        """Maturity of 1y should produce lower RW than 5y."""
        rw_1y = irb_risk_weight(pd=0.02, lgd=0.45, asset_class="corporate", maturity=1.0)
        rw_5y = irb_risk_weight(pd=0.02, lgd=0.45, asset_class="corporate", maturity=5.0)
        assert rw_1y < rw_5y
        # Difference should be meaningful (>10% relative)
        ratio = rw_5y / rw_1y
        assert ratio > 1.10, f"5y/1y ratio only {ratio:.2f}"
