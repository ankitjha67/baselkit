"""Integration tests with realistic 10k+ exposure portfolios."""

import numpy as np
import pytest

from creditriskengine.core.types import IFRS9Stage
from creditriskengine.ecl.ifrs9.ecl_calc import calculate_ecl, ecl_12_month
from creditriskengine.rwa.irb.formulas import PD_FLOOR, irb_risk_weight

# ============================================================
# Synthetic portfolio generation
# ============================================================

ASSET_CLASSES = ["corporate", "sovereign", "bank", "residential_mortgage", "qrre", "other_retail"]
NON_RETAIL = {"corporate", "sovereign", "bank"}

NUM_EXPOSURES = 10_000


@pytest.fixture(scope="module")
def synthetic_portfolio():
    """Generate a 10k-exposure synthetic portfolio with numpy seed 42."""
    rng = np.random.default_rng(42)

    # PDs: lognormal-ish, clipped to [PD_FLOOR, 0.30]
    pds = np.clip(rng.lognormal(mean=-4.5, sigma=1.5, size=NUM_EXPOSURES), PD_FLOOR, 0.30)

    # LGDs: beta distribution centred around 0.40
    lgds = np.clip(rng.beta(4, 6, size=NUM_EXPOSURES), 0.05, 0.95)

    # EADs: lognormal, typical corporate/retail mix
    eads = rng.lognormal(mean=12.0, sigma=2.0, size=NUM_EXPOSURES)

    # Maturities: uniform [1, 5] for non-retail, ignored for retail
    maturities = rng.uniform(1.0, 5.0, size=NUM_EXPOSURES)

    # Asset classes: weighted random choice
    weights = [0.35, 0.10, 0.10, 0.25, 0.10, 0.10]
    asset_class_indices = rng.choice(len(ASSET_CLASSES), size=NUM_EXPOSURES, p=weights)
    asset_classes = [ASSET_CLASSES[i] for i in asset_class_indices]

    return {
        "pds": pds,
        "lgds": lgds,
        "eads": eads,
        "maturities": maturities,
        "asset_classes": asset_classes,
    }


# ============================================================
# Full portfolio IRB risk weight calculation
# ============================================================

class TestPortfolioIRB:
    """Run IRB risk weights for 10k exposures and verify portfolio-level sanity."""

    def test_all_risk_weights_compute(self, synthetic_portfolio):
        """All 10k risk weights must compute without error."""
        port = synthetic_portfolio
        rws = []
        for i in range(NUM_EXPOSURES):
            ac = port["asset_classes"][i]
            mat = float(port["maturities"][i]) if ac in NON_RETAIL else 2.5
            rw = irb_risk_weight(
                pd=float(port["pds"][i]),
                lgd=float(port["lgds"][i]),
                asset_class=ac,
                maturity=mat,
            )
            rws.append(rw)

        rws = np.array(rws)
        assert len(rws) == NUM_EXPOSURES
        # All risk weights must be non-negative
        assert np.all(rws >= 0.0)

    def test_portfolio_rwa_sensible(self, synthetic_portfolio):
        """Total RWA should be positive and within a sensible range vs total EAD."""
        port = synthetic_portfolio
        rwa_total = 0.0
        for i in range(NUM_EXPOSURES):
            ac = port["asset_classes"][i]
            mat = float(port["maturities"][i]) if ac in NON_RETAIL else 2.5
            rw = irb_risk_weight(
                pd=float(port["pds"][i]),
                lgd=float(port["lgds"][i]),
                asset_class=ac,
                maturity=mat,
            )
            rwa_total += (rw / 100.0) * port["eads"][i]

        total_ead = float(np.sum(port["eads"]))
        avg_rw_pct = (rwa_total / total_ead) * 100.0

        # Average risk weight should be in a reasonable range (5-200%)
        assert 5.0 < avg_rw_pct < 200.0, f"Average RW: {avg_rw_pct:.1f}%"
        # Total RWA must be positive
        assert rwa_total > 0.0

    def test_rw_distribution_reasonable(self, synthetic_portfolio):
        """RW distribution should have reasonable percentiles."""
        port = synthetic_portfolio
        rws = []
        for i in range(NUM_EXPOSURES):
            ac = port["asset_classes"][i]
            mat = float(port["maturities"][i]) if ac in NON_RETAIL else 2.5
            rws.append(
                irb_risk_weight(
                    pd=float(port["pds"][i]),
                    lgd=float(port["lgds"][i]),
                    asset_class=ac,
                    maturity=mat,
                )
            )
        rws = np.array(rws)

        p10 = np.percentile(rws, 10)
        p50 = np.percentile(rws, 50)
        p90 = np.percentile(rws, 90)

        # 10th percentile should be low but positive
        assert 0.0 < p10 < 50.0, f"P10={p10:.1f}%"
        # Median in a reasonable range
        assert 5.0 < p50 < 150.0, f"P50={p50:.1f}%"
        # 90th percentile should be elevated but not extreme
        assert 30.0 < p90 < 400.0, f"P90={p90:.1f}%"


# ============================================================
# Stress test: double PDs, increase LGDs
# ============================================================

class TestStressScenario:
    """Stress test: PDs doubled, LGDs increased by 10pp."""

    def test_stress_rwa_exceeds_base(self, synthetic_portfolio):
        """Stressed RWA must exceed base-case RWA."""
        port = synthetic_portfolio

        base_rwa = 0.0
        stress_rwa = 0.0

        for i in range(NUM_EXPOSURES):
            ac = port["asset_classes"][i]
            pd_base = float(port["pds"][i])
            lgd_base = float(port["lgds"][i])
            ead = float(port["eads"][i])
            mat = float(port["maturities"][i]) if ac in NON_RETAIL else 2.5

            rw_base = irb_risk_weight(pd=pd_base, lgd=lgd_base, asset_class=ac, maturity=mat)
            base_rwa += (rw_base / 100.0) * ead

            # Stressed parameters
            pd_stress = min(pd_base * 2.0, 0.9999)
            lgd_stress = min(lgd_base + 0.10, 1.0)
            rw_stress = irb_risk_weight(pd=pd_stress, lgd=lgd_stress, asset_class=ac, maturity=mat)
            stress_rwa += (rw_stress / 100.0) * ead

        assert stress_rwa > base_rwa, "Stress RWA must exceed base RWA"
        # Stress should increase RWA by at least 20%
        increase_pct = (stress_rwa - base_rwa) / base_rwa * 100.0
        assert increase_pct > 20.0, f"Stress increase only {increase_pct:.1f}%"


# ============================================================
# ECL calculation on portfolio subset
# ============================================================

class TestPortfolioECL:
    """ECL calculation over a portfolio subset."""

    def test_12m_ecl_positive(self, synthetic_portfolio):
        """12-month ECL for all Stage 1 exposures must be positive."""
        port = synthetic_portfolio
        total_ecl = 0.0
        n = min(1000, NUM_EXPOSURES)  # test a subset for speed
        for i in range(n):
            ecl = ecl_12_month(
                pd_12m=float(port["pds"][i]),
                lgd=float(port["lgds"][i]),
                ead=float(port["eads"][i]),
                eir=0.05,
            )
            assert ecl >= 0.0
            total_ecl += ecl
        assert total_ecl > 0.0

    def test_lifetime_ecl_exceeds_12m(self, synthetic_portfolio):
        """Lifetime ECL (Stage 2) should exceed 12-month ECL for same exposure."""
        port = synthetic_portfolio
        for i in range(100):
            pd_12m = float(port["pds"][i])
            lgd = float(port["lgds"][i])
            ead = float(port["eads"][i])

            ecl_stage1 = calculate_ecl(
                stage=IFRS9Stage.STAGE_1,
                pd_12m=pd_12m,
                lgd=lgd,
                ead=ead,
                eir=0.05,
            )

            # Create a simple marginal PD curve (5 years)
            # Marginal PDs derived from cumulative: survival decays
            survival = np.cumprod(np.full(5, 1.0 - pd_12m))
            marginal_pds = np.diff(np.concatenate(([1.0], survival)))
            marginal_pds = np.abs(marginal_pds)  # ensure positive

            ecl_stage2 = calculate_ecl(
                stage=IFRS9Stage.STAGE_2,
                pd_12m=pd_12m,
                lgd=lgd,
                ead=ead,
                eir=0.05,
                marginal_pds=marginal_pds,
            )

            # Lifetime ECL should be >= 12-month ECL
            assert ecl_stage2 >= ecl_stage1 * 0.99, (
                f"Exposure {i}: lifetime ECL ({ecl_stage2:.2f}) < "
                f"12m ECL ({ecl_stage1:.2f})"
            )
