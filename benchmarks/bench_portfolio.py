"""
Performance benchmarks for CreditRiskEngine.

Run with: pytest benchmarks/ --benchmark-only
Or standalone: python benchmarks/bench_portfolio.py

Measures throughput for key operations on realistic portfolio sizes.
"""

import time

import numpy as np


def bench_irb_risk_weight_single() -> float:
    """Benchmark single IRB risk weight calculation."""
    from creditriskengine.rwa.irb.formulas import irb_risk_weight

    start = time.perf_counter()
    n = 100_000
    for _ in range(n):
        irb_risk_weight(pd=0.01, lgd=0.45, asset_class="corporate", maturity=2.5)
    elapsed = time.perf_counter() - start
    rate = n / elapsed
    print(f"IRB risk weight (single): {rate:,.0f} calc/sec ({elapsed:.3f}s for {n:,})")
    return rate


def bench_irb_portfolio_10k() -> float:
    """Benchmark IRB RW for 10,000 exposures."""
    from creditriskengine.rwa.irb.formulas import irb_risk_weight

    np.random.seed(42)
    n = 10_000
    pds = np.clip(np.random.lognormal(-4.5, 1.2, n), 0.0003, 0.99)
    lgds = np.random.uniform(0.10, 0.60, n)
    maturities = np.random.uniform(1.0, 5.0, n)

    start = time.perf_counter()
    for i in range(n):
        irb_risk_weight(
            pd=float(pds[i]),
            lgd=float(lgds[i]),
            asset_class="corporate",
            maturity=float(maturities[i]),
        )
    elapsed = time.perf_counter() - start
    rate = n / elapsed
    print(f"IRB portfolio (10k): {rate:,.0f} exposures/sec ({elapsed:.3f}s)")
    return rate


def bench_sa_risk_weight_10k() -> float:
    """Benchmark SA risk weight for 10,000 exposures."""
    from creditriskengine.core.types import CreditQualityStep, SAExposureClass
    from creditriskengine.rwa.standardized.credit_risk_sa import assign_sa_risk_weight

    n = 10_000
    classes = [SAExposureClass.CORPORATE] * n
    cqs_vals = [CreditQualityStep.CQS_2, CreditQualityStep.CQS_3,
                CreditQualityStep.UNRATED, CreditQualityStep.CQS_1]

    start = time.perf_counter()
    for i in range(n):
        assign_sa_risk_weight(classes[i], cqs_vals[i % 4])
    elapsed = time.perf_counter() - start
    rate = n / elapsed
    print(f"SA risk weight (10k): {rate:,.0f} exposures/sec ({elapsed:.3f}s)")
    return rate


def bench_ecl_calculation() -> float:
    """Benchmark IFRS 9 ECL calculation for 10,000 exposures."""
    from creditriskengine.ecl.ifrs9.ecl_calc import ecl_12_month

    n = 10_000
    start = time.perf_counter()
    for _ in range(n):
        ecl_12_month(pd_12m=0.02, lgd=0.40, ead=1_000_000, eir=0.05)
    elapsed = time.perf_counter() - start
    rate = n / elapsed
    print(f"IFRS 9 ECL (10k): {rate:,.0f} calc/sec ({elapsed:.3f}s)")
    return rate


def bench_stress_test_projection() -> float:
    """Benchmark multi-period stress test with 10k exposures."""
    from creditriskengine.portfolio.stress_testing import multi_period_projection

    np.random.seed(42)
    n = 10_000
    pds = np.random.uniform(0.005, 0.10, n)
    lgds = np.random.uniform(0.20, 0.60, n)
    eads = np.random.lognormal(13, 1.5, n)
    pd_mult = np.array([1.5, 2.0, 2.5])
    lgd_add = np.array([0.05, 0.10, 0.08])

    start = time.perf_counter()
    multi_period_projection(pds, lgds, eads, pd_mult, lgd_add)
    elapsed = time.perf_counter() - start
    print(f"Stress test (10k × 3yr): {elapsed:.3f}s")
    return elapsed


def bench_monte_carlo_simulation() -> float:
    """Benchmark copula simulation with 10k exposures."""
    from creditriskengine.portfolio.copula import simulate_single_factor

    np.random.seed(42)
    n = 10_000
    pds = np.random.uniform(0.005, 0.10, n)
    lgds = np.random.uniform(0.20, 0.60, n)
    eads = np.random.lognormal(13, 1.5, n)

    start = time.perf_counter()
    simulate_single_factor(pds, lgds, eads, rho=0.15, n_simulations=10_000)
    elapsed = time.perf_counter() - start
    print(f"Monte Carlo (10k exp × 10k sims): {elapsed:.3f}s")
    return elapsed


def bench_1m_portfolio() -> float:
    """Benchmark IRB RW for 1,000,000 exposures (vectorized where possible)."""
    from creditriskengine.rwa.irb.formulas import irb_risk_weight

    np.random.seed(42)
    n = 1_000_000
    pds = np.clip(np.random.lognormal(-4.5, 1.2, n), 0.0003, 0.99)
    lgds = np.random.uniform(0.10, 0.60, n)
    maturities = np.random.uniform(1.0, 5.0, n)

    start = time.perf_counter()
    for i in range(n):
        irb_risk_weight(
            pd=float(pds[i]),
            lgd=float(lgds[i]),
            asset_class="corporate",
            maturity=float(maturities[i]),
        )
    elapsed = time.perf_counter() - start
    rate = n / elapsed
    print(f"IRB portfolio (1M): {rate:,.0f} exposures/sec ({elapsed:.3f}s)")
    return rate


if __name__ == "__main__":
    print("=" * 60)
    print("CreditRiskEngine Performance Benchmarks")
    print("=" * 60)
    print()

    bench_irb_risk_weight_single()
    bench_irb_portfolio_10k()
    bench_sa_risk_weight_10k()
    bench_ecl_calculation()
    bench_stress_test_projection()
    bench_monte_carlo_simulation()

    print()
    print("Large portfolio benchmark (may take ~30s):")
    bench_1m_portfolio()

    print()
    print("=" * 60)
    print("Benchmarks complete.")
