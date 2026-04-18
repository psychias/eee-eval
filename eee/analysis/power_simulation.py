"""
power_simulation.py — bootstrap power simulation by number of SOURCES k.

For a given observed score delta μ, simulates power to detect the delta as
a function of k (the number of independent source reports), using 5000
bootstrap replicates per k value.

Power is estimated as: fraction of replicates where the 95% CI of the mean
delta across k sources excludes zero.

Output: analysis_output/power_simulation.csv
Columns: k, mean_power, ci_lo_95, ci_hi_95, k_80
  (k_80 is the minimum k achieving ≥ 80% power, appended as metadata row)
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

OUT_DIR = _ROOT / "analysis_output"

N_RESAMPLES = 2000
K_RANGE = list(range(2, 21))  # k = 2..20 sources
ALPHA = 0.05
POWER_TARGET = 0.80
RNG = np.random.default_rng(42)


def estimate_power(deltas: np.ndarray, k: int, n_resamples: int = N_RESAMPLES) -> tuple[float, float, float]:
    """
    Bootstrap power for detecting a non-zero mean delta using k sources.

    For each resample:
    - draw k observations from the empirical delta distribution
    - compute bootstrap 95% CI of the mean
    - power = fraction of resamples where CI excludes 0

    Returns (mean_power, ci_lo, ci_hi) based on the binomial variance of the power estimate.
    """
    n_delta = len(deltas)
    reject = 0
    for _ in range(n_resamples):
        sample = RNG.choice(deltas, size=k, replace=True)
        # Vectorised inner bootstrap
        boot_samples = RNG.choice(sample, size=(100, k), replace=True)
        boot_means = boot_samples.mean(axis=1)
        lo = np.percentile(boot_means, 2.5)
        hi = np.percentile(boot_means, 97.5)
        if lo > 0 or hi < 0:
            reject += 1

    power = reject / n_resamples
    se = np.sqrt(power * (1 - power) / n_resamples)
    return round(power, 4), round(power - 1.96 * se, 4), round(power + 1.96 * se, 4)


def simulate_per_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """Run power simulation per benchmark (for harness_differs predictor)."""
    # Only benchmarks with >= 3 collision pairs
    bench_counts = df.groupby("benchmark")["delta"].count()
    valid = bench_counts[bench_counts >= 3].index.tolist()

    rows = []
    for bench in valid:
        sub = df[df["benchmark"] == bench]
        deltas = sub["delta"].values
        # Compute observed R² for harness_differs
        harness_diff = (sub["harness_a"] != sub["harness_b"]).astype(float).values
        if harness_diff.std() > 0 and len(deltas) > 2:
            from numpy.polynomial.polynomial import polyfit
            ss_tot = np.sum((np.abs(deltas) - np.mean(np.abs(deltas))) ** 2)
            if ss_tot > 0:
                corr = np.corrcoef(harness_diff, np.abs(deltas))[0, 1]
                r2_obs = corr ** 2 if not np.isnan(corr) else 0.0
            else:
                r2_obs = 0.0
        else:
            r2_obs = 0.0

        print(f"\n  Benchmark: {bench} (n={len(deltas)}, R²_obs={r2_obs:.4f})")
        k_80 = None
        for k in K_RANGE:
            pwr, ci_lo, ci_hi = estimate_power(deltas, k)
            # Also compute power at R²±0.15 (scale deltas accordingly)
            for r2_shift_label, r2_shift in [("lo", -0.15), ("hi", +0.15)]:
                scale = max(0.5, 1 + r2_shift)
                scaled_deltas = deltas * scale
                pwr_s, _, _ = estimate_power(scaled_deltas, k,
                                             n_resamples=500)
                rows.append({
                    "benchmark": bench, "k": k,
                    "mean_power": pwr, "ci_lo_95": ci_lo, "ci_hi_95": ci_hi,
                    "r2_obs": round(r2_obs, 4),
                    "r2_shift": r2_shift_label,
                    "power_shifted": pwr_s,
                })
            if k_80 is None and pwr >= POWER_TARGET:
                k_80 = k
            print(f"    k={k:2d} power={pwr:.3f}", end="")

        print(f"\n    k_80 = {k_80}")

    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    collisions_path = OUT_DIR / "collision_pairs.csv"
    if not collisions_path.exists():
        print("Run collision_detection.py first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(collisions_path)
    if df.empty:
        print("No collision pairs; using synthetic delta distribution.", file=sys.stderr)
        deltas = RNG.normal(loc=0.05, scale=0.1, size=50)
    else:
        deltas = df["delta"].values

    print(f"Delta distribution: n={len(deltas)}, mean={np.mean(deltas):.4f}, "
          f"std={np.std(deltas):.4f}")

    # ── Global power curve (backward compat) ──
    rows = []
    k_80 = None
    for k in K_RANGE:
        print(f"  k={k:2d} ...", end="", flush=True)
        pwr, ci_lo, ci_hi = estimate_power(deltas, k)
        print(f" power={pwr:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")
        rows.append({"k": k, "mean_power": pwr, "ci_lo_95": ci_lo, "ci_hi_95": ci_hi})
        if k_80 is None and pwr >= POWER_TARGET:
            k_80 = k

    result = pd.DataFrame(rows)
    result["k_80"] = k_80 if k_80 is not None else -1

    out_path = OUT_DIR / "power_simulation.csv"
    result.to_csv(out_path, index=False)
    print(f"\nPower simulation saved → {out_path}")
    print(f"k_80 (first k with ≥80% power) = {k_80}")
    print(result.to_string(index=False))

    # ── Per-benchmark power curves (for fig6) ──
    print("\n=== Per-benchmark power simulation ===")
    bench_df = simulate_per_benchmark(df)
    bench_out = OUT_DIR / "power_simulation_per_bench.csv"
    bench_df.to_csv(bench_out, index=False)
    print(f"\nPer-benchmark power simulation saved → {bench_out}")


if __name__ == "__main__":
    main()
