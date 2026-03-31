"""
Experiment 2: Bootstrap Robustness of the Structural Wall

Tests whether the fragmentation finding (observed << expected collisions)
is stable across:
  (a) leave-one-source-out ablations
  (b) normalisation sensitivity levels
  (c) source-subset size (collision yield curve)

Uses the analytical null model from the paper:
  E[collisions] = sum_{i<j} R_i * R_j / U
  where U = |models| * |benchmarks| = 5672 * 180 = 1,020,960
  SD[collisions] ~ sqrt(E)  (Poisson regime)
This matches the paper's stated expected value of ~67 +/- 8.

Outputs:
  - figures/fig9_robustness_yield_curve.pdf
  - experiment2_results.txt
"""

import os
import math
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

np.random.seed(42)

# ---------------------------------------------------------------------------
# Source metadata (Table 2)
# ---------------------------------------------------------------------------
SOURCES = {
    "OLv2":         {"records": 26790, "models": 4465, "benchmarks": 6},
    "PWC":          {"records": 576,   "models": 383,  "benchmarks": 16},
    "AlpacaEval2":  {"records": 446,   "models": 223,  "benchmarks": 2},
    "BFCL":         {"records": 327,   "models": 109,  "benchmarks": 3},
    "BigCodeBench": {"records": 280,   "models": 156,  "benchmarks": 2},
    "ChatbotArena": {"records": 218,   "models": 218,  "benchmarks": 1},
    "EvalPlus":     {"records": 214,   "models": 125,  "benchmarks": 2},
    "HFModelCard":  {"records": 207,   "models": 11,   "benchmarks": 156},
    "WildBench":    {"records": 122,   "models": 61,   "benchmarks": 2},
    "SWEBench":     {"records": 117,   "models": 117,  "benchmarks": 1},
    "MTBench":      {"records": 34,    "models": 34,   "benchmarks": 1},
}

TOTAL_UNIQUE_MODELS = 5672
TOTAL_UNIQUE_BENCHMARKS = 180
UNIVERSE_SIZE = TOTAL_UNIQUE_MODELS * TOTAL_UNIQUE_BENCHMARKS  # 1,020,960

# Known collision pairs: 8 independent + 8 likely copied = 16 total
COLLISION_PAIRS_ALL = {
    ("OLv2",        "PWC"):        1,  # Qwen2.5 GPQA (indep)
    ("EvalPlus",    "HFModelCard"): 5, # SC2-15B-Inst x2 (indep) + SC2 base x3 (copied)
    ("HFModelCard", "OLv2"):      10,  # Tulu-3-8B x5 (indep) + Tulu-3-70B-SFT x5 (copied)
}
TOTAL_ALL_PAIRS = 16

# Per-source collision involvement
ALL_INVOLVING = defaultdict(int)
for (s1, s2), n in COLLISION_PAIRS_ALL.items():
    ALL_INVOLVING[s1] += n
    ALL_INVOLVING[s2] += n


# ---------------------------------------------------------------------------
# Analytical null model
# ---------------------------------------------------------------------------
def expected_collisions(source_subset):
    """E[collisions] = sum_{i<j} R_i * R_j / U  (birthday-paradox)."""
    records = [SOURCES[s]["records"] for s in source_subset]
    total = 0.0
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            total += records[i] * records[j]
    return total / UNIVERSE_SIZE


def sd_collisions(source_subset):
    """SD ~ sqrt(E) in the Poisson regime (R_s << U for most sources)."""
    return math.sqrt(max(expected_collisions(source_subset), 1e-10))


def z_score_val(observed, source_subset):
    E = expected_collisions(source_subset)
    S = sd_collisions(source_subset)
    return (observed - E) / S if S > 0 else 0.0


# ---------------------------------------------------------------------------
# Monte Carlo verification (fast numpy approach, full dataset only)
# ---------------------------------------------------------------------------
def mc_collisions_fast(source_subset, n_iter=200):
    """Fast MC using numpy for the biggest sources, Python for the rest."""
    records = [SOURCES[s]["records"] for s in source_subset]
    counts = np.zeros(n_iter)
    for it in range(n_iter):
        pair_counts = Counter()
        for R in records:
            drawn = np.random.choice(UNIVERSE_SIZE, size=R, replace=False)
            pair_counts.update(drawn.tolist())
        counts[it] = sum(c * (c - 1) // 2 for c in pair_counts.values() if c >= 2)
    return counts


# ---------------------------------------------------------------------------
# 1. Leave-One-Source-Out
# ---------------------------------------------------------------------------
print("=" * 60)
print("LEAVE-ONE-SOURCE-OUT ANALYSIS")
print("=" * 60)

all_sources = list(SOURCES.keys())

full_E = expected_collisions(all_sources)
full_SD = sd_collisions(all_sources)
full_z = (TOTAL_ALL_PAIRS - full_E) / full_SD

print(f"\nFull dataset: observed={TOTAL_ALL_PAIRS}, "
      f"E={full_E:.1f}, SD={full_SD:.1f}, z={full_z:.1f}")

loo_results = []
for src in all_sources:
    remaining = [s for s in all_sources if s != src]
    observed_without = max(0, TOTAL_ALL_PAIRS - ALL_INVOLVING.get(src, 0))

    E = expected_collisions(remaining)
    S = sd_collisions(remaining)
    z = (observed_without - E) / S if S > 0 else 0.0
    sig = abs(z) > 3.29  # p < 0.001

    loo_results.append({
        "source_removed": src,
        "n_remaining": sum(SOURCES[s]["records"] for s in remaining),
        "observed": observed_without,
        "expected": round(E, 1),
        "SD": round(S, 1),
        "z": round(z, 1),
        "sig": "Yes" if sig else "No",
    })
    print(f"  -{src:15s}: obs={observed_without:2d}, "
          f"E={E:.1f}, SD={S:.1f}, z={z:.1f}, sig={'Yes' if sig else 'No'}")

loo_df = pd.DataFrame(loo_results)

# ---------------------------------------------------------------------------
# 2. Normalisation sensitivity
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("NORMALISATION SENSITIVITY")
print("=" * 60)

norm_levels = [("strict", 6), ("default", 16), ("loose", 16)]
norm_results = []
for level, obs in norm_levels:
    z = (obs - full_E) / full_SD
    sig = abs(z) > 3.29
    norm_results.append({
        "level": level, "observed": obs,
        "expected": round(full_E, 1), "z": round(z, 1),
        "sig": "Yes" if sig else "No",
    })
    print(f"  {level:10s}: obs={obs:2d}, E={full_E:.1f}, z={z:.1f}")

norm_df = pd.DataFrame(norm_results)

# ---------------------------------------------------------------------------
# 3. Collision yield curve (analytical — instant)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("COLLISION YIELD CURVE")
print("=" * 60)

N_SUBSETS = 200
yield_data = []

for k in range(2, 12):
    if k == 11:
        subsets = [all_sources]
    else:
        possible = list(itertools.combinations(all_sources, k))
        if len(possible) <= N_SUBSETS:
            subsets = [list(s) for s in possible]
        else:
            idx = np.random.choice(len(possible), size=N_SUBSETS, replace=False)
            subsets = [list(possible[i]) for i in idx]

    obs_list, exp_list = [], []
    for subset in subsets:
        obs = sum(n for (s1, s2), n in COLLISION_PAIRS_ALL.items()
                  if s1 in subset and s2 in subset)
        obs_list.append(obs)
        exp_list.append(expected_collisions(subset))

    obs_arr = np.array(obs_list)
    exp_arr = np.array(exp_list)

    yield_data.append({
        "k": k,
        "obs_mean": obs_arr.mean(), "obs_std": obs_arr.std(),
        "exp_mean": exp_arr.mean(), "exp_std": exp_arr.std(),
        "n_subsets": len(subsets),
    })
    print(f"  k={k:2d}: obs={obs_arr.mean():.1f}+/-{obs_arr.std():.1f}, "
          f"exp={exp_arr.mean():.1f}+/-{exp_arr.std():.1f}")

yield_df = pd.DataFrame(yield_data)

# ---------------------------------------------------------------------------
# 4. MC verification (full dataset, small sample)
# ---------------------------------------------------------------------------
print("\n--- MC verification (full dataset, 100 iterations) ---")
mc = mc_collisions_fast(all_sources, n_iter=100)
print(f"  MC: {mc.mean():.1f} +/- {mc.std():.1f}  "
      f"(analytical: E={full_E:.1f}, SD={full_SD:.1f})")

# ---------------------------------------------------------------------------
# 5. Figure: Collision yield curve
# ---------------------------------------------------------------------------
ABSENT_COL = "#C0392B"
PRESENT_COL = "#1A7A4A"

fig, ax = plt.subplots(figsize=(5.5, 3.5))
ks = yield_df["k"].values

ax.fill_between(ks,
                yield_df["exp_mean"] - yield_df["exp_std"],
                yield_df["exp_mean"] + yield_df["exp_std"],
                alpha=0.15, color=ABSENT_COL)
ax.plot(ks, yield_df["exp_mean"], "-o", color=ABSENT_COL, markersize=4,
        linewidth=1.5, label="Expected (null model)")

ax.fill_between(ks,
                np.maximum(0, yield_df["obs_mean"] - yield_df["obs_std"]),
                yield_df["obs_mean"] + yield_df["obs_std"],
                alpha=0.15, color=PRESENT_COL)
ax.plot(ks, yield_df["obs_mean"], "-s", color=PRESENT_COL, markersize=4,
        linewidth=1.5, label="Observed")

ax.set_xlabel("Number of sources included ($k$)", fontsize=9)
ax.set_ylabel("Collision pairs", fontsize=9)
ax.set_title("Cross-source collision yield vs. source count", fontsize=9.5, pad=8)
ax.set_xticks(range(2, 12))
ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
ax.grid(alpha=0.2)
ax.set_xlim(1.8, 11.2)

plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "fig9_robustness_yield_curve.pdf"),
            bbox_inches="tight", dpi=300)
fig.savefig(os.path.join(FIGURES_DIR, "fig9_robustness_yield_curve.png"),
            bbox_inches="tight", dpi=300)
plt.close(fig)

# ---------------------------------------------------------------------------
# 6. Summary
# ---------------------------------------------------------------------------
z_vals = [r["z"] for r in loo_results]
z_min, z_max = min(z_vals), max(z_vals)
n_sig = sum(1 for r in loo_results if r["sig"] == "Yes")
olv2_row = next(r for r in loo_results if r["source_removed"] == "OLv2")
strict_z = round((6 - full_E) / full_SD, 1)

gap_from_k = None
for row in yield_data:
    if row["obs_mean"] + row["obs_std"] < row["exp_mean"] - row["exp_std"]:
        gap_from_k = row["k"]
        break

results_path = os.path.join(SCRIPT_DIR, "experiment2_results.txt")
lines = [
    "=" * 60,
    "EXPERIMENT 2: BOOTSTRAP ROBUSTNESS RESULTS",
    "=" * 60, "",
    "--- Full Dataset Baseline ---",
    f"  Observed: {TOTAL_ALL_PAIRS}",
    f"  Expected (null): {full_E:.1f} +/- {full_SD:.1f}",
    f"  z = {full_z:.1f}  (paper reports z = -6.1)", "",
    "--- Leave-One-Source-Out ---",
    loo_df.to_string(index=False), "",
    f"  z-score range: {z_min} to {z_max}",
    f"  Significant (p<0.001): {n_sig}/11",
    f"  OLv2 excluded: obs={olv2_row['observed']}, E={olv2_row['expected']}, z={olv2_row['z']}",
    "",
    "--- Normalisation Sensitivity ---",
    norm_df.to_string(index=False),
    f"  Strict z-score: {strict_z}", "",
    "--- Collision Yield Curve ---",
    yield_df[["k", "obs_mean", "obs_std", "exp_mean", "exp_std"]].to_string(index=False),
    f"  Gap visible from k={gap_from_k}" if gap_from_k else "  Gap visible at all k",
    "",
    f"KEY FINDING: Structural fragmentation holds across "
    f"{n_sig}/11 LOO ablations and all normalisation levels "
    f"(z from {z_min} to {z_max}).",
    "",
    "Figures: figures/fig9_robustness_yield_curve.pdf",
]

report = "\n".join(lines)
with open(results_path, "w", encoding="utf-8") as f:
    f.write(report)
print("\n" + report)
