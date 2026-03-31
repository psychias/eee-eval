"""
Experiment 3: Leave-Two-Out Analysis

Tests whether the structural fragmentation finding holds when OLv2
(the dominant source at 91% of records) is excluded *together with*
each of the remaining 10 sources, one at a time.

This directly tests whether the fragmentation signal is an artefact
of OLv2's dominance or a property of the broader ecosystem.

Uses the same analytical null model as experiment 2:
  E[collisions] = sum_{i<j} R_i * R_j / U
  where U = |models| * |benchmarks| = 5672 * 180 = 1,020,960
  SD[collisions] ~ sqrt(E)  (Poisson regime)

Outputs:
  - figures/fig10_leave_two_out.pdf / .png
  - experiment3_results.txt
"""

import os
import math
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

np.random.seed(42)

# ---------------------------------------------------------------------------
# Source metadata (Table 2 of the paper)
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

# Known collision pairs grouped by the two sources involved
# 8 independent + 8 likely copied = 16 total
COLLISION_PAIRS_BY_SOURCES = {
    ("OLv2",        "PWC"):         1,  # Qwen2.5 GPQA
    ("EvalPlus",    "HFModelCard"): 5,  # SC2-15B-Inst x2 + SC2-base x3
    ("HFModelCard", "OLv2"):       10,  # Tulu-3-8B x5 + Tulu-3-70B-SFT x5
}
TOTAL_ALL_PAIRS = 16

# Track how many collision pairs each individual source is involved in
ALL_INVOLVING = defaultdict(int)
for (s1, s2), n in COLLISION_PAIRS_BY_SOURCES.items():
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
    """SD ~ sqrt(E) in the Poisson regime."""
    return math.sqrt(max(expected_collisions(source_subset), 1e-10))


def z_score(observed, source_subset):
    E = expected_collisions(source_subset)
    S = sd_collisions(source_subset)
    return (observed - E) / S if S > 0 else 0.0


def p_value_from_z(z_val):
    """One-sided p-value for testing obs < expected (left tail)."""
    return stats.norm.cdf(z_val)


# ---------------------------------------------------------------------------
# Non-OLv2 baseline
# ---------------------------------------------------------------------------
all_sources = list(SOURCES.keys())
non_olv2 = [s for s in all_sources if s != "OLv2"]

# Count collision pairs that don't involve OLv2
non_olv2_observed = sum(
    n for (s1, s2), n in COLLISION_PAIRS_BY_SOURCES.items()
    if s1 != "OLv2" and s2 != "OLv2"
)
# That's only EvalPlus-HFModelCard = 5

non_olv2_E = expected_collisions(non_olv2)
non_olv2_SD = sd_collisions(non_olv2)
non_olv2_z = z_score(non_olv2_observed, non_olv2)
non_olv2_records = sum(SOURCES[s]["records"] for s in non_olv2)

print("=" * 70)
print("EXPERIMENT 3: LEAVE-TWO-OUT ANALYSIS")
print("OLv2 + one other source excluded, iterated over all 10 non-OLv2 sources")
print("=" * 70)

print(f"\n--- Non-OLv2 Baseline (10 sources) ---")
print(f"  Observed: {non_olv2_observed}")
print(f"  Expected: {non_olv2_E:.2f} +/- {non_olv2_SD:.2f}")
print(f"  z = {non_olv2_z:.2f}")
print(f"  Records: {non_olv2_records}")
print(f"  Ratio obs/exp: {non_olv2_observed / non_olv2_E:.2f}")

# ---------------------------------------------------------------------------
# Leave-two-out: remove OLv2 + one other, iterate
# ---------------------------------------------------------------------------
print(f"\n--- Leave-Two-Out Results ---")

l2o_results = []
for exclude_src in non_olv2:
    remaining = [s for s in all_sources if s != "OLv2" and s != exclude_src]

    # Count observed collisions among remaining sources
    observed = 0
    for (s1, s2), n in COLLISION_PAIRS_BY_SOURCES.items():
        if s1 in remaining and s2 in remaining:
            observed += n

    E = expected_collisions(remaining)
    S = sd_collisions(remaining)
    z_val = (observed - E) / S if S > 0 else 0.0
    p_left = p_value_from_z(z_val)  # P(Z < z) for testing fragmentation
    p_right = 1 - p_left  # P(Z > z) for testing observed > expected
    n_records = sum(SOURCES[s]["records"] for s in remaining)

    direction = "below" if observed < E else "above"
    sig_label = "n.s."
    if direction == "below" and p_left < 0.001:
        sig_label = "p < 0.001"
    elif direction == "below" and p_left < 0.01:
        sig_label = "p < 0.01"
    elif direction == "below" and p_left < 0.05:
        sig_label = "p < 0.05"
    elif direction == "above" and p_right < 0.05:
        sig_label = "p < 0.05 (above)"

    l2o_results.append({
        "excluded_with_OLv2": exclude_src,
        "n_remaining_sources": len(remaining),
        "n_remaining_records": n_records,
        "observed": observed,
        "expected": round(E, 2),
        "SD": round(S, 2),
        "z": round(z_val, 2),
        "p_left": round(p_left, 4),
        "direction": direction,
        "sig": sig_label,
    })
    print(f"  -OLv2 -{exclude_src:15s}: obs={observed:2d}, "
          f"E={E:.2f}, SD={S:.2f}, z={z_val:.2f}, dir={direction}, sig={sig_label}")

l2o_df = pd.DataFrame(l2o_results)

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
n_below = sum(1 for r in l2o_results if r["direction"] == "below")
n_sig_frag = sum(1 for r in l2o_results
                 if r["direction"] == "below" and r["p_left"] < 0.05)
z_range = (min(r["z"] for r in l2o_results),
           max(r["z"] for r in l2o_results))

# Key finding: is fragmentation directionally consistent?
# When HFModelCard is excluded, all 5 EvalPlus-HFModelCard collisions vanish
# so observed drops to 0, which should be below expected
hfm_row = next(r for r in l2o_results if r["excluded_with_OLv2"] == "HFModelCard")
ep_row = next(r for r in l2o_results if r["excluded_with_OLv2"] == "EvalPlus")

print(f"\n--- Summary ---")
print(f"  {n_below}/10 ablations show observed < expected (fragmentation direction)")
print(f"  {n_sig_frag}/10 significant at p < 0.05 (left tail)")
print(f"  z-score range: {z_range[0]:.2f} to {z_range[1]:.2f}")
print(f"  When HFModelCard excluded: obs={hfm_row['observed']}, E={hfm_row['expected']}")
print(f"  When EvalPlus excluded: obs={ep_row['observed']}, E={ep_row['expected']}")

# ---------------------------------------------------------------------------
# Figure: Leave-two-out bar chart
# ---------------------------------------------------------------------------
BELOW_COL = "#C0392B"  # Red for fragmentation direction
ABOVE_COL = "#2E86C1"  # Blue for above expected
EXPECTED_COL = "#666666"

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5),
                          gridspec_kw={"width_ratios": [3, 1.2]})

# Panel (a): Leave-two-out bars
ax = axes[0]
sources_sorted = l2o_df.sort_values("z")
y_pos = range(len(sources_sorted))
colors = [BELOW_COL if r["direction"] == "below" else ABOVE_COL
          for _, r in sources_sorted.iterrows()]

bars = ax.barh(list(y_pos), sources_sorted["z"].values,
               color=colors, edgecolor="white", linewidth=0.5, alpha=0.85)

# Significance markers
for i, (_, row) in enumerate(sources_sorted.iterrows()):
    marker = ""
    if "p < 0.001" in row["sig"]:
        marker = "***"
    elif "p < 0.01" in row["sig"]:
        marker = "**"
    elif "p < 0.05" in row["sig"]:
        marker = "*"
    if marker:
        x_pos = row["z"] + (0.08 if row["z"] >= 0 else -0.08)
        ax.text(x_pos, i, marker, va="center",
                ha="left" if row["z"] >= 0 else "right",
                fontsize=10, fontweight="bold", color=colors[i])

ax.set_yticks(list(y_pos))
ax.set_yticklabels([f"$-$OLv2 $-${r['excluded_with_OLv2']}"
                     for _, r in sources_sorted.iterrows()], fontsize=8)
ax.set_xlabel("$z$-score (negative = fewer collisions than expected)", fontsize=9)
ax.set_title("(a) Leave-two-out: OLv2 + one source excluded", fontsize=10,
             fontweight="bold")
ax.axvline(0, color="black", linewidth=0.8, linestyle="-")
ax.axvline(-1.96, color=BELOW_COL, linewidth=0.6, linestyle="--", alpha=0.5)
ax.axvline(1.96, color=ABOVE_COL, linewidth=0.6, linestyle="--", alpha=0.5)
ax.grid(axis="x", alpha=0.2)

# Panel (b): Non-OLv2 baseline comparison
ax2 = axes[1]
categories = ["Full\n(11 sources)", "Non-OLv2\n(10 sources)"]

# Full dataset values
full_E = expected_collisions(all_sources)
full_obs = TOTAL_ALL_PAIRS
full_z = z_score(full_obs, all_sources)

x = [0, 1]
observed_vals = [full_obs, non_olv2_observed]
expected_vals = [full_E, non_olv2_E]

bar_width = 0.3
ax2.bar([xi - bar_width/2 for xi in x], observed_vals, bar_width,
        color=BELOW_COL, alpha=0.8, label="Observed")
ax2.bar([xi + bar_width/2 for xi in x], expected_vals, bar_width,
        color=EXPECTED_COL, alpha=0.5, label="Expected (null)")

# Error bars for expected
ax2.errorbar([xi + bar_width/2 for xi in x], expected_vals,
             yerr=[sd_collisions(all_sources), non_olv2_SD],
             fmt="none", ecolor="black", capsize=3, linewidth=1)

for i, (obs, exp) in enumerate(zip(observed_vals, expected_vals)):
    z_val = full_z if i == 0 else non_olv2_z
    ax2.text(i, max(obs, exp) + 5, f"$z$={z_val:.1f}",
             ha="center", fontsize=8, fontweight="bold")

ax2.set_xticks(x)
ax2.set_xticklabels(categories, fontsize=8)
ax2.set_ylabel("Collision pairs", fontsize=9)
ax2.set_title("(b) Full vs. non-OLv2", fontsize=10, fontweight="bold")
ax2.legend(fontsize=7, loc="upper right")
ax2.grid(axis="y", alpha=0.2)

fig.suptitle("Leave-Two-Out Robustness: Is Fragmentation OLv2-Dependent?",
             fontsize=11, fontweight="bold", y=1.02)
fig.tight_layout()

fig.savefig(os.path.join(FIGURES_DIR, "fig10_leave_two_out.pdf"),
            bbox_inches="tight", dpi=300)
fig.savefig(os.path.join(FIGURES_DIR, "fig10_leave_two_out.png"),
            bbox_inches="tight", dpi=300)
plt.close(fig)
print(f"\nFigure saved: figures/fig10_leave_two_out.pdf/.png")

# ---------------------------------------------------------------------------
# Write results file
# ---------------------------------------------------------------------------
results_path = os.path.join(SCRIPT_DIR, "experiment3_results.txt")
lines = [
    "=" * 70,
    "EXPERIMENT 3: LEAVE-TWO-OUT ANALYSIS",
    "OLv2 + one other source excluded, iterated over all 10 non-OLv2 sources",
    "=" * 70,
    "",
    "--- Leave-Two-Out Results ---",
    l2o_df.to_string(index=False),
    "",
    f"--- Non-OLv2 Baseline (10 sources) ---",
    f"  Observed: {non_olv2_observed}",
    f"  Expected: {non_olv2_E:.2f} +/- {non_olv2_SD:.2f}",
    f"  z = {non_olv2_z:.2f}",
    f"  Records: {non_olv2_records}",
    f"  Ratio obs/exp = {non_olv2_observed / non_olv2_E:.2f}",
    "",
    f"--- Summary ---",
    f"  {n_below}/10 ablations show observed < expected",
    f"  Significant at p < 0.05 (fragmentation direction): {n_sig_frag}/10",
    f"  z-score range: {z_range[0]:.2f} to {z_range[1]:.2f}",
    "",
    "KEY FINDING:",
    "When OLv2 is excluded, the non-OLv2 ecosystem shows 5 observed",
    f"collisions vs. {non_olv2_E:.2f} expected (z = {non_olv2_z:.2f}, n.s.).",
    "The ratio obs/exp = 1.83 indicates the non-OLv2 ecosystem is NOT",
    "fragmented in the same direction as the full dataset — observed",
    "collisions actually exceed expectations, likely because the remaining",
    "sources (especially HFModelCard with 156 benchmarks) create more",
    "overlap than expected by the null model.",
    "",
    "The fragmentation finding (H2) is therefore CONDITIONAL on OLv2's",
    "presence: OLv2 evaluates 4,465 models on only 6 benchmarks via a",
    "single harness, creating a massive, isolated evaluation silo.",
    "The statistical significance of z = -6.1 is driven by the contrast",
    "between OLv2's large record count and its near-zero model overlap",
    "with other sources.",
    "",
    "This does not invalidate H2 — OLv2 IS the dominant source and its",
    "isolation IS real. But the abstract and conclusions should frame",
    "fragmentation as a property of an OLv2-dominated ecosystem, not",
    "a universal property of all evaluation sources.",
    "",
    "Figures: figures/fig10_leave_two_out.pdf",
]

report = "\n".join(lines)
with open(results_path, "w", encoding="utf-8") as f:
    f.write(report)
print("\n" + report)
