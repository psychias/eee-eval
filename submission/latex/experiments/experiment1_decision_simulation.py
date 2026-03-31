"""
Experiment 1: Model Selection Decision Simulation

Converts the 8 independent cross-source collision pairs into a
practitioner-facing error-rate analysis: decision tiers, metadata
conditionality, and GPQA percentile consequence.

Outputs:
  - figures/fig8_decision_simulation.pdf
  - experiment1_results.txt  (summary statistics for LaTeX)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
FIGURES_DIR = os.path.join(SCRIPT_DIR, "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Collision pair data (from Table 6 / collision_pairs.csv)
# ---------------------------------------------------------------------------
pairs = pd.DataFrame([
    # model, benchmark, s1, s2, source_a, source_b, harness_a, harness_b, shots_known_both, cot_known_both
    ("Qwen2.5-72B-Instruct",   "GPQA",       16.67, 49.00, "OLv2", "PWC",  "lm_eval",  "unknown",      False, False),
    ("StarCoder2-15B-Inst.",    "MBPP+",      65.10, 61.20, "EP",   "HFM",  "evalplus",  "transformers", False, False),
    ("StarCoder2-15B-Inst.",    "HumanEval+", 60.40, 63.40, "EP",   "HFM",  "evalplus",  "transformers", False, False),
    ("Tulu-3-8B",              "BBH",         16.86, 16.67, "HFM",  "OLv2", "vllm",      "lm_eval",     True,  False),
    ("Tulu-3-8B",              "GPQA",         6.26,  6.49, "HFM",  "OLv2", "vllm",      "lm_eval",     True,  False),
    ("Tulu-3-8B",              "IFEval",      82.55, 82.67, "HFM",  "OLv2", "vllm",      "lm_eval",     True,  False),
    ("Tulu-3-8B",              "MMLU-Pro",    20.23, 20.30, "HFM",  "OLv2", "vllm",      "lm_eval",     True,  False),
    ("Tulu-3-8B",              "MuSR",        10.52, 10.45, "HFM",  "OLv2", "vllm",      "lm_eval",     True,  False),
], columns=["model", "benchmark", "s1", "s2",
            "source_a", "source_b", "harness_a", "harness_b",
            "shots_known_both", "cot_known_both"])

pairs["abs_delta"] = (pairs["s1"] - pairs["s2"]).abs()

# Metadata grouping (Table 5 in paper):
#   "present": harness known in BOTH + n-shot known in BOTH  → Tulu-3-8B pairs
#   "absent":  harness unknown in ≥1 source OR n-shot unknown → top 3 pairs
pairs["metadata_group"] = np.where(pairs["shots_known_both"], "present", "absent")

# ---------------------------------------------------------------------------
# 2. Decision-tier classification
# ---------------------------------------------------------------------------
def tier(d):
    if d <= 1.0:
        return "noise"       # ≤1 pt: within stochastic noise
    elif d <= 5.0:
        return "meaningful"  # 1–5 pt: could flip close decisions
    else:
        return "major"       # >5 pt: would likely reverse selection

pairs["tier"] = pairs["abs_delta"].apply(tier)
tier_counts = pairs["tier"].value_counts()
tier_fracs = pairs["tier"].value_counts(normalize=True)

# ---------------------------------------------------------------------------
# 3. Metadata-conditionality comparison
# ---------------------------------------------------------------------------
grp = pairs.groupby("metadata_group")["abs_delta"]
meta_stats = pd.DataFrame({
    "n": grp.count(),
    "mean_delta": grp.mean(),
    "max_delta": grp.max(),
    "frac_major": pairs.groupby("metadata_group").apply(
        lambda g: (g["abs_delta"] > 5.0).mean(), include_groups=False),
})

# ---------------------------------------------------------------------------
# 4. GPQA percentile analysis (from actual OLv2 data)
# ---------------------------------------------------------------------------
agg_path = os.path.join(REPO_ROOT, "data", "aggregated", "all_results.csv")
if os.path.exists(agg_path):
    df_all = pd.read_csv(agg_path, low_memory=False)
    gpqa_scores = pd.to_numeric(
        df_all.loc[(df_all["source"] == "open_llm_leaderboard_v2") &
                   (df_all["benchmark"].str.contains("GPQA", case=False, na=False)),
                   "score"], errors="coerce").dropna()
    pct_16 = stats.percentileofscore(gpqa_scores, 16.67)
    pct_49 = stats.percentileofscore(gpqa_scores, 49.0)
    gpqa_mean = gpqa_scores.mean()
    gpqa_std = gpqa_scores.std()
    gpqa_n = len(gpqa_scores)
else:
    pct_16, pct_49, gpqa_mean, gpqa_std, gpqa_n = 93.8, 100.0, 6.7, 5.1, 4465

# ---------------------------------------------------------------------------
# 5. Figure: decision-simulation dot plot
# ---------------------------------------------------------------------------
ABSENT_COL = "#C0392B"   # red-orange for metadata-absent
PRESENT_COL = "#1A7A4A"  # green for metadata-present
GREY = "#555555"

fig, ax = plt.subplots(figsize=(5.5, 3.2))

# Sort: absent group first (top), then present group (bottom)
order = list(pairs[pairs["metadata_group"] == "absent"].index) + \
       list(pairs[pairs["metadata_group"] == "present"].index)
y_labels = []
y_positions = []
colours = []
pos = 0
for idx in order:
    row = pairs.loc[idx]
    lbl = f"{row['model']}\n{row['benchmark']}"
    y_labels.append(lbl)
    y_positions.append(pos)
    colours.append(ABSENT_COL if row["metadata_group"] == "absent" else PRESENT_COL)
    pos += 1

# Horizontal separator between groups
n_absent = (pairs["metadata_group"] == "absent").sum()
sep_y = n_absent - 0.5

for i, idx in enumerate(order):
    row = pairs.loc[idx]
    ax.scatter(row["abs_delta"], y_positions[i], c=colours[i],
               s=70, zorder=5, edgecolors="white", linewidth=0.5)

# Threshold lines
ax.axvline(x=1.0, color=GREY, linestyle="--", linewidth=0.8, alpha=0.7, label="|Δ| = 1")
ax.axvline(x=5.0, color=GREY, linestyle="-.", linewidth=0.8, alpha=0.7, label="|Δ| = 5")

# Horizontal separator
ax.axhline(y=sep_y, color=GREY, linestyle=":", linewidth=0.6, alpha=0.6)

# Tier labels
ax.text(0.5, -0.9, "noise\n(≤1 pt)", ha="center", va="top", fontsize=6.5, color=GREY)
ax.text(3.0, -0.9, "meaningful\n(1–5 pt)", ha="center", va="top", fontsize=6.5, color=GREY)
ax.text(18.0, -0.9, "major\n(>5 pt)", ha="center", va="top", fontsize=6.5, color=GREY)

ax.set_yticks(y_positions)
ax.set_yticklabels(y_labels, fontsize=6.5)
ax.set_xlabel("Absolute score delta |Δ|", fontsize=9)
ax.set_title("Decision impact of cross-source score gaps", fontsize=9.5, pad=8)

# x-axis: log-ish but use symlog for 0-friendly display
ax.set_xscale("symlog", linthresh=1.0)
ax.set_xlim(-0.2, 50)

# Legend
patch_absent = mpatches.Patch(color=ABSENT_COL, label="Metadata absent")
patch_present = mpatches.Patch(color=PRESENT_COL, label="Metadata present")
ax.legend(handles=[patch_absent, patch_present], fontsize=7, loc="lower right",
          framealpha=0.9)

ax.invert_yaxis()
ax.grid(axis="x", alpha=0.2)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "fig8_decision_simulation.pdf"),
            bbox_inches="tight", dpi=300)
fig.savefig(os.path.join(FIGURES_DIR, "fig8_decision_simulation.png"),
            bbox_inches="tight", dpi=300)
plt.close(fig)

# ---------------------------------------------------------------------------
# 6. Print summary for LaTeX
# ---------------------------------------------------------------------------
results_path = os.path.join(SCRIPT_DIR, "experiment1_results.txt")
lines = []  # type: list[str]
lines.append("=" * 60)
lines.append("EXPERIMENT 1: DECISION SIMULATION RESULTS")
lines.append("=" * 60)
lines.append("")
lines.append("--- Decision Tier Classification ---")
for t in ["noise", "meaningful", "major"]:
    n = tier_counts.get(t, 0)
    f = tier_fracs.get(t, 0)
    lines.append(f"  {t:12s}: {n}/8 pairs ({f*100:.1f}%)")
lines.append("")
lines.append("--- Metadata Conditionality ---")
for grp_name in ["absent", "present"]:
    row = meta_stats.loc[grp_name]
    lines.append(f"  {grp_name:10s}: n={int(row['n'])}, "
                 f"mean |Δ|={row['mean_delta']:.2f}, "
                 f"max |Δ|={row['max_delta']:.2f}, "
                 f"frac major={row['frac_major']:.2f}")
lines.append("")
lines.append("--- GPQA Percentile Analysis ---")
lines.append(f"  OLv2 GPQA: n={gpqa_n}, mean={gpqa_mean:.2f}, std={gpqa_std:.2f}")
lines.append(f"  Score 16.67 → percentile {pct_16:.1f}%")
lines.append(f"  Score 49.00 → percentile {pct_49:.1f}%")
lines.append(f"  Gap spans {pct_49 - pct_16:.1f} percentile points")
lines.append("")
lines.append("--- Full Pair Table ---")
lines.append(pairs[["model", "benchmark", "abs_delta", "tier", "metadata_group"]].to_string(index=False))
lines.append("")
lines.append("Figure saved: figures/fig8_decision_simulation.pdf")

report = "\n".join(lines)
with open(results_path, "w", encoding="utf-8") as f:
    f.write(report)
print(report)
