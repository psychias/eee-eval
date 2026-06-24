"""
Metadata-coverage horizontal bar chart — §4.3 (replaces tab:coverage in body).

Output: LLM_Evaluation_Report/figures/coverage_bars.pdf

Visual takeaway: harness, prompt_template, and temperature have zero valid
coverage in the 10-source leaderboard subset (bars flush at 0).
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, "..")
OUT_PATH   = os.path.join(OUT_DIR, "coverage_bars.pdf")
# Single source of truth: regenerate via `python reproduce_paper.py`.
PAPER_NUMBERS = os.path.join(SCRIPT_DIR, "..", "..", "..",
                             "analysis_output", "paper_numbers.json")

BLUE_HDR   = "#4A90D9"
BLUE_LIGHT = "#A8C8EC"
ORNG_HDR   = "#E89A4F"
DGREY      = "#444444"
MGREY      = "#CCCCCC"
WHITE      = "#FFFFFF"
BLACK      = "#111111"
SANS       = "DejaVu Sans"
MONO       = "DejaVu Sans Mono"

# (field, full_dataset_%, leaderboard_subset_%) — ordered bottom-to-top.
# Loaded from analysis_output/paper_numbers.json so the figure always matches
# the paper's coverage table (regenerate both via `python reproduce_paper.py`).
with open(PAPER_NUMBERS, encoding="utf-8") as _fh:
    _PN = json.load(_fh)
_COV = _PN["coverage"]
_DS = _PN["dataset"]
FIELDS = [
    (name, _COV["full_dataset"][name], _COV["leaderboard_subset"][name])
    for name in ("temperature", "prompt_template", "chain_of_thought", "harness", "shots")
]
TOTAL_RECORDS = _DS["total_records"]
SUBSET_RECORDS = _DS["leaderboard_subtotal"]["records"]
N_SOURCES = len(_DS["per_source"])

fig, ax = plt.subplots(figsize=(6.5, 3.4), facecolor=WHITE)

y_pos      = list(range(len(FIELDS)))
bar_height = 0.36
gap        = 0.02

full_vals   = [v for _, v, _ in FIELDS]
subset_vals = [v for _, _, v in FIELDS]
labels      = [f for f, _, _ in FIELDS]

ax.barh([y + (bar_height + gap) / 2 for y in y_pos], full_vals,
        height=bar_height, color=BLUE_LIGHT, edgecolor=BLUE_HDR, linewidth=0.8,
        zorder=3)
ax.barh([y - (bar_height + gap) / 2 for y in y_pos], subset_vals,
        height=bar_height, color=ORNG_HDR, edgecolor=ORNG_HDR, linewidth=0.8,
        zorder=3)

# Value labels
for y, v in zip(y_pos, full_vals):
    ax.text(v + 1.2, y + (bar_height + gap) / 2, f"{v:.1f}%",
            va="center", ha="left",
            fontsize=7.2, fontfamily=SANS, color=DGREY, zorder=4)
for y, v in zip(y_pos, subset_vals):
    label = f"{v:.1f}%"
    if v == 0:
        ax.text(1.2, y - (bar_height + gap) / 2, label,
                va="center", ha="left",
                fontsize=7.5, fontfamily=SANS, color=ORNG_HDR,
                fontweight="bold", zorder=4)
    else:
        ax.text(v + 1.2, y - (bar_height + gap) / 2, label,
                va="center", ha="left",
                fontsize=7.2, fontfamily=SANS, color=DGREY, zorder=4)

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=8.5, fontfamily=MONO, color=BLACK)
ax.set_xlim(0, 110)
ax.set_xlabel('Coverage (% of records with non-null, non-"unknown" value)',
              fontsize=8, fontfamily=SANS, color=DGREY)
ax.tick_params(axis="x", labelsize=7.5, colors=DGREY)
ax.tick_params(axis="y", length=0)

for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color(MGREY)
ax.spines["left"].set_color(MGREY)
ax.grid(axis="x", linestyle=":", color=MGREY, linewidth=0.5, zorder=1)
ax.set_axisbelow(True)

legend_full = mpatches.Patch(facecolor=BLUE_LIGHT, edgecolor=BLUE_HDR,
                             label=f"Full dataset ({TOTAL_RECORDS:,} records, {N_SOURCES} sources)")
legend_sub  = mpatches.Patch(facecolor=ORNG_HDR, edgecolor=ORNG_HDR,
                             label=f"10-source leaderboard subset ({SUBSET_RECORDS:,} records)")
ax.legend(handles=[legend_full, legend_sub],
          loc="lower right", fontsize=7.5, frameon=False,
          handlelength=1.2, handleheight=0.9, borderpad=0.4)

plt.tight_layout()
plt.savefig(OUT_PATH, format="pdf", bbox_inches="tight", dpi=300)
plt.savefig(OUT_PATH.replace(".pdf", ".png"), format="png", bbox_inches="tight", dpi=200)
print(f"Saved: {OUT_PATH}")
