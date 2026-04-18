"""
fig3_completeness_vs_delta.py
Generates Figure 3: metadata completeness vs |Δ| scatter plot.

Data hardcoded from Table 6 (tab:attribution_extended) in acl_latex.tex.
Exports:
  - figures/fig_completeness_vs_delta.pdf  (LaTeX inclusion)
  - figures/fig_completeness_vs_delta.png  (300 dpi)

Requirements:
  matplotlib, scipy, numpy, matplotlib.patches
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from scipy.stats import spearmanr
from pathlib import Path

# ---------------------------------------------------------------------------
# Data from Table 6
# ---------------------------------------------------------------------------
records = [
    # (completeness, abs_delta, model_label, benchmark_label, is_tulu)
    (2, 0.23, "Tulu-3-8B", "GPQA",      True),
    (2, 0.19, "Tulu-3-8B", "BBH",       True),
    (2, 0.12, "Tulu-3-8B", "IFEval",    True),
    (2, 0.07, "Tulu-3-8B", "MMLU-Pro",  True),
    (2, 0.07, "Tulu-3-8B", "MuSR",      True),
    (1, 3.90, "SC2-15B",   "MBPP+",     False),
    (1, 3.00, "SC2-15B",   "HumanEval+",False),
    (0, 32.33,"Qwen2.5-72B","GPQA",     False),
]

completeness = np.array([r[0] for r in records])
abs_delta    = np.array([r[1] for r in records])
labels       = [f"{r[2]} / {r[3]}" for r in records]
is_tulu      = np.array([r[4] for r in records])

rho, p_val = spearmanr(completeness, abs_delta)

# ---------------------------------------------------------------------------
# Colorblind-safe palette (seaborn colorblind / tab10 accessible subset)
# Tulu cluster = blue (#0173B2), others = orange (#DE8F05), Qwen = red (#CC3311)
# ---------------------------------------------------------------------------
TULU_COLOR  = "#0173B2"
SC2_COLOR   = "#DE8F05"
QWEN_COLOR  = "#CC3311"

def point_color(r):
    if r[4]:
        return TULU_COLOR
    elif r[2].startswith("Qwen"):
        return QWEN_COLOR
    else:
        return SC2_COLOR

colors = [point_color(r) for r in records]

# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 5.0))

# Plot points
for i, (c, d, lbl, col) in enumerate(zip(completeness, abs_delta, labels, colors)):
    ax.scatter(c, d, color=col, s=80, zorder=5)
    # Offset labels to avoid overlap
    x_off = 0.06
    y_off = d * 0.08 if d > 0.5 else 0.015
    ha = "left"
    ax.text(c + x_off, d + y_off, lbl, fontsize=7, ha=ha, va="bottom",
            color=col)

# y-axis: log10 scale
ax.set_yscale("log")
ax.set_ylim(0.04, 80)
ax.set_xlim(-0.4, 2.8)
ax.set_xticks([0, 1, 2])
ax.set_xlabel("Metadata completeness (fields documented out of 5)", fontsize=10)
ax.set_ylabel(r"$|\Delta|$ (absolute score divergence, pp)", fontsize=10)
ax.set_title("Metadata completeness vs.\ cross-source score divergence", fontsize=11)

# Shaded ellipse around Tulu-3-8B cluster
tulu_x = completeness[is_tulu]
tulu_y = abs_delta[is_tulu]
# Ellipse centred on mean; width/height = 2*std + padding
ell = Ellipse(
    xy=(np.mean(tulu_x), np.mean(np.log10(tulu_y))),  # centre in log-space for drawing
    width=0.55,
    height=0.55,
    angle=0,
    transform=ax.transData,
    linewidth=0,
)
# Draw in data coordinates using the log-scale axes
from matplotlib.patches import FancyBboxPatch
# Use a simple rectangle in log-space
ax.axvspan(1.72, 2.28, ymin=0, ymax=1, alpha=0.0)  # invisible; ellipse below

# Re-draw ellipse in display coords via ax.transData with log y
# Simpler: draw rectangle patch manually
rect = mpatches.FancyBboxPatch(
    (1.72, 0.055), 0.56, 0.22,
    boxstyle="round,pad=0.02",
    linewidth=1.5, edgecolor=TULU_COLOR,
    facecolor=TULU_COLOR, alpha=0.10,
    zorder=2
)
ax.add_patch(rect)
ax.text(1.78, 0.31, "Single model family\n(Tulu-3-8B)",
        fontsize=7, color=TULU_COLOR, va="bottom", style="italic")

# Spearman annotation (top right)
ax.text(0.97, 0.97,
        f"Spearman $\\rho$ = {rho:.3f}\n$p$ = {p_val:.3f}\n(n=8; see §5.5 for\nconfound discussion)",
        transform=ax.transAxes, fontsize=7.5,
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="gray", alpha=0.8))

# ---------------------------------------------------------------------------
# Inset panel (lower right): zoom into 0.05–0.25 |Δ| range
# ---------------------------------------------------------------------------
ax_ins = ax.inset_axes([0.55, 0.04, 0.43, 0.38])  # [left, bottom, width, height]
for i, (c, d, lbl, col) in enumerate(zip(completeness, abs_delta, labels, colors)):
    if 0.05 <= d <= 0.25:
        ax_ins.scatter(c, d, color=col, s=60, zorder=5)
        ax_ins.text(c + 0.04, d + 0.005, lbl.split(" / ")[1],
                    fontsize=6, color=col, ha="left", va="bottom")

ax_ins.set_xlim(1.5, 2.5)
ax_ins.set_ylim(0.05, 0.27)
ax_ins.set_xticks([2])
ax_ins.set_yticks([0.07, 0.12, 0.19, 0.23])
ax_ins.yaxis.set_tick_params(labelsize=6)
ax_ins.xaxis.set_tick_params(labelsize=6)
ax_ins.set_xlabel("Compl.", fontsize=6)
ax_ins.set_ylabel(r"$|\Delta|$", fontsize=6)
ax_ins.set_title("Tulu-3-8B\nzoom", fontsize=6.5, pad=3)
ax_ins.tick_params(axis="both", which="major", length=2)

# Rectangle border
for spine in ax_ins.spines.values():
    spine.set_edgecolor(TULU_COLOR)
    spine.set_linewidth(1.2)

# Legend
legend_elements = [
    mpatches.Patch(color=TULU_COLOR, label="Tulu-3-8B (single model family)"),
    mpatches.Patch(color=SC2_COLOR,  label="SC2-15B"),
    mpatches.Patch(color=QWEN_COLOR, label="Qwen2.5-72B"),
]
ax.legend(handles=legend_elements, fontsize=7.5, loc="upper left",
          framealpha=0.85)

plt.tight_layout()

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
out_dir = Path(__file__).parent
pdf_path = out_dir / "fig_completeness_vs_delta.pdf"
png_path = out_dir / "fig_completeness_vs_delta.png"

plt.savefig(pdf_path, bbox_inches="tight")
plt.savefig(png_path, dpi=300, bbox_inches="tight")
print(f"Saved:\n  {pdf_path}\n  {png_path}")
plt.close()
