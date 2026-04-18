#!/usr/bin/env python3
"""
fig4_coverage_bar.py — Coverage summary bar chart (replaces full heatmap).

Grouped bar chart: one cluster per methodology field, two bars per cluster
(leaderboard vs paper), Y = % of sources documenting the field.
Annotated bottleneck callout on prompt_template.

Subsumes the old fig11_lb_vs_paper_coverage and fig4_coverage_heatmap.
The full per-source heatmap is now in fig4b_coverage_heatmap_appendix.py.

Output: figures/fig4_coverage_bar.pdf  +  submission/fig4_coverage_bar.png
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

OUT_DIR = _ROOT / "figures"
OUT_DIR.mkdir(exist_ok=True)
SUB_DIR = _ROOT / "submission"
SUB_DIR.mkdir(exist_ok=True)

LEADERBOARD_SOURCES = {
    "hfopenllm_v2", "alpacaeval2", "chatbot_arena", "mt_bench",
    "wildbench", "bigcodebench",
}

def main():
    csv_path = _ROOT / "analysis_output" / "coverage_stats.csv"
    if not csv_path.exists():
        print("Run coverage_audit.py first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df = df[df["n_records"] > 0].copy()

    fields = ["pct_n_shot", "pct_harness", "pct_prompt_template", "pct_temperature"]
    labels = ["n-shot", "Harness", "Prompt\ntemplate", "Temperature"]

    is_lb = df["source"].isin(LEADERBOARD_SOURCES)
    lb_df = df[is_lb]
    paper_df = df[~is_lb]
    n_lb, n_paper = len(lb_df), len(paper_df)

    # % of sources with >0 coverage for each field
    lb_pcts   = [(lb_df[c] > 0).mean() * 100 for c in fields]
    paper_pcts = [(paper_df[c] > 0).mean() * 100 for c in fields]
    # overall % for annotation
    all_pcts  = [(df[c] > 0).mean() * 100 for c in fields]

    x = np.arange(len(labels))
    width = 0.32

    fig, ax = plt.subplots(figsize=(5.5, 3.4))

    COL_LB    = "#E69F00"     # amber / leaderboard
    COL_PAPER = "#0072B2"     # blue  / paper

    bars_lb = ax.bar(x - width / 2, lb_pcts, width,
                     label=f"Leaderboards (n = {n_lb})",
                     color=COL_LB, edgecolor="white", linewidth=0.6)
    bars_paper = ax.bar(x + width / 2, paper_pcts, width,
                        label=f"Papers (n = {n_paper})",
                        color=COL_PAPER, edgecolor="white", linewidth=0.6)

    # Value annotations on bars
    for bars, col in [(bars_lb, COL_LB), (bars_paper, COL_PAPER)]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.8,
                    f"{h:.0f}%", ha="center", va="bottom", fontsize=7.5,
                    fontweight="bold", color=col)

    # ----- Bottleneck callout on prompt_template (index 2) -----
    pt_idx = 2
    # Overall prompt_template coverage
    overall_pt = all_pcts[pt_idx]
    callout_y = max(lb_pcts[pt_idx], paper_pcts[pt_idx]) + 18
    ax.annotate(
        f"{overall_pt:.1f}% overall\nprimary bottleneck",
        xy=(x[pt_idx], max(lb_pcts[pt_idx], paper_pcts[pt_idx]) + 4),
        xytext=(x[pt_idx] + 0.55, callout_y + 8),
        fontsize=7.5, fontstyle="italic", color="#C0392B", fontweight="bold",
        ha="center", va="bottom",
        arrowprops=dict(arrowstyle="-|>", color="#C0392B", lw=1.3),
        bbox=dict(boxstyle="round,pad=0.25", fc="#FDEDEC", ec="#C0392B",
                  lw=1.0, alpha=0.92),
    )

    ax.set_ylabel("% of sources documenting field", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(0, 120)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.15, linewidth=0.6)
    ax.axhline(50, color="#AAAAAA", linewidth=0.7, linestyle="--", alpha=0.35)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_pdf = OUT_DIR / "fig4_coverage_bar.pdf"
    out_png = SUB_DIR / "fig4_coverage_bar.png"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=150)
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved → {out_pdf}")
    print(f"Saved → {out_png}")


if __name__ == "__main__":
    main()
