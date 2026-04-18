#!/usr/bin/env python3
"""
fig11_lb_vs_paper_coverage.py — Leaderboard vs Paper coverage grouped bar chart.

X-axis = 4 methodology fields (n_shot, harness, prompt_template, temperature).
Y-axis = % sources with non-zero coverage.
Two bar groups: leaderboards (n=6) vs papers (n=52).

Replaces Table 4 (leaderboard recommendations) with a visual that
directly shows the documentation gap between source types.

Output: figures/fig11_lb_vs_paper_coverage.pdf + submission/fig11_lb_vs_paper_coverage.png
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

    cols = ["pct_n_shot", "pct_harness", "pct_prompt_template", "pct_temperature"]
    labels = ["n-shot", "Harness", "Prompt\ntemplate", "Temperature"]

    is_lb = df["source"].isin(LEADERBOARD_SOURCES)
    lb_df = df[is_lb]
    paper_df = df[~is_lb]

    # Compute % sources with >0 coverage for each field
    lb_pcts = []
    paper_pcts = []
    for col in cols:
        lb_pcts.append((lb_df[col] > 0).mean() * 100)
        paper_pcts.append((paper_df[col] > 0).mean() * 100)

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))

    bars_lb = ax.bar(x - width/2, lb_pcts, width,
                     label=f"Leaderboards (n={len(lb_df)})",
                     color="#E69F00", edgecolor="white", linewidth=0.5, alpha=0.85)
    bars_paper = ax.bar(x + width/2, paper_pcts, width,
                        label=f"Papers (n={len(paper_df)})",
                        color="#0072B2", edgecolor="white", linewidth=0.5, alpha=0.85)

    # Annotations on bars
    for bar in bars_lb:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.5,
                f"{h:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold",
                color="#E69F00")
    for bar in bars_paper:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.5,
                f"{h:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold",
                color="#0072B2")

    ax.set_ylabel("% Sources Documenting Field", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 115)
    ax.set_title("Methodology Documentation: Leaderboards vs Papers",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.2)

    # Add a horizontal reference line at 50%
    ax.axhline(50, color="#999999", linewidth=0.8, linestyle="--", alpha=0.4)

    fig.tight_layout()
    out_pdf = OUT_DIR / "fig11_lb_vs_paper_coverage.pdf"
    out_png = SUB_DIR / "fig11_lb_vs_paper_coverage.png"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=150)
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved → {out_pdf}")
    print(f"Saved → {out_png}")


if __name__ == "__main__":
    main()
