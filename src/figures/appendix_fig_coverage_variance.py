#!/usr/bin/env python3
"""
fig9_coverage_variance_scatter.py — Coverage vs. Score Variance scatter.

x = % collision pairs with documented prompt_template for each benchmark
y = median |Δ| for that benchmark

Shows the relationship between documentation quality and observed variance.
Each point is one benchmark; labeled and sized by n.

Output: figures/fig9_coverage_variance.pdf + submission/fig9_coverage_variance.png
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


def main():
    cp_path = _ROOT / "analysis_output" / "collision_pairs.csv"
    if not cp_path.exists():
        print("Run collision_detection.py first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(cp_path)
    benchmarks = sorted(df["benchmark"].unique())

    rows = []
    for bench in benchmarks:
        bdf = df[df["benchmark"] == bench]
        n = len(bdf)
        if n < 2:
            continue
        med_abs_delta = bdf["delta"].abs().median()

        # Count pairs where at least one side has documented (non-standard) prompt_template
        has_pt = 0
        for _, r in bdf.iterrows():
            pt_a = str(r.get("prompt_template_a", "standard"))
            pt_b = str(r.get("prompt_template_b", "standard"))
            if (pt_a not in ("standard", "", "nan", "None") or
                pt_b not in ("standard", "", "nan", "None")):
                has_pt += 1

        pt_pct = has_pt / n * 100

        rows.append({
            "benchmark": bench,
            "n_pairs": n,
            "median_abs_delta": med_abs_delta,
            "pct_documented_pt": pt_pct,
        })

    data = pd.DataFrame(rows)
    if data.empty:
        print("No data to plot.", file=sys.stderr)
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Size by n_pairs (scaled for visibility)
    sizes = data["n_pairs"].values * 8 + 40

    scatter = ax.scatter(data["pct_documented_pt"], data["median_abs_delta"],
                         s=sizes, c=data["median_abs_delta"],
                         cmap="RdYlGn_r", edgecolors="black", linewidths=0.8,
                         zorder=5, alpha=0.85, vmin=0, vmax=0.1)

    # Label each point
    for _, row in data.iterrows():
        ax.annotate(f"{row['benchmark']}\n(n={row['n_pairs']:.0f})",
                    xy=(row["pct_documented_pt"], row["median_abs_delta"]),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=7.5, fontweight="bold", color="#333333",
                    zorder=6)

    ax.set_xlabel("% collision pairs with documented prompt template", fontsize=10)
    ax.set_ylabel("Median |Δ| (score delta)", fontsize=10)
    ax.set_title("Documentation Coverage vs. Observed Score Variance\nby Benchmark",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(-5, 105)
    ax.set_ylim(-0.005, max(data["median_abs_delta"]) * 1.3 + 0.01)
    ax.grid(True, alpha=0.3, linestyle="--")

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Median |Δ|", fontsize=9)

    fig.tight_layout()
    out_pdf = OUT_DIR / "fig9_coverage_variance.pdf"
    out_png = SUB_DIR / "fig9_coverage_variance.png"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=150)
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved → {out_pdf}")
    print(f"Saved → {out_png}")


if __name__ == "__main__":
    main()
