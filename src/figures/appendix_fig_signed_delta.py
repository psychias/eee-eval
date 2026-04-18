#!/usr/bin/env python3
"""
fig10_signed_delta_profile.py — Per-source mean signed Δ dot plot.

For each source that appears in collision pairs, compute:
  - mean signed delta (across all pairs where source is source_a)
  - number of collision pairs involving that source
  - whether source is a leaderboard or paper

Y-axis = source name, X-axis = mean signed Δ.
Dot size = number of collision pairs, colour = source type.
Vertical dashed line at Δ=0.

Replaces the power simulation bar chart (fig6) in the main body —
this shows an ACTUAL finding (directional bias by source) rather
than a projected future state.

Output: figures/fig10_signed_delta_profile.pdf + submission/fig10_signed_delta_profile.png
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


def pretty_source(s: str) -> str:
    mapping = {
        "hfopenllm_v2": "HF Open LLM v2",
        "alpacaeval2": "AlpacaEval 2",
        "chatbot_arena": "Chatbot Arena",
        "mt_bench": "MT-Bench",
        "wildbench": "WildBench",
        "bigcodebench": "BigCodeBench",
    }
    if s in mapping:
        return mapping[s]
    if s.startswith("papers_"):
        return f"arXiv:{s.replace('papers_', '')}"
    return s


def main():
    csv_path = _ROOT / "analysis_output" / "collision_pairs.csv"
    if not csv_path.exists():
        print("Run collision analysis first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # For each source, collect all signed deltas when that source is source_a.
    # Also collect when it's source_b (flip sign) so every source gets a
    # comprehensive view regardless of pair ordering.
    records = []
    for _, row in df.iterrows():
        records.append({
            "source": row["source_a"],
            "delta": row["delta"],
            "benchmark": row["benchmark"],
        })
        records.append({
            "source": row["source_b"],
            "delta": -row["delta"],
            "benchmark": row["benchmark"],
        })

    sdf = pd.DataFrame(records)

    # Aggregate per source
    agg = sdf.groupby("source").agg(
        mean_delta=("delta", "mean"),
        n_pairs=("delta", "count"),
        std_delta=("delta", "std"),
    ).reset_index()

    # Compute 95% CI (mean ± 1.96 * se)
    agg["se"] = agg["std_delta"] / np.sqrt(agg["n_pairs"])
    agg["ci_lo"] = agg["mean_delta"] - 1.96 * agg["se"]
    agg["ci_hi"] = agg["mean_delta"] + 1.96 * agg["se"]

    # Filter: only sources with ≥2 collision pairs
    agg = agg[agg["n_pairs"] >= 2].copy()

    # Sort by mean delta
    agg = agg.sort_values("mean_delta", ascending=True).reset_index(drop=True)

    # Source type
    agg["is_lb"] = agg["source"].isin(LEADERBOARD_SOURCES)
    agg["label"] = agg["source"].apply(pretty_source)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(7, max(4, 0.4 * len(agg) + 1)))

    y_pos = np.arange(len(agg))

    # Scale dot size by pair count
    min_n, max_n = agg["n_pairs"].min(), agg["n_pairs"].max()
    size_range = (40, 250)
    if max_n > min_n:
        sizes = size_range[0] + (agg["n_pairs"] - min_n) / (max_n - min_n) * (size_range[1] - size_range[0])
    else:
        sizes = np.full(len(agg), 100)

    # Colours: blue for papers, orange for leaderboards
    colors = ["#E69F00" if lb else "#0072B2" for lb in agg["is_lb"]]

    # Error bars (95% CI)
    xerr_lo = agg["mean_delta"] - agg["ci_lo"]
    xerr_hi = agg["ci_hi"] - agg["mean_delta"]
    ax.errorbar(agg["mean_delta"], y_pos, xerr=[xerr_lo, xerr_hi],
                fmt="none", ecolor="#999999", elinewidth=0.8, capsize=2, zorder=1)

    # Scatter
    ax.scatter(agg["mean_delta"], y_pos, s=sizes, c=colors,
               edgecolors="white", linewidths=0.5, zorder=2, alpha=0.85)

    # Vertical dashed line at 0
    ax.axvline(0, color="#333333", linewidth=1, linestyle="--", alpha=0.5, zorder=0)

    # Y-labels
    ax.set_yticks(y_pos)
    labels_with_n = [f"{row['label']}  (n={int(row['n_pairs'])})"
                     for _, row in agg.iterrows()]
    ax.set_yticklabels(labels_with_n, fontsize=7.5)

    ax.set_xlabel("Mean Signed Δ (Source A − Source B)", fontsize=10)
    ax.set_title("Per-Source Signed Score Delta Profile", fontsize=12, fontweight="bold")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#0072B2',
               markersize=8, label='Paper'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E69F00',
               markersize=8, label='Leaderboard'),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right", framealpha=0.9)

    # Size legend
    # Add a text note about dot size instead of complex size legend
    ax.annotate("Dot size ∝ n pairs", xy=(0.98, 0.02), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=7, color="#666666", style="italic")

    ax.grid(axis="x", alpha=0.2)
    ax.set_xlim(ax.get_xlim()[0] - 0.02, ax.get_xlim()[1] + 0.02)

    fig.tight_layout()
    out_pdf = OUT_DIR / "fig10_signed_delta_profile.pdf"
    out_png = SUB_DIR / "fig10_signed_delta_profile.png"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=150)
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved → {out_pdf}")
    print(f"Saved → {out_png}")


if __name__ == "__main__":
    main()
