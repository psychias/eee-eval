"""
fig3_rank_heatmap.py — Kendall τ heatmap between source pairs.

Reads analysis_output/rank_instability.csv and renders a heatmap of
Kendall τ between all source pairs:
  - each cell shows τ AND sample size n
  - dashed borders for cells with n < 10
  - diagonal annotated '---' (self-comparison)
  - RdYlGn colour scale: red = low agreement, green = high agreement

Output: figures/fig3_rank_heatmap.pdf  +  submission/fig3_rank_instability.png
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

OUT_DIR = _ROOT / "figures"
OUT_DIR.mkdir(exist_ok=True)
SUB_DIR = _ROOT / "submission"
SUB_DIR.mkdir(exist_ok=True)


def short_label(s: str) -> str:
    s = s.replace("papers_", "arXiv:")
    s = s.replace("hfopenllm_v2", "HF Open LLM v2")
    s = s.replace("alpacaeval2", "AlpacaEval 2")
    s = s.replace("mt_bench", "MT-Bench")
    s = s.replace("chatbot_arena", "Chatbot Arena")
    s = s.replace("wildbench", "WildBench")
    s = s.replace("bigcodebench", "BigCodeBench")
    s = s.replace("global-mmlu-lite", "Global MMLU Lite")
    s = s.replace("reward-bench", "RewardBench")
    return s


def main():
    csv_path = _ROOT / "analysis_output" / "rank_instability.csv"
    if not csv_path.exists():
        print("Run rank_instability.py first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    if df.empty:
        print("No rank instability data.", file=sys.stderr)
        sys.exit(1)

    # Build symmetric τ matrix and n_models matrix
    sources = sorted(set(df["source_a"].tolist() + df["source_b"].tolist()))
    n = len(sources)
    idx = {s: i for i, s in enumerate(sources)}

    tau_matrix = np.full((n, n), np.nan)
    n_matrix   = np.full((n, n), 0, dtype=int)
    np.fill_diagonal(tau_matrix, 1.0)

    # Aggregate: median τ over benchmarks for each source pair
    tau_col = "tau_b" if "tau_b" in df.columns else "tau"
    grouped = df.groupby(["source_a", "source_b"]).agg(
        tau_med=(tau_col, "median"),
        n_total=("n_models", "sum"),
    ).reset_index()

    for _, row in grouped.iterrows():
        i = idx[row["source_a"]]
        j = idx[row["source_b"]]
        tau_matrix[i, j] = row["tau_med"]
        tau_matrix[j, i] = row["tau_med"]
        n_matrix[i, j]   = int(row["n_total"])
        n_matrix[j, i]   = int(row["n_total"])

    labels = [short_label(s) for s in sources]

    fig, ax = plt.subplots(figsize=(max(7, 0.7 * n + 1), max(6, 0.65 * n + 1)))

    cmap = plt.cm.RdYlGn
    im = ax.imshow(tau_matrix, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=7)

    # Annotate each cell
    for i in range(n):
        for j in range(n):
            if i == j:
                # Diagonal
                ax.text(j, i, "---", ha="center", va="center",
                        fontsize=7, color="black", fontweight="bold")
                continue
            val = tau_matrix[i, j]
            nij = n_matrix[i, j]
            if np.isnan(val):
                continue
            text_color = "white" if abs(val) > 0.7 else "black"
            ax.text(j, i, f"τ={val:.2f}\nn={nij}",
                    ha="center", va="center", fontsize=5.5,
                    color=text_color, linespacing=1.2)

            # Dashed border for n < 10
            if nij < 10:
                rect = mpatches.FancyBboxPatch(
                    (j - 0.48, i - 0.48), 0.96, 0.96,
                    boxstyle="square,pad=0", linewidth=1.5,
                    edgecolor="black", facecolor="none",
                    linestyle="--", zorder=3)
                ax.add_patch(rect)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Kendall τ (median across benchmarks)", fontsize=9)

    ax.set_title("Rank Correlation Between Evaluation Sources\n"
                 "(Kendall τ; dashed border = n < 10)",
                 fontsize=11, fontweight="bold", pad=12)

    fig.tight_layout()
    out_pdf = OUT_DIR / "fig3_rank_heatmap.pdf"
    out_png = SUB_DIR / "fig3_rank_instability.png"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=150)
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved → {out_pdf}")
    print(f"Saved → {out_png}")


if __name__ == "__main__":
    main()
