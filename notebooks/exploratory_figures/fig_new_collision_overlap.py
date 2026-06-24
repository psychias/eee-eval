"""
Bipartite heatmap: sources × models, colored by collision existence.
Shows the structural isolation problem — leaderboards don't overlap.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = _ROOT / "figures"
SUB_DIR = _ROOT / "submission"

LEADERBOARD_SOURCES = {
    "hfopenllm_v2", "alpacaeval2", "chatbot_arena",
    "mt_bench", "wildbench", "bigcodebench",
}

_SHORT_NAMES = {
    "hfopenllm_v2": "HF LLM v2",
    "alpacaeval2": "AlpacaEval",
    "chatbot_arena": "Chatbot Arena",
    "mt_bench": "MT-Bench",
    "wildbench": "WildBench",
    "bigcodebench": "BigCodeBench",
}


def shorten(s):
    if s in LEADERBOARD_SOURCES:
        return _SHORT_NAMES.get(s, s)
    return f"arXiv:{s.replace('papers_', '')[:10]}"


def main():
    matrix_path = _ROOT / "analysis_output" / "collision_overlap_matrix.csv"
    if not matrix_path.exists():
        print(f"Missing {matrix_path} — run collision_overlap_matrix.py first.")
        return

    matrix = pd.read_csv(matrix_path, index_col=0)
    if matrix.empty:
        print("collision_overlap_matrix.csv is empty — skipping figure.")
        return

    # Sort: leaderboards first, then papers; models by frequency
    sources = matrix.index.tolist()
    lb_sources = [s for s in sources if s in LEADERBOARD_SOURCES]
    paper_sources = [s for s in sources if s not in LEADERBOARD_SOURCES]
    ordered_sources = lb_sources + paper_sources

    model_freq = matrix.sum(axis=0).sort_values(ascending=False)
    top_models = model_freq.head(30).index.tolist()

    # Ensure ordered_sources are all in the matrix index
    ordered_sources = [s for s in ordered_sources if s in matrix.index]
    top_models = [m for m in top_models if m in matrix.columns]

    if not ordered_sources or not top_models:
        print("Not enough data for collision overlap figure.")
        return

    mat = matrix.loc[ordered_sources, top_models].values

    fig_height = max(4, 0.4 * len(ordered_sources) + 2)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    # Color: 0=white, 1=colored by source type
    colors = np.zeros((*mat.shape, 4))
    lb_color = np.array([0.914, 0.624, 0.0, 0.8])    # amber
    paper_color = np.array([0.0, 0.447, 0.698, 0.8])  # blue

    n_lb = len(lb_sources)
    for i, src in enumerate(ordered_sources):
        c = lb_color if src in LEADERBOARD_SOURCES else paper_color
        for j in range(len(top_models)):
            if mat[i, j]:
                colors[i, j] = c
            else:
                colors[i, j] = [1, 1, 1, 1]

    ax.imshow(colors, aspect="auto")

    # Divider between leaderboards and papers
    if n_lb > 0 and len(paper_sources) > 0:
        ax.axhline(n_lb - 0.5, color="black", lw=2)

    ax.set_yticks(range(len(ordered_sources)))
    ax.set_yticklabels([shorten(s) for s in ordered_sources], fontsize=7)
    ax.set_xticks(range(len(top_models)))
    ax.set_xticklabels([m.split("/")[-1][:20] for m in top_models],
                        rotation=45, ha="right", fontsize=6.5)

    ax.set_title("Source × Model Coverage in Collision Pairs\n"
                 "(colored = model appears in ≥1 collision pair for that source)",
                 fontsize=11, fontweight="bold")

    # Annotations
    if n_lb > 0:
        ax.text(-0.5, n_lb / 2 - 0.5, "Leaderboards",
                rotation=90, va="center", ha="right", fontsize=8, fontweight="bold",
                color="#E69F00", transform=ax.transData)
    if paper_sources:
        ax.text(-0.5, n_lb + len(paper_sources) / 2 - 0.5, "Papers",
                rotation=90, va="center", ha="right", fontsize=8, fontweight="bold",
                color="#0072B2", transform=ax.transData)

    from matplotlib.patches import Patch
    fig.legend(handles=[
        Patch(color="#E69F00", alpha=0.8, label="Leaderboard source"),
        Patch(color="#0072B2", alpha=0.8, label="Paper source"),
        Patch(color="white", ec="grey", label="Not in collision pairs"),
    ], loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.05))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_collision_overlap.pdf",
                bbox_inches="tight", dpi=150)
    fig.savefig(SUB_DIR / "fig_collision_overlap.png",
                bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Saved → fig_collision_overlap")


if __name__ == "__main__":
    main()
