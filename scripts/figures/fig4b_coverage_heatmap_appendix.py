"""
fig4_coverage_heatmap.py — metadata field coverage heatmap per source.

Four-color scheme for prompt_template column:
  green (>=80%), yellow (1-79%), amber ("standard" placeholder), red (absent).
Other columns: green, yellow, red.
Sorted by total coverage score within LB/paper groups.

Output: figures/fig4_coverage_heatmap.pdf + submission/fig4_coverage_heatmap.png
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

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


def classify_prompt_template_sources() -> dict[str, str]:
    """Classify each source's prompt_template usage."""
    DATA = _ROOT / "data"
    source_pt: dict[str, set[str]] = defaultdict(set)

    for sub in sorted(DATA.iterdir()):
        if not sub.is_dir():
            continue
        for f in sub.rglob("*.json"):
            try:
                rec = json.loads(f.read_text())
                for er in rec.get("evaluation_results", []):
                    gc = er.get("generation_config", {})
                    ad = gc.get("additional_details", {})
                    pt = ad.get("prompt_template", "")
                    source_pt[sub.name].add(pt if pt else "__MISSING__")
            except Exception:
                pass

    result = {}
    for src, vals in source_pt.items():
        non_standard = vals - {"standard", "__MISSING__"}
        if non_standard:
            result[src] = "documented"
        elif "standard" in vals:
            result[src] = "standard_only"
        else:
            result[src] = "absent"
    return result


def main():
    csv_path = _ROOT / "analysis_output" / "coverage_stats.csv"
    if not csv_path.exists():
        print("Run coverage_audit.py first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df = df[df["n_records"] > 0].copy()

    cols = ["pct_n_shot", "pct_harness", "pct_prompt_template", "pct_temperature"]
    col_labels = ["n_shot", "harness", "prompt\ntemplate", "temperature"]

    pt_class = classify_prompt_template_sources()

    df["total_coverage"] = df[cols].sum(axis=1)
    is_lb = df["source"].isin(LEADERBOARD_SOURCES)
    lb_df = df[is_lb].sort_values("total_coverage", ascending=False).reset_index(drop=True)
    paper_df = df[~is_lb].sort_values("total_coverage", ascending=False).reset_index(drop=True)
    ordered = pd.concat([lb_df, paper_df], ignore_index=True)
    n_lb = len(lb_df)

    matrix = ordered[cols].values.copy().astype(float)
    row_labels = ordered["source"].tolist()
    n_records = ordered["n_records"].tolist()

    # For prompt_template column (idx 2): mark "standard_only" as -1
    pt_col = 2
    for i, src in enumerate(row_labels):
        if matrix[i, pt_col] == 0:
            cls = pt_class.get(src, "absent")
            if cls == "standard_only":
                matrix[i, pt_col] = -1

    # --- Plot ---
    fig_h = max(4, 0.42 * len(row_labels) + 1.5)
    fig, ax = plt.subplots(figsize=(7.5, fig_h))

    # 4-color: red (absent), amber (std placeholder), yellow (partial), green (>=80)
    colors_4 = ["#E74C3C", "#F5CBA7", "#F5B041", "#27AE60"]
    bounds = [-1.5, -0.5, 0.5, 80, 100.5]
    cmap = ListedColormap(colors_4)
    norm = BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(col_labels, fontsize=10)
    for label in ax.get_xticklabels():
        if "harness" in label.get_text():
            label.set_fontweight("bold")

    ax.set_yticks(range(len(row_labels)))
    row_tick_labels = [f"{pretty_source(r)}  (n={n})"
                       for r, n in zip(row_labels, n_records)]
    ax.set_yticklabels(row_tick_labels, fontsize=7.5, family="monospace")

    for i in range(len(row_labels)):
        for j in range(len(cols)):
            val = matrix[i, j]
            if val == -1:
                txt, tc = '"std"', "#8B4513"
            elif val == 0:
                txt, tc = "0%", "white"
            elif val >= 80:
                txt, tc = f"{val:.0f}%", "white"
            else:
                txt, tc = f"{val:.0f}%", "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=7.5, color=tc, fontweight="bold")

    if 0 < n_lb < len(row_labels):
        ax.axhline(n_lb - 0.5, color="black", linewidth=2, linestyle="--")
        ax.text(-0.85, (n_lb - 1) / 2, "Leaderboards",
                ha="center", va="center", fontsize=8, fontweight="bold",
                rotation=90, color="#333333")
        ax.text(-0.85, n_lb + (len(row_labels) - n_lb - 1) / 2, "Papers",
                ha="center", va="center", fontsize=8, fontweight="bold",
                rotation=90, color="#333333")

    legend_elements = [
        Patch(facecolor="#27AE60", edgecolor="grey", label="\u226580% documented"),
        Patch(facecolor="#F5B041", edgecolor="grey", label="1\u201379% partial"),
        Patch(facecolor="#F5CBA7", edgecolor="grey", label='\u201cstandard\u201d placeholder'),
        Patch(facecolor="#E74C3C", edgecolor="grey", label="0% absent"),
    ]
    ax.legend(handles=legend_elements, fontsize=7.5, loc="lower right",
              bbox_to_anchor=(1.0, -0.08), ncol=4, framealpha=0.9)

    ax.set_title("Metadata Coverage by Source and Field",
                 fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

    fig.tight_layout()
    out_pdf = OUT_DIR / "fig4_coverage_heatmap.pdf"
    out_png = SUB_DIR / "fig4_coverage_heatmap.png"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=150)
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved -> {out_pdf}")
    print(f"Saved -> {out_png}")


if __name__ == "__main__":
    main()
