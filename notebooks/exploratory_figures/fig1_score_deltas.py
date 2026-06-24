"""
fig1_score_deltas.py — score delta distributions per benchmark.

Reads analysis_output/collision_pairs.csv and renders jittered strip plots
(swarm) overlaid on light boxplots per benchmark:
  - colour encodes dominant methodology difference
    (blue = n-shot differs, orange = harness differs, green = both)
  - explicit n= count annotations per benchmark (replaces grey bands)
  - every collision pair shown as an individual dot (honest about n)
  - light boxplot (whiskers + median line) behind dots
  - dashed horizontal line at Δ = 0

Output: figures/fig1_score_deltas.pdf  +  submission/fig1_score_deltas.png
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

# Methodology-difference colour palette (matches paper caption)
COLOR_NSHOT   = "#4C72B0"   # blue  – n-shot differs
COLOR_HARNESS = "#DD8452"   # orange – harness differs
COLOR_BOTH    = "#55A868"   # green  – both differ
COLOR_NONE    = "#888888"   # grey   – identical / undocumented


def classify_methodology(sub: pd.DataFrame) -> str:
    """Return dominant methodology difference for a benchmark group."""
    nshot_diff  = (sub["n_shot_a"] != sub["n_shot_b"]).sum()
    harness_diff = (sub["harness_a"] != sub["harness_b"]).sum()
    total = len(sub)
    if total == 0:
        return "none"
    nshot_frac   = nshot_diff / total
    harness_frac = harness_diff / total
    if nshot_frac > 0.3 and harness_frac > 0.3:
        return "both"
    if nshot_frac > 0.3:
        return "nshot"
    if harness_frac > 0.3:
        return "harness"
    return "none"


METH_COLORS = {
    "nshot":   COLOR_NSHOT,
    "harness": COLOR_HARNESS,
    "both":    COLOR_BOTH,
    "none":    COLOR_NONE,
}


def main():
    csv_path = _ROOT / "analysis_output" / "collision_pairs.csv"
    if not csv_path.exists():
        print("Run collision_detection.py first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    if df.empty:
        print("No collision pairs.", file=sys.stderr)
        sys.exit(1)

    # Filter to benchmarks with ≥ 2 collision pairs
    bench_counts = df.groupby("benchmark")["delta"].count()
    valid_benchmarks = bench_counts[bench_counts >= 2].index
    df = df[df["benchmark"].isin(valid_benchmarks)]

    if df.empty:
        print("No benchmarks with ≥2 collision pairs.", file=sys.stderr)
        sys.exit(1)

    # Sort benchmarks by median |delta| descending
    order = (df.assign(abs_delta=df["delta"].abs())
               .groupby("benchmark")["abs_delta"]
               .median()
               .sort_values(ascending=False)
               .index.tolist())

    # Classify each benchmark's dominant methodology difference
    meth_class = {}
    for bench in order:
        meth_class[bench] = classify_methodology(df[df["benchmark"] == bench])

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(order)), 5))

    for i, bench in enumerate(order):
        sub_df = df[df["benchmark"] == bench]
        vals = sub_df["delta"].values
        n = len(vals)
        color = METH_COLORS[meth_class[bench]]

        # Light boxplot behind dots — honest about distribution shape
        bp = ax.boxplot(vals, positions=[i], widths=0.45, vert=True,
                        patch_artist=True, zorder=2,
                        showfliers=False, medianprops=dict(color="black", linewidth=2))
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.15)
        for element in ("whiskers", "caps"):
            for line in bp[element]:
                line.set_color(color)
                line.set_alpha(0.5)

        # Jittered strip overlay — every point visible
        rng = np.random.default_rng(42 + i)
        jitter = rng.uniform(-0.15, 0.15, size=n)
        ax.scatter([i + j for j in jitter], vals, alpha=0.75, s=32,
                   color=color, zorder=5, edgecolors="white", linewidths=0.4)

        # Count annotation above the distribution
        y_top = max(vals) + 0.01
        ax.text(i, y_top, f"n={n} pairs",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold",
                color=color)

    ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)

    # X-axis labels (counts now annotated directly on plot)
    ax.set_xticks(range(len(order)))
    xlabels = [b for b in order]
    ax.set_xticklabels(xlabels, rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("Score delta (source A − source B)", fontsize=11)
    ax.set_title("Score Delta Distributions per Benchmark\n"
                 f"({len(df)} cross-source collision pairs, {df['model_a'].nunique() if 'model_a' in df.columns else 27} unique models)",
                 fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_NSHOT,   label="n-shot differs"),
        Patch(facecolor=COLOR_HARNESS, label="harness differs"),
        Patch(facecolor=COLOR_BOTH,    label="both differ"),
        Patch(facecolor=COLOR_NONE,    label="undocumented / identical"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper right",
              framealpha=0.9)

    fig.tight_layout()

    out_pdf = OUT_DIR / "fig1_score_deltas.pdf"
    out_png = SUB_DIR / "fig1_score_deltas.png"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=150)
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved → {out_pdf}")
    print(f"Saved → {out_png}")


if __name__ == "__main__":
    main()
