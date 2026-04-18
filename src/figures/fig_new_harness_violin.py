"""
Paired violin plot: |delta| when harness is same vs different.
The single most compelling visual for the main finding.
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


def main():
    df = pd.read_csv(_ROOT / "analysis_output" / "harness_effect.csv")

    benchmarks = sorted(df["benchmark"].unique())
    n_bench = len(benchmarks)

    if n_bench == 0:
        print("No benchmarks found in harness_effect.csv — skipping figure.")
        return

    fig, axes = plt.subplots(1, n_bench, figsize=(2.5 * n_bench, 5),
                              sharey=True, squeeze=False)
    axes = axes[0]  # flatten from 2D to 1D

    COL_SAME = "#4C72B0"   # blue — same harness
    COL_DIFF = "#DD8452"   # orange — different harness

    for ax, bench in zip(axes, benchmarks):
        sub = df[df["benchmark"] == bench]
        same = sub[sub["harness_same"]]["abs_delta"].values
        diff = sub[~sub["harness_same"]]["abs_delta"].values

        data = []
        colors = []
        positions = []
        if len(same) > 1:
            data.append(same)
            colors.append(COL_SAME)
            positions.append(1)
        if len(diff) > 1:
            data.append(diff)
            colors.append(COL_DIFF)
            positions.append(2)

        if data:
            parts = ax.violinplot(data, positions=positions,
                                   showmedians=True, showextrema=True)
            for pc, col in zip(parts["bodies"], colors):
                pc.set_facecolor(col)
                pc.set_alpha(0.7)

        # Overlay jittered points
        rng = np.random.default_rng(42)
        if len(same) > 0:
            jitter = rng.uniform(-0.08, 0.08, len(same))
            ax.scatter(1 + jitter, same, alpha=0.5, s=15,
                      color=COL_SAME, zorder=3, edgecolors="white", lw=0.3)
        if len(diff) > 0:
            jitter = rng.uniform(-0.08, 0.08, len(diff))
            ax.scatter(2 + jitter, diff, alpha=0.5, s=15,
                      color=COL_DIFF, zorder=3, edgecolors="white", lw=0.3)

        # Sample size annotations
        ylim_top = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.1
        ax.text(1, ylim_top * 0.95,
                f"n={len(same)}", ha="center", fontsize=7, color=COL_SAME)
        ax.text(2, ylim_top * 0.95,
                f"n={len(diff)}", ha="center", fontsize=7, color=COL_DIFF)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Same\nharness", "Diff.\nharness"], fontsize=8)
        ax.set_title(bench, fontsize=9, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("|Δ| (absolute score delta)", fontsize=10)
    fig.suptitle("Score Delta: Same Harness vs. Different Harness",
                 fontsize=12, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    fig.legend(handles=[
        Patch(facecolor=COL_SAME, alpha=0.7, label="Same harness"),
        Patch(facecolor=COL_DIFF, alpha=0.7, label="Different harness"),
    ], loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_harness_violin.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(SUB_DIR / "fig_harness_violin.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Saved → fig_harness_violin")


if __name__ == "__main__":
    main()
