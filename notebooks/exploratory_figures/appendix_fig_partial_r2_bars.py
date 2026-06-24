"""
fig2_variance_decomp.py — per-benchmark partial R² grouped bar chart.

Reads analysis_output/per_benchmark_ols.csv and renders grouped horizontal
bars of partial R² for each methodology predictor, broken out per benchmark:
  - bar length = unique variance explained in simple OLS
  - error bars = 95% CI
  - dashed line at R² = 0.35 (large-effect threshold, f² = 0.54)
  - note about GSM8K harness/n-shot collinearity

Output: figures/fig2_variance_decomp.pdf  +  submission/fig2_variance_decomp.png
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

PREDICTOR_LABELS = {
    "harness_differs":         "harness differs",
    "n_shot_diff":             "n-shot differs",
    "n_shot_differs":          "n-shot differs",
    "prompt_template_differs": "prompt template differs",
}
PREDICTOR_COLORS = {
    "harness_differs":         "#DD8452",
    "n_shot_diff":             "#4C72B0",
    "n_shot_differs":          "#4C72B0",
    "prompt_template_differs": "#55A868",
}


def r2_ci_simple(r2: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Simple delta-method 95 %% CI for R² (Olkin & Finn)."""
    if n <= 3 or r2 <= 0:
        return (0.0, min(0.05, r2 + 0.05))
    se = np.sqrt(4 * r2 * (1 - r2) ** 2 / (n - 3))
    z = 1.96
    return (max(0, r2 - z * se), min(1, r2 + z * se))


def main():
    csv_path = _ROOT / "analysis_output" / "per_benchmark_ols.csv"
    if not csv_path.exists():
        csv_path = _ROOT / "analysis_output" / "variance_decomp.csv"
    if not csv_path.exists():
        print("Run per_benchmark_ols.py or variance_decomposition.py first.",
              file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    has_benchmark = "benchmark" in df.columns
    if not has_benchmark:
        df["benchmark"] = "All benchmarks"

    benchmarks = sorted(df["benchmark"].unique().tolist())
    predictors = sorted(df["predictor"].unique().tolist(),
                        key=lambda p: PREDICTOR_LABELS.get(p, p))
    n_bench = len(benchmarks)
    n_pred  = len(predictors)

    bar_height = 0.22
    group_gap  = 0.6
    fig_h = max(4, (bar_height * n_pred + group_gap) * n_bench + 1.5)
    fig, ax = plt.subplots(figsize=(8, fig_h))

    for bi, bench in enumerate(benchmarks):
        for pi, pred in enumerate(predictors):
            row = df[(df["benchmark"] == bench) & (df["predictor"] == pred)]
            y = bi * (n_pred * bar_height + group_gap) + pi * bar_height

            if row.empty:
                continue

            r2 = float(row["partial_r2"].values[0])
            n  = int(row["n"].values[0]) if "n" in row.columns else 50
            ci_lo, ci_hi = r2_ci_simple(r2, n)
            pval = float(row["p_value"].values[0]) if "p_value" in row.columns else 1.0

            color = PREDICTOR_COLORS.get(pred, "#999999")
            label_text = PREDICTOR_LABELS.get(pred, pred)

            # Make harness bars prominent, n_shot/prompt_template thinner
            is_harness = "harness" in pred
            this_height = bar_height if is_harness else bar_height * 0.55
            this_alpha = 0.85 if is_harness else 0.5
            y_offset = 0 if is_harness else (bar_height - this_height) / 2

            ax.barh(y + y_offset, r2, height=this_height, color=color,
                    alpha=this_alpha,
                    edgecolor="white", linewidth=0.5,
                    label=label_text if bi == 0 else None)
            ax.errorbar(r2, y + y_offset + this_height / 2,
                        xerr=[[r2 - ci_lo], [ci_hi - r2]],
                        fmt="none", ecolor="black", capsize=3, linewidth=1)

            star = ""
            if pval < 0.001:
                star = " ***"
            elif pval < 0.01:
                star = " **"
            elif pval < 0.05:
                star = " *"
            else:
                star = " n.s."
            ax.text(max(r2, ci_hi) + 0.01, y,
                    f"{r2:.3f}{star}", va="center", fontsize=7.5, color="black")

    # Best observed result reference line (replaces unachieved 0.35 threshold)
    best_r2 = df["partial_r2"].max()
    ax.axvline(best_r2, color="#0072B2", linestyle="--", linewidth=1.5, alpha=0.7,
               label=f"best observed R² = {best_r2:.3f}")

    # Y-axis: benchmark group labels with sample size
    ytick_pos, ytick_lbl = [], []
    for bi, bench in enumerate(benchmarks):
        mid = bi * (n_pred * bar_height + group_gap) + (n_pred - 1) * bar_height / 2
        ytick_pos.append(mid)
        # Add sample size annotation
        bench_n = df.loc[df["benchmark"] == bench, "n"].iloc[0] if "n" in df.columns else ""
        lbl = f"{bench} (n={bench_n})" if bench_n != "" else bench
        ytick_lbl.append(lbl)
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_lbl, fontsize=10, fontweight="bold")

    for bi in range(1, n_bench):
        y_div = bi * (n_pred * bar_height + group_gap) - group_gap / 2
        ax.axhline(y_div, color="#CCCCCC", linewidth=0.5)

    ax.set_xlabel("Partial R² (unique variance explained)", fontsize=10)
    ax.set_title("Partial R² of Methodology Predictors per Benchmark\n"
                 "(exploratory OLS; GSM8K harness/n-shot collinear)",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(0, min(1.0, df["partial_r2"].max() * 1.5 + 0.1))
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    ax.invert_yaxis()

    # Deduplicated legend
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uh, ul = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            uh.append(h)
            ul.append(l)
    ax.legend(uh, ul, fontsize=8, loc="lower right", framealpha=0.9)

    fig.tight_layout()
    out_pdf = OUT_DIR / "fig2_variance_decomp.pdf"
    out_png = SUB_DIR / "fig2_variance_decomp.png"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=150)
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved → {out_pdf}")
    print(f"Saved → {out_png}")


if __name__ == "__main__":
    main()
