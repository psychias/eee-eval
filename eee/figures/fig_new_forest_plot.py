"""
Forest plot of partial R² with CIs per benchmark × predictor.
Replaces Table 2 in main body.
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


def r2_ci(r2, n, alpha=0.05):
    """Approximate 95% CI for partial R² via delta-method SE."""
    if n <= 3 or r2 <= 0:
        return 0.0, min(0.05, r2 + 0.03)
    se = np.sqrt(4 * r2 * (1 - r2)**2 / max(n - 3, 1))
    return max(0, r2 - 1.96 * se), min(1, r2 + 1.96 * se)


def main():
    ols_path = _ROOT / "analysis_output" / "per_benchmark_ols.csv"
    if not ols_path.exists():
        print(f"Missing {ols_path} — run per_benchmark_ols.py first.")
        return

    df = pd.read_csv(ols_path)
    if df.empty:
        print("per_benchmark_ols.csv is empty — skipping figure.")
        return

    benchmarks = sorted(df["benchmark"].unique(),
                        key=lambda b: df[df["benchmark"] == b]["partial_r2"].max(),
                        reverse=True)
    predictors = ["harness_differs", "n_shot_diff", "prompt_template_differs"]
    pred_labels = {"harness_differs": "Harness",
                   "n_shot_diff": "n-shot",
                   "prompt_template_differs": "Prompt template"}
    pred_colors = {"harness_differs": "#DD8452",
                   "n_shot_diff": "#4C72B0",
                   "prompt_template_differs": "#55A868"}

    fig, ax = plt.subplots(figsize=(8, max(3, 0.9 * len(benchmarks) * len(predictors) * 0.3 + 2)))

    y_positions = {}
    y = 0
    for bench in benchmarks:
        for pred in predictors:
            row = df[(df["benchmark"] == bench) & (df["predictor"] == pred)]
            if row.empty:
                continue
            r2 = float(row["partial_r2"].values[0])
            n = int(row["n"].values[0]) if "n" in row.columns else 30
            pval = float(row["p_value"].values[0]) if "p_value" in row.columns else 1.0
            ci_lo, ci_hi = r2_ci(r2, n)

            color = pred_colors[pred]
            is_harness = pred == "harness_differs"
            height = 0.25 if is_harness else 0.15
            alpha = 0.85 if is_harness else 0.55

            ax.barh(y, r2, height=height, color=color, alpha=alpha,
                   edgecolor="white")
            ax.errorbar(r2, y, xerr=[[r2 - ci_lo], [ci_hi - r2]],
                       fmt="none", ecolor="black", capsize=3, lw=1)

            # Significance stars
            star = ("***" if pval < 0.001 else "**" if pval < 0.01 else
                    "*" if pval < 0.05 else "n.s.")
            ax.text(max(r2, ci_hi) + 0.005, y,
                   f"R²={r2:.3f} {star}", va="center", fontsize=7)

            y_positions[(bench, pred)] = y
            y += 0.3
        y += 0.2  # gap between benchmarks

    # Y-axis benchmark labels
    bench_midpoints = []
    for bench in benchmarks:
        ys = [y_positions[(bench, p)] for p in predictors
              if (bench, p) in y_positions]
        if ys:
            bench_midpoints.append((np.mean(ys), bench))

    ax.set_yticks([m for m, _ in bench_midpoints])
    ax.set_yticklabels(
        [f"{b}\n(n={int(df[df['benchmark'] == b]['n'].iloc[0])})"
         for _, b in bench_midpoints],
        fontsize=9
    )

    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Partial R² (unique variance explained)", fontsize=10)
    ax.set_title("Effect of Methodology Predictors on Score Variance\n"
                 "(exploratory OLS; no effect survives BH-FDR correction)",
                 fontsize=11, fontweight="bold")

    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=c, label=pred_labels[p])
                        for p, c in pred_colors.items()],
               loc="lower right", fontsize=8)

    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_forest_plot.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(SUB_DIR / "fig_forest_plot.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Saved → fig_forest_plot")


if __name__ == "__main__":
    main()
