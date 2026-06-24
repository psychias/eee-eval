#!/usr/bin/env python3
"""
Collision-pair analysis: metadata completeness vs. score divergence.
Focuses on the 8 independent collision pairs from Table 5/6.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib


def main():
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy import stats

    # Fix Windows console encoding
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    OUT = os.path.join(ROOT, "outputs")
    os.makedirs(OUT, exist_ok=True)

    # ── Hard-coded data from Table 5/6 ──────────────────────────
    data = pd.DataFrame([
        {"model": "Qwen2.5-72B",  "benchmark": "GPQA",     "s1": 16.67, "s2": 49.00, "H": 0, "S": 0, "C": 0, "T": 0, "P": 0},
        {"model": "SC2-15B",      "benchmark": "MBPP+",     "s1": 65.10, "s2": 61.20, "H": 1, "S": 0, "C": 0, "T": 0, "P": 0},
        {"model": "SC2-15B",      "benchmark": "HumanE+",   "s1": 60.40, "s2": 63.40, "H": 1, "S": 0, "C": 0, "T": 0, "P": 0},
        {"model": "Tulu-3-8B",    "benchmark": "BBH",       "s1": 16.86, "s2": 16.67, "H": 1, "S": 1, "C": 0, "T": 0, "P": 0},
        {"model": "Tulu-3-8B",    "benchmark": "GPQA",      "s1":  6.26, "s2":  6.49, "H": 1, "S": 1, "C": 0, "T": 0, "P": 0},
        {"model": "Tulu-3-8B",    "benchmark": "IFEval",    "s1": 82.55, "s2": 82.67, "H": 1, "S": 1, "C": 0, "T": 0, "P": 0},
        {"model": "Tulu-3-8B",    "benchmark": "MMLU-Pro",  "s1": 20.23, "s2": 20.30, "H": 1, "S": 1, "C": 0, "T": 0, "P": 0},
        {"model": "Tulu-3-8B",    "benchmark": "MuSR",      "s1": 10.52, "s2": 10.45, "H": 1, "S": 1, "C": 0, "T": 0, "P": 0},
    ])

    # ══════════════════════════════════════════════════════════════
    #  TASK 1 — Metadata completeness score
    # ══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("TASK 1 — Metadata Completeness Score")
    print("=" * 60)

    data["completeness"] = data["H"] + data["S"] + data["C"] + data["T"] + data["P"]
    data["abs_delta"] = (data["s1"] - data["s2"]).abs()

    print(data[["model", "benchmark", "completeness", "abs_delta"]].to_string(index=False))
    print(f"\nCompleteness range: {data['completeness'].min()} – {data['completeness'].max()}")
    print(f"|Δ| range: {data['abs_delta'].min():.2f} – {data['abs_delta'].max():.2f}")

    # ══════════════════════════════════════════════════════════════
    #  TASK 2 — Spearman correlation
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("TASK 2 — Spearman Correlation")
    print("=" * 60)

    rho, p_val = stats.spearmanr(data["completeness"], data["abs_delta"])
    print(f"Spearman rho = {rho:.4f}")
    print(f"p-value       = {p_val:.4f}")

    if rho < 0:
        direction = "negative"
        interp = ("Higher metadata completeness is associated with smaller score "
                  "gaps, consistent with the hypothesis that better-documented "
                  "evaluation setups yield more reproducible scores.")
    else:
        direction = "positive"
        interp = ("Higher metadata completeness is associated with larger score "
                  "gaps, which is counter to the expected direction and may "
                  "reflect confounding factors.")

    print(f"Direction: {direction}")
    print(f"Interpretation: {interp}")
    print(f"\nNote: n=8; this is illustrative, not definitive. The small sample "
          f"size limits statistical power and generalizability.")

    # ══════════════════════════════════════════════════════════════
    #  TASK 3 — First vs third party directional analysis
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("TASK 3 — First vs Third Party Directional Analysis (Tulu-3-8B)")
    print("=" * 60)

    tulu = data[data["model"] == "Tulu-3-8B"].copy()
    tulu["diff_s2_s1"] = tulu["s2"] - tulu["s1"]  # positive = third-party (OLv2) higher

    print(tulu[["benchmark", "s1", "s2", "diff_s2_s1"]].to_string(index=False))

    all_positive = (tulu["diff_s2_s1"] > 0).all()
    all_negative = (tulu["diff_s2_s1"] < 0).all()
    consistent = all_positive or all_negative

    mean_diff = tulu["diff_s2_s1"].mean()
    min_diff = tulu["diff_s2_s1"].min()
    max_diff = tulu["diff_s2_s1"].max()

    n_positive = (tulu["diff_s2_s1"] > 0).sum()
    n_negative = (tulu["diff_s2_s1"] < 0).sum()

    print(f"\nDirection consistent across all 5 pairs? {'Yes' if consistent else 'No'}")
    print(f"  Pairs where third-party > first-party (s2 > s1): {n_positive}")
    print(f"  Pairs where first-party > third-party (s1 > s2): {n_negative}")
    print(f"Mean difference (s2 - s1): {mean_diff:+.3f}")
    print(f"Range: [{min_diff:+.2f}, {max_diff:+.2f}]")

    if not consistent:
        print("\nThe direction is NOT consistent: most differences are positive "
              "(third-party slightly higher), but BBH and MuSR show the opposite.")

    # ══════════════════════════════════════════════════════════════
    #  TASK 4 — Figures
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("TASK 4 — Generating Figures")
    print("=" * 60)

    # --- Figure A: Scatter plot ---
    fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.scatter(data["completeness"], data["abs_delta"],
               s=60, color="#2c3e50", zorder=5, edgecolors="white", linewidths=0.5)

    # Label each point
    for _, row in data.iterrows():
        label = f"{row['model']}\n{row['benchmark']}"
        # Offset labels to avoid overlap
        x_off, y_off = 0.08, 0
        if row["benchmark"] == "GPQA" and row["model"] == "Qwen2.5-72B":
            y_off = -1.5
        elif row["benchmark"] == "HumanE+":
            y_off = 1.0
        elif row["benchmark"] == "MBPP+":
            y_off = -1.5
        ax.annotate(label, (row["completeness"] + x_off, row["abs_delta"] + y_off),
                    fontsize=7, color="#555555", ha="left", va="center")

    # Spearman annotation
    ax.text(0.97, 0.97,
            f"Spearman $\\rho$ = {rho:.3f}\n$p$ = {p_val:.3f}\n$n$ = 8",
            transform=ax.transAxes, fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0",
                      edgecolor="#cccccc", alpha=0.9))

    ax.set_xlabel("Metadata completeness (0–5)", fontsize=10)
    ax.set_ylabel("|$\\Delta$|  (absolute score gap)", fontsize=10)
    ax.set_title("Metadata completeness vs. score gap\nacross 8 independent collision pairs",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(-0.3, 5.3)
    ax.set_xticks(range(6))

    # Clean academic style: remove top/right spines, no gridlines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")

    plt.tight_layout()
    fig.savefig(f"{OUT}/figA_completeness_vs_delta.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/figA_completeness_vs_delta.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Figure A saved.")

    # --- Figure B: Dot plot for Tulu-3-8B ---
    fig, ax = plt.subplots(figsize=(6, 4))

    benchmarks = tulu["benchmark"].values
    y_pos = np.arange(len(benchmarks))

    for i, (_, row) in enumerate(tulu.iterrows()):
        color = "#27ae60" if row["s2"] > row["s1"] else "#e74c3c"
        ax.plot([row["s1"], row["s2"]], [i, i], color=color, linewidth=1.5, zorder=3)

    # Plot markers
    ax.scatter(tulu["s1"], y_pos, marker="o", s=50, color="#2c3e50",
               zorder=5, label="OLv2 (third-party)")
    ax.scatter(tulu["s2"], y_pos, marker="^", s=50, color="#8e44ad",
               zorder=5, label="HF Model Card (first-party)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(benchmarks, fontsize=9)
    ax.set_xlabel("Score", fontsize=10)
    ax.set_title("First-party vs. third-party scores: Tulu-3-8B",
                 fontsize=11, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")

    # Legend
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)

    # Line color legend annotation
    ax.text(0.97, 0.03,
            "Green: first-party > third-party\nRed: third-party > first-party",
            transform=ax.transAxes, fontsize=7, va="bottom", ha="right",
            color="#666666")

    plt.tight_layout()
    fig.savefig(f"{OUT}/figB_tulu3_first_vs_third.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/figB_tulu3_first_vs_third.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Figure B saved.")

    # ══════════════════════════════════════════════════════════════
    #  TASK 5 — Updated Table 6 (LaTeX)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("TASK 5 — Updated Table 6 (LaTeX)")
    print("=" * 60)

    # Sort by completeness descending, then by abs_delta descending
    table6 = data.sort_values(["completeness", "abs_delta"], ascending=[False, False]).copy()

    # Find row with largest delta
    max_delta_idx = table6["abs_delta"].idxmax()

    latex_lines = []
    latex_lines.append(r"\begin{table}[t]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{Collision pairs with metadata completeness and score divergence, sorted by completeness.}")
    latex_lines.append(r"\label{tab:collision_completeness}")
    latex_lines.append(r"\small")
    latex_lines.append(r"\begin{tabular}{llrrrrrrrrr}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"Model & Bench & $s_1$ & $s_2$ & H & S & C & T & P & Compl. & $|\Delta|$ \\")
    latex_lines.append(r"\midrule")

    for idx, row in table6.iterrows():
        bold = idx == max_delta_idx
        fmt = lambda x: f"\\textbf{{{x}}}" if bold else str(x)
        fmtf = lambda x, d: f"\\textbf{{{x:.{d}f}}}" if bold else f"{x:.{d}f}"

        line = (f"{fmt(row['model'])} & {fmt(row['benchmark'])} & "
                f"{fmtf(row['s1'], 2)} & {fmtf(row['s2'], 2)} & "
                f"{fmt(int(row['H']))} & {fmt(int(row['S']))} & "
                f"{fmt(int(row['C']))} & {fmt(int(row['T']))} & "
                f"{fmt(int(row['P']))} & "
                f"{fmt(int(row['completeness']))} & "
                f"{fmtf(row['abs_delta'], 2)} \\\\")
        latex_lines.append(line)

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")

    latex_table6 = "\n".join(latex_lines)
    print(latex_table6)

    with open(f"{OUT}/table6_updated.tex", "w") as f:
        f.write(latex_table6)
    print("\ntable6_updated.tex saved.")

    # ══════════════════════════════════════════════════════════════
    #  TASK 6 — Summary paragraph
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("TASK 6 — Summary Paragraph")
    print("=" * 60)

    paragraph = (
        f"To examine whether metadata documentation quality relates to cross-source score "
        f"agreement, we analyzed the {len(data)} independent collision pairs identified in "
        f"Tables~5 and~6, computing a per-pair metadata completeness score (sum of documented "
        f"fields across harness, n-shot, chain-of-thought, temperature, and prompt template; "
        f"range 0--5) and the absolute score divergence $|\\Delta|$ between the two sources. "
        f"Completeness varied from {int(data['completeness'].min())} (no fields documented) to "
        f"{int(data['completeness'].max())} (harness and n-shot documented), while $|\\Delta|$ "
        f"ranged from {data['abs_delta'].min():.2f} to {data['abs_delta'].max():.2f} percentage "
        f"points. A Spearman rank correlation yielded $\\rho = {rho:.3f}$ ($p = {p_val:.3f}$, "
        f"$n = 8$), suggesting a {direction} association between completeness and score "
        f"divergence---i.e., pairs with {'more' if rho < 0 else 'less'} complete metadata "
        f"documentation tended to exhibit {'smaller' if rho < 0 else 'larger'} cross-source "
        f"discrepancies. A directional analysis restricted to the five Tulu-3-8B pairs "
        f"(where OLv2 serves as a third-party source and the HF Model Card as a first-party "
        f"source) revealed a mean first-party advantage of {-mean_diff:+.3f} points "
        f"(range: [{-max_diff:+.2f}, {-min_diff:+.2f}]), with {n_negative} of 5 pairs showing "
        f"higher first-party scores, though the {'inconsistent' if not consistent else 'consistent'} "
        f"direction across benchmarks and the small magnitude caution against strong directional "
        f"claims. Given the small sample ($n = 8$), these results should be treated as "
        f"illustrative evidence of a plausible relationship between documentation practices and "
        f"reproducibility, rather than as definitive proof of a causal link."
    )

    print(paragraph)

    with open(f"{OUT}/analysis_paragraph.txt", "w") as f:
        f.write(paragraph)
    print("\nanalysis_paragraph.txt saved.")

    print("\n" + "=" * 60)
    print("[DONE] All collision-pair analysis tasks complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
