"""
generate_figures.py — produce all paper figures from aggregated pipeline output.

Reads:  data/aggregated/all_results.csv
        data/aggregated/coverage_stats.json
Writes: data/figures/*.png  (300 dpi, publication-ready)

Figures produced (matching the doc-3 review):
  fig1_coverage_heatmap.png       — metadata documentation rates per benchmark
  fig2_score_delta_distributions.png — |Δ| split by harness-same vs. different
  fig3_collision_matrix.png       — source × model collision overlap
  fig4_forest_plot.png            — effect sizes (partial R²) per predictor
  fig5_coverage_power_projection.png — statistical power vs. new sources
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = pathlib.Path(__file__).resolve().parent.parent
AGG_DIR   = ROOT / "data" / "aggregated"
FIG_DIR   = ROOT / "data" / "figures"
CSV_PATH  = AGG_DIR / "all_results.csv"
COV_PATH  = AGG_DIR / "coverage_stats.json"

FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.15)
PALETTE = {
    "same":    "#4878CF",   # blue — harness same
    "diff":    "#D65F5F",   # red  — harness differs
    "high":    "#6ACC65",
    "medium":  "#F5C518",
    "low":     "#D65F5F",
}


# ─────────────────────────────────────────────────────────────────────────────
def load_data() -> tuple[pd.DataFrame, dict]:
    if not CSV_PATH.exists():
        print(f"  ERROR: {CSV_PATH} not found — run 'pipeline: 3 — aggregate results' first")
        sys.exit(1)
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df.columns = df.columns.str.strip()
    cov: dict = {}
    if COV_PATH.exists():
        cov = json.loads(COV_PATH.read_text())
    print(f"  Loaded {len(df)} records: "
          f"{df['model'].nunique()} models, "
          f"{df['benchmark'].nunique()} benchmarks, "
          f"{df['source'].nunique()} sources")
    return df, cov


# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 — Metadata documentation rates (grouped heatmap)
# ─────────────────────────────────────────────────────────────────────────────
def fig1_coverage_heatmap(df: pd.DataFrame, cov: dict) -> None:
    fields = ["shots", "temperature", "prompt_template", "harness"]
    benchmarks = df["benchmark"].value_counts().head(20).index.tolist()

    rates = []
    for b in benchmarks:
        sub = df[df["benchmark"] == b]
        n = len(sub)
        row = {"benchmark": b}
        for f in fields:
            filled = sub[f].notna() & (sub[f].astype(str).str.strip() != "")
            row[f] = filled.sum() / n if n > 0 else 0.0
        rates.append(row)
    heat_df = pd.DataFrame(rates).set_index("benchmark")[fields]

    fig, ax = plt.subplots(figsize=(10, max(5, len(benchmarks) * 0.45)))
    sns.heatmap(
        heat_df, ax=ax, vmin=0, vmax=1, cmap="RdYlGn",
        annot=True, fmt=".0%", linewidths=0.5,
        cbar_kws={"label": "Documentation rate"},
    )
    ax.set_title("Metadata documentation rates per benchmark (top 20)", pad=14)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    out = FIG_DIR / "fig1_coverage_heatmap.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — Score delta distributions: harness-same vs. harness-different
# ─────────────────────────────────────────────────────────────────────────────
def fig2_score_delta_distributions(df: pd.DataFrame) -> None:
    # Build collision pairs: same (model, benchmark), different sources
    needed = ["model", "benchmark", "score", "source", "harness"]
    sub = df[needed].dropna(subset=["score", "model", "benchmark"])
    sub["score"] = pd.to_numeric(sub["score"], errors="coerce")
    sub = sub.dropna(subset=["score"])

    pairs = []
    grouped = sub.groupby(["model", "benchmark"])
    for (model, bench), grp in grouped:
        grp = grp.reset_index(drop=True)
        if len(grp) < 2:
            continue
        for i in range(len(grp)):
            for j in range(i + 1, len(grp)):
                a, b = grp.iloc[i], grp.iloc[j]
                delta = abs(float(a["score"]) - float(b["score"]))
                harness_same = (
                    str(a["harness"]).strip().lower() ==
                    str(b["harness"]).strip().lower()
                ) and str(a["harness"]).strip() not in ("", "nan")
                pairs.append({
                    "model": model, "benchmark": bench,
                    "delta": delta,
                    "harness_condition": "Same harness" if harness_same else "Different harness",
                })
    if not pairs:
        print("  SKIP fig2 — no collision pairs found (need ≥2 sources per model×benchmark)")
        return

    pair_df = pd.DataFrame(pairs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    # Panel a — boxplot per benchmark
    top_benches = pair_df["benchmark"].value_counts().head(10).index
    box_data = pair_df[pair_df["benchmark"].isin(top_benches)]
    sns.boxplot(
        data=box_data, x="benchmark", y="delta", ax=ax1,
        palette="Blues", order=top_benches,
    )
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=40, ha="right", fontsize=9)
    ax1.set_title("(a) |Δ| spread per benchmark")
    ax1.set_xlabel("")
    ax1.set_ylabel("|Score delta|")

    # Panel b — violin: harness same vs different
    colors = [PALETTE["same"], PALETTE["diff"]]
    sns.violinplot(
        data=pair_df, x="harness_condition", y="delta", ax=ax2,
        palette=colors, inner="box", cut=0,
    )
    ax2.set_title("(b) |Δ| by harness condition")
    ax2.set_xlabel("")
    ax2.set_ylabel("|Score delta|")

    fig.suptitle("Score delta distributions across sources", fontsize=14, y=1.01)
    plt.tight_layout()
    out = FIG_DIR / "fig2_score_delta_distributions.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — Collision overlap matrix (sources × models)
# ─────────────────────────────────────────────────────────────────────────────
def fig3_collision_matrix(df: pd.DataFrame) -> None:
    top_sources = df["source"].value_counts().head(15).index.tolist()
    top_models  = df["model"].value_counts().head(20).index.tolist()

    sub = df[df["source"].isin(top_sources) & df["model"].isin(top_models)]
    matrix = sub.groupby(["source", "model"])["benchmark"].count().unstack(fill_value=0)
    # binarize: 1 = at least one collision exists
    matrix = (matrix > 0).astype(int)
    # reindex to keep top-N order
    matrix = matrix.reindex(index=top_sources, columns=top_models, fill_value=0)

    fig, ax = plt.subplots(figsize=(max(10, len(top_models) * 0.55),
                                    max(5, len(top_sources) * 0.45)))
    sns.heatmap(
        matrix, ax=ax, cmap="Blues", linewidths=0.4,
        cbar_kws={"label": "Source covers model"},
        annot=False,
    )
    ax.set_title("Source × model coverage matrix\n(blue = source has result for model)", pad=12)
    ax.set_xlabel("Model")
    ax.set_ylabel("Source")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    plt.tight_layout()
    out = FIG_DIR / "fig3_collision_matrix.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 — Forest plot of effect sizes (partial R² proxy per predictor)
# ─────────────────────────────────────────────────────────────────────────────
def fig4_forest_plot(df: pd.DataFrame) -> None:
    from scipy import stats as sp_stats

    needed = ["model", "benchmark", "score", "harness", "shots",
              "temperature", "prompt_template", "eval_library"]
    sub = df[needed].copy()
    sub["score"] = pd.to_numeric(sub["score"], errors="coerce")

    # Build collision pairs first
    pairs = []
    grouped = sub.groupby(["model", "benchmark"])
    for (model, bench), grp in grouped:
        grp = grp.dropna(subset=["score"]).reset_index(drop=True)
        if len(grp) < 2:
            continue
        for i in range(len(grp)):
            for j in range(i + 1, len(grp)):
                a, b_row = grp.iloc[i], grp.iloc[j]
                delta = abs(float(a["score"]) - float(b_row["score"]))
                pairs.append({
                    "delta": delta,
                    "harness_diff":  int(str(a["harness"]).strip() != str(b_row["harness"]).strip()),
                    "shots_diff":    int(str(a["shots"]).strip()   != str(b_row["shots"]).strip()),
                    "temp_diff":     int(str(a["temperature"]).strip() != str(b_row["temperature"]).strip()),
                    "pt_diff":       int(str(a["prompt_template"]).strip() != str(b_row["prompt_template"]).strip()),
                    "lib_diff":      int(str(a["eval_library"]).strip() != str(b_row["eval_library"]).strip()),
                })
    if not pairs:
        print("  SKIP fig4 — no collision pairs for effect size computation")
        return

    pair_df = pd.DataFrame(pairs)
    predictors = {
        "harness_diff":  "Evaluation harness",
        "shots_diff":    "Shot count",
        "temp_diff":     "Temperature",
        "pt_diff":       "Prompt template",
        "lib_diff":      "Eval library",
    }

    results_list = []
    for col, label in predictors.items():
        grp0 = pair_df[pair_df[col] == 0]["delta"].dropna()
        grp1 = pair_df[pair_df[col] == 1]["delta"].dropna()
        if len(grp0) < 2 or len(grp1) < 2:
            continue
        _, pval = sp_stats.mannwhitneyu(grp0, grp1, alternative="two-sided")
        effect = grp1.mean() - grp0.mean()
        # bootstrap 95% CI on effect
        rng = np.random.default_rng(42)
        boots = []
        for _ in range(500):
            s0 = rng.choice(grp0.values, size=len(grp0), replace=True)
            s1 = rng.choice(grp1.values, size=len(grp1), replace=True)
            boots.append(s1.mean() - s0.mean())
        ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
        results_list.append({
            "label": label, "effect": effect,
            "ci_lo": ci_lo, "ci_hi": ci_hi, "pval": pval,
            "n_diff": len(grp1),
        })

    if not results_list:
        print("  SKIP fig4 — insufficient data for effect size computation")
        return

    res_df = pd.DataFrame(results_list).sort_values("effect", ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(4, len(res_df) * 0.7)))
    y_pos = np.arange(len(res_df))
    colors = [PALETTE["diff"] if e > 0 else PALETTE["same"] for e in res_df["effect"]]

    ax.barh(y_pos, res_df["effect"], color=colors, alpha=0.75, height=0.5)
    ax.errorbar(
        res_df["effect"], y_pos,
        xerr=[res_df["effect"] - res_df["ci_lo"], res_df["ci_hi"] - res_df["effect"]],
        fmt="none", color="black", capsize=4, linewidth=1.5,
    )
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")

    # Significance markers
    for i, (_, row) in enumerate(res_df.iterrows()):
        marker = "***" if row["pval"] < 0.001 else ("**" if row["pval"] < 0.01 else ("*" if row["pval"] < 0.05 else "ns"))
        ax.text(res_df["effect"].max() * 1.05, y_pos[i], marker, va="center", fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(res_df["label"], fontsize=10)
    ax.set_xlabel("Mean |Δ| increase when predictor differs (95% CI)")
    ax.set_title("Effect of metadata differences on score delta\n(forest plot — longer bar = stronger predictor)", pad=12)
    plt.tight_layout()
    out = FIG_DIR / "fig4_forest_plot.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 5 — Statistical power projection vs. number of new sources
# ─────────────────────────────────────────────────────────────────────────────
def fig5_coverage_power_projection(df: pd.DataFrame) -> None:
    n_sources_range = np.arange(1, 20)
    # Simulate power using a simple paired-t power approximation:
    # effect size estimated from observed delta distribution
    sub = df[["model", "benchmark", "score", "source"]].copy()
    sub["score"] = pd.to_numeric(sub["score"], errors="coerce")
    deltas = []
    for (m, b), grp in sub.groupby(["model", "benchmark"]):
        grp = grp.dropna(subset=["score"])
        if len(grp) >= 2:
            scores = grp["score"].values
            for i in range(len(scores)):
                for j in range(i + 1, len(scores)):
                    deltas.append(abs(scores[i] - scores[j]))

    if not deltas:
        print("  SKIP fig5 — no delta data available for power projection")
        return

    observed_delta = np.mean(deltas)
    observed_std   = np.std(deltas) if np.std(deltas) > 0 else 0.01

    from scipy import stats as sp_stats
    alpha = 0.05
    powers = []
    for n in n_sources_range:
        # non-centrality parameter for paired t
        ncp = (observed_delta / observed_std) * np.sqrt(n)
        df_t = n - 1 if n > 1 else 1
        t_crit = sp_stats.t.ppf(1 - alpha / 2, df_t)
        power = 1 - sp_stats.nct.cdf(t_crit, df_t, ncp)
        powers.append(float(np.clip(power, 0, 1)))

    pct_covered = [min(100, n / df["source"].nunique() * 100) for n in n_sources_range]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: power vs. new sources
    ax1.plot(n_sources_range, powers, "o-", color="#4878CF", linewidth=2)
    ax1.axhline(0.80, color="grey", linestyle="--", linewidth=1, label="0.80 threshold")
    ax1.fill_between(n_sources_range, powers, alpha=0.12, color="#4878CF")
    target_n = next((n for n, p in zip(n_sources_range, powers) if p >= 0.80), None)
    if target_n:
        ax1.axvline(target_n, color=PALETTE["diff"], linestyle=":", linewidth=1.5,
                    label=f"Target: n={target_n}")
    ax1.set_xlabel("Number of comparison sources")
    ax1.set_ylabel("Statistical power (1−β)")
    ax1.set_title("(a) Power vs. new sources")
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=9)

    # Right: power vs. metadata coverage %
    ax2.plot(pct_covered, powers, "s-", color="#6ACC65", linewidth=2)
    ax2.axhline(0.80, color="grey", linestyle="--", linewidth=1, label="0.80 threshold")
    ax2.set_xlabel("Metadata coverage (%)")
    ax2.set_ylabel("Statistical power (1−β)")
    ax2.set_title("(b) Power vs. coverage %")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=9)

    fig.suptitle("Statistical power projection", fontsize=14, y=1.02)
    plt.tight_layout()
    out = FIG_DIR / "fig5_coverage_power_projection.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("── Loading aggregated data ──────────────────────────────────────────")
    df, cov = load_data()

    print("── Generating figures ───────────────────────────────────────────────")
    fig1_coverage_heatmap(df, cov)
    fig2_score_delta_distributions(df)
    fig3_collision_matrix(df)
    fig4_forest_plot(df)
    fig5_coverage_power_projection(df)

    print(f"\n✓ All figures written to {FIG_DIR}/")
    print("  Open data/figures/ in your file explorer to review.")


if __name__ == "__main__":
    main()
