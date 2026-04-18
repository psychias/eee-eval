#!/usr/bin/env python3
"""
generate_all_figures.py — Publication-quality figures for the EEE dataset paper.

Reads:  data/aggregated/all_results.csv
        analysis_output/coverage_stats.csv
        analysis_output/collision_pairs.csv
        analysis_output/rank_instability.csv

Writes: figures/*.pdf  +  submission/*.png  (300 dpi, publication-ready)

Figures produced:
  fig1  — Dataset composition treemap / overview
  fig2  — Source × benchmark coverage heatmap
  fig3  — Metadata documentation rates (coverage bar chart)
  fig4  — Score distributions per benchmark (ridgeline / violin)
  fig5  — Developer ecosystem: top developers by benchmark coverage
  fig6  — Benchmark correlation heatmap (Spearman across models)
  fig7  — Model coverage landscape: #benchmarks vs #sources per model
  fig8  — Score comparison across sources (for overlapping model×benchmark)
  fig9  — n-shot distribution per benchmark
  fig10 — Dataset growth / source contribution (stacked bar)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent
AGG_CSV = ROOT / "data" / "aggregated" / "all_results.csv"
COV_CSV = ROOT / "analysis_output" / "coverage_stats.csv"
FIG_DIR = ROOT / "figures"
SUB_DIR = ROOT / "submission"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SUB_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Colorblind-friendly palette
SOURCE_COLORS = {
    "open_llm_leaderboard_v2": "#0072B2",
    "papers_with_code":        "#E69F00",
    "alpacaeval2":             "#009E73",
    "bfcl":                    "#D55E00",
    "bigcodebench":            "#CC79A7",
    "chatbot_arena":           "#56B4E9",
    "evalplus":                "#F0E442",
    "wildbench":               "#999999",
    "swe_bench":               "#882255",
    "mt_bench":                "#44AA99",
}

SOURCE_LABELS = {
    "open_llm_leaderboard_v2": "HF Open LLM v2",
    "papers_with_code":        "Papers with Code",
    "alpacaeval2":             "AlpacaEval 2",
    "bfcl":                    "BFCL v3",
    "bigcodebench":            "BigCodeBench",
    "chatbot_arena":           "Chatbot Arena",
    "evalplus":                "EvalPlus",
    "wildbench":               "WildBench",
    "swe_bench":               "SWE-Bench",
    "mt_bench":                "MT-Bench",
}


def pretty_source(s: str) -> str:
    return SOURCE_LABELS.get(s, s)


def save_fig(fig, name: str) -> None:
    pdf = FIG_DIR / f"{name}.pdf"
    png = SUB_DIR / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight", dpi=150)
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  OK {pdf.name}  +  {png.name}")


# ═══════════════════════════════════════════════════════════════════════════════
def load_data() -> pd.DataFrame:
    if not AGG_CSV.exists():
        print(f"ERROR: {AGG_CSV} not found. Run aggregate_results.py first.")
        sys.exit(1)
    df = pd.read_csv(AGG_CSV, low_memory=False)
    df.columns = df.columns.str.strip()
    df["score_num"] = pd.to_numeric(df["score"], errors="coerce")
    print(f"Loaded {len(df):,} records | "
          f"{df['model'].nunique():,} models | "
          f"{df['benchmark'].nunique()} benchmarks | "
          f"{df['source'].nunique()} sources")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Dataset composition overview (horizontal bar + pie inset)
# ═══════════════════════════════════════════════════════════════════════════════
def fig1_dataset_overview(df: pd.DataFrame) -> None:
    """Two-panel overview: (a) records per source, (b) benchmarks per source."""
    src_counts = df["source"].value_counts()
    sources = src_counts.index.tolist()
    colors = [SOURCE_COLORS.get(s, "#888888") for s in sources]
    labels = [pretty_source(s) for s in sources]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel (a) — records per source
    bars = ax1.barh(range(len(sources)), src_counts.values, color=colors,
                    edgecolor="white", linewidth=0.5)
    ax1.set_yticks(range(len(sources)))
    ax1.set_yticklabels(labels)
    ax1.invert_yaxis()
    ax1.set_xlabel("Number of evaluation records")
    ax1.set_title("(a) Records per source", fontweight="bold")
    # Annotate counts
    for i, (bar, v) in enumerate(zip(bars, src_counts.values)):
        ax1.text(v + src_counts.values.max() * 0.01, i,
                 f"{v:,}", va="center", fontsize=8, fontweight="bold")
    ax1.set_xlim(0, src_counts.values.max() * 1.15)

    # Panel (b) — unique benchmarks per source
    bench_per_src = df.groupby("source")["benchmark"].nunique().reindex(sources)
    bars2 = ax2.barh(range(len(sources)), bench_per_src.values, color=colors,
                     edgecolor="white", linewidth=0.5)
    ax2.set_yticks(range(len(sources)))
    ax2.set_yticklabels(labels)
    ax2.invert_yaxis()
    ax2.set_xlabel("Number of distinct benchmarks")
    ax2.set_title("(b) Benchmark coverage per source", fontweight="bold")
    for i, (bar, v) in enumerate(zip(bars2, bench_per_src.values)):
        ax2.text(v + 0.3, i, str(v), va="center", fontsize=8, fontweight="bold")
    ax2.set_xlim(0, bench_per_src.values.max() * 1.25)

    fig.suptitle(f"EEE Dataset Overview: {len(df):,} Records Across "
                 f"{df['source'].nunique()} Sources",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig1_dataset_overview")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Source x Benchmark coverage heatmap
# ═══════════════════════════════════════════════════════════════════════════════
def fig2_source_benchmark_heatmap(df: pd.DataFrame) -> None:
    """Binary heatmap: which sources cover which benchmarks."""
    ct = pd.crosstab(df["source"], df["benchmark"])
    # Sort: benchmarks by total coverage desc, sources by total records desc
    bench_order = ct.sum(axis=0).sort_values(ascending=False).index.tolist()
    src_order = ct.sum(axis=1).sort_values(ascending=False).index.tolist()
    ct = ct.reindex(index=src_order, columns=bench_order)

    # Log-scale for visibility
    ct_log = np.log10(ct.replace(0, np.nan))

    fig, ax = plt.subplots(figsize=(max(10, len(bench_order) * 0.5),
                                    max(4, len(src_order) * 0.55)))

    # Custom annotation: show actual counts
    annot = ct.copy().astype(str)
    annot[ct == 0] = ""

    sns.heatmap(ct_log, ax=ax, cmap="YlOrRd", linewidths=0.5,
                annot=annot, fmt="", annot_kws={"fontsize": 6},
                cbar_kws={"label": "log10(record count)"})

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    pretty_labels = [pretty_source(s) for s in src_order]
    ax.set_yticklabels(pretty_labels, fontsize=8)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Source x Benchmark Coverage Matrix\n"
                 "(cell values = record count; colour = log10 scale)",
                 fontweight="bold", pad=12)
    fig.tight_layout()
    save_fig(fig, "fig2_source_benchmark_heatmap")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Metadata documentation rates per source
# ═══════════════════════════════════════════════════════════════════════════════
def fig3_metadata_coverage(df: pd.DataFrame) -> None:
    """Grouped bar chart: documentation rate for key fields per source."""
    fields = ["shots", "temperature", "prompt_template", "harness",
              "chain_of_thought"]
    field_labels = ["n-shot", "Temperature", "Prompt\ntemplate", "Harness",
                    "Chain of\nthought"]

    sources = df["source"].value_counts().index.tolist()
    rates = {}
    for src in sources:
        sub = df[df["source"] == src]
        n = len(sub)
        src_rates = []
        for f in fields:
            filled = sub[f].notna() & (sub[f].astype(str).str.strip() != "")
            src_rates.append(filled.sum() / n * 100 if n > 0 else 0)
        rates[pretty_source(src)] = src_rates

    rate_df = pd.DataFrame(rates, index=field_labels).T

    fig, ax = plt.subplots(figsize=(10, max(5, len(sources) * 0.5)))
    rate_df.plot(kind="barh", ax=ax, width=0.8,
                 color=["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"])
    ax.set_xlabel("Documentation rate (%)")
    ax.set_title("Metadata Documentation Rates by Source\n"
                 "(% of records with non-empty field value)",
                 fontweight="bold", pad=12)
    ax.legend(title="Field", bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=8)
    ax.set_xlim(0, 105)
    ax.axvline(x=50, color="grey", linestyle="--", alpha=0.5, linewidth=0.8)
    fig.tight_layout()
    save_fig(fig, "fig3_metadata_coverage")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Score distributions per benchmark (violin + box)
# ═══════════════════════════════════════════════════════════════════════════════
def fig4_score_distributions(df: pd.DataFrame) -> None:
    """Violin plots of score distributions for top-15 benchmarks."""
    sub = df.dropna(subset=["score_num"]).copy()

    # Filter to benchmarks with enough data and reasonable range
    bench_counts = sub.groupby("benchmark")["score_num"].count()
    top_bench = bench_counts[bench_counts >= 20].sort_values(ascending=False).head(15).index.tolist()
    sub = sub[sub["benchmark"].isin(top_bench)]

    # Order by median score
    order = (sub.groupby("benchmark")["score_num"]
               .median()
               .reindex(top_bench)
               .sort_values(ascending=False)
               .index.tolist())

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.violinplot(data=sub, x="benchmark", y="score_num", order=order,
                   ax=ax, inner="box", cut=0, palette="Set2", linewidth=0.5)

    # Overlay n= annotations
    for i, bench in enumerate(order):
        n = len(sub[sub["benchmark"] == bench])
        ax.text(i, ax.get_ylim()[1] * 0.98, f"n={n}",
                ha="center", va="top", fontsize=6, fontstyle="italic",
                color="#555555")

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=40, ha="right", fontsize=8)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_title("Score Distributions by Benchmark (top 15 by sample size)",
                 fontweight="bold", pad=12)
    fig.tight_layout()
    save_fig(fig, "fig4_score_distributions")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Developer ecosystem: coverage heatmap
# ═══════════════════════════════════════════════════════════════════════════════
def fig5_developer_benchmark_heatmap(df: pd.DataFrame) -> None:
    """Heatmap: top developers vs top benchmarks, cell = # models evaluated."""
    # Filter known developers
    dev_df = df[df["developer"].notna() & (df["developer"] != "unknown")].copy()

    top_devs = dev_df["developer"].value_counts().head(15).index.tolist()
    top_bench = dev_df["benchmark"].value_counts().head(15).index.tolist()

    sub = dev_df[dev_df["developer"].isin(top_devs) & dev_df["benchmark"].isin(top_bench)]
    # Count unique models per developer-benchmark
    ct = sub.groupby(["developer", "benchmark"])["model"].nunique().unstack(fill_value=0)
    ct = ct.reindex(index=top_devs, columns=top_bench, fill_value=0)

    fig, ax = plt.subplots(figsize=(max(10, len(top_bench) * 0.6),
                                    max(5, len(top_devs) * 0.45)))
    sns.heatmap(ct, ax=ax, cmap="Blues", linewidths=0.5,
                annot=True, fmt="d", annot_kws={"fontsize": 7},
                cbar_kws={"label": "# unique models evaluated"})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    ax.set_title("Developer x Benchmark Evaluation Landscape\n"
                 "(cell = # distinct models evaluated per developer)",
                 fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    save_fig(fig, "fig5_developer_benchmark_heatmap")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Benchmark correlation matrix (Spearman)
# ═══════════════════════════════════════════════════════════════════════════════
def fig6_benchmark_correlation(df: pd.DataFrame) -> None:
    """Spearman rank correlation heatmap between benchmarks.

    For each model, pivot (model, benchmark) -> score, then compute
    pairwise Spearman correlations. Only includes benchmarks with >= 20
    shared models.
    """
    sub = df[["model", "benchmark", "score_num"]].dropna()

    # If multiple scores per (model, benchmark), take mean
    pivot = sub.groupby(["model", "benchmark"])["score_num"].mean().unstack()

    # Keep benchmarks with at least 20 models
    keep = pivot.columns[pivot.notna().sum() >= 20]
    pivot = pivot[keep]

    if len(keep) < 3:
        print("  SKIP fig6 - insufficient benchmark overlap for correlation")
        return

    corr = pivot.corr(method="spearman")

    # Cluster for readability
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform

    dist = 1 - corr.fillna(0).values
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2  # symmetrize
    dist = np.clip(dist, 0, None)
    try:
        link = linkage(squareform(dist), method="average")
        order = leaves_list(link)
        corr = corr.iloc[order, order]
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(max(8, len(keep) * 0.55),
                                    max(7, len(keep) * 0.5)))

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, ax=ax, mask=mask, cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, linewidths=0.3,
                annot=True, fmt=".2f", annot_kws={"fontsize": 6},
                cbar_kws={"label": "Spearman rho", "shrink": 0.8},
                square=True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
    ax.set_title("Benchmark Correlation Matrix (Spearman rho)\n"
                 "Lower triangle; clustered by similarity",
                 fontweight="bold", pad=12)
    fig.tight_layout()
    save_fig(fig, "fig6_benchmark_correlation")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7 — Model evaluation coverage (# benchmarks vs # sources)
# ═══════════════════════════════════════════════════════════════════════════════
def fig7_model_coverage_scatter(df: pd.DataFrame) -> None:
    """Scatter: how many benchmarks and sources each model appears in."""
    model_stats = df.groupby("model").agg(
        n_bench=("benchmark", "nunique"),
        n_source=("source", "nunique"),
        n_records=("score", "count"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        model_stats["n_bench"], model_stats["n_source"],
        s=np.clip(model_stats["n_records"] * 0.5, 5, 200),
        alpha=0.4, c=model_stats["n_records"],
        cmap="viridis", edgecolors="white", linewidth=0.3)

    ax.set_xlabel("Number of distinct benchmarks")
    ax.set_ylabel("Number of distinct sources")
    ax.set_title("Model Evaluation Coverage Landscape\n"
                 f"({len(model_stats):,} models; dot size = record count)",
                 fontweight="bold", pad=12)

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("Total evaluation records")

    # Annotate some notable outliers
    top_bench = model_stats.nlargest(5, "n_bench")
    for _, row in top_bench.iterrows():
        name = row["model"]
        if len(name) > 25:
            name = name[:22] + "..."
        ax.annotate(name, (row["n_bench"], row["n_source"]),
                    fontsize=5.5, alpha=0.7,
                    xytext=(5, 5), textcoords="offset points")

    # Add marginal histograms using twinx/twiny would be complex;
    # instead add summary stats as text
    ax.text(0.98, 0.02,
            f"Median: {model_stats['n_bench'].median():.0f} benchmarks, "
            f"{model_stats['n_source'].median():.0f} source(s)\n"
            f"Max: {model_stats['n_bench'].max()} benchmarks, "
            f"{model_stats['n_source'].max()} sources",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, bbox=dict(boxstyle="round,pad=0.3",
                                  fc="lightyellow", ec="gray", alpha=0.8))

    fig.tight_layout()
    save_fig(fig, "fig7_model_coverage_scatter")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 8 — n-shot distribution per benchmark
# ═══════════════════════════════════════════════════════════════════════════════
def fig8_nshot_distribution(df: pd.DataFrame) -> None:
    """Stacked bar chart showing n-shot configurations per benchmark."""
    sub = df.copy()
    sub["shots_str"] = sub["shots"].astype(str).str.strip()
    sub.loc[sub["shots_str"].isin(["", "nan", "None"]), "shots_str"] = "unknown"

    # Simplify shots to categories
    def categorize_shots(s):
        if s in ("unknown", "nan", "None", ""):
            return "Undocumented"
        try:
            v = int(float(s))
            if v == 0:
                return "0-shot"
            elif v <= 3:
                return "1-3 shot"
            elif v <= 5:
                return "5-shot"
            elif v <= 10:
                return "6-10 shot"
            else:
                return ">10 shot"
        except (ValueError, TypeError):
            return "Other"

    sub["shot_cat"] = sub["shots_str"].apply(categorize_shots)

    top_bench = sub["benchmark"].value_counts().head(15).index.tolist()
    sub = sub[sub["benchmark"].isin(top_bench)]

    ct = pd.crosstab(sub["benchmark"], sub["shot_cat"])
    # Normalize to %
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

    # Order by largest 0-shot fraction
    cat_order = ["0-shot", "1-3 shot", "5-shot", "6-10 shot", ">10 shot",
                 "Other", "Undocumented"]
    cat_order = [c for c in cat_order if c in ct_pct.columns]
    ct_pct = ct_pct.reindex(columns=cat_order, fill_value=0)

    colors = ["#0072B2", "#56B4E9", "#009E73", "#E69F00",
              "#D55E00", "#CC79A7", "#BBBBBB"][:len(cat_order)]

    fig, ax = plt.subplots(figsize=(12, 6))
    ct_pct.plot(kind="barh", stacked=True, ax=ax, color=colors,
                edgecolor="white", linewidth=0.3)
    ax.set_xlabel("% of records")
    ax.set_ylabel("")
    ax.set_title("n-shot Configuration Distribution per Benchmark",
                 fontweight="bold", pad=12)
    ax.legend(title="n-shot category", bbox_to_anchor=(1.02, 1),
              loc="upper left", fontsize=8)
    ax.set_xlim(0, 100)
    fig.tight_layout()
    save_fig(fig, "fig8_nshot_distribution")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 9 — Source contribution Sankey-style (stacked bar)
# ═══════════════════════════════════════════════════════════════════════════════
def fig9_source_contribution(df: pd.DataFrame) -> None:
    """Stacked bar showing source composition for each benchmark."""
    top_bench = df["benchmark"].value_counts().head(20).index.tolist()
    sub = df[df["benchmark"].isin(top_bench)]

    ct = pd.crosstab(sub["benchmark"], sub["source"])
    bench_order = ct.sum(axis=1).sort_values(ascending=False).index.tolist()
    src_order = ct.sum(axis=0).sort_values(ascending=False).index.tolist()
    ct = ct.reindex(index=bench_order, columns=src_order)

    colors = [SOURCE_COLORS.get(s, "#888888") for s in src_order]
    pretty_labels_src = [pretty_source(s) for s in src_order]

    fig, ax = plt.subplots(figsize=(12, 7))
    ct.plot(kind="barh", stacked=True, ax=ax, color=colors,
            edgecolor="white", linewidth=0.3, legend=False)
    ax.set_xlabel("Number of evaluation records")
    ax.set_ylabel("")
    ax.set_title("Source Contribution per Benchmark\n"
                 "(top 20 benchmarks by record count)",
                 fontweight="bold", pad=12)

    # Custom legend
    handles = [mpatches.Patch(color=c, label=l)
               for c, l in zip(colors, pretty_labels_src)]
    ax.legend(handles=handles, title="Source", bbox_to_anchor=(1.02, 1),
              loc="upper left", fontsize=7)

    fig.tight_layout()
    save_fig(fig, "fig9_source_contribution")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 10 — Top models comparison radar / multi-benchmark bar
# ═══════════════════════════════════════════════════════════════════════════════
def fig10_top_models_comparison(df: pd.DataFrame) -> None:
    """Grouped bar chart comparing top models across key benchmarks."""
    # Select benchmarks from Open LLM Leaderboard v2 (most standardized)
    ollm_benchmarks = ["IFEval", "GPQA", "MMLU-Pro", "BBH", "MATH-500", "MuSR"]
    sub = df[df["benchmark"].isin(ollm_benchmarks)].copy()
    sub = sub.dropna(subset=["score_num"])

    # Get top 10 models by mean score across these benchmarks
    model_means = (sub.groupby("model")["score_num"]
                     .agg(["mean", "count"])
                     .query("count >= 4")  # require at least 4 benchmarks
                     .nlargest(12, "mean"))

    top_models = model_means.index.tolist()
    sub = sub[sub["model"].isin(top_models)]

    pivot = sub.pivot_table(index="model", columns="benchmark",
                            values="score_num", aggfunc="mean")
    pivot = pivot.reindex(index=top_models, columns=ollm_benchmarks)

    fig, ax = plt.subplots(figsize=(14, 6))
    pivot.plot(kind="bar", ax=ax, width=0.85,
               color=["#0072B2", "#E69F00", "#009E73", "#D55E00",
                      "#CC79A7", "#56B4E9"])

    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_xticklabels([m[:30] for m in top_models],
                       rotation=35, ha="right", fontsize=8)
    ax.set_title("Top Models Across Open LLM Leaderboard v2 Benchmarks",
                 fontweight="bold", pad=12)
    ax.legend(title="Benchmark", bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=7)
    fig.tight_layout()
    save_fig(fig, "fig10_top_models_comparison")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 11 — Evaluation library adoption across sources
# ═══════════════════════════════════════════════════════════════════════════════
def fig11_eval_library_adoption(df: pd.DataFrame) -> None:
    """Pie chart + bar showing eval library distribution."""
    lib_counts = df["eval_library"].value_counts()
    # Remove empty
    lib_counts = lib_counts[lib_counts.index.astype(str).str.strip() != ""]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                    gridspec_kw={"width_ratios": [1, 1.3]})

    # Panel (a) — pie chart
    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(lib_counts)))
    wedges, texts, autotexts = ax1.pie(
        lib_counts.values, labels=lib_counts.index, autopct="%1.1f%%",
        colors=colors_pie, startangle=90,
        pctdistance=0.85, textprops={"fontsize": 7})
    for t in autotexts:
        t.set_fontsize(6)
    ax1.set_title("(a) Evaluation library\ndistribution", fontweight="bold")

    # Panel (b) — library per source (stacked)
    ct = pd.crosstab(df["source"], df["eval_library"])
    src_order = ct.sum(axis=1).sort_values(ascending=False).index.tolist()
    ct = ct.reindex(index=src_order)
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

    ct_pct.plot(kind="barh", stacked=True, ax=ax2,
                color=plt.cm.Set3(np.linspace(0, 1, ct_pct.shape[1])),
                edgecolor="white", linewidth=0.3)
    pretty_ylabels = [pretty_source(s) for s in src_order]
    ax2.set_yticklabels(pretty_ylabels)
    ax2.set_xlabel("% of records")
    ax2.set_title("(b) Library usage per source", fontweight="bold")
    ax2.legend(title="Library", bbox_to_anchor=(1.02, 1), loc="upper left",
               fontsize=6)
    ax2.set_xlim(0, 100)

    fig.suptitle("Evaluation Framework Adoption",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig11_eval_library_adoption")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 12 — Score scale heterogeneity (max_score distribution)
# ═══════════════════════════════════════════════════════════════════════════════
def fig12_score_scale_heterogeneity(df: pd.DataFrame) -> None:
    """Show the heterogeneity of score scales across benchmarks."""
    sub = df.dropna(subset=["score_num"]).copy()

    bench_stats = sub.groupby("benchmark").agg(
        min_score=("score_num", "min"),
        max_score=("score_num", "max"),
        median_score=("score_num", "median"),
        iqr_low=("score_num", lambda x: x.quantile(0.25)),
        iqr_high=("score_num", lambda x: x.quantile(0.75)),
        n=("score_num", "count")
    ).reset_index()

    # Filter to benchmarks with enough data
    bench_stats = bench_stats[bench_stats["n"] >= 10].sort_values(
        "median_score", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(5, len(bench_stats) * 0.35)))

    y = range(len(bench_stats))

    # Draw range bars
    for i, (_, row) in enumerate(bench_stats.iterrows()):
        ax.plot([row["min_score"], row["max_score"]], [i, i],
                color="#CCCCCC", linewidth=2, solid_capstyle="round", zorder=1)
        ax.plot([row["iqr_low"], row["iqr_high"]], [i, i],
                color="#4C72B0", linewidth=4, solid_capstyle="round", zorder=2)
        ax.scatter([row["median_score"]], [i], color="#D55E00",
                   s=40, zorder=3, edgecolors="white", linewidth=0.5)

    ax.set_yticks(range(len(bench_stats)))
    ax.set_yticklabels(bench_stats["benchmark"].values, fontsize=7)
    ax.set_xlabel("Score value")
    ax.set_title("Score Scale Heterogeneity Across Benchmarks\n"
                 "(grey = full range, blue = IQR, orange dot = median)",
                 fontweight="bold", pad=12)

    # Legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="#CCCCCC", lw=2, label="Full range"),
        Line2D([0], [0], color="#4C72B0", lw=4, label="IQR (25-75%)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#D55E00",
               markersize=7, label="Median"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7)

    fig.tight_layout()
    save_fig(fig, "fig12_score_heterogeneity")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 13 — Model family performance comparison
# ═══════════════════════════════════════════════════════════════════════════════
def fig13_model_family_comparison(df: pd.DataFrame) -> None:
    """Box plots comparing model families (developers) on key benchmarks."""
    # Only look at benchmarks with wide developer coverage
    dev_df = df[(df["developer"] != "unknown") & df["developer"].notna()].copy()
    dev_df = dev_df.dropna(subset=["score_num"])

    # Key benchmarks from Open LLM Leaderboard
    key_bench = ["MMLU-Pro", "GPQA", "BBH", "MATH-500", "IFEval", "MuSR"]
    dev_df = dev_df[dev_df["benchmark"].isin(key_bench)]

    top_devs = dev_df["developer"].value_counts().head(8).index.tolist()
    dev_df = dev_df[dev_df["developer"].isin(top_devs)]

    if len(dev_df) < 10:
        print("  SKIP fig13 - insufficient data for model family comparison")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharey=False)
    axes = axes.flatten()

    for idx, bench in enumerate(key_bench):
        if idx >= len(axes):
            break
        ax = axes[idx]
        bdata = dev_df[dev_df["benchmark"] == bench]
        if len(bdata) < 5:
            ax.set_visible(False)
            continue

        # Order developers by median score
        dev_order = (bdata.groupby("developer")["score_num"]
                         .median()
                         .sort_values(ascending=False)
                         .index.tolist())

        sns.boxplot(data=bdata, x="developer", y="score_num",
                    order=dev_order, ax=ax, palette="Set2", linewidth=0.6,
                    fliersize=2)
        ax.set_xticks(range(len(dev_order)))
        ax.set_xticklabels(dev_order, rotation=40, ha="right", fontsize=7)
        ax.set_xlabel("")
        ax.set_ylabel("Score")
        ax.set_title(bench, fontweight="bold", fontsize=10)

    fig.suptitle("Model Family Performance Comparison\n"
                 "(by developer, key benchmarks from Open LLM Leaderboard v2)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig13_model_family_comparison")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 14 — Cross-source score agreement (for overlapping models)
# ═══════════════════════════════════════════════════════════════════════════════
def fig14_cross_source_agreement(df: pd.DataFrame) -> None:
    """Scatter plot: pairwise score comparison for models in 2+ sources."""
    sub = df.dropna(subset=["score_num"]).copy()

    # Find (model, benchmark) combos from 2+ sources
    grouped = sub.groupby(["model", "benchmark"])
    pairs = []
    for (model, bench), grp in grouped:
        if grp["source"].nunique() < 2:
            continue
        source_scores = grp.groupby("source")["score_num"].mean()
        srcs = source_scores.index.tolist()
        for i in range(len(srcs)):
            for j in range(i + 1, len(srcs)):
                pairs.append({
                    "model": model, "benchmark": bench,
                    "source_a": srcs[i], "source_b": srcs[j],
                    "score_a": source_scores.iloc[i],
                    "score_b": source_scores.iloc[j],
                    "delta": abs(source_scores.iloc[i] - source_scores.iloc[j]),
                })

    if not pairs:
        print("  SKIP fig14 - no overlapping model x benchmark across sources")
        return

    pair_df = pd.DataFrame(pairs)
    print(f"  Found {len(pair_df)} cross-source comparison pairs "
          f"({pair_df['model'].nunique()} models, "
          f"{pair_df['benchmark'].nunique()} benchmarks)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel (a) — scatter: score_a vs score_b
    sc = ax1.scatter(pair_df["score_a"], pair_df["score_b"],
                     alpha=0.5, s=30, c="#0072B2", edgecolors="white",
                     linewidth=0.3)
    lims = [min(pair_df[["score_a","score_b"]].min()),
            max(pair_df[["score_a","score_b"]].max())]
    ax1.plot(lims, lims, "--", color="grey", linewidth=1, alpha=0.7)
    ax1.set_xlabel("Score (Source A)")
    ax1.set_ylabel("Score (Source B)")
    ax1.set_title("(a) Cross-source score agreement", fontweight="bold")
    from scipy import stats
    r, p = stats.spearmanr(pair_df["score_a"], pair_df["score_b"])
    ax1.text(0.05, 0.95, f"Spearman rho = {r:.3f}\np = {p:.2e}\nn = {len(pair_df)}",
             transform=ax1.transAxes, fontsize=8, va="top",
             bbox=dict(boxstyle="round", fc="lightyellow", ec="gray", alpha=0.8))

    # Panel (b) — histogram of |delta|
    ax2.hist(pair_df["delta"], bins=30, color="#E69F00", edgecolor="white",
             linewidth=0.5, alpha=0.8)
    ax2.axvline(pair_df["delta"].median(), color="#D55E00", linestyle="--",
                linewidth=1.5, label=f"Median = {pair_df['delta'].median():.2f}")
    ax2.set_xlabel("|Score delta| between sources")
    ax2.set_ylabel("Count")
    ax2.set_title("(b) Distribution of cross-source score differences",
                   fontweight="bold")
    ax2.legend(fontsize=8)

    fig.suptitle("Cross-Source Score Agreement for Overlapping Models",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig14_cross_source_agreement")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 15 — Schema completeness per source (radar chart)
# ═══════════════════════════════════════════════════════════════════════════════
def fig15_schema_completeness_radar(df: pd.DataFrame) -> None:
    """Radar (spider) chart showing schema field completeness per source."""
    fields = ["model", "benchmark", "score", "shots", "temperature",
              "prompt_template", "harness", "chain_of_thought", "eval_library",
              "developer"]
    field_labels = ["Model", "Benchmark", "Score", "n-shot", "Temp",
                    "Prompt tpl", "Harness", "CoT", "Eval lib", "Developer"]

    sources = df["source"].value_counts().head(8).index.tolist()

    # Compute completeness per source per field
    data = {}
    for src in sources:
        sub = df[df["source"] == src]
        n = len(sub)
        rates = []
        for f in fields:
            filled = sub[f].notna() & (sub[f].astype(str).str.strip() != "")
            rates.append(filled.sum() / n * 100 if n > 0 else 0)
        data[pretty_source(src)] = rates

    # Radar chart
    angles = np.linspace(0, 2 * np.pi, len(fields), endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    colors_radar = plt.cm.tab10(np.linspace(0, 1, len(sources)))
    for i, (src_name, rates) in enumerate(data.items()):
        values = rates + rates[:1]
        ax.plot(angles, values, 'o-', linewidth=1.5, label=src_name,
                color=colors_radar[i], markersize=3)
        ax.fill(angles, values, alpha=0.05, color=colors_radar[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(field_labels, fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=7)
    ax.set_title("Schema Field Completeness by Source\n(% of records documenting each field)",
                 fontweight="bold", pad=20)

    fig.tight_layout()
    save_fig(fig, "fig15_schema_completeness_radar")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 16 — Source complementarity analysis (overlap matrix)
# ═══════════════════════════════════════════════════════════════════════════════
def fig16_source_overlap(df: pd.DataFrame) -> None:
    """Heatmap showing model overlap (Jaccard index) between sources."""
    sources = df["source"].value_counts().index.tolist()
    n = len(sources)

    # Compute Jaccard index between all source pairs (model sets)
    source_models = {}
    for src in sources:
        source_models[src] = set(df[df["source"] == src]["model"].unique())

    jaccard = np.zeros((n, n))
    overlap_count = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            si, sj = source_models[sources[i]], source_models[sources[j]]
            inter = len(si & sj)
            union = len(si | sj)
            jaccard[i, j] = inter / union if union > 0 else 0
            overlap_count[i, j] = inter

    labels = [pretty_source(s) for s in sources]
    jaccard_df = pd.DataFrame(jaccard, index=labels, columns=labels)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                    gridspec_kw={"width_ratios": [1, 1]})

    # Panel (a) — Jaccard similarity heatmap
    mask = np.zeros_like(jaccard, dtype=bool)
    np.fill_diagonal(mask, True)
    sns.heatmap(jaccard_df, ax=ax1, cmap="YlOrRd", vmin=0, vmax=0.5,
                annot=True, fmt=".2f", annot_kws={"fontsize": 7},
                linewidths=0.5, mask=mask,
                cbar_kws={"label": "Jaccard index"})
    ax1.set_title("(a) Model Overlap (Jaccard index)\nbetween sources",
                   fontweight="bold")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right",
                         fontsize=7)
    ax1.set_yticklabels(ax1.get_yticklabels(), fontsize=7)

    # Panel (b) — Absolute overlap counts
    overlap_arr = overlap_count.copy()
    np.fill_diagonal(overlap_arr, 0)
    overlap_df = pd.DataFrame(overlap_arr, index=labels, columns=labels)
    annot_str = overlap_df.copy().astype(int).astype(str)
    annot_str = annot_str.replace("0", "")

    sns.heatmap(overlap_df, ax=ax2, cmap="Blues", linewidths=0.5,
                annot=annot_str, fmt="", annot_kws={"fontsize": 7},
                cbar_kws={"label": "# shared models"})
    ax2.set_title("(b) Shared Models Count\nbetween sources",
                   fontweight="bold")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right",
                         fontsize=7)
    ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=7)

    fig.suptitle("Source Complementarity: How Much Do Sources Overlap?",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig16_source_overlap")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 17 — Performance distribution by model size (if available)
# ═══════════════════════════════════════════════════════════════════════════════
def fig17_model_count_per_benchmark(df: pd.DataFrame) -> None:
    """Horizontal bar chart: number of unique models evaluated per benchmark,
    colored by the dominant source."""
    bench_stats = df.groupby("benchmark").agg(
        n_models=("model", "nunique"),
        n_records=("score", "count"),
        n_sources=("source", "nunique"),
    ).sort_values("n_models", ascending=True)

    # Dominant source per benchmark
    dom_src = df.groupby("benchmark")["source"].agg(
        lambda x: x.value_counts().idxmax()
    ).reindex(bench_stats.index)

    colors = [SOURCE_COLORS.get(s, "#888888") for s in dom_src.values]

    fig, ax = plt.subplots(figsize=(10, max(6, len(bench_stats) * 0.35)))
    bars = ax.barh(range(len(bench_stats)), bench_stats["n_models"].values,
                   color=colors, edgecolor="white", linewidth=0.3)

    ax.set_yticks(range(len(bench_stats)))
    ax.set_yticklabels(bench_stats.index, fontsize=7)
    ax.set_xlabel("Number of unique models evaluated")
    ax.set_title("Evaluation Breadth: Unique Models per Benchmark\n"
                 "(bar color = dominant source)",
                 fontweight="bold", pad=12)

    # Annotate n_sources on each bar
    for i, (n_mod, n_src) in enumerate(zip(bench_stats["n_models"],
                                            bench_stats["n_sources"])):
        ax.text(n_mod + bench_stats["n_models"].max() * 0.01, i,
                f"{n_mod:,} ({n_src}s)", va="center", fontsize=6)

    # Legend for source colors
    unique_srcs = dom_src.unique()
    handles = [mpatches.Patch(color=SOURCE_COLORS.get(s, "#888888"),
                              label=pretty_source(s))
               for s in unique_srcs]
    ax.legend(handles=handles, title="Dominant source", fontsize=6,
              loc="lower right")

    fig.tight_layout()
    save_fig(fig, "fig17_model_count_per_benchmark")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 18 — Dataset summary table (rendered as figure for paper inclusion)
# ═══════════════════════════════════════════════════════════════════════════════
def fig18_summary_table(df: pd.DataFrame) -> None:
    """Render a summary statistics table as a figure."""
    # Source-level stats
    stats = []
    for src in df["source"].value_counts().index:
        sub = df[df["source"] == src]
        n_rec = len(sub)
        n_mod = sub["model"].nunique()
        n_bench = sub["benchmark"].nunique()
        n_dev = sub[sub["developer"] != "unknown"]["developer"].nunique()
        lib = sub["eval_library"].mode().iloc[0] if len(sub) > 0 else ""

        # Metadata completeness
        fields = ["shots", "temperature", "prompt_template"]
        meta_pct = 0
        for f in fields:
            filled = sub[f].notna() & (sub[f].astype(str).str.strip() != "")
            meta_pct += filled.sum()
        meta_pct = meta_pct / (n_rec * len(fields)) * 100 if n_rec > 0 else 0

        stats.append([
            pretty_source(src), f"{n_rec:,}", str(n_mod), str(n_bench),
            str(n_dev), lib, f"{meta_pct:.1f}%"
        ])

    # Totals row
    stats.append([
        "TOTAL", f"{len(df):,}", str(df["model"].nunique()),
        str(df["benchmark"].nunique()),
        str(df[df["developer"] != "unknown"]["developer"].nunique()),
        f"{df['eval_library'].nunique()} libs",
        ""
    ])

    col_labels = ["Source", "Records", "Models", "Benchmarks",
                  "Developers", "Eval Library", "Meta %"]

    fig, ax = plt.subplots(figsize=(12, max(3, len(stats) * 0.4 + 1)))
    ax.axis("off")

    table = ax.table(
        cellText=stats, colLabels=col_labels,
        cellLoc="center", loc="center",
        colColours=["#E8E8E8"] * len(col_labels),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.6)

    # Bold the header and totals row
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#4C72B0")
            cell.set_text_props(color="white", fontweight="bold")
        elif row == len(stats):
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#F0F0F0")

    ax.set_title("EEE Dataset Summary Statistics",
                 fontsize=13, fontweight="bold", pad=20)
    fig.tight_layout()
    save_fig(fig, "fig18_summary_table")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 19 — Benchmark difficulty (score distribution as ridgeline)
# ═══════════════════════════════════════════════════════════════════════════════
def fig19_benchmark_ridgeline(df: pd.DataFrame) -> None:
    """Ridgeline (joy) plot of score distributions for top benchmarks,
    normalized to [0,1] scale for comparability."""
    sub = df.dropna(subset=["score_num"]).copy()

    # Normalize scores per benchmark to [0,1]
    bench_mins = sub.groupby("benchmark")["score_num"].transform("min")
    bench_maxs = sub.groupby("benchmark")["score_num"].transform("max")
    ranges = bench_maxs - bench_mins
    sub["score_norm"] = np.where(ranges > 0,
                                  (sub["score_num"] - bench_mins) / ranges,
                                  0.5)

    bench_counts = sub["benchmark"].value_counts()
    top_bench = bench_counts[bench_counts >= 50].head(15).index.tolist()
    sub = sub[sub["benchmark"].isin(top_bench)]

    # Sort by median normalized score
    order = (sub.groupby("benchmark")["score_norm"]
                .median()
                .sort_values()
                .index.tolist())

    # Plot
    n_bench = len(order)
    fig, ax = plt.subplots(figsize=(10, max(6, n_bench * 0.5)))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_bench))

    for i, bench in enumerate(order):
        bdata = sub[sub["benchmark"] == bench]["score_norm"].values
        from scipy.stats import gaussian_kde
        try:
            kde = gaussian_kde(bdata, bw_method=0.2)
        except Exception:
            continue
        x_grid = np.linspace(0, 1, 200)
        density = kde(x_grid)
        # Normalize density for display
        density = density / density.max() * 0.8

        ax.fill_between(x_grid, i, i + density, alpha=0.6, color=colors[i])
        ax.plot(x_grid, i + density, color="black", linewidth=0.5)

    ax.set_yticks(range(n_bench))
    ax.set_yticklabels(order, fontsize=7)
    ax.set_xlabel("Normalized score (0 = min, 1 = max per benchmark)")
    ax.set_ylim(-0.5, n_bench + 0.5)
    ax.set_xlim(0, 1)
    ax.set_title("Benchmark Difficulty Landscape\n"
                 "(score distributions normalized to [0,1]; ridge = density)",
                 fontweight="bold", pad=12)
    fig.tight_layout()
    save_fig(fig, "fig19_benchmark_ridgeline")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 20 — Evaluator relationship & source type composition
# ═══════════════════════════════════════════════════════════════════════════════
def fig20_source_type_composition(df: pd.DataFrame) -> None:
    """Source type and evaluator relationship breakdown."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel (a) — source_type if available from original JSONs
    # Use eval_library as proxy for source type:
    #   lm_eval = standardized framework, others = ad-hoc
    sub = df.copy()
    sub["framework_type"] = sub["eval_library"].apply(
        lambda x: "Standardized\n(lm_eval)" if str(x).strip() == "lm_eval"
        else ("Specialized\nFramework" if str(x).strip() not in ("", "unknown", "nan")
              else "Unknown /\nUndocumented"))

    ft_counts = sub["framework_type"].value_counts()
    colors_ft = ["#0072B2", "#E69F00", "#BBBBBB"][:len(ft_counts)]
    wedges, texts, autotexts = ax1.pie(
        ft_counts.values, labels=ft_counts.index,
        autopct=lambda p: f"{p:.1f}%\n({int(p*len(sub)/100):,})",
        colors=colors_ft, startangle=90,
        pctdistance=0.75, textprops={"fontsize": 8})
    for t in autotexts:
        t.set_fontsize(7)
    ax1.set_title("(a) Evaluation framework type\n(by record count)",
                   fontweight="bold")

    # Panel (b) — records per evaluator_relationship
    er_counts = sub.groupby(["source", "evaluator_relationship"]).size().unstack(
        fill_value=0)
    er_counts.index = [pretty_source(s) for s in er_counts.index]
    src_order = er_counts.sum(axis=1).sort_values(ascending=True).index.tolist()
    er_counts = er_counts.reindex(index=src_order)

    er_counts.plot(kind="barh", stacked=True, ax=ax2,
                   color=["#0072B2", "#E69F00", "#009E73", "#D55E00"],
                   edgecolor="white", linewidth=0.3)
    ax2.set_xlabel("Number of records")
    ax2.set_title("(b) Evaluator relationship by source", fontweight="bold")
    ax2.legend(title="Relationship", fontsize=7, loc="lower right")

    fig.suptitle("Evaluation Methodology Landscape",
                  fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig20_source_type_composition")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("EEE Dataset — Publication Figure Generation")
    print("=" * 60)

    df = load_data()
    print()

    generators = [
        ("Fig 1:  Dataset overview",             fig1_dataset_overview),
        ("Fig 2:  Source x benchmark heatmap",    fig2_source_benchmark_heatmap),
        ("Fig 3:  Metadata coverage",             fig3_metadata_coverage),
        ("Fig 4:  Score distributions",           fig4_score_distributions),
        ("Fig 5:  Developer x benchmark heatmap", fig5_developer_benchmark_heatmap),
        ("Fig 6:  Benchmark correlation",         fig6_benchmark_correlation),
        ("Fig 7:  Model coverage scatter",        fig7_model_coverage_scatter),
        ("Fig 8:  n-shot distribution",           fig8_nshot_distribution),
        ("Fig 9:  Source contribution per bench",  fig9_source_contribution),
        ("Fig 10: Top models comparison",          fig10_top_models_comparison),
        ("Fig 11: Eval library adoption",          fig11_eval_library_adoption),
        ("Fig 12: Score scale heterogeneity",      fig12_score_scale_heterogeneity),
        ("Fig 13: Model family comparison",        fig13_model_family_comparison),
        ("Fig 14: Cross-source agreement",         fig14_cross_source_agreement),
        ("Fig 15: Schema completeness radar",      fig15_schema_completeness_radar),
        ("Fig 16: Source overlap/complementarity",  fig16_source_overlap),
        ("Fig 17: Models per benchmark",             fig17_model_count_per_benchmark),
        ("Fig 18: Summary statistics table",         fig18_summary_table),
        ("Fig 19: Benchmark ridgeline",              fig19_benchmark_ridgeline),
        ("Fig 20: Source type composition",          fig20_source_type_composition),
    ]

    success = 0
    for label, func in generators:
        print(f"\n--- {label} ---")
        try:
            func(df)
            success += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"Done: {success}/{len(generators)} figures generated")
    print(f"  PDF: {FIG_DIR}")
    print(f"  PNG: {SUB_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
