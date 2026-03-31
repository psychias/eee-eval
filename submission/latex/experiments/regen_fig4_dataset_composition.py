"""
Regenerate fig4_dataset_composition with correct model count (5,672).

The original figure used df['model'].nunique() which gives 5,753 (display names).
The correct count is df['model_id'].nunique() = 5,672 (canonical identifiers).
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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

SOURCE_COLORS = {
    "open_llm_leaderboard_v2": "#0072B2",
    "papers_with_code":        "#E69F00",
    "alpacaeval2":             "#009E73",
    "bfcl":                    "#D55E00",
    "bigcodebench":            "#CC79A7",
    "chatbot_arena":           "#56B4E9",
    "evalplus":                "#F0E442",
    "hf_model_card":           "#117733",
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
    "hf_model_card":           "HF Model Card",
    "wildbench":               "WildBench",
    "swe_bench":               "SWE-Bench",
    "mt_bench":                "MT-Bench",
}

def pretty_source(s):
    return SOURCE_LABELS.get(s, s)

ROOT = Path(__file__).resolve().parent.parent.parent.parent
AGG_CSV = ROOT / "data" / "aggregated" / "all_results.csv"
OUT_DIR = Path(__file__).resolve().parent.parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not AGG_CSV.exists():
    print(f"ERROR: {AGG_CSV} not found.")
    sys.exit(1)

df = pd.read_csv(AGG_CSV, low_memory=False)
df.columns = df.columns.str.strip()

# Use model_id for canonical model count (5,672), not model (5,753)
n_models = df["model_id"].nunique()
n_records = len(df)
n_sources = df["source"].nunique()
n_benchmarks = df["benchmark"].nunique()
print(f"Loaded {n_records:,} records | {n_models:,} models (model_id) | "
      f"{n_benchmarks} benchmarks | {n_sources} sources")

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

# Corrected suptitle: use model_id count (5,672)
fig.suptitle(f"EEE Dataset Overview: {n_records:,} Records, "
             f"{n_models:,} Models, {n_benchmarks} Benchmarks "
             f"Across {n_sources} Sources",
             fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()

for fmt in ("pdf", "png"):
    path = OUT_DIR / f"fig4_dataset_composition.{fmt}"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    print(f"  Saved: {path}")

plt.close(fig)
print("Done.")
