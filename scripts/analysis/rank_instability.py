"""
rank_instability.py — Kendall τ rank correlation between source pairs.

For each pair of source directories sharing ≥3 models on the same benchmark,
computes Kendall τ_b (tie-corrected) with a 1000-resample bootstrap 95% CI.

Output: analysis_output/rank_instability.csv
Columns: source_a, source_b, benchmark, n_models, tau_b, ci_lo, ci_hi, pvalue, rq_type
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

DATA_DIR = _ROOT / "data"
OUT_DIR = _ROOT / "analysis_output"
BOOTSTRAP_RESAMPLES = 1000
MIN_SHARED_MODELS = 3
RNG = np.random.default_rng(42)


def load_scores() -> pd.DataFrame:
    """Load all (source, model_id, benchmark, score) rows."""
    rows = []
    for source_dir in sorted(DATA_DIR.iterdir()):
        if not source_dir.is_dir():
            continue
        source = source_dir.name
        for fpath in source_dir.rglob("*.json"):
            try:
                rec = json.loads(fpath.read_text())
            except Exception:
                continue
            model_id = rec.get("model_info", {}).get("id", "")
            for result in rec.get("evaluation_results", []):
                bench = result.get("evaluation_name", "")
                score = result.get("score_details", {}).get("score")
                if score is not None:
                    rows.append({
                        "source": source,
                        "model_id": model_id,
                        "benchmark": bench,
                        "score": float(score),
                    })
    return pd.DataFrame(rows)


def bootstrap_tau_ci(scores_a: np.ndarray, scores_b: np.ndarray,
                     n_resamples: int = BOOTSTRAP_RESAMPLES,
                     alpha: float = 0.05) -> tuple[float, float]:
    n = len(scores_a)
    if n < 3:
        return float("nan"), float("nan")
    taus = []
    for _ in range(n_resamples):
        idx = RNG.integers(0, n, size=n)
        t, _ = kendalltau(scores_a[idx], scores_b[idx], variant='b')
        if not np.isnan(t):
            taus.append(t)
    if len(taus) < 10:
        return float("nan"), float("nan")
    taus = np.array(taus)
    return float(np.percentile(taus, 100 * alpha / 2)), float(np.percentile(taus, 100 * (1 - alpha / 2)))


def classify_rq(source_a: str, source_b: str) -> str:
    """Classify source pair type for RQ annotation."""
    leaderboards = {"hfopenllm_v2", "chatbot_arena", "alpacaeval2", "mt_bench", "wildbench", "bigcodebench"}
    a_lb = any(lb in source_a for lb in leaderboards)
    b_lb = any(lb in source_b for lb in leaderboards)
    if a_lb and b_lb:
        return "leaderboard_vs_leaderboard"
    elif a_lb or b_lb:
        return "leaderboard_vs_paper"
    else:
        return "paper_vs_paper"


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading scores...")
    df = load_scores()
    print(f"  {len(df)} rows, {df['source'].nunique()} sources")

    sources = sorted(df["source"].unique())
    rows = []

    for src_a, src_b in combinations(sources, 2):
        df_a = df[df["source"] == src_a]
        df_b = df[df["source"] == src_b]
        shared_benchmarks = set(df_a["benchmark"].unique()) & set(df_b["benchmark"].unique())
        for bench in sorted(shared_benchmarks):
            # Take mean score per model in case of duplicates
            sub_a = (df_a[df_a["benchmark"] == bench]
                     .groupby("model_id")["score"].mean())
            sub_b = (df_b[df_b["benchmark"] == bench]
                     .groupby("model_id")["score"].mean())
            shared_models = sub_a.index.intersection(sub_b.index)
            if len(shared_models) < MIN_SHARED_MODELS:
                continue
            sa = sub_a.loc[shared_models].values
            sb = sub_b.loc[shared_models].values
            tau, pval = kendalltau(sa, sb, variant='b')
            ci_lo, ci_hi = bootstrap_tau_ci(sa, sb)
            rows.append({
                "source_a": src_a,
                "source_b": src_b,
                "benchmark": bench,
                "n_models": len(shared_models),
                "tau_b": round(float(tau), 4),
                "ci_lo": round(ci_lo, 4),
                "ci_hi": round(ci_hi, 4),
                "pvalue": round(float(pval), 4),
                "rq_type": classify_rq(src_a, src_b),
            })

    result = pd.DataFrame(rows)
    out_path = OUT_DIR / "rank_instability.csv"
    result.to_csv(out_path, index=False)
    print(f"  {len(result)} source-pair / benchmark combinations")
    print(f"Rank instability saved → {out_path}")
    if not result.empty:
        print("\nSample (sorted by tau_b):")
        print(result.sort_values("tau_b").head(10)[
            ["source_a", "source_b", "benchmark", "n_models", "tau_b", "ci_lo", "ci_hi"]
        ].to_string(index=False))


if __name__ == "__main__":
    main()
