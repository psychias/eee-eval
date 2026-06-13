#!/usr/bin/env python3
"""
Recompute all key statistics from the EEE paper after incorporating
new arxiv_extraction_general data (~100 papers).

Outputs: updated_statistics.txt
"""

import csv
import json
import os
import re
import sys
import warnings
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# Force UTF-8 output even when piped on Windows
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent.parent


# ── Inline normalization (eee.normalization not available) ─────────
def normalize_model_id(raw: str) -> str:
    """Lowercase, strip org prefix, collapse whitespace.

    ArXiv papers report 'Llama-3.1-8B-Instruct'; leaderboards report
    'meta-llama/Llama-3.1-8B-Instruct'. Stripping the org/ prefix and
    lowercasing makes them match.
    """
    s = str(raw).strip()
    # Strip HF-style org prefix: 'meta-llama/Llama-3.1-8B' -> 'Llama-3.1-8B'
    if "/" in s:
        s = s.rsplit("/", 1)[-1]
    # Strip parenthesized suffixes added by some leaderboards: '(Prompt)', '(FC)'
    s = re.sub(r"\s*\((?:Prompt|FC|Chat)\)\s*$", "", s, flags=re.IGNORECASE)
    s = s.lower()
    # Normalize underscores to hyphens (papers use Gemma_2_9B, HF uses Gemma-2-9B)
    s = s.replace("_", "-")
    s = re.sub(r"\s+", "-", s)
    return s


def normalize_benchmark_for_matching(raw: str) -> str:
    """Normalize benchmark names: lowercase, strip shot specs, strip parenthesized suffixes."""
    s = str(raw).strip().lower()
    # Remove e.g. "(0-shot)" or "(5-Shot)" suffix
    s = re.sub(r"\s*\(\d+-shot\)\s*$", "", s, flags=re.IGNORECASE)
    # Remove trailing shot specs like " 0-shot", " 5-shot"
    s = re.sub(r"\s+\d+-shot\s*$", "", s, flags=re.IGNORECASE)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    # Normalize hyphens/underscores in benchmark names to spaces for alias lookup
    s_lookup = s.replace("-", " ").replace("_", " ").strip()
    # Benchmark alias mapping — map known variants to canonical forms
    _BENCH_ALIASES = {
        "bigbench hard": "bbh",
        "big bench hard": "bbh",
        "gpqa diamond": "gpqa-diamond",
        "gpqa d": "gpqa-diamond",
    }
    s = _BENCH_ALIASES.get(s_lookup, s)
    # Normalize common aliases
    s = re.sub(r"^humaneval\+?\s*$", "humaneval+", s)
    s = re.sub(r"^mbpp\+?\s*$", "mbpp+", s)
    return s

# ═══════════════════════════════════════════════════════════════
#  STEP 1 — Re-aggregate: load ALL JSONs (existing + new arxiv)
# ═══════════════════════════════════════════════════════════════

# ── Post-hoc benchmark name fixup for stale JSON files ──────────────────
_BENCHMARK_FIXUP = {
    "ARC-C": "ARC-Challenge", "ARC-E": "ARC-Easy",
    "ARC-c": "ARC-Challenge", "ARC-e": "ARC-Easy",
    "arc-c": "ARC-Challenge", "arc-e": "ARC-Easy",
    # BigBench-Hard variants → BBH
    "BigBench-Hard": "BBH", "bigbench-hard": "BBH",
    "Big-Bench Hard": "BBH", "big-bench hard": "BBH",
    "Big-Bench-Hard": "BBH", "big-bench-hard": "BBH",
    "Bigbench Hard": "BBH", "bigbench hard": "BBH",
    # GPQA variants → canonical
    "GPQA Diamond": "GPQA-Diamond", "GPQA-D": "GPQA-Diamond",
    "gpqa diamond": "GPQA-Diamond", "gpqa-d": "GPQA-Diamond",
}



def main():
    print("Step 1: Re-aggregating all data (existing + new arxiv papers)...")

    DATA_DIR = ROOT / "data"
    rows = []
    skip_dirs = {"aggregated", "arxiv_ids_test.txt", "arxiv_ids_full.txt", "arxiv_ids.txt"}

    for f in DATA_DIR.rglob("*.json"):
        if "aggregated" in str(f):
            continue
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
            mi = rec.get("model_info", {})
            src = rec.get("source_metadata", {})
            lib = rec.get("eval_library", {})

            for er in rec.get("evaluation_results", []):
                mc = er.get("metric_config", {})
                sd = er.get("score_details", {})
                gc = er.get("generation_config", {})
                ga = gc.get("generation_args") or {}
                ad = gc.get("additional_details") or {}
                sd_details = sd.get("details") or {}

                row = {
                    "model": mi.get("name", ""),
                    "model_id": mi.get("id", ""),
                    "developer": mi.get("developer", ""),
                    "parameter_count": mi.get("parameter_count", ""),
                    "benchmark": _BENCHMARK_FIXUP.get(er.get("evaluation_name", ""), er.get("evaluation_name", "")),
                    "score": sd.get("score", ""),
                    "lower_is_better": mc.get("lower_is_better", ""),
                    "min_score": mc.get("min_score", ""),
                    "max_score": mc.get("max_score", ""),
                    "metric_name": mc.get("metric_name", ""),
                    "shots": ga.get("shots", ""),
                    "temperature": ga.get("temperature", ""),
                    "top_p": ga.get("top_p", ""),
                    "chain_of_thought": ga.get("chain_of_thought", ""),
                    "prompt_template": ga.get("prompt_template", ""),
                    "harness": ad.get("harness", "") or lib.get("name", ""),
                    "scoring_method": ad.get("scoring_method", ""),
                    "source": ad.get("source", src.get("source_name", "")),
                    "source_name": src.get("source_name", ""),
                    "evaluator_relationship": src.get("evaluator_relationship", ""),
                    "eval_library": lib.get("name", ""),
                    "eval_library_version": lib.get("version", ""),
                    "evaluation_id": rec.get("evaluation_id", ""),
                    "file": str(f.relative_to(ROOT)),
                }
                rows.append(row)
        except Exception:
            continue

    df = pd.DataFrame(rows)
    print(f"  Total records loaded (before filter): {len(df):,}")

    # ── Exclude PWC data ──────────────────────────────────────────
    df = df[df["source"] != "papers_with_code"].reset_index(drop=True)
    print(f"  After excluding PWC: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")

    # ── Fix OLv2 GPQA-Diamond metric mismatch ─────────────────────
    # OLv2 reports GPQA-Diamond using acc_norm (normalized by answer-option byte
    # length) but mislabels it "accuracy". The max OLv2 GPQA-Diamond score is ~29
    # while papers report raw accuracy up to 92. Relabel the benchmark so these
    # incomparable metrics don't create spurious cross-source collisions.
    _gpqa_olv2_mask = (
        (df["benchmark"] == "GPQA-Diamond") &
        (df["source"] == "open_llm_leaderboard_v2")
    )
    n_gpqa_relabeled = _gpqa_olv2_mask.sum()
    df.loc[_gpqa_olv2_mask, "benchmark"] = "GPQA-Diamond (acc_norm)"
    df.loc[_gpqa_olv2_mask, "metric_name"] = "acc_norm"
    print(f"  Relabeled {n_gpqa_relabeled:,} OLv2 GPQA-Diamond records → GPQA-Diamond (acc_norm)")


    # ═══════════════════════════════════════════════════════════════
    #  STEP 1b — Score normalization (fix 0-1 vs 0-100 scale mismatch)
    # ═══════════════════════════════════════════════════════════════
    #
    # Some papers report accuracy as 0-1 (e.g. 0.78) while most report
    # as percentage (e.g. 78.0). We detect and rescale 0-1 scores to
    # percentage for benchmarks where the majority of scores are > 1.
    # This prevents fake 77-pt "conflicts" that are just scale artefacts.

    print("Step 1b: Normalizing score scales (0-1 -> 0-100 where appropriate)...")

    df["score_num"] = pd.to_numeric(df["score"], errors="coerce")

    # Build per-benchmark scale profile: what fraction of scores are > 1?
    _bench_norm_tmp = df["benchmark"].apply(normalize_benchmark_for_matching)
    n_rescaled = 0
    for bench in _bench_norm_tmp.unique():
        mask = _bench_norm_tmp == bench
        bench_scores = df.loc[mask, "score_num"].dropna()
        if len(bench_scores) < 2:
            continue
        n_above_1 = (bench_scores.abs() > 1.0).sum()
        n_in_01 = ((bench_scores >= 0) & (bench_scores <= 1.0)).sum()
        frac_above = n_above_1 / len(bench_scores)
        # If the majority (>60%) of scores for this benchmark are > 1,
        # then scores in [0,1] are likely on a 0-1 scale and should be × 100
        if frac_above >= 0.6 and n_in_01 > 0:
            rescale_mask = mask & (df["score_num"] >= 0) & (df["score_num"] <= 1.0)
            n_rescaled += rescale_mask.sum()
            df.loc[rescale_mask, "score"] = (df.loc[rescale_mask, "score_num"] * 100).round(4)
            df.loc[rescale_mask, "score_num"] = df.loc[rescale_mask, "score_num"] * 100

    print(f"  Rescaled {n_rescaled} scores from 0-1 to 0-100 scale")

    # Extract paper name and extraction bucket from file path for arxiv records.
    # Path: data/arxiv_extraction_general/{bucket}/{paper_name}/{model}/{bench}/file.json
    # Segment indices after split on 'arxiv_extraction_general': [0]='', [1]=bucket, [2]=paper
    def _parse_arxiv_path(x):
        s = str(x)
        if "arxiv_extraction_general" not in s:
            return "", ""
        parts = s.split("arxiv_extraction_general")[1].split(os.sep)
        bucket = parts[1] if len(parts) > 1 else ""
        paper = parts[2] if len(parts) > 2 else ""
        return paper, bucket

    df[["paper", "extraction_bucket"]] = df["file"].apply(
        lambda x: pd.Series(_parse_arxiv_path(x))
    )

    # Records in model_name_errors / score_errors are known-bad extractions.
    # Exclude them from all analysis; they'd produce artefactual conflicts.
    _error_buckets = {"model_name_errors", "score_errors"}
    n_before_bucket_filter = len(df)
    df = df[~df["extraction_bucket"].isin(_error_buckets)].reset_index(drop=True)
    n_error_excluded = n_before_bucket_filter - len(df)
    print(f"  Excluded {n_error_excluded:,} records from error buckets "
          f"({', '.join(sorted(_error_buckets))})")
    print(f"  Records after exclusion: {len(df):,}")


    # ═══════════════════════════════════════════════════════════════
    #  STEP 2 — Identify old vs new data
    # ═══════════════════════════════════════════════════════════════

    # NOTE: We previously loaded df_old from the aggregated CSV, but Step 1
    # overwrites that CSV.  For a meaningful old-vs-new comparison we'd need
    # a snapshot saved *before* this run.  Instead we simply note the current
    # dataset IS the latest version and skip the self-comparison.
    df_old = df.copy()  # placeholder — no real "old" available
    print(f"\n  (No prior snapshot for comparison — deltas will show +0)")

    # The new arxiv data has source = "arxiv_html_llm"
    arxiv_mask = df["source"].str.contains("arxiv", case=False, na=False)
    df_arxiv_new = df[arxiv_mask]
    df_leaderboard = df[~arxiv_mask]
    print(f"  Leaderboard records: {len(df_leaderboard):,}")
    print(f"  New arxiv records: {len(df_arxiv_new):,}")


    # ═══════════════════════════════════════════════════════════════
    #  HELPER FUNCTIONS
    # ═══════════════════════════════════════════════════════════════

    def is_valid(series):
        """Non-null, non-empty, non-'unknown' values."""
        return (series.notna() &
                (series.astype(str).str.strip() != "") &
                (series.astype(str).str.lower() != "unknown"))


    def normalise_benchmark(raw):
        return normalize_benchmark_for_matching(str(raw))


    def normalise_model(raw):
        return normalize_model_id(str(raw))


    SOURCE_LABELS = {
        "open_llm_leaderboard_v2": "OLv2",
        "papers_with_code": "PWC",
        "alpacaeval2": "AlpacaEval2",
        "chatbot_arena": "Arena",
        "bigcodebench": "BigCodeBench",
        "evalplus": "EvalPlus",
        "bfcl": "BFCL",
        "wildbench": "WildBench",
        "swe_bench": "SWE-bench",
        "mt_bench": "MT-Bench",
        "hf_model_card": "HF-Card",
        "arxiv_html_llm": "ArXiv-LLM",
    }


    def src_label(s):
        return SOURCE_LABELS.get(s, s)


    # ═══════════════════════════════════════════════════════════════
    #  STEP 3 — Compute all statistics
    # ═══════════════════════════════════════════════════════════════

    output_lines = []


    def out(line=""):
        output_lines.append(line)
        print(line)


    out("=" * 76)
    out("  EEE UPDATED STATISTICS — With New ArXiv Data (Excluding PWC)")
    out("=" * 76)
    out()

    # ──────────────────────────────────────────────────────────────
    #  Section A: Dataset Overview
    # ──────────────────────────────────────────────────────────────

    out("─" * 76)
    out("  A. DATASET OVERVIEW")
    out("─" * 76)
    out()

    n_total = len(df)
    n_models = df["model_id"].nunique()
    n_benchmarks = df["benchmark"].nunique()
    n_sources = df["source"].nunique()
    sources_list = sorted(df["source"].unique())

    out(f"  Total records:       {n_total:>10,}")
    out(f"  Unique models:       {n_models:>10,}")
    out(f"  Unique benchmarks:   {n_benchmarks:>10,}")
    out(f"  Sources:             {n_sources:>10}")
    out()

    # Comparison with old
    out(f"  Change from previous:")
    out(f"    Records:     {len(df_old):,} → {n_total:,}  (+{n_total - len(df_old):,})")
    out(f"    Models:      {df_old['model_id'].nunique()} → {n_models}  (+{n_models - df_old['model_id'].nunique()})")
    out(f"    Benchmarks:  {df_old['benchmark'].nunique()} → {n_benchmarks}  (+{n_benchmarks - df_old['benchmark'].nunique()})")
    out(f"    Sources:     {df_old['source'].nunique()} → {n_sources}  (+{n_sources - df_old['source'].nunique()})")
    out()

    out("  Records per source:")
    for src, count in df["source"].value_counts().items():
        out(f"    {src_label(src):20s} {count:>8,}")
    out()

    # ArXiv new data breakdown
    if len(df_arxiv_new) > 0:
        arxiv_sources = df_arxiv_new["source"].unique()
        out(f"  New ArXiv data breakdown:")
        out(f"    ArXiv records:     {len(df_arxiv_new):,}")
        out(f"    ArXiv models:      {df_arxiv_new['model_id'].nunique()}")
        out(f"    ArXiv benchmarks:  {df_arxiv_new['benchmark'].nunique()}")
        # Count papers from the already-computed paper column
        arxiv_papers = df_arxiv_new["paper"].loc[df_arxiv_new["paper"] != ""].nunique()
        out(f"    ArXiv papers:      {arxiv_papers}")
        out()


    # ──────────────────────────────────────────────────────────────
    #  Section B: Metadata Coverage (H1 — Metadata Gap)
    # ──────────────────────────────────────────────────────────────

    out("─" * 76)
    out("  B. METADATA COVERAGE (H1 — Metadata Gap)")
    out("─" * 76)
    out()

    fields = ["shots", "temperature", "prompt_template", "harness",
              "chain_of_thought", "scoring_method"]

    out("  Overall coverage (combined dataset):")
    out(f"  {'Field':25s} {'Filled':>8s} {'Total':>8s} {'Pct':>8s}")
    out(f"  {'─' * 25} {'─' * 8} {'─' * 8} {'─' * 8}")
    for field in fields:
        if field in df.columns:
            filled = is_valid(df[field]).sum()
            pct = 100 * filled / len(df)
            out(f"  {field:25s} {filled:>8,} {len(df):>8,} {pct:>7.1f}%")
        else:
            out(f"  {field:25s}     N/A")
    out()

    # Coverage in old data vs new
    out("  Coverage comparison — Old (leaderboard-only) vs New (combined):")
    out(f"  {'Field':25s} {'Old %':>8s} {'New %':>8s} {'Delta':>8s}")
    out(f"  {'─' * 25} {'─' * 8} {'─' * 8} {'─' * 8}")
    for field in fields:
        if field in df_old.columns and field in df.columns:
            old_pct = 100 * is_valid(df_old[field]).sum() / len(df_old)
            new_pct = 100 * is_valid(df[field]).sum() / len(df)
            delta = new_pct - old_pct
            sign = "+" if delta > 0 else ""
            out(f"  {field:25s} {old_pct:>7.1f}% {new_pct:>7.1f}% {sign}{delta:>6.1f}pp")
    out()

    # Per-source coverage
    out("  Per-source coverage:")
    out(f"  {'Source':20s} {'N':>7s} {'shots':>8s} {'temp':>8s} {'prompt':>8s} {'harness':>8s} {'CoT':>8s}")
    out(f"  {'─' * 20} {'─' * 7} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8}")
    for src in sorted(df["source"].unique()):
        sub = df[df["source"] == src]
        n = len(sub)
        vals = {}
        for field in ["shots", "temperature", "prompt_template", "harness", "chain_of_thought"]:
            if field in sub.columns:
                vals[field] = f"{100 * is_valid(sub[field]).sum() / n:.1f}%"
            else:
                vals[field] = "N/A"
        out(f"  {src_label(src):20s} {n:>7,} {vals['shots']:>8s} {vals['temperature']:>8s} "
            f"{vals['prompt_template']:>8s} {vals['harness']:>8s} {vals['chain_of_thought']:>8s}")
    out()

    # ArXiv-specific coverage vs leaderboard
    if len(df_arxiv_new) > 0:
        out("  ArXiv papers vs Leaderboards — metadata documentation rate:")
        out(f"  {'Field':25s} {'Leaderboard':>12s} {'ArXiv':>12s}")
        out(f"  {'─' * 25} {'─' * 12} {'─' * 12}")
        for field in fields:
            if field in df.columns:
                lb_pct = 100 * is_valid(df_leaderboard[field]).sum() / max(1, len(df_leaderboard))
                ax_pct = 100 * is_valid(df_arxiv_new[field]).sum() / max(1, len(df_arxiv_new))
                out(f"  {field:25s} {lb_pct:>11.1f}% {ax_pct:>11.1f}%")
        out()


    # ──────────────────────────────────────────────────────────────
    #  Section C: Cross-Source Collisions (H2 — Structural Fragmentation)
    # ──────────────────────────────────────────────────────────────

    out("─" * 76)
    out("  C. CROSS-SOURCE COLLISIONS (H2 — Structural Fragmentation)")
    out("─" * 76)
    out()

    # Normalize
    df["model_id_norm"] = df["model_id"].apply(normalise_model)
    df["benchmark_norm"] = df["benchmark"].apply(normalise_benchmark)


    def _safe_eq(a, b):
        a, b = str(a).strip(), str(b).strip()
        if not a or not b or a == "nan" or b == "nan":
            return False
        return a == b


    def _most_common(series):
        counts = Counter(series.dropna().astype(str).tolist())
        return counts.most_common(1)[0][0] if counts else ""


    # Deduplicate within source
    dedup_rows = []
    for (source, mid_norm, bench_norm), grp in df.groupby(
        ["source", "model_id_norm", "benchmark_norm"], sort=False
    ):
        rep = grp.iloc[0]
        dedup_rows.append({
            "source": source,
            "model_id": rep["model_id"],
            "model_id_norm": mid_norm,
            "benchmark": rep["benchmark"],
            "benchmark_norm": bench_norm,
            "score": float(pd.to_numeric(grp["score"], errors="coerce").median()),
            "n_shot": _most_common(grp["shots"]),
            "cot": _most_common(grp["chain_of_thought"]),
            "harness": _most_common(grp["harness"]),
            "prompt_template": _most_common(grp["prompt_template"]),
        })
    df_dedup = pd.DataFrame(dedup_rows)

    out(f"  Records after within-source dedup: {len(df_dedup):,}")

    # Find cross-source pairs
    multi = df_dedup.groupby(["model_id_norm", "benchmark_norm"])["source"].nunique()
    n_multi = (multi >= 2).sum()
    out(f"  (model, benchmark) pairs in ≥2 sources: {n_multi}")

    # Build collision pairs
    collision_pairs = []
    for (mid_norm, bench_norm), group in df_dedup.groupby(["model_id_norm", "benchmark_norm"]):
        sources = sorted(group["source"].unique())
        if len(sources) < 2:
            continue
        src_rows = {}
        for src in sources:
            src_rows[src] = group[group["source"] == src].iloc[0]
        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                sa, sb = sources[i], sources[j]
                ra, rb = src_rows[sa], src_rows[sb]
                score_a = float(ra["score"]) if pd.notna(ra["score"]) else 0
                score_b = float(rb["score"]) if pd.notna(rb["score"]) else 0
                delta = round(score_a - score_b, 6)

                same_harness = _safe_eq(ra["harness"], rb["harness"])
                same_shots = _safe_eq(ra["n_shot"], rb["n_shot"])
                same_cot = _safe_eq(ra["cot"], rb["cot"])

                # Skip citation duplicates (same score, no known config difference)
                is_cite_dup = (abs(delta) < 1e-9 and
                               not (_safe_eq(ra["harness"], "x") and not same_harness) and
                               not (_safe_eq(ra["n_shot"], "x") and not same_shots))
                # More precisely: skip if delta ≈ 0 and no field *differs* knowably
                harness_differs = (_safe_eq(ra["harness"], rb["harness"]) is False and
                                   str(ra["harness"]).strip() not in ("", "nan", "unknown") and
                                   str(rb["harness"]).strip() not in ("", "nan", "unknown"))
                shots_differs = (_safe_eq(ra["n_shot"], rb["n_shot"]) is False and
                                 str(ra["n_shot"]).strip() not in ("", "nan", "unknown") and
                                 str(rb["n_shot"]).strip() not in ("", "nan", "unknown"))
                cot_differs = (_safe_eq(ra["cot"], rb["cot"]) is False and
                               str(ra["cot"]).strip() not in ("", "nan", "unknown") and
                               str(rb["cot"]).strip() not in ("", "nan", "unknown"))

                if abs(delta) < 1e-9 and not harness_differs and not shots_differs and not cot_differs:
                    continue  # citation duplicate

                collision_pairs.append({
                    "model_id_norm": mid_norm,
                    "benchmark_norm": bench_norm,
                    "source_a": sa,
                    "source_b": sb,
                    "score_a": score_a,
                    "score_b": score_b,
                    "delta": delta,
                    "same_harness": same_harness,
                    "same_shots": same_shots,
                    "same_cot": same_cot,
                    "config_match": same_harness and same_shots and same_cot,
                })

    collisions = pd.DataFrame(collision_pairs)
    n_collisions = len(collisions)
    out(f"  Collision pairs (excl. citation duplicates): {n_collisions}")

    if n_collisions > 0:
        genuine = collisions[collisions["delta"].abs() > 1e-9]
        n_genuine = len(genuine)
        n_method_diff = n_collisions - n_genuine
        out(f"    With score difference (|Δ| > 0):  {n_genuine}")
        out(f"    Same score, method differs:        {n_method_diff}")

        if n_genuine > 0:
            ds = genuine["delta"].abs()
            out(f"\n  Score delta statistics (genuine collisions):")
            out(f"    Mean |Δ|:    {ds.mean():.2f}")
            out(f"    Median |Δ|:  {ds.median():.2f}")
            out(f"    Max |Δ|:     {ds.max():.2f}")
            out(f"    |Δ| > 1:     {(ds > 1).sum()}")
            out(f"    |Δ| > 5:     {(ds > 5).sum()}")
            out(f"    |Δ| > 10:    {(ds > 10).sum()}")

        out(f"\n  Collisions by source pair:")
        src_pairs = collisions.groupby(["source_a", "source_b"]).agg(
            n_pairs=("delta", "size"),
            n_score_diff=("delta", lambda x: (x.abs() > 1e-9).sum()),
            mean_abs_delta=("delta", lambda x: x.abs().mean()),
            max_abs_delta=("delta", lambda x: x.abs().max()),
        ).reset_index().sort_values("n_pairs", ascending=False)
        for _, r in src_pairs.iterrows():
            out(f"    {src_label(r['source_a']):20s} × {src_label(r['source_b']):15s}: "
                f"{int(r['n_pairs']):>3d} pairs, {int(r['n_score_diff']):>3d} score-diff, "
                f"mean|Δ|={r['mean_abs_delta']:.2f}, max|Δ|={r['max_abs_delta']:.2f}")

        # Top collisions
        top = collisions.sort_values("delta", key=abs, ascending=False).head(10)
        out(f"\n  Top 10 collisions by |Δ|:")
        for _, r in top.iterrows():
            out(f"    {r['model_id_norm'][:40]:40s} | {r['benchmark_norm']:15s} | "
                f"{src_label(r['source_a']):12s} vs {src_label(r['source_b']):12s} | "
                f"Δ={r['delta']:+.2f}")
    else:
        out("  No collision pairs found.")
    out()


    # ──────────────────────────────────────────────────────────────
    #  Section D: Permutation Null Model
    # ──────────────────────────────────────────────────────────────

    out("─" * 76)
    out("  D. PERMUTATION NULL MODEL")
    out("─" * 76)
    out()


    def count_collisions(data):
        grouped = data.groupby(["model_id_norm", "benchmark_norm"])["source"].nunique()
        return (grouped >= 2).sum()


    def permutation_test(data, n_iter=200, seed=42):
        """Permutation test using integer-coded keys for speed.
        Shuffle model IDs within each source, count cross-source collisions."""
        rng = np.random.default_rng(seed)
        null_counts = np.zeros(n_iter, dtype=int)

        # Encode strings as integer codes for fast hashing
        all_models = data["model_id_norm"].astype(str).unique()
        all_benches = data["benchmark_norm"].astype(str).unique()
        model_to_int = {m: i for i, m in enumerate(all_models)}
        bench_to_int = {b: i for i, b in enumerate(all_benches)}
        n_benches = len(all_benches)

        # Pre-compute per-source integer arrays
        source_arrays = []
        for src_idx, (src, grp) in enumerate(data.groupby("source")):
            mids = np.array([model_to_int[m] for m in grp["model_id_norm"].astype(str)], dtype=np.int32)
            bids = np.array([bench_to_int[b] for b in grp["benchmark_norm"].astype(str)], dtype=np.int32)
            source_arrays.append((mids, bids, src_idx))

        for i in range(n_iter):
            if (i + 1) % 50 == 0:
                print(f"    Permutation {i+1}/{n_iter}...")

            # Use a dict[int, set] where key = model_int * n_benches + bench_int
            pair_sources: dict = {}
            for model_ids, bench_ids, src_idx in source_arrays:
                ids = model_ids.copy()
                rng.shuffle(ids)
                for j in range(len(ids)):
                    key = int(ids[j]) * n_benches + int(bench_ids[j])
                    s = pair_sources.get(key)
                    if s is None:
                        pair_sources[key] = 1 << src_idx  # bitmask
                    else:
                        pair_sources[key] = s | (1 << src_idx)

            # Count keys with 2+ sources (popcount > 1)
            cnt = sum(1 for v in pair_sources.values() if v & (v - 1))
            null_counts[i] = cnt

        return null_counts


    # Run on dedup data
    obs_all = count_collisions(df_dedup)
    df_dedup_no_olv2 = df_dedup[df_dedup["source"] != "open_llm_leaderboard_v2"]
    obs_no_olv2 = count_collisions(df_dedup_no_olv2)

    out(f"  Observed collisions (all sources):       {obs_all}")
    out(f"  Observed collisions (excl. OLv2):        {obs_no_olv2}")

    print("  Running permutation test (all sources, n=200)...")
    null_all = permutation_test(df_dedup, n_iter=200)
    print("  Running permutation test (excl. OLv2, n=200)...")
    null_no_olv2 = permutation_test(df_dedup_no_olv2, n_iter=200)


    def perm_stats(observed, null_dist):
        mean_n = null_dist.mean()
        sd_n = null_dist.std()
        ci_lo = np.percentile(null_dist, 2.5)
        ci_hi = np.percentile(null_dist, 97.5)
        z = (observed - mean_n) / sd_n if sd_n > 0 else np.inf
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        return mean_n, sd_n, ci_lo, ci_hi, z, p


    stats_all = perm_stats(obs_all, null_all)
    stats_no_olv2 = perm_stats(obs_no_olv2, null_no_olv2)

    out(f"\n  Permutation test (n=200 iterations):")
    out(f"  {'Subset':20s} {'Obs':>6s} {'Mean':>8s} {'SD':>6s} {'95% CI':>16s} {'z':>8s} {'p':>12s}")
    out(f"  {'─' * 20} {'─' * 6} {'─' * 8} {'─' * 6} {'─' * 16} {'─' * 8} {'─' * 12}")
    out(f"  {'All sources':20s} {obs_all:>6d} {stats_all[0]:>8.1f} {stats_all[1]:>6.1f} "
        f"[{stats_all[2]:.1f}, {stats_all[3]:.1f}]{' ':>3s} {stats_all[4]:>8.2f} {stats_all[5]:>12.2e}")
    out(f"  {'Excluding OLv2':20s} {obs_no_olv2:>6d} {stats_no_olv2[0]:>8.1f} {stats_no_olv2[1]:>6.1f} "
        f"[{stats_no_olv2[2]:.1f}, {stats_no_olv2[3]:.1f}]{' ':>3s} {stats_no_olv2[4]:>8.2f} {stats_no_olv2[5]:>12.2e}")
    out()


    # ──────────────────────────────────────────────────────────────
    #  Section E: OLv2 Isolation Analysis
    # ──────────────────────────────────────────────────────────────

    out("─" * 76)
    out("  E. OLv2 ISOLATION ANALYSIS")
    out("─" * 76)
    out()

    sources = sorted(df["source"].unique())
    model_sets = {src: set(df[df["source"] == src]["model_id"].unique()) for src in sources}
    bench_sets = {src: set(df[df["source"] == src]["benchmark"].unique()) for src in sources}

    olv2_models = model_sets.get("open_llm_leaderboard_v2", set())
    other_models = set()
    for src in sources:
        if src != "open_llm_leaderboard_v2":
            other_models |= model_sets[src]
    olv2_only = olv2_models - other_models
    olv2_shared = olv2_models & other_models

    olv2_records = len(df[df["source"] == "open_llm_leaderboard_v2"])
    olv2_pct = 100 * olv2_records / len(df)

    out(f"  OLv2 records:             {olv2_records:,} ({olv2_pct:.1f}% of dataset)")
    out(f"  OLv2 unique models:       {len(olv2_models):,}")
    out(f"  OLv2-exclusive models:    {len(olv2_only):,} ({100 * len(olv2_only) / max(1, len(olv2_models)):.1f}%)")
    out(f"  OLv2 models shared:       {len(olv2_shared):,}")
    out()

    # Jaccard
    out("  Jaccard similarity (model sets) — top 20 pairs:")
    jaccard_pairs = []
    for s1, s2 in combinations(sources, 2):
        inter = len(model_sets[s1] & model_sets[s2])
        union = len(model_sets[s1] | model_sets[s2])
        j = inter / union if union > 0 else 0
        jaccard_pairs.append((s1, s2, j, inter))
    jaccard_pairs.sort(key=lambda x: -x[2])
    for s1, s2, j, inter in jaccard_pairs[:20]:
        out(f"    {src_label(s1):20s} × {src_label(s2):15s}: J={j:.4f} (shared={inter})")
    out()

    # Benchmark overlap
    out("  Benchmark overlap — top 20 pairs:")
    bench_pairs = []
    for s1, s2 in combinations(sources, 2):
        shared = len(bench_sets[s1] & bench_sets[s2])
        bench_pairs.append((s1, s2, shared))
    bench_pairs.sort(key=lambda x: -x[2])
    for s1, s2, shared in bench_pairs[:20]:
        out(f"    {src_label(s1):20s} × {src_label(s2):15s}: {shared} shared benchmarks")
    out()


    # ──────────────────────────────────────────────────────────────
    #  Section F: Subset Sensitivity Analysis
    # ──────────────────────────────────────────────────────────────

    out("─" * 76)
    out("  F. SUBSET SENSITIVITY ANALYSIS")
    out("─" * 76)
    out()

    subsets = {
        "All": df,
        "- OLv2": df[df["source"] != "open_llm_leaderboard_v2"],
        "- PWC": df[df["source"] != "papers_with_code"],
        "- OLv2 - PWC": df[~df["source"].isin(["open_llm_leaderboard_v2", "papers_with_code"])],
        "Leaderboards only": df_leaderboard,
        "ArXiv only": df_arxiv_new,
    }

    out(f"  {'Subset':20s} {'Records':>9s} {'Models':>8s} {'Bench':>8s} {'Harness':>9s} {'N-shot':>9s} "
        f"{'Temp':>9s} {'Prompt':>9s}")
    out(f"  {'─' * 20} {'─' * 9} {'─' * 8} {'─' * 8} {'─' * 9} {'─' * 9} {'─' * 9} {'─' * 9}")
    for name, sub in subsets.items():
        if len(sub) == 0:
            continue
        n = len(sub)
        nm = sub["model_id"].nunique()
        nb = sub["benchmark"].nunique()
        h = f"{100 * is_valid(sub['harness']).sum() / n:.1f}%" if "harness" in sub.columns else "N/A"
        s = f"{100 * is_valid(sub['shots']).sum() / n:.1f}%" if "shots" in sub.columns else "N/A"
        t = f"{100 * is_valid(sub['temperature']).sum() / n:.1f}%" if "temperature" in sub.columns else "N/A"
        p = f"{100 * is_valid(sub['prompt_template']).sum() / n:.1f}%" if "prompt_template" in sub.columns else "N/A"
        out(f"  {name:20s} {n:>9,} {nm:>8} {nb:>8} {h:>9s} {s:>9s} {t:>9s} {p:>9s}")
    out()


    # ──────────────────────────────────────────────────────────────
    #  Section G: PWC Artefact Analysis
    # ──────────────────────────────────────────────────────────────

    out("─" * 76)
    out("  G. PWC ARTEFACT ANALYSIS")
    out("─" * 76)
    out()

    pwc = df[df["source"] == "papers_with_code"].copy()
    out(f"  PWC records: {len(pwc)}")

    if len(pwc) > 0:
        pwc["score_num"] = pd.to_numeric(pwc["score"], errors="coerce")
        artefact_a = pwc[pwc["score_num"] > 1000]
        score_str = pwc["score"].astype(str)
        artefact_a2 = pwc[score_str.str.contains(r"\d+[Bb]", na=False)]
        prompt_str = pwc["prompt_template"].astype(str).str.lower()
        artefact_b_mask = pd.Series(False, index=pwc.index)
        for pat in ["cot", "chat", "instruct", "-v", "template"]:
            artefact_b_mask |= prompt_str.str.contains(pat, na=False)
        artefact_b_mask &= is_valid(pwc["prompt_template"])
        artefact_b = pwc[artefact_b_mask]
        dup_mask = pwc.duplicated(subset=["model_id", "benchmark", "score"], keep=False)
        artefact_c = pwc[dup_mask]

        all_artefact_idx = set(artefact_a.index) | set(artefact_a2.index) | set(artefact_b.index) | set(artefact_c.index)
        n_artefacts = len(all_artefact_idx)
        artefact_pct = 100 * n_artefacts / len(pwc)

        out(f"  Artefact A (score > 1000):         {len(artefact_a)}")
        out(f"  Artefact A2 (param pattern):       {len(artefact_a2)}")
        out(f"  Artefact B (variant-tag leakage):  {len(artefact_b)}")
        out(f"  Artefact C (exact duplicates):     {len(artefact_c)}")
        out(f"  Total unique artefacts:            {n_artefacts} ({artefact_pct:.1f}%)")
        out(f"  Clean records:                     {len(pwc) - n_artefacts}")
    else:
        out("  No PWC data found.")
    out()


    # ──────────────────────────────────────────────────────────────
    #  Section H: Cross-Resource Conflicts (all resources vs all, scores normalized)
    # ──────────────────────────────────────────────────────────────

    out("─" * 76)
    out("  H. CROSS-RESOURCE CONFLICTS (all resources compared, scores normalized)")
    out("─" * 76)
    out()
    out("  Each arxiv paper and each leaderboard is treated as an independent")
    out("  resource. We find every (model, benchmark) pair reported by 2+ resources")
    out("  with different scores.")
    out()

    df["model_id_norm"] = df["model_id"].apply(normalise_model)
    df["benchmark_norm"] = df["benchmark"].apply(normalise_benchmark)

    # Create a unified "resource" column:
    #   - For leaderboard data: the source name (e.g. "open_llm_leaderboard_v2")
    #   - For arxiv data: the paper name (e.g. "OLMo__Accelerating_the_Science_...")
    df["resource"] = df.apply(
        lambda r: r["paper"] if r["paper"] else r["source"], axis=1
    )

    # Pretty labels for resources
    def resource_label(r):
        if r in SOURCE_LABELS:
            return SOURCE_LABELS[r]
        # It's a paper name — shorten
        return r.replace("_", " ")[:50]

    n_resources = df["resource"].nunique()
    out(f"  Total unique resources: {n_resources}")
    out(f"    Leaderboard resources: {df[~df['source'].str.contains('arxiv', case=False, na=False)]['resource'].nunique()}")
    out(f"    ArXiv paper resources: {df[df['source'].str.contains('arxiv', case=False, na=False)]['resource'].nunique()}")
    out()

    # ── H.1 Find all cross-resource conflicts ─────────────────────
    # For each (model_norm, benchmark_norm), aggregate across resources.
    # Take the median score per resource to collapse within-resource duplicates.
    per_resource = (
        df.assign(score_num=pd.to_numeric(df["score"], errors="coerce"))
        .dropna(subset=["score_num"])
        .groupby(["model_id_norm", "benchmark_norm", "resource"])
        .agg(
            score_median=("score_num", "median"),
            score_mean=("score_num", "mean"),
            n_records=("score_num", "size"),
            source=("source", "first"),
        )
        .reset_index()
    )

    # Find (model, benchmark) pairs present in 2+ resources
    pair_resource_count = per_resource.groupby(["model_id_norm", "benchmark_norm"])["resource"].nunique()
    multi_resource_pairs = pair_resource_count[pair_resource_count >= 2].index

    conflict_rows = []
    for (mid_norm, bench_norm) in multi_resource_pairs:
        grp = per_resource[
            (per_resource["model_id_norm"] == mid_norm) &
            (per_resource["benchmark_norm"] == bench_norm)
        ].sort_values("score_median")

        resources_list = grp["resource"].tolist()
        scores = grp["score_median"].values
        score_range = scores.max() - scores.min()

        # Build per-resource detail
        resource_details = []
        for _, row in grp.iterrows():
            resource_details.append({
                "resource": row["resource"],
                "source": row["source"],
                "score": row["score_median"],
                "n_records": int(row["n_records"]),
            })

        conflict_rows.append({
            "model_id_norm": mid_norm,
            "benchmark_norm": bench_norm,
            "n_resources": len(resources_list),
            "resources": resources_list,
            "resource_details": resource_details,
            "score_min": scores.min(),
            "score_max": scores.max(),
            "score_range": score_range,
            "scores": sorted(scores.tolist()),
        })

    xr_conflicts = pd.DataFrame(conflict_rows)
    # Only keep pairs where scores actually differ (range > 0)
    xr_diff = xr_conflicts[xr_conflicts["score_range"] > 0.01].copy()

    out(f"  H.1 -- Overview:")
    out(f"    (model, benchmark) pairs in 2+ resources: {len(xr_conflicts)}")
    out(f"    Pairs with score difference (range > 0.01): {len(xr_diff)}")
    out(f"    Pairs with identical scores across resources: {len(xr_conflicts) - len(xr_diff)}")
    if len(xr_diff) > 0:
        out(f"    Unique models in conflicts:     {xr_diff['model_id_norm'].nunique()}")
        out(f"    Unique benchmarks in conflicts: {xr_diff['benchmark_norm'].nunique()}")
    out()

    if len(xr_diff) > 0:
        # ── H.2 Score range distribution ───────────────────────────
        out(f"  H.2 -- Score range distribution:")
        rv = xr_diff["score_range"]
        out(f"    range (0, 1] pt:     {((rv > 0) & (rv <= 1)).sum():>4d}")
        out(f"    range (1, 5] pt:     {((rv > 1) & (rv <= 5)).sum():>4d}")
        out(f"    range (5, 10] pt:    {((rv > 5) & (rv <= 10)).sum():>4d}")
        out(f"    range (10, 20] pt:   {((rv > 10) & (rv <= 20)).sum():>4d}")
        out(f"    range (20, 50] pt:   {((rv > 20) & (rv <= 50)).sum():>4d}")
        out(f"    range > 50 pt:       {(rv > 50).sum():>4d}")
        out(f"    Mean range:          {rv.mean():.2f}")
        out(f"    Median range:        {rv.median():.2f}")
        out()

        # ── H.3 Distribution by number of resources ───────────────
        out(f"  H.3 -- How many resources per conflict:")
        for n, cnt in xr_diff["n_resources"].value_counts().sort_index().items():
            out(f"    {n} resources: {cnt} conflicts")
        out()

        # ── H.4 Top 30 cross-resource conflicts by range ──────────
        out(f"  H.4 -- Top 30 cross-resource conflicts by score range:")
        out(f"    {'Model':<35s} | {'Benchmark':<18s} | {'#R':>3s} | {'Range':>7s} | {'Min':>8s} | {'Max':>8s} | Resources")
        out(f"    {'-' * 130}")
        for _, r in xr_diff.sort_values("score_range", ascending=False).head(30).iterrows():
            res_str = ", ".join(resource_label(x)[:25] for x in r["resources"][:4])
            if len(r["resources"]) > 4:
                res_str += f" (+{len(r['resources'])-4} more)"
            out(f"    {r['model_id_norm'][:35]:<35s} | {r['benchmark_norm'][:18]:<18s} | "
                f"{r['n_resources']:>3d} | {r['score_range']:>7.2f} | {r['score_min']:>8.2f} | "
                f"{r['score_max']:>8.2f} | {res_str}")
        out()

        # ── H.5 Detailed drill-down: top 10 ───────────────────────
        out(f"  H.5 -- Detailed drill-down (top 10 conflicts):")
        for _, r in xr_diff.sort_values("score_range", ascending=False).head(10).iterrows():
            out(f"    Model: {r['model_id_norm']}")
            out(f"    Benchmark: {r['benchmark_norm']}")
            out(f"    Range: {r['score_range']:.2f}  ({r['n_resources']} resources)")
            out(f"    Per-resource scores:")
            for rd in sorted(r["resource_details"], key=lambda x: x["score"]):
                rlbl = resource_label(rd["resource"])
                src_type = "leaderboard" if not rd["source"].startswith("arxiv") else "arxiv paper"
                out(f"      {rd['score']:>8.2f}  [{src_type}]  {rlbl}  (n={rd['n_records']})")
            out()

        # ── H.6 Most common benchmarks and models ─────────────────
        out(f"  H.6 -- Most common benchmarks in cross-resource conflicts:")
        for bench, cnt in xr_diff["benchmark_norm"].value_counts().head(15).items():
            out(f"    {bench:<25s}: {cnt}")
        out()

        out(f"  H.7 -- Most common models in cross-resource conflicts:")
        for model, cnt in xr_diff["model_id_norm"].value_counts().head(15).items():
            out(f"    {model:<40s}: {cnt}")
        out()

        # ── H.8 Leaderboard-vs-Paper conflicts ─────────────────────
        # Find conflicts where at least one resource is a leaderboard and
        # at least one is an arxiv paper
        leaderboard_resources = set(df[~df["source"].str.contains("arxiv", case=False, na=False)]["resource"].unique())

        lb_vs_paper_rows = []
        for _, r in xr_diff.iterrows():
            resources = set(r["resources"])
            has_lb = bool(resources & leaderboard_resources)
            has_paper = bool(resources - leaderboard_resources)
            if has_lb and has_paper:
                lb_vs_paper_rows.append(r)
        lb_vs_paper = pd.DataFrame(lb_vs_paper_rows) if lb_vs_paper_rows else pd.DataFrame()

        out(f"  H.8 -- Leaderboard vs ArXiv paper conflicts:")
        out(f"    Total: {len(lb_vs_paper)}")
        if len(lb_vs_paper) > 0:
            out(f"    Mean range: {lb_vs_paper['score_range'].mean():.2f}")
            out(f"    Median range: {lb_vs_paper['score_range'].median():.2f}")
            out()
            out(f"    Top 15:")
            out(f"    {'Model':<35s} | {'Benchmark':<18s} | {'Range':>7s} | Resources")
            out(f"    {'-' * 110}")
            for _, r in lb_vs_paper.sort_values("score_range", ascending=False).head(15).iterrows():
                res_parts = []
                for rd in sorted(r["resource_details"], key=lambda x: x["score"]):
                    tag = "LB" if rd["resource"] in leaderboard_resources else "AP"
                    res_parts.append(f"{rd['score']:.1f}[{tag}:{resource_label(rd['resource'])[:20]}]")
                out(f"    {r['model_id_norm'][:35]:<35s} | {r['benchmark_norm'][:18]:<18s} | "
                    f"{r['score_range']:>7.2f} | {' vs '.join(res_parts[:4])}")
        out()

        # ── H.9 Paper-vs-Paper only conflicts ──────────────────────
        paper_only_rows = []
        for _, r in xr_diff.iterrows():
            resources = set(r["resources"])
            if not (resources & leaderboard_resources):
                paper_only_rows.append(r)
        paper_only = pd.DataFrame(paper_only_rows) if paper_only_rows else pd.DataFrame()

        out(f"  H.9 -- Paper vs Paper conflicts (no leaderboard involved):")
        out(f"    Total: {len(paper_only)}")
        if len(paper_only) > 0:
            out(f"    Mean range: {paper_only['score_range'].mean():.2f}")
            out(f"    Median range: {paper_only['score_range'].median():.2f}")
            out()
            out(f"    Top 20:")
            out(f"    {'Model':<35s} | {'Benchmark':<18s} | {'#P':>3s} | {'Range':>7s} | Papers")
            out(f"    {'-' * 120}")
            for _, r in paper_only.sort_values("score_range", ascending=False).head(20).iterrows():
                papers_str = ", ".join(resource_label(x)[:28] for x in r["resources"][:3])
                if len(r["resources"]) > 3:
                    papers_str += f" (+{len(r['resources'])-3} more)"
                out(f"    {r['model_id_norm'][:35]:<35s} | {r['benchmark_norm'][:18]:<18s} | "
                    f"{r['n_resources']:>3d} | {r['score_range']:>7.2f} | {papers_str}")
        out()

        # ── H.10 Summary statistics ────────────────────────────────
        out(f"  H.10 -- Summary:")
        out(f"    Total cross-resource conflicts:    {len(xr_diff)}")
        out(f"    Paper-vs-Paper:                    {len(paper_only)}")
        out(f"    Leaderboard-vs-Paper:              {len(lb_vs_paper)}")
        lb_only = len(xr_diff) - len(paper_only) - len(lb_vs_paper)
        out(f"    Leaderboard-vs-Leaderboard:        {lb_only}")
        out(f"    Conflicts with range > 5 pt:       {(xr_diff['score_range'] > 5).sum()}")
        out(f"    Conflicts with range > 10 pt:      {(xr_diff['score_range'] > 10).sum()}")
        out(f"    Conflicts with 3+ resources:       {(xr_diff['n_resources'] >= 3).sum()}")

        # ── Write collision.txt with ALL collision pairs ───────────
        collision_path = ROOT / "analysis_output" / "collision.txt"
        with open(collision_path, "w", encoding="utf-8") as cf:
            cf.write("=" * 100 + "\n")
            cf.write("  ALL CROSS-RESOURCE COLLISION PAIRS\n")
            cf.write(f"  Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
            cf.write(f"  Total collision pairs (score range > 0.01): {len(xr_diff)}\n")
            cf.write("=" * 100 + "\n\n")

            for idx, (_, r) in enumerate(
                xr_diff.sort_values("score_range", ascending=False).iterrows(), 1
            ):
                cf.write(f"--- Collision #{idx} ---\n")
                cf.write(f"  Model:      {r['model_id_norm']}\n")
                cf.write(f"  Benchmark:  {r['benchmark_norm']}\n")
                cf.write(f"  Resources:  {r['n_resources']}\n")
                cf.write(f"  Score range: {r['score_range']:.2f}  (min={r['score_min']:.2f}, max={r['score_max']:.2f})\n")
                cf.write(f"  Per-resource scores:\n")
                for rd in sorted(r["resource_details"], key=lambda x: x["score"]):
                    rlbl = resource_label(rd["resource"])
                    src_type = "leaderboard" if rd["resource"] in leaderboard_resources else "arxiv paper"
                    cf.write(f"    {rd['score']:>8.2f}  [{src_type:<14s}]  {rlbl}  (n={rd['n_records']})\n")
                cf.write("\n")

        out(f"  Collision pairs written to: {collision_path}")
        print(f"  collision.txt written: {len(xr_diff)} pairs")
    out()


    # ──────────────────────────────────────────────────────────────
    #  Section I: Power Simulation
    # ──────────────────────────────────────────────────────────────

    out("─" * 76)
    out("  I. POWER SIMULATION (How many sources needed for 80% power?)")
    out("─" * 76)
    out()

    # Use the observed deltas from collisions
    if n_collisions > 0 and len(genuine) > 0:
        observed_deltas = genuine["delta"].abs().values
        n_boot = 2000
        rng = np.random.default_rng(42)

        out(f"  Observed delta distribution: n={len(observed_deltas)}, "
            f"mean={observed_deltas.mean():.2f}, SD={observed_deltas.std():.2f}")
        out()
        out(f"  {'k sources':>12s} {'Power':>8s} {'95% CI':>16s}")
        out(f"  {'─' * 12} {'─' * 8} {'─' * 16}")

        k_80 = -1
        for k in range(2, 21):
            sig_count = 0
            for _ in range(n_boot):
                sample = rng.choice(observed_deltas, size=k, replace=True)
                t_stat, p_val = stats.ttest_1samp(sample, 0)
                if p_val < 0.05:
                    sig_count += 1
            power = sig_count / n_boot
            ci_lo = power - 1.96 * np.sqrt(power * (1 - power) / n_boot)
            ci_hi = power + 1.96 * np.sqrt(power * (1 - power) / n_boot)
            if power >= 0.80 and k_80 == -1:
                k_80 = k
            out(f"  {k:>12d} {power:>8.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")

        out(f"\n  k_80 (80% power threshold): {k_80 if k_80 > 0 else 'not reached at k=20'}")
    else:
        out("  Insufficient collision data for power simulation.")
    out()


    # ──────────────────────────────────────────────────────────────
    #  Section J: Rank Instability
    # ──────────────────────────────────────────────────────────────

    out("─" * 76)
    out("  J. RANK INSTABILITY (Kendall τ between sources)")
    out("─" * 76)
    out()

    rank_results = []
    for (mid_norm, bench_norm), group in df_dedup.groupby(["model_id_norm", "benchmark_norm"]):
        pass  # Just checking structure

    # Find benchmarks shared between source pairs with enough models
    for s1, s2 in combinations(sources, 2):
        sub1 = df_dedup[df_dedup["source"] == s1]
        sub2 = df_dedup[df_dedup["source"] == s2]
        shared_bench = set(sub1["benchmark_norm"]) & set(sub2["benchmark_norm"])
        for bench in shared_bench:
            b1 = sub1[sub1["benchmark_norm"] == bench][["model_id_norm", "score"]].drop_duplicates("model_id_norm")
            b2 = sub2[sub2["benchmark_norm"] == bench][["model_id_norm", "score"]].drop_duplicates("model_id_norm")
            merged = b1.merge(b2, on="model_id_norm", suffixes=("_1", "_2"))
            if len(merged) >= 4:
                tau, p = stats.kendalltau(merged["score_1"], merged["score_2"])
                rank_results.append({
                    "source_a": s1,
                    "source_b": s2,
                    "benchmark": bench,
                    "n_models": len(merged),
                    "tau_b": tau,
                    "pvalue": p,
                })

    if rank_results:
        rank_df = pd.DataFrame(rank_results).sort_values("tau_b")
        out(f"  Benchmark×source pairs with ≥4 shared models: {len(rank_df)}")
        out(f"\n  {'Source A':20s} {'Source B':15s} {'Benchmark':15s} {'n':>4s} {'τ_b':>8s} {'p':>10s}")
        out(f"  {'─' * 20} {'─' * 15} {'─' * 15} {'─' * 4} {'─' * 8} {'─' * 10}")
        for _, r in rank_df.iterrows():
            out(f"  {src_label(r['source_a']):20s} {src_label(r['source_b']):15s} "
                f"{r['benchmark']:15s} {r['n_models']:>4d} {r['tau_b']:>8.3f} {r['pvalue']:>10.4f}")
    else:
        out("  No benchmark×source pairs found with ≥4 shared models.")
    out()


    # ──────────────────────────────────────────────────────────────
    #  Section K: Documentation Gap (Incentive Problem)
    # ──────────────────────────────────────────────────────────────

    out("─" * 76)
    out("  K. DOCUMENTATION GAP (Incentive Problem)")
    out("─" * 76)
    out()

    out(f"  {'Source':20s} {'Total':>8s} {'Gap temp':>10s} {'Gap prompt':>11s} {'Gap harness':>12s}")
    out(f"  {'─' * 20} {'─' * 8} {'─' * 10} {'─' * 11} {'─' * 12}")
    for src in sorted(df["source"].unique()):
        sub = df[df["source"] == src]
        n = len(sub)
        gap_temp = n - is_valid(sub["temperature"]).sum()
        gap_prompt = n - is_valid(sub["prompt_template"]).sum()
        gap_harness = n - is_valid(sub["harness"]).sum()
        out(f"  {src_label(src):20s} {n:>8,} {gap_temp:>10,} {gap_prompt:>11,} {gap_harness:>12,}")
    out()

    # Total gaps
    n_total = len(df)
    total_gap_temp = n_total - is_valid(df["temperature"]).sum()
    total_gap_prompt = n_total - is_valid(df["prompt_template"]).sum()
    total_gap_harness = n_total - is_valid(df["harness"]).sum()
    out(f"  {'TOTAL':20s} {n_total:>8,} {total_gap_temp:>10,} {total_gap_prompt:>11,} {total_gap_harness:>12,}")
    out()


    # ──────────────────────────────────────────────────────────────
    #  Section L: Reproducibility Checklist
    # ──────────────────────────────────────────────────────────────

    out("─" * 76)
    out("  L. REPRODUCIBILITY CHECKLIST")
    out("─" * 76)
    out()

    checklist_fields = [
        ("eval_library.name", "harness"),
        ("eval_library.version", "eval_library_version"),
        ("generation_args.n_shot", "shots"),
        ("generation_args.temperature", "temperature"),
        ("generation_args.prompt_template", "prompt_template"),
        ("additional_details.cot", "chain_of_thought"),
        ("scoring_method", "scoring_method"),
    ]

    out(f"  {'EvalSpec Field':35s} {'Coverage':>10s} {'Old Coverage':>13s} {'Delta':>8s}")
    out(f"  {'─' * 35} {'─' * 10} {'─' * 13} {'─' * 8}")
    for field_name, col in checklist_fields:
        if col in df.columns:
            new_cov = 100 * is_valid(df[col]).sum() / len(df)
            if col in df_old.columns:
                old_cov = 100 * is_valid(df_old[col]).sum() / len(df_old)
                delta = new_cov - old_cov
                sign = "+" if delta > 0 else ""
                out(f"  {field_name:35s} {new_cov:>9.1f}% {old_cov:>12.1f}% {sign}{delta:>6.1f}pp")
            else:
                out(f"  {field_name:35s} {new_cov:>9.1f}%           N/A")
        else:
            out(f"  {field_name:35s}       N/A")
    out()


    # ──────────────────────────────────────────────────────────────
    #  Section M: Score Sanity Check
    # ──────────────────────────────────────────────────────────────

    out("─" * 76)
    out("  M. SCORE SANITY CHECK (after 0-1 -> 0-100 normalization)")
    out("─" * 76)
    out()
    out(f"  Scores rescaled in Step 1b: {n_rescaled}")
    out()

    df["score_num"] = pd.to_numeric(df["score"], errors="coerce")
    valid_scores = df["score_num"].dropna()
    out(f"  Numeric scores: {len(valid_scores):,} / {len(df):,}")
    out(f"  Score range: {valid_scores.min():.4f} – {valid_scores.max():.4f}")
    out(f"  Scores > 100:  {(valid_scores > 100).sum()}")
    out(f"  Scores > 1000: {(valid_scores > 1000).sum()}")
    out(f"  Scores == 0:   {(valid_scores == 0).sum()}")
    out(f"  Negative:      {(valid_scores < 0).sum()}")
    out()

    out("  Top 15 benchmarks by record count:")
    for bench, count in df["benchmark"].value_counts().head(15).items():
        out(f"    {bench:40s}: {count:>6,}")
    out()


    # ──────────────────────────────────────────────────────────────
    #  Section N: Key Hypothesis Summary
    # ──────────────────────────────────────────────────────────────

    out("─" * 76)
    out("  N. KEY HYPOTHESIS SUMMARY (Updated)")
    out("─" * 76)
    out()

    # H1
    temp_cov = 100 * is_valid(df["temperature"]).sum() / len(df)
    prompt_cov = 100 * is_valid(df["prompt_template"]).sum() / len(df)
    harness_cov = 100 * is_valid(df["harness"]).sum() / len(df)
    shots_cov = 100 * is_valid(df["shots"]).sum() / len(df)

    out("  H1 (Metadata Gap):")
    out(f"    Temperature coverage:      {temp_cov:.1f}%")
    out(f"    Prompt template coverage:  {prompt_cov:.1f}%")
    out(f"    Harness coverage:          {harness_cov:.1f}%  (but ≈all = 'unknown' for non-leaderboard)")
    out(f"    N-shot coverage:           {shots_cov:.1f}%")
    if temp_cov < 1.0 and prompt_cov < 1.0:
        out(f"    → H1 CONFIRMED: temperature ({temp_cov:.1f}%) and prompt template ({prompt_cov:.1f}%) "
            f"remain near-zero even with {n_total:,} records across {n_sources} sources")
    out()

    # H2
    out("  H2 (Structural Fragmentation):")
    out(f"    Cross-source collision pairs: {n_collisions} (observed)")
    out(f"    Permutation null mean:        {stats_all[0]:.1f} ± {stats_all[1]:.1f}")
    out(f"    z = {stats_all[4]:.2f}, p = {stats_all[5]:.2e}")
    direction = "fewer than" if obs_all < stats_all[0] else "more than"
    sig = "significantly" if stats_all[5] < 0.05 else "not significantly"
    out(f"    → Observed collisions are {sig} {direction} expected")
    out()

    # H4
    if len(pwc) > 0:
        out("  H4 (Pipeline Fragility — PWC):")
        out(f"    PWC artefact rate: {artefact_pct:.1f}% ({n_artefacts}/{len(pwc)})")
        out(f"    → {'CONFIRMED' if artefact_pct > 40 else 'PARTIALLY CONFIRMED'}: "
            f"{'majority' if artefact_pct > 50 else 'substantial fraction'} of PWC records are extraction artefacts")
    out()


    # ═══════════════════════════════════════════════════════════════
    #  WRITE OUTPUT
    # ═══════════════════════════════════════════════════════════════

    output_path = ROOT / "analysis_output" / "updated_statistics.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
        f.write("\n")

    print(f"\n{'=' * 76}")
    print(f"  Results saved to: {output_path}")
    print(f"{'=' * 76}")


if __name__ == "__main__":
    main()
