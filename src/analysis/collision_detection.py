"""
collision_detection.py -- find cross-source score collisions for
(model_id, benchmark) pairs that appear in 2+ source directories.

Algorithm:
  1. Load every JSON evaluation record from data/<source_dir>/**/*.json.
  2. Normalise model_id and benchmark so that trivial spelling/case differences
     (e.g. "Meta-Llama/Llama-2-7b-hf" vs "meta-llama/Llama-2-7B") are resolved.
  3. For each (source, normalised_model, normalised_benchmark) group that has
     multiple rows (multiple JSON files for the same model in one source),
     collapse to a single canonical row: median score, most-common config.
  4. For each (normalised_model, normalised_benchmark) that appears in >=2
     distinct source directories, emit every pairwise combination.
     - Classify each pair: same harness? same shots? same CoT?
     - Exclude citation duplicates: identical score AND identical config.
  5. Save all genuine collision pairs to analysis_output/collision_pairs.csv.

Output columns: model_id, model_id_norm, benchmark, benchmark_norm,
  source_a, source_b, score_a, score_b, delta,
  harness_a, harness_b, same_harness,
  n_shot_a, n_shot_b, same_shots,
  cot_a, cot_b, same_cot,
  config_match (all three match)
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path
from collections import Counter

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from eee.normalization import normalize_model_id, normalize_benchmark_for_matching

DATA_DIR = _ROOT / "data"
OUT_DIR = _ROOT / "analysis_output"

# ---------------------------------------------------------------------------
# Normalisation — delegates to eee.normalization (single source of truth)
# ---------------------------------------------------------------------------

def normalise_model_id(raw: str) -> str:
    """Normalize model ID for cross-source matching."""
    return normalize_model_id(raw)


def normalise_benchmark(raw: str) -> str:
    """Normalize benchmark name for cross-source matching."""
    return normalize_benchmark_for_matching(raw)


# ---------------------------------------------------------------------------
# Field extraction
# ---------------------------------------------------------------------------

def _get_gen_field(gen_cfg: dict | None, field: str) -> str:
    if gen_cfg is None:
        return ""
    gen_args = gen_cfg.get("generation_args") or {}
    field_map = {"n_shot": "shots", "temperature": "temperature", "top_p": "top_p"}
    mapped = field_map.get(field, field)
    val = gen_args.get(mapped)
    if val is not None and val != "":
        return str(val)
    details = gen_cfg.get("additional_details") or {}
    return str(details.get(field, ""))


def load_all_records() -> pd.DataFrame:
    """Load every JSON record into a flat DataFrame with normalised keys."""
    rows = []
    for source_dir in sorted(DATA_DIR.iterdir()):
        if not source_dir.is_dir() or source_dir.name == "aggregated":
            continue
        source = source_dir.name
        for fpath in source_dir.rglob("*.json"):
            try:
                rec = json.loads(fpath.read_text(encoding="utf-8"))
            except Exception:
                continue
            model_id_raw = rec.get("model_info", {}).get("id", "")
            harness_top = rec.get("eval_library", {}).get("name", "") or ""
            for result in rec.get("evaluation_results", []):
                bench_raw = result.get("evaluation_name", "")
                score = result.get("score_details", {}).get("score")
                gen_cfg = result.get("generation_config")
                n_shot = _get_gen_field(gen_cfg, "n_shot")
                cot = _get_gen_field(gen_cfg, "chain_of_thought")
                if not cot:
                    cot = _get_gen_field(gen_cfg, "reasoning_mode")
                prompt_template = _get_gen_field(gen_cfg, "prompt_template")
                harness_result = _get_gen_field(gen_cfg, "harness")
                harness = (harness_result
                           if harness_result and harness_result not in ("unknown", "")
                           else harness_top)
                if score is not None and model_id_raw and bench_raw:
                    rows.append({
                        "model_id": model_id_raw,
                        "model_id_norm": normalise_model_id(model_id_raw),
                        "benchmark": bench_raw,
                        "benchmark_norm": normalise_benchmark(bench_raw),
                        "source": source,
                        "score": float(score),
                        "n_shot": n_shot,
                        "cot": cot,
                        "harness": harness,
                        "prompt_template": prompt_template,
                        "file": str(fpath),
                    })
    return pd.DataFrame(rows)


def _most_common(series: pd.Series) -> str:
    counts = Counter(series.dropna().astype(str).tolist())
    if not counts:
        return ""
    return counts.most_common(1)[0][0]


def _safe_eq(a: str, b: str) -> bool:
    """Compare two config values; treat empty/missing as non-comparable (not equal)."""
    a, b = str(a).strip(), str(b).strip()
    if not a or not b or a == "nan" or b == "nan":
        return False  # can't confirm match if either side is missing
    return a == b


def deduplicate_within_source(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse (source, model_id_norm, benchmark_norm) groups to one canonical row.

    Uses median score, most-common config values.
    """
    result = []
    for (source, mid_norm, bench_norm), grp in df.groupby(
        ["source", "model_id_norm", "benchmark_norm"], sort=False
    ):
        # Pick one representative raw model_id and benchmark string
        rep = grp.iloc[0]
        result.append({
            "source": source,
            "model_id": rep["model_id"],
            "model_id_norm": mid_norm,
            "benchmark": rep["benchmark"],
            "benchmark_norm": bench_norm,
            "score": float(grp["score"].median()),
            "n_shot": _most_common(grp["n_shot"]),
            "cot": _most_common(grp["cot"]),
            "harness": _most_common(grp["harness"]),
            "prompt_template": _most_common(grp["prompt_template"]),
        })
    return pd.DataFrame(result)


def is_citation_duplicate(row: dict) -> bool:
    """Return True when both sources report the same underlying evaluation run.

    A citation duplicate has identical score AND identical known config on
    every field that both sides report.  If config is missing on both sides,
    we still need score equality.
    """
    if abs(row["delta"]) > 1e-9:
        return False  # different score => can't be same run

    # Same score.  Check if any config dimension differs.
    if _differs(row, "harness") or _differs(row, "n_shot") or _differs(row, "cot"):
        return False  # known methodological difference => not a citation dup

    return True  # same score, no known config difference


def _differs(row: dict, field: str) -> bool:
    """Return True if field_a != field_b AND both are non-empty."""
    a = str(row.get(f"{field}_a", "")).strip()
    b = str(row.get(f"{field}_b", "")).strip()
    if not a or not b or a == "nan" or b == "nan":
        return False  # can't tell
    return a != b


def detect_collisions(df: pd.DataFrame) -> pd.DataFrame:
    """Find (model_id_norm, benchmark_norm) pairs in 2+ source directories.

    For each such pair, emits every pairwise source combination and classifies
    each by config-match status.  Excludes citation duplicates (same score,
    same config).  No collapsing -- every genuine pair is kept to preserve
    granularity for downstream analyses.
    """
    pairs = []
    grouped = df.groupby(["model_id_norm", "benchmark_norm"])
    for (mid_norm, bench_norm), group in grouped:
        sources = sorted(group["source"].unique())
        if len(sources) < 2:
            continue
        # Build lookup: source -> canonical row
        src_rows = {}
        for src in sources:
            src_rows[src] = group[group["source"] == src].iloc[0]

        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                sa, sb = sources[i], sources[j]
                ra, rb = src_rows[sa], src_rows[sb]
                score_a, score_b = float(ra["score"]), float(rb["score"])
                delta = round(score_a - score_b, 6)
                harness_a = str(ra["harness"])
                harness_b = str(rb["harness"])
                n_shot_a = str(ra["n_shot"])
                n_shot_b = str(rb["n_shot"])
                cot_a = str(ra["cot"])
                cot_b = str(rb["cot"])

                same_harness = _safe_eq(harness_a, harness_b)
                same_shots = _safe_eq(n_shot_a, n_shot_b)
                same_cot = _safe_eq(cot_a, cot_b)

                candidate = {
                    "model_id": ra["model_id"],
                    "model_id_norm": mid_norm,
                    "benchmark": ra["benchmark"],
                    "benchmark_norm": bench_norm,
                    "source_a": sa,
                    "source_b": sb,
                    "score_a": score_a,
                    "score_b": score_b,
                    "delta": delta,
                    "harness_a": harness_a,
                    "harness_b": harness_b,
                    "same_harness": same_harness,
                    "n_shot_a": n_shot_a,
                    "n_shot_b": n_shot_b,
                    "same_shots": same_shots,
                    "cot_a": cot_a,
                    "cot_b": cot_b,
                    "same_cot": same_cot,
                    "config_match": same_harness and same_shots and same_cot,
                }
                if not is_citation_duplicate(candidate):
                    pairs.append(candidate)

    if not pairs:
        return pd.DataFrame()

    result = pd.DataFrame(pairs)
    result = result.sort_values("delta", key=abs, ascending=False).reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(df_raw: pd.DataFrame, df_dedup: pd.DataFrame,
                  collisions: pd.DataFrame) -> None:
    """Print a comprehensive summary of the collision landscape."""
    print(f"\n{'='*72}")
    print("COLLISION DETECTION SUMMARY")
    print(f"{'='*72}")

    # Record counts
    print(f"\nRecords loaded:          {len(df_raw):>7,}")
    print(f"Sources:                 {df_raw['source'].nunique():>7}")
    print(f"After within-src dedup:  {len(df_dedup):>7,}")

    # Cross-source overlap
    multi = df_dedup.groupby(["model_id_norm", "benchmark_norm"])["source"].nunique()
    n_multi = (multi >= 2).sum()
    print(f"\n(model, bench) in 2+ sources: {n_multi}")

    if collisions.empty:
        print("No collisions found.")
        return

    n_total = len(collisions)
    n_genuine = int((collisions["delta"].abs() > 1e-9).sum())
    n_methodology = n_total - n_genuine
    print(f"\nCollision pairs (excl. citation duplicates): {n_total}")
    print(f"  With score difference (|delta| > 0):      {n_genuine}")
    print(f"  Same score, different methodology:         {n_methodology}")

    # Config match breakdown
    if n_genuine > 0:
        gen = collisions[collisions["delta"].abs() > 1e-9]
        n_cfg_match = gen["config_match"].sum()
        n_cfg_diff = len(gen) - n_cfg_match
        print(f"\nAmong score-different pairs:")
        print(f"  Config match (harness+shots+CoT all agree): {n_cfg_match}")
        print(f"  Config differs on at least one field:       {n_cfg_diff}")

        print(f"\n  same_harness: {gen['same_harness'].sum()}/{len(gen)}")
        print(f"  same_shots:   {gen['same_shots'].sum()}/{len(gen)}")
        print(f"  same_cot:     {gen['same_cot'].sum()}/{len(gen)}")

    # Delta distribution
    gen = collisions[collisions["delta"].abs() > 1e-9]
    if not gen.empty:
        ds = gen["delta"].abs()
        print(f"\n|delta| statistics (genuine):")
        print(f"  mean:   {ds.mean():.2f}")
        print(f"  median: {ds.median():.2f}")
        print(f"  max:    {ds.max():.2f}")
        print(f"  |d| > 1:   {(ds > 1).sum()}")
        print(f"  |d| > 5:   {(ds > 5).sum()}")
        print(f"  |d| > 10:  {(ds > 10).sum()}")

    # Source pair distribution
    print(f"\nCollisions by source pair:")
    src_pairs = collisions.groupby(["source_a", "source_b"]).size().sort_values(ascending=False)
    for (sa, sb), count in src_pairs.items():
        sub = collisions[(collisions["source_a"] == sa) & (collisions["source_b"] == sb)]
        gen_sub = sub[sub["delta"].abs() > 1e-9]
        print(f"  {sa:30s} x {sb:25s}: {count:3d} pairs "
              f"({len(gen_sub)} score-diff, {len(sub)-len(gen_sub)} method-diff)")

    # Top collisions
    print(f"\nTop 20 collisions by |delta|:")
    cols = ["model_id_norm", "benchmark_norm", "source_a", "source_b",
            "score_a", "score_b", "delta", "same_harness", "same_shots"]
    print(collisions.head(20)[cols].to_string(index=False))

    # Unique coverage
    print(f"\nUnique models in collisions:     {collisions['model_id_norm'].nunique()}")
    print(f"Unique benchmarks in collisions: {collisions['benchmark_norm'].nunique()}")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading records...")
    df = load_all_records()
    print(f"  {len(df)} benchmark rows from {df['source'].nunique()} sources")

    # Per-source stats
    for src, grp in df.groupby("source"):
        print(f"    {src:30s}: {len(grp):6d} rows, "
              f"{grp['model_id_norm'].nunique():5d} models, "
              f"{grp['benchmark_norm'].nunique():3d} benchmarks")

    # ── Structural overlap diagnostic ──
    print("\n--- Benchmark overlap matrix (why cross-source collisions are bounded) ---")
    sources = sorted(df["source"].unique())
    bench_by_src = {s: set(df[df["source"] == s]["benchmark_norm"].unique()) for s in sources}
    model_by_src = {s: set(df[df["source"] == s]["model_id_norm"].unique()) for s in sources}
    print(f"{'':30s} | {'benchmarks':>10s} | {'models':>6s} | benchmark overlap with other sources")
    for s in sources:
        overlaps = []
        for s2 in sources:
            if s2 == s:
                continue
            b_overlap = bench_by_src[s] & bench_by_src[s2]
            m_overlap = model_by_src[s] & model_by_src[s2]
            if b_overlap and m_overlap:
                overlaps.append(f"{s2}({len(b_overlap)}b,{len(m_overlap)}m)")
        print(f"  {s:28s} | {len(bench_by_src[s]):>10d} | {len(model_by_src[s]):>6d} | "
              f"{', '.join(overlaps) if overlaps else '(no benchmark+model overlap)'}")

    # ── Within-source conflict report ──
    print("\n--- Within-source score conflicts (same model+bench, different scores) ---")
    within_conflicts = []
    for (src, mid_norm, bench_norm), grp in df.groupby(
        ["source", "model_id_norm", "benchmark_norm"]
    ):
        if len(grp) < 2:
            continue
        scores = grp["score"].unique()
        if len(scores) > 1:
            within_conflicts.append({
                "source": src,
                "model_id_norm": mid_norm,
                "benchmark_norm": bench_norm,
                "n_scores": len(scores),
                "score_min": scores.min(),
                "score_max": scores.max(),
                "score_range": scores.max() - scores.min(),
            })
    wc_df = pd.DataFrame(within_conflicts)
    if not wc_df.empty:
        print(f"  {len(wc_df)} (source, model, bench) groups with conflicting scores")
        for src, grp in wc_df.groupby("source"):
            print(f"    {src:30s}: {len(grp):4d} conflicts, "
                  f"max range={grp['score_range'].max():.2f}")
        wc_path = OUT_DIR / "within_source_conflicts.csv"
        wc_df.sort_values("score_range", ascending=False).to_csv(wc_path, index=False)
        print(f"  Saved -> {wc_path}")
    else:
        print("  None found.")

    print("\nDeduplicating within-source...")
    df_dedup = deduplicate_within_source(df)
    print(f"  {len(df_dedup)} rows after within-source dedup")

    print("\nDetecting cross-source collisions...")
    collisions = detect_collisions(df_dedup)

    out_path = OUT_DIR / "collision_pairs.csv"
    collisions.to_csv(out_path, index=False)
    print(f"  Saved {len(collisions)} pairs -> {out_path}")

    # Also save source-pair summary
    if not collisions.empty:
        src_summary = (
            collisions.groupby(["source_a", "source_b"])
            .agg(
                n_pairs=("delta", "size"),
                n_score_diff=("delta", lambda x: (x.abs() > 1e-9).sum()),
                mean_abs_delta=("delta", lambda x: x.abs().mean()),
                max_abs_delta=("delta", lambda x: x.abs().max()),
            )
            .sort_values("n_pairs", ascending=False)
        )
        src_path = OUT_DIR / "collision_source_pairs.csv"
        src_summary.to_csv(src_path)
        print(f"  Saved source-pair summary -> {src_path}")

    print_summary(df, df_dedup, collisions)


if __name__ == "__main__":
    main()
