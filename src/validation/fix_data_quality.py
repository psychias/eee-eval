"""
fix_data_quality.py — Post-hoc data quality fixes for all EEE JSON records.

Fixes applied:
  1. Remove parameter-leak evaluation results from Papers With Code
     (metric_name containing "Parameters" or scores >> max_score)
  2. Canonicalize benchmark names across all sources (e.g. arc_challenge → ARC-Challenge)
  3. Deduplicate within-source (model, benchmark) conflicts in PWC
     (prefer most recent arxiv paper; break ties by score within expected range)
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import sys
from collections import defaultdict

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"

# ──────────────────────────────────────────────────────────────────
# 1. PARAMETER-LEAK DETECTION
# ──────────────────────────────────────────────────────────────────

# Metric names that are NOT benchmark metrics — these are metadata fields
# that the PWC CSV sometimes mixes in with real scores.
_PARAM_METRIC_PATTERNS = [
    re.compile(r"param", re.IGNORECASE),       # "Parameters (Billions)", "Params"
    re.compile(r"size", re.IGNORECASE),         # "Model Size"
    re.compile(r"flop", re.IGNORECASE),         # "FLOPs"
    re.compile(r"training.*(token|step|hour|cost)", re.IGNORECASE),
    re.compile(r"latency", re.IGNORECASE),
    re.compile(r"throughput", re.IGNORECASE),
    re.compile(r"memory", re.IGNORECASE),
]


def _is_metadata_metric(metric_name: str) -> bool:
    """Check if a metric_name looks like model metadata rather than a benchmark score."""
    for pat in _PARAM_METRIC_PATTERNS:
        if pat.search(metric_name):
            return True
    return False


def _is_score_out_of_range(score: float, min_score: float, max_score: float) -> bool:
    """Check if score is implausibly outside declared range (parameter leak signal)."""
    if max_score <= 100 and score > 100:
        return True
    if max_score <= 1.0 and score > 1.5:
        return True
    return False


# ──────────────────────────────────────────────────────────────────
# 2. BENCHMARK NAME CANONICALIZATION
# ──────────────────────────────────────────────────────────────────

# Maps non-canonical names → canonical names.
# Applied to evaluation_name, source_data.dataset_name fields.
_BENCH_CANONICAL: dict[str, str] = {
    # HF Model Card uses lowercase/underscore variants
    "arc_challenge":   "ARC-Challenge",
    "arc_easy":        "ARC-Easy",
    "hellaswag":       "HellaSwag",
    "humaneval":       "HumanEval",
    "winogrande":      "WinoGrande",
    "truthfulqa":      "TruthfulQA",
    "mmlu":            "MMLU",
    "gsm8k":           "GSM8K",
    "mathqa":          "MathQA",
    "openbookqa":      "OpenBookQA",
    "triviaqa":        "TriviaQA",
    "pubmedqa":        "PubMedQA",
    "piqa":            "PIQA",
    "sciq":            "SciQ",
    "headqa":          "HeadQA",
    "logiqa":          "LogiQA",
    "race":            "RACE",
    "webqs":           "WebQS",
    "boolq":           "BoolQ",
    "copa":            "COPA",
    "multirc":         "MultiRC",
}


def _canonicalize_benchmark(name: str) -> str:
    """Return canonical benchmark name if a mapping exists, else return as-is."""
    return _BENCH_CANONICAL.get(name, name)


# ──────────────────────────────────────────────────────────────────
# 3. PWC DEDUPLICATION
# ──────────────────────────────────────────────────────────────────

def _extract_arxiv_date(record: dict) -> str:
    """Extract ISO date or arxiv-derived date for sorting."""
    # Try evaluation_timestamp first
    ts = record.get("evaluation_timestamp", "")
    if ts:
        return ts
    # Try arxiv_id from source_metadata
    arxiv_id = record.get("source_metadata", {}).get("additional_details", {}).get("arxiv_id", "")
    if arxiv_id:
        m = re.match(r"(\d{2})(\d{2})\.", arxiv_id)
        if m:
            yy, mm = int(m.group(1)), int(m.group(2))
            return f"{2000 + yy}-{mm:02d}-01"
    return "1970-01-01"


def _score_is_plausible(score: float, max_score: float) -> bool:
    """Check if a score is within the plausible range."""
    if max_score is not None and score > max_score * 1.1:
        return False
    if score < 0 and max_score > 0:
        return False
    return True


# ──────────────────────────────────────────────────────────────────
# MAIN FIX LOGIC
# ──────────────────────────────────────────────────────────────────

def fix_all() -> dict:
    """Apply all fixes across all data sources. Returns stats dict."""
    stats = {
        "files_scanned": 0,
        "param_leaks_removed": 0,
        "files_deleted_empty": 0,
        "benchmarks_canonicalized": 0,
        "pwc_duplicates_removed": 0,
        "files_modified": 0,
    }

    source_dirs = [d for d in DATA_DIR.iterdir()
                   if d.is_dir() and d.name not in ("aggregated",)]

    # ────────────────────────────────────────────────────
    # Pass 1: Fix parameter leaks + canonicalize benchmarks
    # ────────────────────────────────────────────────────
    print("Pass 1: Fixing parameter leaks + canonicalizing benchmark names...")

    for src_dir in sorted(source_dirs):
        source = src_dir.name
        for json_path in sorted(src_dir.rglob("*.json")):
            stats["files_scanned"] += 1
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            modified = False
            eval_results = data.get("evaluation_results", [])
            cleaned_results = []

            for er in eval_results:
                bench_name = er.get("evaluation_name", "")
                metric_name = er.get("metric_config", {}).get("metric_name", "")
                score = er.get("score_details", {}).get("score")
                max_score = er.get("metric_config", {}).get("max_score")
                min_score = er.get("metric_config", {}).get("min_score", 0.0)

                # Fix 1a: Remove entries with metadata-like metric names
                if _is_metadata_metric(metric_name):
                    stats["param_leaks_removed"] += 1
                    modified = True
                    continue  # skip this eval result

                # Fix 1b: Remove entries with scores impossibly above max_score
                if score is not None and max_score is not None:
                    if _is_score_out_of_range(score, min_score, max_score):
                        stats["param_leaks_removed"] += 1
                        modified = True
                        continue

                # Fix 2: Canonicalize benchmark name
                canonical = _canonicalize_benchmark(bench_name)
                if canonical != bench_name:
                    er["evaluation_name"] = canonical
                    # Also fix source_data.dataset_name if it matches
                    sd = er.get("source_data", {})
                    if sd.get("dataset_name") == bench_name:
                        sd["dataset_name"] = canonical
                    stats["benchmarks_canonicalized"] += 1
                    modified = True

                cleaned_results.append(er)

            if modified:
                if not cleaned_results:
                    # All eval results were removed — delete the file
                    json_path.unlink()
                    stats["files_deleted_empty"] += 1
                else:
                    data["evaluation_results"] = cleaned_results
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                    stats["files_modified"] += 1

    print(f"  Parameter leaks removed:       {stats['param_leaks_removed']}")
    print(f"  Files deleted (now empty):      {stats['files_deleted_empty']}")
    print(f"  Benchmarks canonicalized:       {stats['benchmarks_canonicalized']}")
    print(f"  Files modified:                 {stats['files_modified']}")

    # ────────────────────────────────────────────────────
    # Pass 2: Deduplicate PWC within-source conflicts
    # ────────────────────────────────────────────────────
    print("\nPass 2: Deduplicating PWC within-source conflicts...")

    pwc_dir = DATA_DIR / "papers_with_code"
    if pwc_dir.exists():
        # Build index: (model_id, benchmark) → [(score, max_score, arxiv_date, file_path, er_index)]
        pwc_index: dict[tuple[str, str], list[dict]] = defaultdict(list)

        for json_path in sorted(pwc_dir.rglob("*.json")):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            model_id = data.get("model_info", {}).get("id", "?")
            arxiv_date = _extract_arxiv_date(data)

            for idx, er in enumerate(data.get("evaluation_results", [])):
                bench = er.get("evaluation_name", "?")
                score = er.get("score_details", {}).get("score")
                max_score = er.get("metric_config", {}).get("max_score")
                key = (model_id, bench)
                pwc_index[key].append({
                    "score": score,
                    "max_score": max_score,
                    "arxiv_date": arxiv_date,
                    "file_path": str(json_path),
                    "er_index": idx,
                })

        # Find duplicates and decide which to keep
        entries_to_remove: dict[str, set[int]] = defaultdict(set)  # file_path → set of er_indices to remove

        for (model_id, bench), entries in pwc_index.items():
            if len(entries) <= 1:
                continue

            # Multiple entries for same (model, benchmark) — deduplicate
            # Strategy: keep the one from the most recent paper with a plausible score
            def sort_key(e):
                plausible = _score_is_plausible(e["score"], e["max_score"]) if e["score"] is not None and e["max_score"] is not None else True
                return (plausible, e["arxiv_date"])

            sorted_entries = sorted(entries, key=sort_key, reverse=True)
            # Keep the first (best), mark rest for removal
            for e in sorted_entries[1:]:
                entries_to_remove[e["file_path"]].add(e["er_index"])
                stats["pwc_duplicates_removed"] += 1

        # Apply removals
        for file_path, indices_to_remove in entries_to_remove.items():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            ers = data.get("evaluation_results", [])
            # Remove by index (descending to keep indices valid)
            for idx in sorted(indices_to_remove, reverse=True):
                if idx < len(ers):
                    ers.pop(idx)

            if not ers:
                pathlib.Path(file_path).unlink()
                stats["files_deleted_empty"] += 1
            else:
                data["evaluation_results"] = ers
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                stats["files_modified"] += 1

        print(f"  Duplicate eval records removed: {stats['pwc_duplicates_removed']}")

    # ────────────────────────────────────────────────────
    # Summary
    # ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FIX SUMMARY")
    print(f"{'='*60}")
    print(f"  Files scanned:               {stats['files_scanned']}")
    print(f"  Parameter leaks removed:     {stats['param_leaks_removed']}")
    print(f"  Benchmarks canonicalized:    {stats['benchmarks_canonicalized']}")
    print(f"  PWC duplicates removed:      {stats['pwc_duplicates_removed']}")
    print(f"  Files modified:              {stats['files_modified']}")
    print(f"  Files deleted (empty):       {stats['files_deleted_empty']}")

    return stats


if __name__ == "__main__":
    fix_all()
