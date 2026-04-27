"""
run_hf_pipeline.py — complete HF-only data collection + conflict detection.

Three data sources, no PDF extraction required:

  Source 1: HF Leaderboards (add_leaderboard_records.py)
    - Open LLM Leaderboard v1 + v2  → ARC, HellaSwag, MMLU, GSM8K, IFEval, BBH ...
    - Chatbot Arena                  → Arena Elo (4 categories)
    - MT-Bench, AlpacaEval, WildBench, BigCodeBench, EvalPlus, BFCL, SWE-bench
    evaluator_relationship: third_party

  Source 2: HF Model Cards (hf_model_card_fetcher.py)
    - Structured YAML results published by model authors themselves
    evaluator_relationship: first_party

  Source 3: Papers With Code (pwc_fetcher.py)
    - Verified paper-linked benchmark results from PWC archive
    - Filtered to canonical benchmarks only
    evaluator_relationship: third_party

  Conflict detector:
    - Finds (model_id, benchmark) pairs that appear in multiple sources
    - Compares scores accounting for scale differences
    - Writes reports/conflicts.json

Usage
-----
  python run_hf_pipeline.py                    # full run
  python run_hf_pipeline.py --refresh         # bypass all caches
  python run_hf_pipeline.py --skip-leaderboards
  python run_hf_pipeline.py --skip-cards
  python run_hf_pipeline.py --skip-pwc
  python run_hf_pipeline.py --conflicts-only  # re-run conflict detection only

Requirements
------------
  pip install huggingface_hub pandas pyarrow requests pyyaml
  HF_TOKEN in .env for gated leaderboard datasets
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import sys
import time
from dataclasses import dataclass, field
from typing import Any

_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

DATA_DIR    = _ROOT / "data"
REPORTS_DIR = _ROOT / "reports"

# ---------------------------------------------------------------------------
# Import the two fetcher modules
# ---------------------------------------------------------------------------

# Add scripts dir to path so we can import the sibling modules
_SCRIPTS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))

from add_leaderboard_records import run_all as _run_leaderboards
from hf_model_card_fetcher   import (
    HFModelCardFetcher,
    write_model_card_record,
    _already_exists as _card_exists,
    models_from_leaderboards,
)
from pwc_fetcher import run_pwc as _run_pwc


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------

@dataclass
class ConflictRecord:
    """A detected score discrepancy between two sources for the same model+benchmark."""
    model_id:       str
    benchmark:      str
    source_a:       str         # source_dir of first record
    source_b:       str         # source_dir of second record
    score_a:        float
    score_b:        float
    min_score_a:    float
    max_score_a:    float
    min_score_b:    float
    max_score_b:    float
    shots_a:        str
    shots_b:        str
    relationship_a: str         # evaluator_relationship
    relationship_b: str
    relative_diff:  float       # abs(a-b) / max(a, b)
    scale_diff:     bool        # True when a and b are on different scales
    severity:       str         # "exact", "rounding", "minor", "significant", "major"


def _normalise_to_fraction(score: float, min_s: float, max_s: float) -> float:
    """Convert any score to 0-1 for scale-independent comparison."""
    span = max_s - min_s
    if span == 0:
        return score
    return (score - min_s) / span


def _severity(relative_diff: float) -> str:
    if relative_diff == 0.0:
        return "exact"
    if relative_diff < 0.005:
        return "rounding"        # < 0.5% — almost certainly a display rounding
    if relative_diff < 0.02:
        return "minor"           # 0.5–2% — could be prompt/seed variation
    if relative_diff < 0.10:
        return "significant"     # 2–10% — likely configuration difference
    return "major"               # > 10% — strong discrepancy signal


def detect_conflicts(min_severity: str = "minor") -> list[ConflictRecord]:
    """Scan data/ for (model, benchmark) pairs with differing scores.

    Returns conflicts where severity >= min_severity.
    Severity order: exact < rounding < minor < significant < major
    """
    _SEV_ORDER = {"exact": 0, "rounding": 1, "minor": 2, "significant": 3, "major": 4}
    min_sev_rank = _SEV_ORDER.get(min_severity, 2)

    # Index: (model_id, benchmark_lower) → list of record dicts
    index: dict[tuple[str, str], list[dict]] = {}

    if not DATA_DIR.exists():
        return []

    for json_file in DATA_DIR.rglob("*.json"):
        try:
            rec = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        model_id   = rec.get("model_info", {}).get("id", "").strip()
        source_dir = str(json_file.relative_to(DATA_DIR)).split("/")[0]
        src_meta   = rec.get("source_metadata", {})
        relationship = src_meta.get("evaluator_relationship", "unknown")

        if not model_id:
            continue

        for er in rec.get("evaluation_results", []):
            bench      = er.get("evaluation_name", "").strip()
            score_det  = er.get("score_details", {})
            metric_cfg = er.get("metric_config", {})
            gen_cfg    = er.get("generation_config", {}) or {}
            add_det    = gen_cfg.get("additional_details", {}) or {}

            score = score_det.get("score")
            if score is None or not bench:
                continue

            min_s = metric_cfg.get("min_score", 0.0)
            max_s = metric_cfg.get("max_score", 1.0)
            shots = add_det.get("shots", "unknown")

            key = (model_id.lower(), bench.lower())
            index.setdefault(key, []).append({
                "model_id":     model_id,
                "benchmark":    bench,
                "source_dir":   source_dir,
                "score":        float(score),
                "min_score":    float(min_s),
                "max_score":    float(max_s),
                "shots":        str(shots),
                "relationship": relationship,
            })

    # Find pairs with more than one entry
    conflicts: list[ConflictRecord] = []
    seen_pairs: set[frozenset] = set()

    for key, entries in index.items():
        if len(entries) < 2:
            continue

        # Compare all pairs
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                a = entries[i]
                b = entries[j]

                # Skip if same source
                if a["source_dir"] == b["source_dir"]:
                    continue

                pair_key = frozenset([
                    f"{a['source_dir']}:{a['score']}",
                    f"{b['source_dir']}:{b['score']}",
                ])
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                # Normalise to 0-1 for scale-independent comparison
                norm_a = _normalise_to_fraction(a["score"], a["min_score"], a["max_score"])
                norm_b = _normalise_to_fraction(b["score"], b["min_score"], b["max_score"])

                # Check if scales are fundamentally different
                scale_diff = (
                    abs(a["max_score"] - b["max_score"]) > 1.0
                    and not math.isclose(
                        a["max_score"] / max(b["max_score"], 1e-9),
                        1.0, rel_tol=0.05
                    )
                )

                max_norm = max(abs(norm_a), abs(norm_b), 1e-9)
                rel_diff = abs(norm_a - norm_b) / max_norm

                sev = _severity(rel_diff)
                if _SEV_ORDER.get(sev, 0) < min_sev_rank:
                    continue

                conflicts.append(ConflictRecord(
                    model_id=a["model_id"],
                    benchmark=a["benchmark"],
                    source_a=a["source_dir"],
                    source_b=b["source_dir"],
                    score_a=a["score"],
                    score_b=b["score"],
                    min_score_a=a["min_score"],
                    max_score_a=a["max_score"],
                    min_score_b=b["min_score"],
                    max_score_b=b["max_score"],
                    shots_a=a["shots"],
                    shots_b=b["shots"],
                    relationship_a=a["relationship"],
                    relationship_b=b["relationship"],
                    relative_diff=round(rel_diff, 6),
                    scale_diff=scale_diff,
                    severity=sev,
                ))

    # Sort by severity (most severe first), then relative_diff descending
    _SEV_ORDER2 = {"major": 4, "significant": 3, "minor": 2, "rounding": 1, "exact": 0}
    conflicts.sort(
        key=lambda c: (_SEV_ORDER2.get(c.severity, 0), c.relative_diff),
        reverse=True,
    )
    return conflicts


def write_conflict_report(conflicts: list[ConflictRecord]) -> pathlib.Path:
    """Write reports/conflicts.json."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "conflicts.json"

    severity_counts: dict[str, int] = {}
    for c in conflicts:
        severity_counts[c.severity] = severity_counts.get(c.severity, 0) + 1

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_conflicts": len(conflicts),
        "severity_breakdown": severity_counts,
        "note": (
            "relative_diff is computed on normalised 0-1 scores so it is "
            "scale-independent. shots_a/shots_b='unknown' means the shot "
            "count was not recorded — do not flag as a conflict without "
            "verifying the configurations match."
        ),
        "conflicts": [
            {
                "model_id":       c.model_id,
                "benchmark":      c.benchmark,
                "severity":       c.severity,
                "relative_diff":  c.relative_diff,
                "scale_diff":     c.scale_diff,
                "source_a": {
                    "source":       c.source_a,
                    "score":        c.score_a,
                    "min_score":    c.min_score_a,
                    "max_score":    c.max_score_a,
                    "shots":        c.shots_a,
                    "relationship": c.relationship_a,
                },
                "source_b": {
                    "source":       c.source_b,
                    "score":        c.score_b,
                    "min_score":    c.min_score_b,
                    "max_score":    c.max_score_b,
                    "shots":        c.shots_b,
                    "relationship": c.relationship_b,
                },
            }
            for c in conflicts
        ],
    }

    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_leaderboards(force_refresh: bool = False) -> dict[str, int]:
    print("\n" + "═" * 60)
    print("STEP 1 — Fetching leaderboard scores")
    print("═" * 60)
    counts = _run_leaderboards(force_refresh=force_refresh)
    total  = sum(counts.values())
    print(f"\n  Total new leaderboard records: {total}")
    return counts


def step_model_cards(force_refresh: bool = False) -> dict[str, int]:
    print("\n" + "═" * 60)
    print("STEP 2 — Fetching HF model card results")
    print("═" * 60)

    model_ids = models_from_leaderboards()
    if not model_ids:
        print("  No leaderboard models found — run step 1 first.")
        return {}

    print(f"  {len(model_ids)} models from leaderboard records\n")

    fetcher  = HFModelCardFetcher()
    written  = 0
    no_data  = 0
    skipped  = 0

    for i, model_id in enumerate(model_ids, 1):
        print(f"  [{i:3d}/{len(model_ids)}] {model_id}", end=" ")

        if _card_exists(model_id):
            print("-> already exists")
            skipped += 1
            continue

        mcd = fetcher.fetch_and_parse(model_id, force_refresh=force_refresh)
        if mcd is None:
            print("-> no structured results in card")
            no_data += 1
            continue

        n = write_model_card_record(mcd)
        if n:
            written += 1
            benches = ", ".join(r.benchmark for r in mcd.results[:3])
            extra   = f" +{len(mcd.results) - 3} more" if len(mcd.results) > 3 else ""
            print(f"-> {len(mcd.results)} results ({benches}{extra})")
        else:
            print("-> write failed")

        if i % 10 == 0:
            time.sleep(0.5)

    print(f"\n  Written: {written}  |  No data: {no_data}  |  Skipped: {skipped}")
    return {"hf_model_card": written}


def step_pwc(force_refresh: bool = False) -> dict[str, int]:
    print("\n" + "═" * 60)
    print("STEP 3 — Fetching Papers With Code results")
    print("═" * 60)
    counts = _run_pwc(force_refresh=force_refresh)
    return counts


def step_detect_conflicts(min_severity: str = "minor") -> list[ConflictRecord]:
    print("\n" + "═" * 60)
    print("STEP 4 — Detecting conflicts")
    print("═" * 60)

    conflicts = detect_conflicts(min_severity=min_severity)
    path      = write_conflict_report(conflicts)

    _SEV_ORDER = {"major": 4, "significant": 3, "minor": 2, "rounding": 1, "exact": 0}
    by_sev: dict[str, list[ConflictRecord]] = {}
    for c in conflicts:
        by_sev.setdefault(c.severity, []).append(c)

    print(f"\n  Total conflicts found: {len(conflicts)}")
    for sev in ["major", "significant", "minor", "rounding", "exact"]:
        if sev in by_sev:
            print(f"    {sev:12s}: {len(by_sev[sev])}")

    if conflicts:
        print(f"\n  Top 10 by severity:")
        for c in conflicts[:10]:
            print(
                f"    [{c.severity:11s}] {c.model_id}  ×  {c.benchmark}\n"
                f"               {c.source_a}={c.score_a}  vs  "
                f"{c.source_b}={c.score_b}"
                f"  (diff={c.relative_diff:.1%}"
                f"{', SCALE DIFF' if c.scale_diff else ''})"
            )

    print(f"\n  Report written to {path}")
    return conflicts


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

def print_data_summary() -> None:
    if not DATA_DIR.exists():
        return
    print("\n" + "─" * 60)
    print("Data directory summary:")
    total = 0
    for d in sorted(DATA_DIR.iterdir()):
        if d.is_dir():
            cnt = sum(1 for _ in d.rglob("*.json"))
            if cnt:
                print(f"  {cnt:6d}  {d.name}")
                total += cnt
    print(f"  {'─'*20}")
    print(f"  {total:6d}  TOTAL")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the complete HF-only EEE data collection pipeline."
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Bypass all caches and re-fetch everything.",
    )
    parser.add_argument(
        "--skip-leaderboards", action="store_true",
        help="Skip leaderboard fetching (step 1).",
    )
    parser.add_argument(
        "--skip-cards", action="store_true",
        help="Skip model card fetching (step 2).",
    )
    parser.add_argument(
        "--skip-pwc", action="store_true",
        help="Skip Papers With Code fetching (step 3).",
    )
    parser.add_argument(
        "--conflicts-only", action="store_true",
        help="Skip both fetch steps and only re-run conflict detection.",
    )
    parser.add_argument(
        "--min-severity",
        choices=["rounding", "minor", "significant", "major"],
        default="minor",
        help="Minimum conflict severity to report (default: minor).",
    )
    args = parser.parse_args()

    start = time.time()

    if not args.conflicts_only:
        if not args.skip_leaderboards:
            step_leaderboards(force_refresh=args.refresh)

        if not args.skip_cards:
            step_model_cards(force_refresh=args.refresh)

        if not args.skip_pwc:
            step_pwc(force_refresh=args.refresh)

    step_detect_conflicts(min_severity=args.min_severity)
    print_data_summary()

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()