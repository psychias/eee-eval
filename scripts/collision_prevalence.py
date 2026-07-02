"""Corpus-wide cross-source collision prevalence (paper Section 5, "From existence
to prevalence").

Reads data/aggregated/all_results.csv (the flattened union of all sources) and
reports, over distinct model-benchmark cells after canonical name normalisation:
  - how many cells are reported by >=2 independent sources (collisions),
  - what fraction of collisions disagree by >1pp / >5pp,
  - how unattributable the disagreements are (scoring-mode never recorded; harness
    usually absent).

Name normalisation is heuristic (strips the HF org prefix + common suffixes,
applies the canonical benchmark/model maps from src.eee_eval.extraction.constants),
so the collision count is a LOWER BOUND on true overlap. Run from the repo root:

    python scripts/collision_prevalence.py
"""
import csv
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, "src")
try:
    from eee_eval.extraction import constants as C
    BENCH_CANON = C._BENCHMARK_CANONICAL
    MODEL_ALIAS = C._MODEL_ALIASES
except Exception:  # noqa: BLE001 - fall back to no-map normalisation
    BENCH_CANON, MODEL_ALIAS = {}, {}

CSV_PATH = os.path.join("data", "aggregated", "all_results.csv")


def norm_bench(b):
    key = re.sub(r"\s+", " ", (b or "").strip().lower())
    if key in BENCH_CANON:
        return BENCH_CANON[key]
    flat = re.sub(r"[^a-z0-9]", "", key)
    if flat in BENCH_CANON:
        return BENCH_CANON[flat]
    base = re.sub(r"[^a-z0-9]", "", re.sub(r"\s*\(.*?\)", "", key))
    return BENCH_CANON.get(base, base)


def norm_model(model_id, name):
    tail = (model_id or name or "").strip().split("/")[-1]
    k = re.sub(r"[^a-z0-9]", "", tail.lower())
    if k in MODEL_ALIAS:
        return MODEL_ALIAS[k]
    return re.sub(r"(instruct|chat|it|hf|v0|base)$", "", k)


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def has(rec, field):
    v = (rec.get(field) or "").strip().lower()
    return v not in ("", "unknown", "none", "nan", "null")


def main():
    with open(CSV_PATH, encoding="utf-8", errors="replace") as fh:
        rows = list(csv.DictReader(fh))

    cells = defaultdict(list)
    for r in rows:
        m, b = norm_model(r.get("model_id"), r.get("model")), norm_bench(r.get("benchmark"))
        if m and b:
            cells[(m, b)].append(r)

    multi = disagree = big = unattributable = 0
    for recs in cells.values():
        if len({r.get("source") for r in recs}) < 2:
            continue
        multi += 1
        scores = [fnum(r.get("score")) for r in recs if fnum(r.get("score")) is not None]
        if len(scores) < 2:
            continue
        spread = max(scores) - min(scores)
        if spread > 1.0:
            disagree += 1
            if spread > 5.0:
                big += 1
            # unattributable: no record carries both harness and scoring-mode
            if any(not has(r, "harness") or not has(r, "scoring_method") for r in recs):
                unattributable += 1

    print(f"records: {len(rows)}")
    print(f"distinct model-benchmark cells: {len(cells)}")
    print(f"collisions (>=2 sources): {multi} ({100*multi/len(cells):.2f}% of cells)")
    print(f"  disagree >1pp: {disagree} ({100*disagree/max(multi,1):.1f}% of collisions)")
    print(f"  disagree >5pp: {big} ({100*big/max(multi,1):.1f}% of collisions)")
    print(f"  unattributable disagreements: {unattributable}/{disagree} "
          f"({100*unattributable/max(disagree,1):.0f}%)")


if __name__ == "__main__":
    main()
