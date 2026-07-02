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
    # Strip the org prefix + separators only. We deliberately do NOT strip
    # capability suffixes (instruct/chat/base): conflating a base and an
    # instruct checkpoint would manufacture false collisions and inflate the
    # disagreement rate. This keeps the collision count a conservative lower
    # bound on true cross-source overlap.
    tail = (model_id or name or "").strip().split("/")[-1]
    k = re.sub(r"[^a-z0-9]", "", tail.lower())
    return MODEL_ALIAS.get(k, k)


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

    # OLv2 runtime-vs-documented harness signature (Sec 6, "pipeline-attached
    # harness labels"): OLv2 runs lighteval (log-likelihood) though we label it
    # lm_eval; its score should sit systematically below sources that use the
    # documented CoT/generation protocol, one-signed across many models.
    import statistics
    gaps = defaultdict(list)
    for recs in cells.values():
        srcs = {r.get("source") for r in recs}
        if "open_llm_leaderboard_v2" not in srcs or len(srcs) < 2:
            continue
        olv = [fnum(r["score"]) for r in recs
               if r.get("source") == "open_llm_leaderboard_v2" and fnum(r.get("score")) is not None]
        oth = [fnum(r["score"]) for r in recs
               if r.get("source") != "open_llm_leaderboard_v2" and fnum(r.get("score")) is not None]
        if olv and oth:
            b = None
            for r in recs:
                if r.get("source") == "open_llm_leaderboard_v2":
                    b = norm_bench(r.get("benchmark")); break
            gaps[b].append(statistics.median(oth) - olv[0])
    print("\nOLv2 (lighteval) vs other-source gap, per benchmark:")
    for b, gs in sorted(gaps.items(), key=lambda x: -len(x[1])):
        if len(gs) >= 2:
            print(f"  {b:14} n={len(gs):3}  median={statistics.median(gs):+.1f}pp  "
                  f"OLv2 lower in {sum(1 for g in gs if g > 0)}/{len(gs)}")


if __name__ == "__main__":
    main()
