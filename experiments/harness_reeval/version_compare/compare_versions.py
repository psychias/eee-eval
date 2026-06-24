#!/usr/bin/env python3
"""Compare lm-eval version-A vs version-B runs and report per-(model,task) deltas.

Handles multi-seed runs: pairs A/B per seed, then aggregates per model with
mean +/- std across seeds and a paired t-test on the per-seed deltas (so the
delta is tested against seed-to-seed noise). Falls back gracefully to the
single-seed case. delta == version effect at fixed config.

Usage:
    python compare_versions.py results/full_version_runs.jsonl
    python compare_versions.py results/final_runs.jsonl --version-a 0.4.3 --version-b 0.4.11
"""
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict

try:
    from scipy import stats as _scipy_stats
except Exception:  # scipy optional
    _scipy_stats = None


def load(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def paired_ttest(deltas):
    """Return (t, p, df) for H0: mean(delta)=0. p via scipy if available, else
    a normal approximation with a note."""
    n = len(deltas)
    if n < 2:
        return None, None, None
    mean = statistics.mean(deltas)
    sd = statistics.stdev(deltas)
    if sd == 0:
        return (math.inf if mean != 0 else 0.0), (0.0 if mean != 0 else 1.0), n - 1
    t = mean / (sd / math.sqrt(n))
    df = n - 1
    if _scipy_stats is not None:
        p = float(2 * _scipy_stats.t.sf(abs(t), df))
    else:
        p = float(2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2)))))  # normal approx
    return t, p, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--version-a", default="0.4.3")
    ap.add_argument("--version-b", default="0.4.11")
    ap.add_argument("--report", default="version_comparison_report.txt")
    ap.add_argument("--csv", default="version_pairs.csv")
    args = ap.parse_args()

    rows = load(args.jsonl)

    # per-seed pairing: (model, task, n_shot, limit, seed) -> {version: record}
    bycfg = defaultdict(dict)
    for r in rows:
        if r.get("status") != "ok" or r.get("score") is None:
            continue
        cfg = (r["model_id"], r["benchmark"], r.get("n_shot"), r.get("limit"),
               r.get("seed"))
        bycfg[cfg][r["lm_eval_version"]] = r

    pairs = []
    for cfg, byver in bycfg.items():
        a, b = byver.get(args.version_a), byver.get(args.version_b)
        if a and b:
            delta = round(b["score"] - a["score"], 4)
            pairs.append({
                "model_id": cfg[0], "benchmark": cfg[1], "n_shot": cfg[2],
                "limit": cfg[3], "seed": cfg[4],
                "score_a": a["score"], "score_b": b["score"],
                "metric_a": a["metric"], "metric_b": b["metric"],
                "delta": delta, "abs_delta": abs(delta),
            })
    pairs.sort(key=lambda p: (p["model_id"], p["seed"]))

    # per-model aggregation across seeds
    bymodel = defaultdict(lambda: {"a": [], "b": [], "deltas": []})
    for p in pairs:
        m = bymodel[(p["model_id"], p["benchmark"], p["n_shot"], p["limit"])]
        m["a"].append(p["score_a"]); m["b"].append(p["score_b"])
        m["deltas"].append(p["delta"])

    # ---- CSV (per-seed rows) ----
    if pairs:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(pairs[0].keys()))
            w.writeheader(); w.writerows(pairs)

    # ---- report ----
    L = []
    def P(s=""): L.append(s)
    P("=" * 92)
    P(f"LM-EVAL VERSION COMPARISON: {args.version_a} vs {args.version_b}")
    P("=" * 92)
    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    n_bad = sum(1 for r in rows if r.get("status") != "ok")
    P(f"OK runs: {n_ok}   failed/no-score runs: {n_bad}   matched per-seed pairs: {len(pairs)}")
    P("NOTE: 0.4.3 and 0.4.11 run on different dependency stacks (they are not")
    P("      reconcilable on one transformers version); delta conflates harness")
    P("      version with dependency era. See README / report discussion.")
    P("")
    P("-" * 92)
    P("PER-MODEL (mean +/- std across seeds, paired t-test on per-seed deltas)")
    P("-" * 92)
    for (model, bench, ns, lim), m in sorted(bymodel.items()):
        seeds_n = len(m["deltas"])
        ma, sa = statistics.mean(m["a"]), (statistics.stdev(m["a"]) if seeds_n > 1 else 0.0)
        mb, sb = statistics.mean(m["b"]), (statistics.stdev(m["b"]) if seeds_n > 1 else 0.0)
        md = statistics.mean(m["deltas"])
        t, p, df = paired_ttest(m["deltas"])
        P(f"  {model}  [{bench}, n_shot={ns}, limit={lim}, seeds={seeds_n}]")
        P(f"    {args.version_a:>8}: {ma:6.2f} +/- {sa:.2f}")
        P(f"    {args.version_b:>8}: {mb:6.2f} +/- {sb:.2f}")
        if t is None:
            P(f"    delta (B-A): {md:+.2f}   (single seed; no test)")
        else:
            approx = "" if _scipy_stats else "  (normal approx; scipy not installed)"
            sig = "ns" if (p is None or p >= 0.05) else "*"
            P(f"    delta (B-A): {md:+.2f}   paired t={t:+.3f}, df={df}, p={p:.4f} {sig}{approx}")
        P("")
    P("-" * 92)
    P("PER-SEED PAIRS")
    P("-" * 92)
    for p in pairs:
        P(f"  {p['model_id']:<30} seed={p['seed']:<5} "
          f"{args.version_a}={p['score_a']:6.2f}  {args.version_b}={p['score_b']:6.2f}  "
          f"delta={p['delta']:+.2f}")

    report = "\n".join(L)
    with open(args.report, "w") as fh:
        fh.write(report + "\n")
    print(report)
    print(f"\n[wrote {args.report} and {args.csv}]")


if __name__ == "__main__":
    main()
