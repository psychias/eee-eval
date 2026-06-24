"""
Per-benchmark one-way ANOVA on prompt-format and temperature for the
factorial-grid 5-shot cells.

Replaces the pooled GSM8K+MMLU ANOVA previously reported in the paper.

Loads:
  experiments/controlled_eval/results/controlled_eval_results.jsonl

Writes (for reproducibility):
  experiments/controlled_eval/results/anova_per_benchmark.json

Prints a copy-pastable summary to stdout.
"""

import json
import math
import os
import sys
from collections import defaultdict

try:
    from scipy.stats import f_oneway
except ImportError:
    print("ERROR: scipy not installed. Run `pip install scipy`.", file=sys.stderr)
    sys.exit(1)

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
JSONL_PATH  = os.path.join(RESULTS_DIR, "controlled_eval_results.jsonl")
OUT_PATH    = os.path.join(RESULTS_DIR, "anova_per_benchmark.json")


def load_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def eta_squared(groups):
    """One-way eta-squared: SS_between / SS_total."""
    all_vals = [v for g in groups for v in g]
    n = len(all_vals)
    if n == 0:
        return float("nan")
    grand_mean = sum(all_vals) / n
    ss_total = sum((v - grand_mean) ** 2 for v in all_vals)
    if ss_total == 0:
        return float("nan")
    ss_between = sum(len(g) * (sum(g) / len(g) - grand_mean) ** 2
                     for g in groups if g)
    return ss_between / ss_total


def per_format_stats(rows, benchmark, n_shot, factor):
    """Group 5-shot rows for one benchmark by `factor` (e.g. prompt_format).

    Returns: {level: [scores]}, plus per-level mean/min/max/n.
    """
    subset = [r for r in rows
              if r.get("benchmark") == benchmark
              and r.get("n_shot")    == n_shot
              and r.get("status")    == "ok"]
    groups = defaultdict(list)
    for r in subset:
        groups[r[factor]].append(float(r["score"]))
    summary = {}
    for level, vals in sorted(groups.items()):
        summary[str(level)] = {
            "n":     len(vals),
            "mean":  sum(vals) / len(vals) if vals else float("nan"),
            "min":   min(vals)             if vals else float("nan"),
            "max":   max(vals)             if vals else float("nan"),
            "scores": vals,
        }
    return summary


def anova(group_dict, factor_name):
    """Run f_oneway over the level lists, return dict with F, p, eta^2, etc."""
    levels = sorted(group_dict.keys())
    lists  = [group_dict[lvl]["scores"] for lvl in levels]
    n_per  = [len(g) for g in lists]
    sizes_ok = all(n >= 2 for n in n_per)  # f_oneway needs >= 2 per group
    if sizes_ok and len(lists) >= 2:
        F, p = f_oneway(*lists)
        F = float(F)
        p = float(p)
    else:
        F, p = float("nan"), float("nan")
    eta2 = eta_squared(lists)
    return {
        "factor":      factor_name,
        "levels":      levels,
        "n_per_level": dict(zip(levels, n_per)),
        "F":           F,
        "p":           p,
        "eta_squared": eta2,
        "n_total":     sum(n_per),
    }


def per_nshot_stats(rows, benchmark, format_filter=None):
    """Group `benchmark` rows by n_shot (the factor), pooling across models,
    temperatures, and seeds. If `format_filter` is given, restrict to that
    prompt_format (isolates n_shot from the CoT parser-collapse interaction).
    """
    subset = [r for r in rows
              if r.get("benchmark") == benchmark
              and r.get("status")   == "ok"
              and (format_filter is None or r.get("prompt_format") == format_filter)]
    groups = defaultdict(list)
    for r in subset:
        groups[r["n_shot"]].append(float(r["score"]))
    summary = {}
    for level, vals in sorted(groups.items()):
        summary[str(level)] = {
            "n":     len(vals),
            "mean":  sum(vals) / len(vals) if vals else float("nan"),
            "min":   min(vals)             if vals else float("nan"),
            "max":   max(vals)             if vals else float("nan"),
            "scores": vals,
        }
    return summary


def compute_nshot_for_benchmark(rows, benchmark):
    pooled = per_nshot_stats(rows, benchmark, format_filter=None)
    plain  = per_nshot_stats(rows, benchmark, format_filter="plain")
    return {
        "benchmark": benchmark,
        "pooled": {
            "summary": {k: {kk: vv for kk, vv in v.items() if kk != "scores"}
                        for k, v in pooled.items()},
            "anova":   anova(pooled, "n_shot"),
        },
        "plain_only": {
            "summary": {k: {kk: vv for kk, vv in v.items() if kk != "scores"}
                        for k, v in plain.items()},
            "anova":   anova(plain, "n_shot"),
        },
    }


def compute_for_benchmark(rows, benchmark, n_shot):
    fmt_groups  = per_format_stats(rows, benchmark, n_shot, "prompt_format")
    temp_groups = per_format_stats(rows, benchmark, n_shot, "temperature")
    return {
        "benchmark":    benchmark,
        "n_shot":       n_shot,
        "format":       {
            "summary": {k: {kk: vv for kk, vv in v.items() if kk != "scores"}
                        for k, v in fmt_groups.items()},
            "anova":   anova(fmt_groups, "prompt_format"),
        },
        "temperature":  {
            "summary": {k: {kk: vv for kk, vv in v.items() if kk != "scores"}
                        for k, v in temp_groups.items()},
            "anova":   anova(temp_groups, "temperature"),
        },
    }


def fmt_F(v):  return "n/a" if math.isnan(v) else f"{v:.2f}"
def fmt_p(v):  return "n/a" if math.isnan(v) else (f"{v:.4f}" if v >= 0.0001 else f"{v:.2e}")
def fmt_pct(v): return "n/a" if math.isnan(v) else f"{100*v:.2f}%"


def main():
    rows = load_rows(JSONL_PATH)
    print(f"Loaded {len(rows)} rows from {JSONL_PATH}")
    print()

    results = {}
    for benchmark in ("gsm8k", "mmlu"):
        n_shot = 5
        block  = compute_for_benchmark(rows, benchmark, n_shot)
        results[benchmark] = block

        print(f"=== {benchmark.upper()} @ {n_shot}-shot ===")
        # per-format summary
        print(f"Per-format (factor: prompt_format)")
        print(f"  {'format':<10} {'n':>3}  {'mean':>8}  {'min':>8}  {'max':>8}")
        for lvl, s in sorted(block["format"]["summary"].items()):
            print(f"  {lvl:<10} {s['n']:>3}  {s['mean']:>8.4f}  {s['min']:>8.4f}  {s['max']:>8.4f}")
        a = block["format"]["anova"]
        print(f"ANOVA(prompt_format): F={fmt_F(a['F'])}  p={fmt_p(a['p'])}  eta^2={fmt_pct(a['eta_squared'])}")
        print()
        # temperature sanity
        print(f"Per-temperature (factor: temperature)")
        print(f"  {'temp':<10} {'n':>3}  {'mean':>8}  {'min':>8}  {'max':>8}")
        for lvl, s in sorted(block["temperature"]["summary"].items()):
            print(f"  {str(lvl):<10} {s['n']:>3}  {s['mean']:>8.4f}  {s['min']:>8.4f}  {s['max']:>8.4f}")
        a = block["temperature"]["anova"]
        print(f"ANOVA(temperature): F={fmt_F(a['F'])}  p={fmt_p(a['p'])}  eta^2={fmt_pct(a['eta_squared'])}")
        print()

    # n_shot factor: pooled across formats, and within plain format only
    nshot_results = {}
    for benchmark in ("gsm8k", "mmlu"):
        block = compute_nshot_for_benchmark(rows, benchmark)
        nshot_results[benchmark] = block
        print(f"=== {benchmark.upper()} — n_shot factor ===")
        for variant in ("pooled", "plain_only"):
            print(f"[{variant}]")
            print(f"  {'n_shot':<8} {'n':>3}  {'mean':>8}  {'min':>8}  {'max':>8}")
            for lvl, s in sorted(block[variant]["summary"].items(), key=lambda kv: int(kv[0])):
                print(f"  {lvl:<8} {s['n']:>3}  {s['mean']:>8.4f}  {s['min']:>8.4f}  {s['max']:>8.4f}")
            a = block[variant]["anova"]
            print(f"  ANOVA(n_shot): F={fmt_F(a['F'])}  p={fmt_p(a['p'])}  eta^2={fmt_pct(a['eta_squared'])}")
        print()
    results["_nshot"] = nshot_results

    # diagnostics: range over formats per benchmark
    print("=== Descriptive ranges (max format mean - min format mean) ===")
    for benchmark, block in results.items():
        if benchmark.startswith("_"):
            continue
        means = [s["mean"] for s in block["format"]["summary"].values()]
        print(f"  {benchmark.upper()}: {max(means) - min(means):.4f} pp")
    print()

    # write JSON
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
