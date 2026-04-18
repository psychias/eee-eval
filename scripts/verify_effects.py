#!/usr/bin/env python3
"""Find which computation produces the 3.67 and 10.29 values."""
import json, numpy as np
from collections import defaultdict

records = []
for line in open("experiments/controlled_eval/results/controlled_eval_results.jsonl"):
    line = line.strip()
    if not line: continue
    r = json.loads(line)
    if r.get("status") == "ok" and r.get("score") is not None:
        records.append(r)

gsm5 = [r for r in records if r["benchmark"] == "gsm8k" and r["n_shot"] == 5]
print(f"5-shot GSM8K: {len(gsm5)} records")

# Method 1: with seed in grouping (gen_config_vs_crosssource.py style)
def axis_eff_with_seed(axis, pool):
    other = {
        "temperature":   ("model_id", "benchmark", "prompt_format", "n_shot", "random_seed"),
        "prompt_format": ("model_id", "benchmark", "temperature", "n_shot", "random_seed"),
    }
    g = defaultdict(list)
    for r in pool:
        g[tuple(r[a] for a in other[axis])].append(r["score"])
    effs = [max(v)-min(v) for v in g.values() if len(v) >= 2]
    return round(float(np.mean(effs)), 2) if effs else 0.0

# Method 2: without seed (analyse_controlled.py style)
def axis_eff_no_seed(axis, pool):
    other = {
        "temperature":   ("model_id", "benchmark", "prompt_format", "n_shot"),
        "prompt_format": ("model_id", "benchmark", "temperature", "n_shot"),
    }
    g = defaultdict(list)
    for r in pool:
        g[tuple(r[a] for a in other[axis])].append(r["score"])
    effs = [max(v)-min(v) for v in g.values() if len(v) >= 2]
    return round(float(np.mean(effs)), 2) if effs else 0.0

# Method 3: average over seeds, then max-min across axis
def axis_eff_avg_seed(axis, pool):
    cell_scores = defaultdict(list)
    for r in pool:
        key = (r["model_id"], r["benchmark"], r["temperature"], r["prompt_format"], r["n_shot"])
        cell_scores[key].append(r["score"])
    cell_means = {k: np.mean(v) for k, v in cell_scores.items()}
    
    groups = defaultdict(list)
    if axis == "temperature":
        for (m, b, t, f, n), mean_score in cell_means.items():
            groups[(m, b, f, n)].append(mean_score)
    elif axis == "prompt_format":
        for (m, b, t, f, n), mean_score in cell_means.items():
            groups[(m, b, t, n)].append(mean_score)
    
    effs = [max(v)-min(v) for v in groups.values() if len(v) >= 2]
    return round(float(np.mean(effs)), 2) if effs else 0.0

print("Method 1 (with seed in grouping):")
t1 = axis_eff_with_seed("temperature", gsm5)
f1 = axis_eff_with_seed("prompt_format", gsm5)
print(f"  temp: {t1:.2f}  fmt: {f1:.2f}")

print("Method 2 (without seed):")
t2 = axis_eff_no_seed("temperature", gsm5)
f2 = axis_eff_no_seed("prompt_format", gsm5)
print(f"  temp: {t2:.2f}  fmt: {f2:.2f}")

print("Method 3 (avg over seeds first):")
t3 = axis_eff_avg_seed("temperature", gsm5)
f3 = axis_eff_avg_seed("prompt_format", gsm5)
print(f"  temp: {t3:.2f}  fmt: {f3:.2f}")

print(f"\nTarget: temp=3.67, fmt=10.29")

# Let me also try: for each (model, format, seed), compute |score(t=0) - score(t=0.7)|
print("\n--- Individual seed-level temp effects (5-shot GSM8K) ---")
by_key = {}
for r in gsm5:
    key = (r["model_id"].split("/")[-1], r["prompt_format"], r["random_seed"])
    if key not in by_key:
        by_key[key] = {}
    by_key[key][r["temperature"]] = r["score"]

for key in sorted(by_key.keys()):
    temps = by_key[key]
    if 0.0 in temps and 0.7 in temps:
        delta = abs(temps[0.0] - temps[0.7])
        print(f"  {key[0]:30s} fmt={key[1]:10s} seed={key[2]:4d}: t0.0={temps[0.0]:5.1f} t0.7={temps[0.7]:5.1f} |d|={delta:.1f}")

# The paper text at line 668 says: "mean within-cell variation is \tempeffectfive pp across
# temperature and \fmteffectfive pp across prompt format."
# "within-cell variation" suggests averaging over seeds within each cell first.
# But Method 3 gives different results too.

# Let me try: the mean of per-seed deltas
all_temp_deltas = []
for key in sorted(by_key.keys()):
    temps = by_key[key]
    if 0.0 in temps and 0.7 in temps:
        all_temp_deltas.append(abs(temps[0.0] - temps[0.7]))
print(f"\nMean per-seed temp delta: {np.mean(all_temp_deltas):.2f} pp")

# For format: for each (model, temp, seed), max-min across formats
print("\n--- Individual seed-level format effects (5-shot GSM8K) ---")
by_key2 = {}
for r in gsm5:
    key = (r["model_id"].split("/")[-1], r["temperature"], r["random_seed"])
    if key not in by_key2:
        by_key2[key] = {}
    by_key2[key][r["prompt_format"]] = r["score"]

all_fmt_deltas = []
for key in sorted(by_key2.keys()):
    fmts = by_key2[key]
    if len(fmts) >= 2:
        delta = max(fmts.values()) - min(fmts.values())
        all_fmt_deltas.append(delta)
        print(f"  {key[0]:30s} t={key[1]} seed={key[2]:4d}: {dict(sorted(fmts.items()))} |max-min|={delta:.1f}")
print(f"\nMean per-seed format delta: {np.mean(all_fmt_deltas):.2f} pp")
