#!/usr/bin/env python3
"""Verify controlled experiment statistics using the same OLS approach as analyse_controlled.py."""
import json, numpy as np
from collections import defaultdict

# Load data
records = []
for line in open("experiments/results/controlled_eval_result.jsonl"):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    if r.get("status") == "ok" and r.get("score") is not None:
        records.append(r)

print(f"OK records: {len(records)}")

# Subsets
gsm5 = [r for r in records if r["benchmark"] == "gsm8k" and r["n_shot"] == 5]
gsm_all = [r for r in records if r["benchmark"] == "gsm8k"]
mmlu = [r for r in records if r["benchmark"] == "mmlu"]
print(f"5-shot GSM8K: {len(gsm5)}")
print(f"All GSM8K: {len(gsm_all)}")
print(f"MMLU: {len(mmlu)}")
print(f"Gen + LL: {len(gsm_all)} + {len(mmlu)} = {len(gsm_all)+len(mmlu)}")

# Axis effects (matching original _axis_effects logic)
def axis_effects(recs, axis):
    axes_map = {
        "temperature": ("model_id", "benchmark", "prompt_format", "n_shot"),
        "prompt_format": ("model_id", "benchmark", "temperature", "n_shot"),
        "n_shot": ("model_id", "benchmark", "temperature", "prompt_format"),
    }
    by_group = defaultdict(list)
    for r in recs:
        key = tuple(r[a] for a in axes_map[axis])
        by_group[key].append(r["score"])
    effects = []
    for scores in by_group.values():
        if len(scores) >= 2:
            effects.append(max(scores) - min(scores))
    return effects

print("\n=== AXIS EFFECTS (5-shot GSM8K) ===")
temp5 = axis_effects(gsm5, "temperature")
fmt5 = axis_effects(gsm5, "prompt_format")
print(f"  Temp effects: {[round(e, 2) for e in temp5]}")
print(f"  Mean temp: {np.mean(temp5):.2f} pp  (macro: 3.67)")
print(f"  Fmt effects: {[round(e, 2) for e in fmt5]}")
print(f"  Mean fmt: {np.mean(fmt5):.2f} pp  (macro: 10.29)")

print("\n=== AXIS EFFECTS (all GSM8K) ===")
temp_all = axis_effects(gsm_all, "temperature")
fmt_all = axis_effects(gsm_all, "prompt_format")
nshot_all = axis_effects(gsm_all, "n_shot")
print(f"  Mean temp: {np.mean(temp_all):.2f} pp  (macro: 1.83)")
print(f"  Mean fmt: {np.mean(fmt_all):.2f} pp  (macro: 5.15)")
print(f"  Mean nshot: {np.mean(nshot_all):.2f} pp  (macro: 33.08)")

print("\n=== SEED NOISE (5-shot GSM8K) ===")
by_cell = defaultdict(list)
for r in gsm5:
    key = (r["model_id"], r["temperature"], r["prompt_format"])
    by_cell[key].append(r["score"])
ranges = [max(s) - min(s) for s in by_cell.values()]
print(f"  Mean range: {np.mean(ranges):.2f} pp  (macro: 1.50)")
print(f"  Max range: {max(ranges):.2f} pp  (macro: 8.00)")
for key, scores in sorted(by_cell.items()):
    model_short = key[0].split("/")[-1]
    rng = max(scores) - min(scores)
    print(f"    {model_short} temp={key[1]} fmt={key[2]}: scores={[round(s,2) for s in scores]}, range={rng:.2f}")

# Partial R² using numpy OLS (matching analyse_controlled.py exactly)
def partial_r2_numpy(recs):
    models = list({r["model_id"] for r in recs})
    model_idx = {m: i for i, m in enumerate(models)}
    n = len(recs)
    k = len(models)
    y = np.array([r["score"] for r in recs])

    # Model fixed effects (intercept + dummies)
    X = np.zeros((n, k + 1))
    X[:, 0] = 1.0
    for i, r in enumerate(recs):
        X[i, model_idx[r["model_id"]] + 1] = 1.0

    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    ss_res_base = float(np.sum((y - X @ beta) ** 2))

    results = {}
    for var_name, var_vals in [
        ("temperature", np.array([r["temperature"] for r in recs])),
        ("prompt_format", np.array([
            {"plain": 0, "instruct": 1, "cot": 2, "standard": 0, "fewshot": 2}.get(r["prompt_format"], -1)
            for r in recs
        ], dtype=float)),
        ("n_shot", np.array([r["n_shot"] for r in recs], dtype=float)),
    ]:
        X_full = np.column_stack([X, var_vals])
        beta_f = np.linalg.lstsq(X_full, y, rcond=None)[0]
        ss_res_full = float(np.sum((y - X_full @ beta_f) ** 2))
        pr2 = (ss_res_base - ss_res_full) / ss_res_base if ss_res_base > 0 else 0.0
        results[var_name] = max(0.0, pr2)
    return results

print("\n=== PARTIAL R² (5-shot GSM8K, n={}) ===".format(len(gsm5)))
pr2_5 = partial_r2_numpy(gsm5)
print(f"  temperature:   {pr2_5['temperature']:.4f}  (macro: 0.030)")
print(f"  prompt_format: {pr2_5['prompt_format']:.4f}  (macro: 0.285)")

print("\n=== PARTIAL R² (all records, n={}) ===".format(len(records)))
pr2_all = partial_r2_numpy(records)
print(f"  temperature:   {pr2_all['temperature']:.4f}  (macro: 0.002)")
print(f"  prompt_format: {pr2_all['prompt_format']:.4f}  (macro: 0.004)")
print(f"  n_shot:        {pr2_all['n_shot']:.4f}  (macro: 0.935)")

# Also check: controlled_summary.tex says the R² values are for ALL records
# not for 5-shot GSM8K. Let's check both.
print("\n=== PARTIAL R² (all GSM8K, n={}) ===".format(len(gsm_all)))
pr2_gsm = partial_r2_numpy(gsm_all)
print(f"  temperature:   {pr2_gsm['temperature']:.4f}")
print(f"  prompt_format: {pr2_gsm['prompt_format']:.4f}")
print(f"  n_shot:        {pr2_gsm['n_shot']:.4f}")

# Overturn bound
rho = -0.850
overturn = rho ** 2
print(f"\n=== OVERTURN BOUND ===")
print(f"  rho^2 = {overturn:.4f}  (macro: 0.723)")
