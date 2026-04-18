#!/usr/bin/env python3
"""Verify controlled experiment details."""
import json, pandas as pd, numpy as np
from pathlib import Path

recs = [json.loads(l) for l in open("experiments/results/controlled_eval_result.jsonl") if l.strip()]
edf = pd.DataFrame(recs)

print(f"Total records: {len(edf)}")
print(f"Status counts:\n{edf['status'].value_counts().to_string()}\n")

ok = edf[edf["status"] == "ok"]
print(f"OK records: {len(ok)}")
print(f"By benchmark:\n{ok.groupby('benchmark').size().to_string()}\n")

# Check for zero-score or suspect records
for bench in ok["benchmark"].unique():
    sub = ok[ok["benchmark"] == bench]
    zeros = (sub["score"] == 0).sum()
    print(f"  {bench}: n={len(sub)}, min={sub['score'].min():.4f}, max={sub['score'].max():.4f}, zeros={zeros}")

print()
notok = edf[edf["status"] != "ok"]
print(f"Non-ok records: {len(notok)}")
if len(notok) > 0:
    print(notok[["model_id", "benchmark", "temperature", "prompt_format", "n_shot", "score", "status", "reason"]].to_string())

# Check if 83 = OK records minus something
# Paper says "72 generative, 11 loglikelihood"
gsm_ok = ok[ok["benchmark"] == "gsm8k"]
mmlu_ok = ok[ok["benchmark"] == "mmlu"]
print(f"\nGSM8K OK: {len(gsm_ok)} (paper: 72)")
print(f"MMLU OK: {len(mmlu_ok)} (paper: 11)")
print(f"Sum: {len(gsm_ok) + len(mmlu_ok)} (paper: 83)\n")

# Check for zero-score GSM8K that might have been excluded
gsm_zeros = gsm_ok[gsm_ok["score"] == 0]
print(f"GSM8K zero-score OK records: {len(gsm_zeros)}")
if len(gsm_zeros) > 0:
    print(gsm_zeros[["model_id", "benchmark", "temperature", "prompt_format", "n_shot", "random_seed", "score", "reason"]].to_string())

# ======= Now verify partial R² on 5-shot GSM8K =======
print("\n" + "=" * 60)
print("CONTROLLED EXPERIMENT: 5-shot GSM8K analysis")
print("=" * 60)

gsm5 = gsm_ok[gsm_ok["n_shot"] == 5].copy()
print(f"5-shot GSM8K records: {len(gsm5)}")
print(f"Models: {sorted(gsm5['model_id'].unique())}")

# Mean |max-min| per axis
# Temperature effect: for each (model, format) cell, max-min across temps
for axis_name, group_cols, vary_col in [
    ("temperature", ["model_id", "prompt_format"], "temperature"),
    ("prompt_format", ["model_id", "temperature"], "prompt_format"),
]:
    effects = []
    for name, grp in gsm5.groupby(group_cols):
        # For each cell, average over seeds first, then take max-min
        cell_means = grp.groupby(vary_col)["score"].mean()
        if len(cell_means) >= 2:
            effects.append(cell_means.max() - cell_means.min())
    mean_effect = np.mean(effects) * 100  # convert to pp
    print(f"  Mean |max-min| for {axis_name}: {mean_effect:.2f} pp")

# Seed noise floor
print("\n  Seed noise (within-cell SD across seeds):")
cells = gsm5.groupby(["model_id", "prompt_format", "temperature"])["score"]
within_cell = cells.agg(["mean", "std", "count", lambda x: (x.max() - x.min())])
within_cell.columns = ["mean", "std", "count", "range"]
print(f"  Mean within-cell range: {within_cell['range'].mean()*100:.2f} pp")
print(f"  Max within-cell range: {within_cell['range'].max()*100:.2f} pp")

# Partial R² via OLS
import statsmodels.api as sm
from statsmodels.formula.api import ols as smols

gsm5["score_pct"] = gsm5["score"] * 100
gsm5["model_short"] = gsm5["model_id"].apply(lambda x: x.split("/")[-1])

# Full model: score ~ model + temperature + format
try:
    full = smols("score_pct ~ C(model_short) + C(temperature) + C(prompt_format)", data=gsm5).fit()
    # Model-only
    model_only = smols("score_pct ~ C(model_short)", data=gsm5).fit()
    # Partial R² for temperature = (SSR_full - SSR_no_temp) / SSR_full
    no_temp = smols("score_pct ~ C(model_short) + C(prompt_format)", data=gsm5).fit()
    no_fmt = smols("score_pct ~ C(model_short) + C(temperature)", data=gsm5).fit()

    pr2_temp = (no_temp.ssr - full.ssr) / no_temp.ssr
    pr2_fmt = (no_fmt.ssr - full.ssr) / no_fmt.ssr

    print(f"\n  Partial R² (temperature): {pr2_temp:.3f} (paper: 0.073)")
    print(f"  Partial R² (prompt_format): {pr2_fmt:.3f} (paper: 0.159)")
    print(f"  Full model R²: {full.rsquared:.3f}")
except Exception as e:
    print(f"  OLS failed: {e}")

# All GSM8K (0-shot + 5-shot)
print("\n" + "=" * 60)
print("CONTROLLED EXPERIMENT: All GSM8K analysis")
print("=" * 60)

gsm_all = gsm_ok.copy()
gsm_all["score_pct"] = gsm_all["score"] * 100
gsm_all["model_short"] = gsm_all["model_id"].apply(lambda x: x.split("/")[-1])
print(f"All GSM8K records: {len(gsm_all)}")

try:
    full_all = smols("score_pct ~ C(model_short) + C(temperature) + C(prompt_format) + C(n_shot)", data=gsm_all).fit()
    no_temp_all = smols("score_pct ~ C(model_short) + C(prompt_format) + C(n_shot)", data=gsm_all).fit()
    no_fmt_all = smols("score_pct ~ C(model_short) + C(temperature) + C(n_shot)", data=gsm_all).fit()
    no_nshot_all = smols("score_pct ~ C(model_short) + C(temperature) + C(prompt_format)", data=gsm_all).fit()

    pr2_temp_all = (no_temp_all.ssr - full_all.ssr) / no_temp_all.ssr
    pr2_fmt_all = (no_fmt_all.ssr - full_all.ssr) / no_fmt_all.ssr
    pr2_nshot_all = (no_nshot_all.ssr - full_all.ssr) / no_nshot_all.ssr

    print(f"  Partial R² (temperature): {pr2_temp_all:.3f} (paper: 0.002)")
    print(f"  Partial R² (prompt_format): {pr2_fmt_all:.3f} (paper: 0.004)")
    print(f"  Partial R² (n_shot): {pr2_nshot_all:.3f} (paper: 0.935)")
except Exception as e:
    print(f"  OLS failed: {e}")
