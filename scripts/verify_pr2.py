#!/usr/bin/env python3
"""Verify 5-shot GSM8K partial R² using same method as gen_config_vs_crosssource.py."""
import json, numpy as np
import pandas as pd
import statsmodels.api as sm

records = []
for line in open("experiments/controlled_eval/results/controlled_eval_results.jsonl"):
    line = line.strip()
    if not line: continue
    r = json.loads(line)
    if r.get("status") == "ok" and r.get("score") is not None:
        records.append(r)

# 5-shot GSM8K
gsm5 = [r for r in records if r["benchmark"] == "gsm8k" and r["n_shot"] == 5]
print(f"5-shot GSM8K: {len(gsm5)} records")

df = pd.DataFrame(gsm5)
# Base: model fixed effects only (no benchmark FE needed for single benchmark)
dummies_m = pd.get_dummies(df["model_id"], prefix="m", drop_first=True)
X_base = pd.concat([pd.Series(1.0, index=df.index, name="const"), dummies_m], axis=1).astype(float)
y = df["score"].astype(float)
base_m = sm.OLS(y, X_base).fit()
ssr_base = float(np.sum(base_m.resid**2))

for var, col in [
    ("temperature", df["temperature"].astype(float)),
    ("prompt_format", pd.Series(pd.Categorical(df["prompt_format"]).codes, index=df.index).astype(float)),
]:
    Xf = pd.concat([X_base, col.rename(var)], axis=1)
    fm = sm.OLS(y, Xf).fit()
    pr2 = max(0.0, (ssr_base - float(np.sum(fm.resid**2))) / ssr_base)
    print(f"  Partial R² [{var:14s}]: {pr2:.4f}")

print(f"  Overturn bound: {0.871**2:.4f}")

# All generative (GSM8K)
print(f"\nAll GSM8K: {sum(1 for r in records if r['benchmark']=='gsm8k')} records")
gen = [r for r in records if r["benchmark"] == "gsm8k"]
df2 = pd.DataFrame(gen)
dummies_m2 = pd.get_dummies(df2["model_id"], prefix="m", drop_first=True)
X_base2 = pd.concat([pd.Series(1.0, index=df2.index, name="const"), dummies_m2], axis=1).astype(float)
y2 = df2["score"].astype(float)
base_m2 = sm.OLS(y2, X_base2).fit()
ssr_base2 = float(np.sum(base_m2.resid**2))

for var, col in [
    ("temperature", df2["temperature"].astype(float)),
    ("prompt_format", pd.Series(pd.Categorical(df2["prompt_format"]).codes, index=df2.index).astype(float)),
    ("n_shot", df2["n_shot"].astype(float)),
]:
    Xf = pd.concat([X_base2, col.rename(var)], axis=1)
    fm = sm.OLS(y2, Xf).fit()
    pr2 = max(0.0, (ssr_base2 - float(np.sum(fm.resid**2))) / ssr_base2)
    print(f"  Partial R² [{var:14s}]: {pr2:.4f}")

# Check: does adding benchmark FE change things for 5-shot?
print("\n=== 5-shot with benchmark FE (should be same — only 1 benchmark) ===")
dummies_b = pd.get_dummies(df["benchmark"], prefix="b", drop_first=True)
if dummies_b.shape[1] > 0:
    X_base3 = pd.concat([pd.Series(1.0, index=df.index, name="const"), dummies_m, dummies_b], axis=1).astype(float)
else:
    X_base3 = X_base
base_m3 = sm.OLS(y, X_base3).fit()
ssr_base3 = float(np.sum(base_m3.resid**2))
print(f"  Same SSR? base={ssr_base:.4f}, with_bench_FE={ssr_base3:.4f}")

# Check the "all records" R² (including MMLU)
print(f"\nAll records (including MMLU): {len(records)}")
dfall = pd.DataFrame(records)
dummies_m_all = pd.get_dummies(dfall["model_id"], prefix="m", drop_first=True)
dummies_b_all = pd.get_dummies(dfall["benchmark"], prefix="b", drop_first=True)
X_base_all = pd.concat([pd.Series(1.0, index=dfall.index, name="const"), dummies_m_all, dummies_b_all], axis=1).astype(float)
y_all = dfall["score"].astype(float)
base_all = sm.OLS(y_all, X_base_all).fit()
ssr_base_all = float(np.sum(base_all.resid**2))

for var, col in [
    ("temperature", dfall["temperature"].astype(float)),
    ("prompt_format", pd.Series(pd.Categorical(dfall["prompt_format"]).codes, index=dfall.index).astype(float)),
    ("n_shot", dfall["n_shot"].astype(float)),
]:
    Xf = pd.concat([X_base_all, col.rename(var)], axis=1)
    fm = sm.OLS(y_all, Xf).fit()
    pr2 = max(0.0, (ssr_base_all - float(np.sum(fm.resid**2))) / ssr_base_all)
    print(f"  Partial R² [{var:14s}]: {pr2:.4f}")
