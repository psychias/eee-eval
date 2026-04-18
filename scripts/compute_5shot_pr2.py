"""Compute 5-shot GSM8K partial R² for the sensitivity figure with 3 models."""
import json, numpy as np, pandas as pd, statsmodels.api as sm
from pathlib import Path
from collections import defaultdict

recs = []
for line in Path('experiments/results/controlled_eval_results.jsonl').read_text().splitlines():
    if line.strip():
        r = json.loads(line)
        if r.get('status') == 'ok' and r.get('score') is not None:
            recs.append(r)

df = pd.DataFrame(recs)

# 5-shot GSM8K only
df5 = df[(df['benchmark'] == 'gsm8k') & (df['n_shot'] == 5)].copy()
print(f"5-shot GSM8K records: {len(df5)}")
print(f"Models: {sorted(df5['model_id'].unique())}")

# Base model: model FE
dummies_m = pd.get_dummies(df5['model_id'], prefix='m', drop_first=True)
X_base = pd.concat([pd.Series(1.0, index=df5.index, name='const'), dummies_m], axis=1).astype(float)
y = df5['score'].astype(float)
base_m = sm.OLS(y, X_base).fit()
ssr_base = float(np.sum(base_m.resid**2))

for var, col in [
    ('temperature',  df5['temperature'].astype(float)),
    ('prompt_format', pd.Series(pd.Categorical(df5['prompt_format']).codes, index=df5.index).astype(float)),
]:
    Xf = pd.concat([X_base, col.rename(var)], axis=1)
    fm = sm.OLS(y, Xf).fit()
    pr2 = max(0.0, (ssr_base - float(np.sum(fm.resid**2))) / ssr_base)
    print(f"  Partial R2 [{var:14s}]: {pr2:.4f}")

# Per-axis effects for 5-shot
five_recs = [r for r in recs if r['benchmark'] == 'gsm8k' and r['n_shot'] == 5]
def axis_eff(axis, pool):
    other = {
        'temperature':   ('model_id', 'benchmark', 'prompt_format', 'n_shot', 'random_seed'),
        'prompt_format': ('model_id', 'benchmark', 'temperature', 'n_shot', 'random_seed'),
    }
    g = defaultdict(list)
    for r in pool:
        g[tuple(r[a] for a in other[axis])].append(r['score'])
    effs = [max(v) - min(v) for v in g.values() if len(v) >= 2]
    return round(float(np.mean(effs)), 2) if effs else 0.0

print(f"  5-shot temp effect:  {axis_eff('temperature', five_recs):.2f} pp")
print(f"  5-shot fmt effect:   {axis_eff('prompt_format', five_recs):.2f} pp")
