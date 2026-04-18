"""
Generate fig_config_vs_crosssource.pdf/.png and fig_config_variance_generative.pdf/.png
for EMNLP paper §5.7 Controlled Configuration Sensitivity.

Inputs: experiments/results/controlled_eval_results.jsonl
Output: figures/fig_config_vs_crosssource.{pdf,png}
        figures/fig_config_variance_generative.{pdf,png}
        experiments/results/controlled_summary.tex
"""
import json, math
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / 'experiments' / 'results'
RESULTS_FILE = RESULTS_DIR / 'controlled_eval_results.jsonl'
FIGURES_DIR = ROOT / 'submission' / 'latex' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CROSS_SOURCE_PAIRS = [
    ('Qwen2.5-72B-Inst.', 'GPQA',      16.67, 49.00, 32.33),
    ('SC2-15B-Inst.',     'MBPP+',      65.10, 61.20,  3.90),
    ('SC2-15B-Inst.',     'HumanEval+', 60.40, 63.40,  3.00),
    ('Tulu-3-8B',         'BBH',        16.86, 16.67,  0.19),
    ('Tulu-3-8B',         'GPQA',        6.26,  6.49,  0.23),
    ('Tulu-3-8B',         'IFEval',     82.55, 82.67,  0.12),
    ('Tulu-3-8B',         'MMLU-Pro',   20.23, 20.30,  0.07),
    ('Tulu-3-8B',         'MuSR',       10.52, 10.45,  0.07),
]

records = []
for line in RESULTS_FILE.read_text(encoding='utf-8').splitlines():
    if line.strip():
        try:
            r = json.loads(line)
            if r.get('status') == 'ok' and r.get('score') is not None:
                records.append(r)
        except Exception:
            pass
print(f'Valid records: {len(records)}')
assert records, 'No valid records found.'

# Split generative vs loglikelihood for separate analysis
gen_records = [r for r in records if r['benchmark'] in ('gsm8k', 'humaneval')]
ll_records  = [r for r in records if r['benchmark'] in ('mmlu',)]
print(f'  Generative: {len(gen_records)}  |  Loglikelihood: {len(ll_records)}')

# ── Sensitivity analysis (generative only) ─────────────────────────────────
print('\n--- Sensitivity analysis on GENERATIVE runs (GSM8K) ---')
print('    (MMLU excluded: loglikelihood scoring is temperature-invariant by construction)')
sensitivity = {}
if len(gen_records) < 6:
    print(f'  Too few generative records ({len(gen_records)}) to fit model.')
else:
    try:
        import pandas as pd, statsmodels.api as sm
        df = pd.DataFrame(gen_records)
        # Base: model + benchmark fixed effects
        dummies_m = pd.get_dummies(df['model_id'], prefix='m', drop_first=True)
        dummies_b = pd.get_dummies(df['benchmark'], prefix='b', drop_first=True)
        X_base = pd.concat([pd.Series(1.0, index=df.index, name='const'),
                            dummies_m, dummies_b], axis=1).astype(float)
        y = df['score'].astype(float)
        base_m = sm.OLS(y, X_base).fit()
        ssr_base = float(np.sum(base_m.resid**2))
        sst = float(np.sum((y - y.mean())**2))
        r2_base = 1.0 - ssr_base / sst
        print(f'  Base model R² (model+benchmark FE): {r2_base:.4f}')
        for var, col in [
            ('temperature',  df['temperature'].astype(float)),
            ('prompt_format', pd.Series(pd.Categorical(df['prompt_format']).codes, index=df.index).astype(float)),
            ('n_shot',       df['n_shot'].astype(float)),
        ]:
            Xf = pd.concat([X_base, col.rename(var)], axis=1)
            fm = sm.OLS(y, Xf).fit()
            pr2 = max(0.0, (ssr_base - float(np.sum(fm.resid**2))) / ssr_base)
            sensitivity[var] = round(pr2, 4)
        overturn = round(0.871**2, 4)
        sensitivity.update(r2_base=round(r2_base, 4), overturn=overturn)
        for var in ('temperature', 'prompt_format', 'n_shot'):
            flag = '  *** EXCEEDS OVERTURN BOUND' if sensitivity[var] >= overturn else ''
            print(f'  Partial R² [{var:14s}]: {sensitivity[var]:.4f}{flag}')
        print(f'  Overturn bound (0.871²):    {overturn:.4f}')
    except ImportError:
        print('  statsmodels not available — skipping')

# ── Per-axis effect sizes ───────────────────────────────────────────────────
def axis_eff(axis, pool):
    other = {
        'temperature':   ('model_id','benchmark','prompt_format','n_shot','random_seed'),
        'prompt_format': ('model_id','benchmark','temperature','n_shot','random_seed'),
        'n_shot':        ('model_id','benchmark','temperature','prompt_format','random_seed'),
    }
    g = defaultdict(list)
    for r in pool:
        g[tuple(r[a] for a in other[axis])].append(r['score'])
    effs = [max(v)-min(v) for v in g.values() if len(v) >= 2]
    return round(float(np.mean(effs)), 2) if effs else 0.0

print('\n--- Per-axis effect sizes (mean |max-min| within cell) ---')
print('  Generative (GSM8K):')
temp_eff_gen  = axis_eff('temperature', gen_records)
fmt_eff_gen   = axis_eff('prompt_format', gen_records)
nshot_eff_gen = axis_eff('n_shot', gen_records)
print(f'    temperature:   {temp_eff_gen:.2f} pp')
print(f'    prompt_format: {fmt_eff_gen:.2f} pp')
print(f'    n_shot:        {nshot_eff_gen:.2f} pp')

if ll_records:
    print('  Loglikelihood (MMLU):')
    print(f'    temperature:   0.00 pp  (by construction; see discussion)')
    fmt_eff_ll   = axis_eff('prompt_format', ll_records)
    nshot_eff_ll = axis_eff('n_shot', ll_records)
    print(f'    prompt_format: {fmt_eff_ll:.2f} pp')
    print(f'    n_shot:        {nshot_eff_ll:.2f} pp')

# ── Seed-only variance (noise floor) ───────────────────────────────────────
print('\n--- Noise floor (seed variation holding all else constant) ---')
by_cell = defaultdict(list)
for r in gen_records:
    k = (r['model_id'], r['benchmark'], r['temperature'],
         r['prompt_format'], r['n_shot'])
    by_cell[k].append(r['score'])
seed_spreads = [max(v)-min(v) for v in by_cell.values() if len(v) >= 2]
if seed_spreads:
    print(f'  Seed spread: mean={np.mean(seed_spreads):.2f} pp  max={np.max(seed_spreads):.2f} pp')
    print(f'  Number of cells with 2+ seeds: {len(seed_spreads)}')

# ── Figure 1: boxplots by axis (generative) ─────────────────────────────────
if gen_records:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, field, label in [
        (axes[0], 'temperature',  'Temperature'),
        (axes[1], 'prompt_format', 'Prompt format'),
        (axes[2], 'n_shot',        'N-shot'),
    ]:
        vals = sorted({r[field] for r in gen_records}, key=str)
        data = [[r['score'] for r in gen_records if r[field] == v] for v in vals]
        ax.boxplot(data, tick_labels=[str(v) for v in vals])
        ax.set_xlabel(label); ax.set_ylabel('Score (pp)')
        ax.set_title(f'GSM8K scores by {label}')
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig_config_variance_generative.pdf', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'fig_config_variance_generative.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved: fig_config_variance_generative')

# ── Figure 2: controlled effect vs cross-source divergence ────────────────────
cs_deltas = [d for *_, d in CROSS_SOURCE_PAIRS]
cs_labels = [f'{m}/{b}' for m, b, *_ in CROSS_SOURCE_PAIRS]
if gen_records:
    by_mb = defaultdict(list)
    for r in gen_records:
        by_mb[(r['model_id'], r['benchmark'])].append(r['score'])
    all_effs = [max(v)-min(v) for v in by_mb.values() if len(v) >= 2]
    mean_ctrl = float(np.mean(all_effs)) if all_effs else 0.0
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.scatter(range(len(cs_deltas)), cs_deltas, marker='s',
                color='#CC3311', s=60, label='Cross-source divergence (Table 5)')
    ax2.axhline(mean_ctrl, linestyle='--', color='#0173B2', alpha=0.8,
                label=f'Mean controlled config effect on GSM8K ({mean_ctrl:.1f} pp)')
    ax2.set_xticks(range(len(cs_labels)))
    ax2.set_xticklabels(cs_labels, rotation=35, ha='right', fontsize=7)
    ax2.set_ylabel('|Delta| (pp)'); ax2.legend(fontsize=9)
    ax2.set_title('Cross-source divergence vs controlled config effect')
    plt.tight_layout()
    fig2.savefig(FIGURES_DIR / 'fig_config_vs_crosssource.pdf', bbox_inches='tight')
    fig2.savefig(FIGURES_DIR / 'fig_config_vs_crosssource.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print('Saved: fig_config_vs_crosssource')

# ── LaTeX macros ──────────────────────────────────────────────────────────────
pr2_t = sensitivity.get('temperature', float('nan'))
pr2_f = sensitivity.get('prompt_format', float('nan'))
pr2_n = sensitivity.get('n_shot', float('nan'))
overturn_val = sensitivity.get('overturn', 0.759)
vals_finite = [v for v in [pr2_t, pr2_f, pr2_n] if not math.isnan(v)]
max_pr2 = max(vals_finite) if vals_finite else 0.0
robust_word = 'above' if max_pr2 >= overturn_val else 'below'

tex = [
    '% controlled_summary.tex -- auto-generated by gen_config_vs_crosssource.py',
    r'\newcommand{\ctrlnrunsgen}{' + str(len(gen_records)) + '}',
    r'\newcommand{\ctrlnrunsll}{'  + str(len(ll_records))  + '}',
    r'\newcommand{\ctrlnmodels}{'  + str(len({r['model_id'] for r in records})) + '}',
    r'\newcommand{\tempvariance}{'  + f'{temp_eff_gen:.2f}' + '}',
    r'\newcommand{\fmtvariance}{'   + f'{fmt_eff_gen:.2f}'  + '}',
    r'\newcommand{\nshotvariance}{' + f'{nshot_eff_gen:.2f}' + '}',
    r'\newcommand{\prtwotemp}{'     + f'{pr2_t:.4f}' + '}',
    r'\newcommand{\prtwoformat}{'   + f'{pr2_f:.4f}' + '}',
    r'\newcommand{\prtwoshot}{'     + f'{pr2_n:.4f}' + '}',
    r'\newcommand{\overturnbound}{' + f'{overturn_val:.4f}' + '}',
    r'\newcommand{\robustword}{'    + robust_word + '}',
]
tex_path = RESULTS_DIR / 'controlled_summary.tex'
tex_path.write_text('\n'.join(tex), encoding='utf-8')
print(f'LaTeX macros -> {tex_path}')
print('=== Analysis complete ===')
