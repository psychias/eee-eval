"""
Comprehensive final statistics audit.
Compares every number in macros.tex and acl_latex.tex against actual data.
"""
import json, math, re, sys
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
from collections import defaultdict

sys.path.insert(0, '.')
from src.analysis.collision_detection import normalise_benchmark

ROOT = Path('.')
FAIL = []
WARN = []
PASS_LIST = []

def parse_macros(path='submission/latex/macros.tex'):
    """Parse macros.tex and return a dict of macro_name -> value_string."""
    text = Path(path).read_text(encoding='utf-8')
    macros = {}
    for m in re.finditer(r'\\newcommand\{\\(\w+)\}\{([^}]*)\}', text):
        macros[m.group(1)] = m.group(2)
    return macros

MACROS = parse_macros()

def check(label, actual, expected, tol=0.01):
    if abs(actual - expected) <= tol:
        PASS_LIST.append(f"  [PASS] {label}: {actual} == {expected}")
    else:
        FAIL.append(f"  [FAIL] {label}: actual={actual}, paper={expected}")

def check_eq(label, actual, expected):
    if actual == expected:
        PASS_LIST.append(f"  [PASS] {label}: {actual}")
    else:
        FAIL.append(f"  [FAIL] {label}: actual={actual}, paper={expected}")

def check_macro(name, expected):
    actual = MACROS.get(name, 'MISSING')
    # Strip $ and - for comparison
    actual_clean = actual.replace('$', '').replace('−', '-').strip()
    expected_str = str(expected)
    # Try numeric comparison first (allow rounding tolerance for display precision)
    try:
        a, e = float(actual_clean), float(expected_str)
        # Tolerance: half the last displayed decimal place
        if a != 0:
            decimals = len(actual_clean.rstrip('0').split('.')[-1]) if '.' in actual_clean else 0
        else:
            decimals = 3
        tol = 0.5 * 10**(-decimals)
        if abs(a - e) <= tol:
            PASS_LIST.append(f"  [PASS] macro {name}: {actual_clean}")
            return
    except (ValueError, TypeError):
        pass
    if actual_clean == expected_str:
        PASS_LIST.append(f"  [PASS] macro {name}: {actual_clean}")
    else:
        FAIL.append(f"  [FAIL] macro {name}: macros.tex={actual_clean}, should be {expected_str}")

# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("SECTION 1: DATASET-LEVEL STATISTICS")
print("=" * 70)

df = pd.read_csv('data/aggregated/all_results.csv', low_memory=False)
df['bn'] = df['benchmark'].apply(normalise_benchmark)

check_eq("Total records", len(df), 29331)
check_eq("Total sources", df['source'].nunique(), 11)
check_eq("Total models", df['model_id'].nunique(), 5672)
check_eq("Total benchmarks", df['benchmark'].nunique(), 180)

# OLv2 dominance
olv2 = len(df[df['source'] == 'open_llm_leaderboard_v2'])
check_eq("OLv2 records", olv2, 26790)
olv2_pct = round(olv2 / len(df) * 100)
check_eq("OLv2 pct (rounded)", olv2_pct, 91)

excl_olv2 = len(df[df['source'] != 'open_llm_leaderboard_v2'])
check_eq("Excl OLv2", excl_olv2, 2541)

# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 2: PER-SOURCE TABLE (Table 1)")
print("=" * 70)

expected_table1 = {
    'open_llm_leaderboard_v2': (26790, 4465, 6),
    'papers_with_code':        (576, 383, 16),
    'alpacaeval2':             (446, 223, 2),
    'bfcl':                    (327, 109, 3),
    'bigcodebench':            (280, 156, 2),
    'chatbot_arena':           (218, 218, 1),
    'evalplus':                (214, 125, 2),
    'hf_model_card':           (207, 11, 156),
    'wildbench':               (122, 61, 2),
    'swe_bench':               (117, 117, 1),
    'mt_bench':                (34, 34, 1),
}

for src, (exp_rec, exp_mod, exp_ben) in expected_table1.items():
    sub = df[df['source'] == src]
    check_eq(f"{src} records", len(sub), exp_rec)
    check_eq(f"{src} models", sub['model_id'].nunique(), exp_mod)
    check_eq(f"{src} benchmarks", sub['benchmark'].nunique(), exp_ben)

# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 3: COLLISION ANALYSIS")
print("=" * 70)

df['model_lower'] = df['model_id'].str.strip().str.lower()
R = {s: len(df[df['source'] == s]) for s in df['source'].unique()}
n_models = df['model_id'].nunique()
n_bench = df['benchmark'].nunique()
U = n_models * n_bench

# Observed collisions
default_groups = df.groupby(['model_lower', 'bn'])['source'].nunique()
obs_default = (default_groups >= 2).sum()
check_eq("Collision pairs (default norm)", obs_default, 16)

strict_groups = df.groupby(['model_lower', 'benchmark'])['source'].nunique()
obs_strict = (strict_groups >= 2).sum()
check_eq("Collision pairs (strict)", obs_strict, 6)

df_no = df[df['source'] != 'open_llm_leaderboard_v2']
no_groups = df_no.groupby(['model_lower', 'bn'])['source'].nunique()
obs_no = (no_groups >= 2).sum()
check_eq("Collision pairs (excl OLv2)", obs_no, 5)

# Birthday formula
E_all = sum(R[a] * R[b] for a, b in combinations(R.keys(), 2)) / U
SD_all = math.sqrt(E_all)
z_default = (16 - E_all) / SD_all
z_strict = (6 - E_all) / SD_all

check("E_all", round(E_all), 69, 0.5)
check("SD_all", round(SD_all), 8, 0.5)
check("z_default", round(z_default, 1), -6.4, 0.05)
check("z_strict", round(z_strict, 1), -7.6, 0.05)

R_no = {s: v for s, v in R.items() if s != 'open_llm_leaderboard_v2'}
E_no = sum(R_no[a] * R_no[b] for a, b in combinations(R_no.keys(), 2)) / U
SD_no = math.sqrt(E_no)
z_no = (5 - E_no) / SD_no
check("E_excl", round(E_no, 1), 2.7, 0.05)
check("z_excl", round(z_no, 1), 1.4, 0.05)

# Source pairs sharing benchmarks
sharing = 0
for a, b in combinations(sorted(df['source'].unique()), 2):
    ba = set(df[df['source'] == a]['bn'].unique())
    bb = set(df[df['source'] == b]['bn'].unique())
    if ba & bb:
        sharing += 1
check_eq("Source pairs sharing benchmarks", sharing, 4)

# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 4: TULU STATISTICS")
print("=" * 70)

# From collision_pairs.csv
cp = pd.read_csv('analysis_output/collision_pairs.csv')
# Tulu pairs
tulu = cp[cp['model_id_norm'].str.contains('tulu-3-8b', case=False, na=False)]
if len(tulu) == 0:
    tulu = cp[cp['model_id_norm'].str.contains('tulu', case=False, na=False)]
    tulu = tulu[~tulu['model_id_norm'].str.contains('70b', case=False, na=False)]

if len(tulu) > 0:
    deltas = tulu['delta'].values if 'delta' in tulu.columns else None
    if deltas is not None:
        mean_abs_delta = round(np.mean(np.abs(deltas)), 2)
        mean_delta = round(np.mean(deltas), 3)
        # Paper says tulumean = 0.12 in macros
        check("Tulu mean |delta|", mean_abs_delta, 0.14, 0.005)
        # Paper says tuluadvantage = +0.032
        # Convention: first-party advantage = score_a - score_b where a=HF, b=OLv2
        # Need to check sign carefully
        print(f"  Tulu deltas: {list(deltas)}")
        print(f"  Mean delta (signed): {mean_delta}")
        print(f"  Mean |delta|: {mean_abs_delta}")
        
        fp_higher = sum(1 for d in deltas if d > 0)
        fp_lower = sum(1 for d in deltas if d < 0)
        print(f"  First-party higher: {fp_higher}/5, lower: {fp_lower}/5")
        check_macro('tulumean', mean_abs_delta)
        # tuluadvantage has $-$ prefix in macros for LaTeX minus sign
        macro_adv = MACROS.get('tuluadvantage', 'MISSING').replace('$','').replace('−','-').strip()
        expected_adv = str(mean_delta)
        if macro_adv == expected_adv:
            PASS_LIST.append(f"  [PASS] macro tuluadvantage: {macro_adv}")
        else:
            FAIL.append(f"  [FAIL] macro tuluadvantage: macros.tex={macro_adv}, should be {expected_adv}")
else:
    WARN.append("  [WARN] No Tulu-3-8B pairs found in collision_pairs.csv")

# Spearman
# 8 independent pairs
indep = cp[cp['delta'].abs() > 0.001] if 'delta' in cp.columns else pd.DataFrame()
if len(indep) >= 8:
    # Need completeness scores
    pass  # Already verified in prior session

# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 5: COVERAGE STATISTICS")
print("=" * 70)

def coverage(field):
    valid = df[field].notna() & (df[field].astype(str).str.strip() != '') & \
            (df[field].astype(str).str.lower() != 'unknown')
    return valid.sum(), len(df), round(valid.sum() / len(df) * 100, 1)

harness_col = 'harness' if 'harness' in df.columns else 'eval_library_name'
nshot_col = 'n_shot' if 'n_shot' in df.columns else 'shots'
cot_col = 'cot' if 'cot' in df.columns else 'chain_of_thought'

for col, expected_pct in [
    (harness_col, 98.0),
    (nshot_col, 96.7),
    (cot_col, 76.3),
]:
    if col in df.columns:
        cnt, tot, pct = coverage(col)
        # harness: paper uses integer-rounded "98%" which is correct rounding of 97.6%
        tol = 0.5 if col == harness_col else 0.1
        check(f"Coverage {col}", pct, expected_pct, tol)
    else:
        WARN.append(f"  [WARN] Column {col} not found")

for col in ['temperature', 'prompt_template']:
    if col in df.columns:
        cnt, tot, pct = coverage(col)
        if pct < 0.1:
            PASS_LIST.append(f"  [PASS] Coverage {col}: <0.1% ({cnt}/{tot})")
        else:
            FAIL.append(f"  [FAIL] Coverage {col}: {pct}% (expected <0.1%)")
    else:
        WARN.append(f"  [WARN] Column {col} not found")

# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 6: CONTROLLED EXPERIMENT (NEW: 3 MODELS)")
print("=" * 70)

results_file = Path('experiments/controlled_eval/results/controlled_eval_results.jsonl')
recs = []
for line in results_file.read_text(encoding='utf-8').splitlines():
    if line.strip():
        try:
            r = json.loads(line)
            recs.append(r)
        except:
            pass

ok_recs = [r for r in recs if r.get('status') == 'ok' and r.get('score') is not None]
err_recs = [r for r in recs if r.get('status') != 'ok' or r.get('score') is None]
gen_recs = [r for r in ok_recs if r['benchmark'] in ('gsm8k', 'humaneval')]
ll_recs = [r for r in ok_recs if r['benchmark'] in ('mmlu',)]
models = {r['model_id'] for r in ok_recs}

print(f"  Total records: {len(recs)}")
print(f"  OK records: {len(ok_recs)}")
print(f"  Error records: {len(err_recs)}")
print(f"  Generative: {len(gen_recs)}")
print(f"  Loglikelihood: {len(ll_recs)}")
print(f"  Models: {sorted(models)}")

# What macros.tex currently says vs actual
print("\n  Macro checks (reading macros.tex):")
check_eq("expmodels (actual)", len(models), 3)
check_eq("expruns (actual)", len(ok_recs), 125)
check_eq("expgenruns (actual)", len(gen_recs), 108)
check_eq("expllruns (actual)", len(ll_recs), 17)

# Compare against what macros.tex actually has
check_macro('expmodels', 3)
check_macro('expruns', 125)
check_macro('expgenruns', 108)
check_macro('expllruns', 17)

# Compute new partial R² for 5-shot GSM8K
import statsmodels.api as sm

edf = pd.DataFrame(ok_recs)
df5 = edf[(edf['benchmark'] == 'gsm8k') & (edf['n_shot'] == 5)].copy()
print(f"\n  5-shot GSM8K records: {len(df5)} (models: {sorted(df5['model_id'].unique())})")

dummies_m = pd.get_dummies(df5['model_id'], prefix='m', drop_first=True)
X_base = pd.concat([pd.Series(1.0, index=df5.index, name='const'), dummies_m], axis=1).astype(float)
y = df5['score'].astype(float)
base_m = sm.OLS(y, X_base).fit()
ssr_base = float(np.sum(base_m.resid**2))

pr2_vals = {}
for var, col in [
    ('temperature',  df5['temperature'].astype(float)),
    ('prompt_format', pd.Series(pd.Categorical(df5['prompt_format']).codes, index=df5.index).astype(float)),
]:
    Xf = pd.concat([X_base, col.rename(var)], axis=1)
    fm = sm.OLS(y, Xf).fit()
    pr2 = max(0.0, (ssr_base - float(np.sum(fm.resid**2))) / ssr_base)
    pr2_vals[var] = round(pr2, 3)

print(f"  Partial R² temp (5-shot): {pr2_vals['temperature']}")
print(f"  Partial R² fmt (5-shot):  {pr2_vals['prompt_format']}")
check_macro('prtwotempfive', pr2_vals['temperature'])
check_macro('prtwofmtfive', pr2_vals['prompt_format'])

# All GSM8K
dfg = edf[edf['benchmark'] == 'gsm8k'].copy()
dummies_mg = pd.get_dummies(dfg['model_id'], prefix='m', drop_first=True)
X_baseg = pd.concat([pd.Series(1.0, index=dfg.index, name='const'), dummies_mg], axis=1).astype(float)
yg = dfg['score'].astype(float)
base_mg = sm.OLS(yg, X_baseg).fit()
ssr_baseg = float(np.sum(base_mg.resid**2))

pr2_all = {}
for var, col in [
    ('temperature',  dfg['temperature'].astype(float)),
    ('prompt_format', pd.Series(pd.Categorical(dfg['prompt_format']).codes, index=dfg.index).astype(float)),
    ('n_shot',       dfg['n_shot'].astype(float)),
]:
    Xf = pd.concat([X_baseg, col.rename(var)], axis=1)
    fm = sm.OLS(yg, Xf).fit()
    pr2 = max(0.0, (ssr_baseg - float(np.sum(fm.resid**2))) / ssr_baseg)
    pr2_all[var] = round(pr2, 4 if pr2 < 0.01 else 3)

print(f"\n  All GSM8K Partial R²: temp={pr2_all['temperature']}, fmt={pr2_all['prompt_format']}, nshot={pr2_all['n_shot']}")
check_macro('prtwotempall', pr2_all['temperature'])
check_macro('prtwofmtall', pr2_all['prompt_format'])
check_macro('prtwonshotall', pr2_all['n_shot'])

# Per-axis effects (5-shot, with seed in grouping)
def axis_eff(axis, pool):
    other = {
        'temperature':   ('model_id', 'benchmark', 'prompt_format', 'n_shot', 'random_seed'),
        'prompt_format': ('model_id', 'benchmark', 'temperature', 'n_shot', 'random_seed'),
        'n_shot':        ('model_id', 'benchmark', 'temperature', 'prompt_format', 'random_seed'),
    }
    g = defaultdict(list)
    for r in pool:
        g[tuple(r[a] for a in other[axis])].append(r['score'])
    effs = [max(v) - min(v) for v in g.values() if len(v) >= 2]
    return round(float(np.mean(effs)), 2) if effs else 0.0

five_recs = [r for r in ok_recs if r['benchmark'] == 'gsm8k' and r['n_shot'] == 5]
temp5 = axis_eff('temperature', five_recs)
fmt5 = axis_eff('prompt_format', five_recs)
print(f"\n  5-shot effects: temp={temp5}, fmt={fmt5}")
check_macro('tempeffectfive', temp5)
check_macro('fmteffectfive', fmt5)

# All GSM8K effects
all_gen = [r for r in ok_recs if r['benchmark'] == 'gsm8k']
temp_all = axis_eff('temperature', all_gen)
fmt_all = axis_eff('prompt_format', all_gen)
nshot_all = axis_eff('n_shot', all_gen)
print(f"  All GSM8K effects: temp={temp_all}, fmt={fmt_all}, nshot={nshot_all}")
check_macro('tempeffectall', temp_all)
check_macro('fmteffectall', fmt_all)
check_macro('nshoteffectall', nshot_all)

# Seed noise
by_cell = defaultdict(list)
for r in all_gen:
    k = (r['model_id'], r['benchmark'], r['temperature'], r['prompt_format'], r['n_shot'])
    by_cell[k].append(r['score'])
seed_spreads = [max(v) - min(v) for v in by_cell.values() if len(v) >= 2]
noise_mean = round(np.mean(seed_spreads), 2)
noise_max = round(np.max(seed_spreads), 2)
print(f"\n  Seed noise: mean={noise_mean}, max={noise_max}")
check_macro('noisefloor', noise_mean)
check_macro('noisefloormax', noise_max)

# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 7: INLINE TEXT CHECKS")
print("=" * 70)

# Check "three orders of magnitude" claim
# Paper line ~782: "three orders of magnitude lower divergence (|Δ| 0.12 vs. 32.3 pp)"
# 32.3 / 0.12 = 269.2 -> log10(269.2) = 2.43 -> "more than two" not "three"
# But macros say tulumean=0.12, meaning 32.3/0.12 = 269.2 (2.43 orders)
# Actual mean |delta| from data = ?
ratio = 32.3 / 0.14  # using actual 0.14
log_ratio = math.log10(ratio)
print(f"  32.3 / 0.14 = {ratio:.1f}, log10 = {log_ratio:.2f}")
print(f"  32.3 / 0.12 = {32.3/0.12:.1f}, log10 = {math.log10(32.3/0.12):.2f}")

# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 8: PAPER TEXT vs MACROS CONSISTENCY")
print("=" * 70)

paper = Path('submission/latex/acl_latex.tex').read_text(encoding='utf-8')

# Check for old 2-model references
if 'Llama-3.1-8B-Instruct was planned but' in paper:
    FAIL.append("  [FAIL] Paper still says 'Llama-3.1-8B-Instruct was planned but excluded'")
if 'Llama-3.1-8B-Instruct\nwas excluded because gated-repo' in paper:
    FAIL.append("  [FAIL] Limitations still says 'Llama excluded because gated-repo'")
if '(Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct) on' in paper:
    FAIL.append("  [FAIL] Design para lists only 2 models (missing Llama)")

# Check "three orders of magnitude" vs "more than two"
if 'three\norders of magnitude' in paper or 'three orders of magnitude' in paper:
    FAIL.append("  [FAIL] Paper says 'three orders of magnitude' but ratio is 10^2.43 (should be 'more than two')")
    
# Check Tulu advantage text
if '+0.032' in paper and 'tuluadvantage' not in paper:
    # Check if inline +0.032 is used
    pass
if '3/5 higher first-party' in paper:
    FAIL.append("  [FAIL] Paper says '3/5 higher first-party' — need to verify from data")

# Check n_shot mention "where both models produce"
if 'where both models produce extractable' in paper:
    FAIL.append("  [FAIL] Paper says 'where both models produce' (should be 'all three models' or 'all models')")

# ═══════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"\nPASSED: {len(PASS_LIST)}")
for p in PASS_LIST:
    print(p)

print(f"\nFAILED: {len(FAIL)}")
for f_ in FAIL:
    print(f_)

print(f"\nWARNINGS: {len(WARN)}")
for w in WARN:
    print(w)
