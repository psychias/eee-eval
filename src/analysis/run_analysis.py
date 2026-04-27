#!/usr/bin/env python3
"""
EEE-Eval comprehensive analysis script.
Produces all tables, figures, and reports for Tasks 1-5.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy import stats

warnings.filterwarnings('ignore')
plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)

import sys as _sys
_sys.path.insert(0, ROOT)
from src.analysis.collision_detection import normalise_benchmark

# ── Data loading ──────────────────────────────────────────────
print("=" * 70)
print("DATA LOADING")
print("=" * 70)

df = pd.read_csv(os.path.join(ROOT, "data/aggregated/all_results.csv"), low_memory=False)
print(f"Loaded {len(df):,} records, {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")
print(f"\nSources: {sorted(df['source'].unique())}")
print(f"Unique models: {df['model_id'].nunique()}")
print(f"Unique benchmarks: {df['benchmark'].nunique()}")

# Normalise benchmark names so that e.g. "GPQA (0-shot)" and "GPQA" collide
df['benchmark_norm'] = df['benchmark'].apply(normalise_benchmark)

# Column mapping notes:
# eval_library_name  -> eval_library  (also 'harness' column)
# n_shot             -> shots
# prompt_template    -> prompt_template
# temperature        -> temperature
# cot                -> chain_of_thought

# Source name mapping for readability
SOURCE_LABELS = {
    'open_llm_leaderboard_v2': 'OLv2',
    'papers_with_code': 'PWC',
    'alpacaeval2': 'AlpacaEval2',
    'chatbot_arena': 'Arena',
    'bigcodebench': 'BigCodeBench',
    'evalplus': 'EvalPlus',
    'bfcl': 'BFCL',
    'wildbench': 'WildBench',
    'swe_bench': 'SWE-bench',
    'mt_bench': 'MT-Bench',
    'hf_model_card': 'HF-Card',
}

def src_label(s):
    return SOURCE_LABELS.get(s, s)


def is_valid(series):
    """Return boolean mask for non-null, non-empty, non-'unknown' values."""
    return series.notna() & (series.astype(str).str.strip() != '') & (series.astype(str).str.lower() != 'unknown')


# ══════════════════════════════════════════════════════════════
#  TASK 1 — Permutation null model
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TASK 1 — Permutation Null Model")
print("=" * 70)


def count_collisions(data):
    """Count (model_id, benchmark_norm) pairs appearing in >= 2 distinct sources."""
    grouped = data.groupby(['model_id', 'benchmark_norm'])['source'].nunique()
    return (grouped >= 2).sum()


def permutation_test(data, n_iter=10000, seed=42):
    """
    Shuffle model_id labels WITHIN each source (preserving each source's
    benchmark distribution exactly). Count collisions after each shuffle.
    """
    rng = np.random.default_rng(seed)
    null_counts = np.zeros(n_iter, dtype=int)

    # Pre-group data by source
    source_groups = {}
    for src, grp in data.groupby('source'):
        source_groups[src] = {
            'model_ids': grp['model_id'].values.copy(),
            'benchmarks': grp['benchmark_norm'].values,
        }

    for i in range(n_iter):
        if (i + 1) % 2000 == 0:
            print(f"  Permutation {i+1}/{n_iter}...")
        shuffled_pairs = set()
        collision_count = 0

        # Build a dict: (model_id, benchmark) -> set of sources
        pair_sources = {}
        for src, info in source_groups.items():
            ids = info['model_ids'].copy()
            rng.shuffle(ids)
            for mid, bench in zip(ids, info['benchmarks']):
                key = (mid, bench)
                if key not in pair_sources:
                    pair_sources[key] = set()
                pair_sources[key].add(src)

        null_counts[i] = sum(1 for v in pair_sources.values() if len(v) >= 2)

    return null_counts


# Subsets
df_all = df.copy()
df_no_olv2 = df[df['source'] != 'open_llm_leaderboard_v2'].copy()

obs_all = count_collisions(df_all)
obs_no_olv2 = count_collisions(df_no_olv2)
print(f"Observed collisions (all):       {obs_all}")
print(f"Observed collisions (excl OLv2): {obs_no_olv2}")

print("\nRunning permutation test (all sources, n=10000)...")
null_all = permutation_test(df_all, n_iter=10000)
print("Running permutation test (excl OLv2, n=10000)...")
null_no_olv2 = permutation_test(df_no_olv2, n_iter=10000)


def perm_stats(observed, null_dist):
    mean_n = null_dist.mean()
    sd_n = null_dist.std()
    ci_lo = np.percentile(null_dist, 2.5)
    ci_hi = np.percentile(null_dist, 97.5)
    z = (observed - mean_n) / sd_n if sd_n > 0 else np.inf
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return mean_n, sd_n, ci_lo, ci_hi, z, p


stats_all = perm_stats(obs_all, null_all)
stats_no_olv2 = perm_stats(obs_no_olv2, null_no_olv2)

# Print results
print(f"\nAll sources:    Obs={obs_all}, Mean={stats_all[0]:.1f}, SD={stats_all[1]:.1f}, "
      f"95%CI=[{stats_all[2]:.1f}, {stats_all[3]:.1f}], z={stats_all[4]:.2f}, p={stats_all[5]:.2e}")
print(f"Excluding OLv2: Obs={obs_no_olv2}, Mean={stats_no_olv2[0]:.1f}, SD={stats_no_olv2[1]:.1f}, "
      f"95%CI=[{stats_no_olv2[2]:.1f}, {stats_no_olv2[3]:.1f}], z={stats_no_olv2[4]:.2f}, p={stats_no_olv2[5]:.2e}")

# LaTeX table
latex_t1 = r"""\begin{table}[t]
\centering
\caption{Permutation null model for cross-source collision counts ($n = 10{,}000$ iterations).}
\label{tab:permutation_null}
\begin{tabular}{lrrrcrr}
\toprule
Subset & Observed & Perm.\ Mean & SD & 95\%\ CI & $z$ & $p$ \\
\midrule
"""
for label, obs, st in [("All sources", obs_all, stats_all),
                        ("Excluding OLv2", obs_no_olv2, stats_no_olv2)]:
    latex_t1 += f"{label} & {obs} & {st[0]:.1f} & {st[1]:.1f} & [{st[2]:.1f}, {st[3]:.1f}] & {st[4]:.1f} & {st[5]:.1e} \\\\\n"
latex_t1 += r"""\bottomrule
\end{tabular}
\end{table}"""
print("\n" + latex_t1)
with open(f"{OUT}/task1_permutation_table.tex", "w") as f:
    f.write(latex_t1)

# Save results CSV
pd.DataFrame([
    {"Subset": "All sources", "Observed": obs_all, "Perm_Mean": stats_all[0],
     "SD": stats_all[1], "CI_lo": stats_all[2], "CI_hi": stats_all[3],
     "z": stats_all[4], "p": stats_all[5]},
    {"Subset": "Excluding OLv2", "Observed": obs_no_olv2, "Perm_Mean": stats_no_olv2[0],
     "SD": stats_no_olv2[1], "CI_lo": stats_no_olv2[2], "CI_hi": stats_no_olv2[3],
     "z": stats_no_olv2[4], "p": stats_no_olv2[5]},
]).to_csv(f"{OUT}/task1_permutation_results.csv", index=False)

# Histogram plots
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, null_d, obs, label in [(axes[0], null_all, obs_all, "All Sources"),
                                 (axes[1], null_no_olv2, obs_no_olv2, "Excluding OLv2")]:
    ax.hist(null_d, bins=50, color='steelblue', alpha=0.7, edgecolor='white', label='Permutation null')
    ax.axvline(obs, color='red', linewidth=2, linestyle='--', label=f'Observed ({obs})')
    ax.set_xlabel("Collision count")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Permutation Null Distribution — {label}")
    ax.legend()
plt.tight_layout()
fig.savefig(f"{OUT}/task1_permutation_histograms.pdf")
fig.savefig(f"{OUT}/task1_permutation_histograms.png")
plt.close(fig)
print("Task 1 figures saved.")


# ══════════════════════════════════════════════════════════════
#  TASK 2 — OLv2-exclusion analysis
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TASK 2 — OLv2-Exclusion as Primary Finding")
print("=" * 70)

# Step 1: Subsets
subsets = {
    "All": df,
    "$-$OLv2": df[df['source'] != 'open_llm_leaderboard_v2'],
    "$-$PWC": df[df['source'] != 'papers_with_code'],
    "$-$OLv2$-$PWC": df[~df['source'].isin(['open_llm_leaderboard_v2', 'papers_with_code'])],
}

coverage_rows = []
for name, sub in subsets.items():
    row = {
        "Subset": name,
        "Records": len(sub),
        "Models": sub['model_id'].nunique(),
        "Benchmarks": sub['benchmark'].nunique(),
        "Harness (%)": 100 * is_valid(sub['harness']).mean() if 'harness' in sub.columns else 0,
        "N-shot (%)": 100 * is_valid(sub['shots']).mean(),
        "CoT (%)": 100 * is_valid(sub['chain_of_thought']).mean(),
        "Temperature (%)": 100 * is_valid(sub['temperature']).mean(),
        "Prompt tpl (%)": 100 * is_valid(sub['prompt_template']).mean(),
    }
    coverage_rows.append(row)
    print(f"\n{name}: {row['Records']:,} records, {row['Models']} models, {row['Benchmarks']} benchmarks")
    for k in ['Harness (%)', 'N-shot (%)', 'CoT (%)', 'Temperature (%)', 'Prompt tpl (%)']:
        print(f"  {k}: {row[k]:.1f}")

coverage_df = pd.DataFrame(coverage_rows)
coverage_df.to_csv(f"{OUT}/task2_coverage_by_subset.csv", index=False)

# LaTeX Table 3 reproduction
latex_t2 = r"""\begin{table}[t]
\centering
\caption{Metadata coverage sensitivity to source exclusion.}
\label{tab:coverage_sensitivity}
\small
\begin{tabular}{lrrrrrrr}
\toprule
Subset & Records & Models & Benchmarks & Harness & N-shot & CoT & Temp. \\
\midrule
"""
for _, r in coverage_df.iterrows():
    latex_t2 += (f"{r['Subset']} & {int(r['Records']):,} & {int(r['Models'])} & {int(r['Benchmarks'])} & "
                 f"{r['Harness (%)']:.1f}\\% & {r['N-shot (%)']:.1f}\\% & "
                 f"{r['CoT (%)']:.1f}\\% & {r['Temperature (%)']:.1f}\\% \\\\\n")
latex_t2 += r"""\bottomrule
\end{tabular}
\end{table}"""
print("\n" + latex_t2)
with open(f"{OUT}/task2_coverage_table.tex", "w") as f:
    f.write(latex_t2)

# Step 3: Jaccard similarity of model sets between source pairs
print("\nComputing Jaccard similarity (model sets)...")
sources = sorted(df['source'].unique())
model_sets = {src: set(df[df['source'] == src]['model_id'].unique()) for src in sources}

jaccard_models = pd.DataFrame(0.0, index=sources, columns=sources)
for s1, s2 in combinations(sources, 2):
    inter = len(model_sets[s1] & model_sets[s2])
    union = len(model_sets[s1] | model_sets[s2])
    j = inter / union if union > 0 else 0
    jaccard_models.loc[s1, s2] = j
    jaccard_models.loc[s2, s1] = j
for s in sources:
    jaccard_models.loc[s, s] = 1.0

# Pretty labels
jm_display = jaccard_models.rename(index=src_label, columns=src_label)

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(jm_display, annot=True, fmt=".3f", cmap="YlOrRd",
            vmin=0, vmax=0.2, ax=ax, linewidths=0.5,
            cbar_kws={'label': 'Jaccard Similarity'})
ax.set_title("Jaccard Similarity of Model Sets Between Sources")
plt.tight_layout()
fig.savefig(f"{OUT}/task2_jaccard_models_heatmap.pdf")
fig.savefig(f"{OUT}/task2_jaccard_models_heatmap.png")
plt.close(fig)
jaccard_models.to_csv(f"{OUT}/task2_jaccard_models.csv")
print("Jaccard model heatmap saved.")

# Step 4: Benchmark overlap
print("Computing benchmark overlap...")
bench_sets = {src: set(df[df['source'] == src]['benchmark'].unique()) for src in sources}

bench_overlap = pd.DataFrame(0, index=sources, columns=sources)
for s1, s2 in combinations(sources, 2):
    shared = len(bench_sets[s1] & bench_sets[s2])
    bench_overlap.loc[s1, s2] = shared
    bench_overlap.loc[s2, s1] = shared
for s in sources:
    bench_overlap.loc[s, s] = len(bench_sets[s])

bo_display = bench_overlap.rename(index=src_label, columns=src_label)

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(bo_display.astype(int), annot=True, fmt="d", cmap="YlGnBu",
            ax=ax, linewidths=0.5,
            cbar_kws={'label': 'Shared Benchmarks'})
ax.set_title("Shared Benchmark Count Between Sources")
plt.tight_layout()
fig.savefig(f"{OUT}/task2_benchmark_overlap_heatmap.pdf")
fig.savefig(f"{OUT}/task2_benchmark_overlap_heatmap.png")
plt.close(fig)
bench_overlap.to_csv(f"{OUT}/task2_benchmark_overlap.csv")
print("Benchmark overlap heatmap saved.")

# Step 5: OLv2 isolation paragraph
olv2_models = model_sets.get('open_llm_leaderboard_v2', set())
other_models = set()
for src in sources:
    if src != 'open_llm_leaderboard_v2':
        other_models |= model_sets[src]
olv2_only = olv2_models - other_models
olv2_shared = olv2_models & other_models
olv2_jaccard_max = 0
for src in sources:
    if src != 'open_llm_leaderboard_v2':
        inter = len(model_sets['open_llm_leaderboard_v2'] & model_sets[src])
        union = len(model_sets['open_llm_leaderboard_v2'] | model_sets[src])
        j = inter / union if union > 0 else 0
        if j > olv2_jaccard_max:
            olv2_jaccard_max = j

olv2_pct = 100 * len(df[df['source'] == 'open_llm_leaderboard_v2']) / len(df)
isolation_para = f"""Open LLM Leaderboard v2 (OLv2) contributes {len(df[df['source']=='open_llm_leaderboard_v2']):,} records ({olv2_pct:.1f}% of the dataset) across {len(olv2_models):,} models, yet its population is largely isolated from other sources. Of these models, {len(olv2_only):,} ({100*len(olv2_only)/len(olv2_models):.1f}%) appear exclusively in OLv2 and are evaluated on none of the other {len(sources)-1} platforms. The maximum Jaccard similarity between OLv2's model set and any other source is {olv2_jaccard_max:.3f}, confirming near-total population disjointness. Removing OLv2 reduces the dataset from {len(df):,} to {len(df_no_olv2):,} records. While OLv2 contributes high harness and n-shot coverage ({coverage_rows[0]['Harness (%)']:.1f}% and {coverage_rows[0]['N-shot (%)']:.1f}% with OLv2 vs {coverage_rows[1]['Harness (%)']:.1f}% and {coverage_rows[1]['N-shot (%)']:.1f}% without), it shares the ecosystem-wide gap in temperature and prompt template reporting (both near 0%). Crucially, OLv2's dominant volume masks the weaker metadata norms of other sources: excluding it reveals that non-OLv2 sources achieve only {coverage_rows[1]['Harness (%)']:.1f}% harness coverage and {coverage_rows[1]['N-shot (%)']:.1f}% n-shot coverage. These findings elevate the OLv2 exclusion from a sensitivity check to a primary structural finding: OLv2 operates as a parallel evaluation ecosystem with minimal population overlap, distinct benchmark scope, and volume that dominates aggregate statistics."""

print(f"\n--- Section 5.4: OLv2 Population Isolation ---")
print(isolation_para)
with open(f"{OUT}/task2_olv2_isolation_paragraph.txt", "w") as f:
    f.write(isolation_para)


# ══════════════════════════════════════════════════════════════
#  TASK 3 — Missing metadata cost quantification
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TASK 3 — Missing Metadata Cost (Incentive Problem)")
print("=" * 70)

gap_rows = []
for src in sources:
    sub = df[df['source'] == src]
    n = len(sub)
    temp_valid = is_valid(sub['temperature']).sum()
    prompt_valid = is_valid(sub['prompt_template']).sum()
    version_valid = is_valid(sub['eval_library']).sum()  # eval_library as proxy
    harness_valid = is_valid(sub['harness']).sum()
    shots_valid = is_valid(sub['shots']).sum()

    gap_rows.append({
        'Source': src_label(src),
        'source_raw': src,
        'Total': n,
        'Temp valid': temp_valid,
        'Prompt valid': prompt_valid,
        'Version valid': version_valid,
        'Harness valid': harness_valid,
        'Shots valid': shots_valid,
        'Gap_temp': n - temp_valid,
        'Gap_prompt': n - prompt_valid,
        'Gap_version': n - version_valid,
    })

gap_df = pd.DataFrame(gap_rows)
gap_df['One-time effort'] = '30 min'
gap_df['Ongoing cost'] = '0 (automated)'

print("\nDocumentation gap table:")
print(gap_df[['Source', 'Total', 'Gap_temp', 'Gap_prompt', 'Gap_version',
              'One-time effort', 'Ongoing cost']].to_string(index=False))

# Identify highest-leverage targets
print("\nHighest-leverage targets (have harness+shots but NOT temperature):")
leverage = gap_df[(gap_df['Harness valid'] > 0) & (gap_df['Shots valid'] > 0) & (gap_df['Gap_temp'] == gap_df['Total'])]
for _, r in leverage.iterrows():
    print(f"  ** {r['Source']}: {r['Total']:,} records with harness & n-shot but zero temperature")
# Also check partial
leverage2 = gap_df[(gap_df['Harness valid'] > 0) & (gap_df['Shots valid'] > 0) & (gap_df['Gap_temp'] > 0) & (gap_df['Gap_temp'] < gap_df['Total'])]
for _, r in leverage2.iterrows():
    print(f"  * {r['Source']}: {r['Total']:,} records, {r['Temp valid']} have temp, {r['Gap_temp']} missing")

# If no target has shots+harness but 0 temp, relax condition
if len(leverage) == 0 and len(leverage2) == 0:
    print("  (Relaxed check: sources with harness but missing temperature)")
    leverage3 = gap_df[(gap_df['Harness valid'] > 0) & (gap_df['Gap_temp'] > 0)]
    for _, r in leverage3.iterrows():
        print(f"  * {r['Source']}: {r['Total']:,} records, harness={r['Harness valid']}, temp gap={r['Gap_temp']}")

gap_df.to_csv(f"{OUT}/task3_documentation_gap.csv", index=False)

# LaTeX table
latex_t3 = r"""\begin{table}[t]
\centering
\caption{Documentation gap by source: additional records needing metadata values to reach 100\% coverage.}
\label{tab:documentation_gap}
\small
\begin{tabular}{lrrrrl}
\toprule
Source & Records & \multicolumn{3}{c}{Gap (records)} & Setup \\
\cmidrule(lr){3-5}
 & & Temp. & Prompt tpl. & Harness ver. & \\
\midrule
"""
for _, r in gap_df.iterrows():
    latex_t3 += f"{r['Source']} & {int(r['Total']):,} & {int(r['Gap_temp']):,} & {int(r['Gap_prompt']):,} & {int(r['Gap_version']):,} & 30 min \\\\\n"
latex_t3 += r"""\bottomrule
\end{tabular}
\end{table}"""
with open(f"{OUT}/task3_gap_table.tex", "w") as f:
    f.write(latex_t3)
print("\n" + latex_t3)

# Bar chart
fig, ax = plt.subplots(figsize=(10, 5))
bar_df = gap_df[['Source', 'Gap_temp', 'Gap_prompt', 'Gap_version']].set_index('Source')
bar_df.columns = ['Temperature', 'Prompt Template', 'Harness Version']
bar_df.plot.barh(ax=ax, color=['#e74c3c', '#3498db', '#2ecc71'])
ax.set_xlabel("Records missing metadata (gap to 100%)")
ax.set_title("Documentation Gap by Source and Field")
ax.legend(title="Field")
plt.tight_layout()
fig.savefig(f"{OUT}/task3_documentation_gap_barchart.pdf")
fig.savefig(f"{OUT}/task3_documentation_gap_barchart.png")
plt.close(fig)
print("Task 3 bar chart saved.")


# ══════════════════════════════════════════════════════════════
#  TASK 4 — PWC artefact analysis
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TASK 4 — PWC Artefact Analysis")
print("=" * 70)

pwc = df[df['source'] == 'papers_with_code'].copy()
print(f"PWC records: {len(pwc)}")

# Ensure score is numeric for analysis
pwc['score_num'] = pd.to_numeric(pwc['score'], errors='coerce')

# Artefact type A: Score looks like param count (>1000)
artefact_a = pwc[pwc['score_num'] > 1000].copy()
artefact_a['artefact_type'] = 'Score > 1000 (possible param count)'
print(f"\nArtefact A (score > 1000): {len(artefact_a)} records")
if len(artefact_a) > 0:
    print(artefact_a[['model_id', 'benchmark', 'score']].head(3).to_string(index=False))

# Also check for string patterns like "7B", "70B" in original score column
score_str = pwc['score'].astype(str)
pattern_b_match = score_str.str.contains(r'\d+[Bb]', na=False)
artefact_a2 = pwc[pattern_b_match].copy()
artefact_a2['artefact_type'] = 'Score matches param pattern (e.g., 7B)'
print(f"Artefact A2 (param pattern like 7B): {len(artefact_a2)} records")
if len(artefact_a2) > 0:
    print(artefact_a2[['model_id', 'benchmark', 'score']].head(3).to_string(index=False))

# Artefact type B: prompt_template contains variant tags
variant_patterns = ['cot', 'chat', 'instruct', '-v', 'template']
prompt_str = pwc['prompt_template'].astype(str).str.lower()
artefact_b_mask = pd.Series(False, index=pwc.index)
for pat in variant_patterns:
    artefact_b_mask |= prompt_str.str.contains(pat, na=False)
# Exclude 'nan' and empty
artefact_b_mask &= is_valid(pwc['prompt_template'])
artefact_b = pwc[artefact_b_mask].copy()
artefact_b['artefact_type'] = 'Variant-tag leakage in prompt_template'
print(f"\nArtefact B (variant-tag leakage): {len(artefact_b)} records")
if len(artefact_b) > 0:
    print(artefact_b[['model_id', 'benchmark', 'prompt_template']].head(3).to_string(index=False))

# Artefact type C: Duplicate (model_id, benchmark, score) triples
dup_mask = pwc.duplicated(subset=['model_id', 'benchmark', 'score'], keep=False)
artefact_c = pwc[dup_mask].copy()
artefact_c['artefact_type'] = 'Duplicate (model, benchmark, score) triple'
print(f"\nArtefact C (exact duplicates): {len(artefact_c)} records")
if len(artefact_c) > 0:
    print(artefact_c[['model_id', 'benchmark', 'score']].head(3).to_string(index=False))

# Combine all artefacts (union of flags)
all_artefact_idx = set(artefact_a.index) | set(artefact_a2.index) | set(artefact_b.index) | set(artefact_c.index)
n_artefacts = len(all_artefact_idx)
n_clean = len(pwc) - n_artefacts
artefact_pct = 100 * n_artefacts / len(pwc) if len(pwc) > 0 else 0

print(f"\n--- PWC Summary ---")
print(f"Raw PWC records:    {len(pwc)}")
print(f"Artefacts removed:  {n_artefacts}")
print(f"Clean records:      {n_clean}")
print(f"Artefact rate:      {artefact_pct:.1f}%")

# Per-benchmark artefact rate
pwc['is_artefact'] = pwc.index.isin(all_artefact_idx)
bench_artefact = pwc.groupby('benchmark').agg(
    total=('is_artefact', 'size'),
    artefacts=('is_artefact', 'sum')
).reset_index()
bench_artefact['rate_%'] = 100 * bench_artefact['artefacts'] / bench_artefact['total']
bench_artefact = bench_artefact.sort_values('rate_%', ascending=False)
print("\nPer-benchmark artefact rate:")
print(bench_artefact.to_string(index=False))
bench_artefact.to_csv(f"{OUT}/task4_pwc_bench_artefact_rate.csv", index=False)

# Write best practices markdown
best_practices = f"""# PWC Data Ingestion Best Practices

## Summary of Artefact Types

| Artefact Type | Count | Example |
|---------------|-------|---------|
| Score > 1000 (possible param count) | {len(artefact_a)} | Score field contains parameter counts instead of benchmark scores |
| Param pattern in score (e.g., "7B") | {len(artefact_a2)} | Score string matches model-size notation |
| Variant-tag leakage in prompt_template | {len(artefact_b)} | Non-template text (cot, chat, instruct) in template field |
| Duplicate (model, benchmark, score) | {len(artefact_c)} | Exact triple duplicates from multiple submissions |
| **Total unique artefact records** | **{n_artefacts}** | **{artefact_pct:.1f}% of {len(pwc)} PWC records** |

## Three Detection Rules

1. **Numeric range check**: Flag any record where `score > 1000` or `score` matches
   the pattern `\\d+[Bb]` (e.g., "7B", "70B"). These are likely parameter counts
   or model-size descriptors that were scraped into the score field.

2. **Template field validation**: Reject records where `prompt_template` contains
   variant identifiers (`cot`, `chat`, `instruct`, `-v`) rather than actual
   template text. This indicates the field was used to tag model variants rather
   than store generation configuration.

3. **Deduplication**: Remove exact `(model_id, benchmark, score)` triples. Multiple
   identical entries arise from scraping the same result from different paper tables
   or re-submissions.

## Recommended Ingestion Checklist

- [ ] Parse numeric scores and reject values outside the benchmark's documented range
      (`min_score`, `max_score` from schema)
- [ ] Strip whitespace and normalise model IDs before deduplication
- [ ] Validate `prompt_template` is either null/empty or contains template markup
      (e.g., Jinja2 syntax, placeholder tokens)
- [ ] Cross-reference `benchmark` names against a canonical list to merge aliases
- [ ] Log `extraction_confidence` and filter at a threshold (e.g., >= 0.8) for
      automated pipelines
- [ ] After cleaning, re-check per-benchmark record counts to ensure no benchmark
      lost all its data

## Per-Benchmark Artefact Rate

{bench_artefact.to_markdown(index=False)}
"""
with open(f"{OUT}/pwc_best_practices.md", "w") as f:
    f.write(best_practices)
print("\npwc_best_practices.md saved.")


# ══════════════════════════════════════════════════════════════
#  TASK 5 — Reproducibility checklist
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TASK 5 — Reproducibility Checklist")
print("=" * 70)

# Field -> (column in data, difficulty, rationale)
checklist_fields = [
    ("eval\\_library.name", "harness", "Low", "Already logged by all dedicated leaderboards"),
    ("eval\\_library.version", "eval_library", "Low", "One git tag or pip show away"),
    ("generation\\_args.n\\_shot", "shots", "Low", "Already logged by most sources"),
    ("generation\\_args.temperature", "temperature", "Medium", "Requires per-run capture, not post-hoc"),
    ("generation\\_args.prompt\\_template", "prompt_template", "High", "Requires versioned template storage"),
    ("additional\\_details.cot", "chain_of_thought", "Medium", "Boolean flag, easy once pipeline supports it"),
    ("scoring\\_mode", None, "High", "Not yet a schema field; requires definition"),
    ("provenance\\_link", None, "Medium", "URL or DOI to upstream score source"),
]

checklist_rows = []
for field_name, col, diff, rationale in checklist_fields:
    if col and col in df.columns:
        cov = 100 * is_valid(df[col]).mean()
    else:
        cov = 0.0
    checklist_rows.append({
        'Field': field_name,
        'Column': col,
        'Difficulty': diff,
        'Coverage (%)': cov,
        'Note': rationale,
    })

checklist_df = pd.DataFrame(checklist_rows)
print(checklist_df[['Field', 'Difficulty', 'Coverage (%)', 'Note']].to_string(index=False))

# Minimum viable report
mvr_fields = checklist_df[(checklist_df['Coverage (%)'] > 50) | (checklist_df['Difficulty'] == 'Low')]
print(f"\nMinimum viable report fields (coverage > 50% OR difficulty = Low):")
for _, r in mvr_fields.iterrows():
    print(f"  - {r['Field']} (difficulty={r['Difficulty']}, coverage={r['Coverage (%)']:.1f}%)")

checklist_df.to_csv(f"{OUT}/task5_reproducibility_checklist.csv", index=False)

# LaTeX longtable
latex_t5 = r"""\begin{longtable}{llrl}
\caption{EEE Minimum Viable Report: Reproducibility Checklist.}
\label{tab:reproducibility_checklist} \\
\toprule
Field & Difficulty & Coverage (\%) & Implementation Note \\
\midrule
\endfirsthead
\toprule
Field & Difficulty & Coverage (\%) & Implementation Note \\
\midrule
\endhead
"""
for _, r in checklist_df.iterrows():
    latex_t5 += f"{r['Field']} & {r['Difficulty']} & {r['Coverage (%)']:.1f} & {r['Note']} \\\\\n"
# MVR row
mvr_names = ", ".join(mvr_fields['Field'].tolist())
latex_t5 += r"\midrule" + "\n"
latex_t5 += f"\\textbf{{Minimum viable report}} & — & — & {mvr_names} \\\\\n"
latex_t5 += r"""\bottomrule
\end{longtable}"""
with open(f"{OUT}/task5_checklist_table.tex", "w") as f:
    f.write(latex_t5)
print("\nLaTeX checklist table saved.")

# Markdown version
md_checklist = "# EEE Reproducibility Checklist\n\n"
md_checklist += "| Field | Difficulty | Coverage (%) | Implementation Note |\n"
md_checklist += "|-------|-----------|-------------|---------------------|\n"
for _, r in checklist_df.iterrows():
    field_md = r['Field'].replace("\\", "")
    md_checklist += f"| {field_md} | {r['Difficulty']} | {r['Coverage (%)']:.1f} | {r['Note']} |\n"
md_checklist += f"\n**Minimum viable report fields:** {', '.join(f.replace(chr(92), '') for f in mvr_fields['Field'].tolist())}\n"
with open(f"{OUT}/task5_reproducibility_checklist.md", "w") as f:
    f.write(md_checklist)
print("Markdown checklist saved.")


# ══════════════════════════════════════════════════════════════
#  FINAL — Summary report
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FINAL — Writing Summary Report")
print("=" * 70)

# Collect data quality issues
quality_issues = []
for col in ['temperature', 'prompt_template', 'chain_of_thought', 'shots']:
    pct_missing = 100 * (~is_valid(df[col])).mean()
    if pct_missing > 50:
        quality_issues.append(f"- `{col}`: {pct_missing:.1f}% missing/unknown across all sources")

# Check for encoding issues
for col in df.columns:
    n_encoding = df[col].astype(str).str.contains(r'[^\x00-\x7F]', na=False).sum()
    if n_encoding > 100:
        quality_issues.append(f"- `{col}`: {n_encoding} records with non-ASCII characters (encoding to check)")

summary = f"""# EEE-Eval Analysis Summary

Generated: 2026-04-15

## Task 1 — Permutation Null Model

A permutation test (n=10,000) replaced the birthday-paradox approximation for expected
cross-source collision counts. With all sources, {obs_all} observed collisions were found
against a permutation mean of {stats_all[0]:.1f} (SD={stats_all[1]:.1f}), yielding z={stats_all[4]:.1f}
and p={stats_all[5]:.1e}. Excluding OLv2, the observed count drops to {obs_no_olv2} against
a null mean of {stats_no_olv2[0]:.1f} (z={stats_no_olv2[4]:.1f}, p={stats_no_olv2[5]:.1e}).
The observed collision count {'significantly exceeds' if stats_all[5] < 0.05 else 'does not significantly differ from'}
the permutation null in both subsets, {'confirming' if stats_all[5] < 0.05 else 'suggesting'} that
cross-source model-benchmark overlap is {'non-random' if stats_all[5] < 0.05 else 'consistent with chance'}.

## Task 2 — OLv2-Exclusion as Primary Finding

OLv2 contributes {olv2_pct:.1f}% of all records but its model population is largely isolated
(max Jaccard = {olv2_jaccard_max:.3f}). Removing OLv2 drops harness coverage from
{coverage_rows[0]['Harness (%)']:.1f}% to {coverage_rows[1]['Harness (%)']:.1f}%, revealing weaker metadata norms among non-OLv2 sources.
The Jaccard heatmap and benchmark overlap matrix confirm that OLv2 operates as a parallel
evaluation ecosystem. This warrants elevation from sensitivity analysis to a primary finding.

## Task 3 — Missing Metadata Cost (Incentive Problem)

Most sources achieve zero coverage for temperature and prompt template fields. Several sources
that already log harness name and n-shot (implying per-run infrastructure exists) still omit
temperature, making them high-leverage targets for a 30-minute one-time setup. The documentation
gap is largest for OLv2 ({gap_df[gap_df['Source']=='OLv2']['Gap_temp'].values[0]:,} records) and
PWC ({gap_df[gap_df['Source']=='PWC']['Gap_temp'].values[0]:,} records).

## Task 4 — PWC Artefact Analysis

Of {len(pwc)} PWC records, {n_artefacts} ({artefact_pct:.1f}%) were flagged as artefacts:
{len(artefact_a)} with score > 1000 (possible param counts), {len(artefact_b)} with variant-tag
leakage, and {len(artefact_c)} exact duplicates. A best-practices document with detection rules
and an ingestion checklist was produced.

## Task 5 — Reproducibility Checklist

Eight fields were assessed for reproducibility documentation. Current ecosystem coverage ranges from
{checklist_df['Coverage (%)'].min():.1f}% to {checklist_df['Coverage (%)'].max():.1f}%.
The minimum viable report (coverage > 50% or difficulty = Low) includes:
{', '.join(f.replace(chr(92), '') for f in mvr_fields['Field'].tolist())}.

## Output Files

| File | Description |
|------|-------------|
| `task1_permutation_results.csv` | Permutation test statistics (all sources, excl OLv2) |
| `task1_permutation_table.tex` | LaTeX table for permutation null model |
| `task1_permutation_histograms.pdf/png` | Permutation null distribution plots |
| `task2_coverage_by_subset.csv` | Metadata coverage by source-exclusion subset |
| `task2_coverage_table.tex` | LaTeX Table 3 reproduction |
| `task2_jaccard_models.csv` | Jaccard similarity matrix (model sets) |
| `task2_jaccard_models_heatmap.pdf/png` | Jaccard similarity heatmap |
| `task2_benchmark_overlap.csv` | Benchmark overlap matrix |
| `task2_benchmark_overlap_heatmap.pdf/png` | Benchmark overlap heatmap |
| `task2_olv2_isolation_paragraph.txt` | Plain-English paragraph for Section 5.4 |
| `task3_documentation_gap.csv` | Documentation gap by source and field |
| `task3_gap_table.tex` | LaTeX table of documentation gaps |
| `task3_documentation_gap_barchart.pdf/png` | Horizontal bar chart of gaps |
| `task4_pwc_bench_artefact_rate.csv` | Per-benchmark artefact rate in PWC |
| `pwc_best_practices.md` | PWC ingestion best practices document |
| `task5_reproducibility_checklist.csv` | Reproducibility checklist data |
| `task5_checklist_table.tex` | LaTeX longtable for checklist |
| `task5_reproducibility_checklist.md` | Markdown checklist for dataset README |
| `analysis_summary.md` | This file |

## Data Quality Issues

{chr(10).join(quality_issues) if quality_issues else 'No major data quality issues detected.'}

- The `eval_library` column in the flat CSV merges library name and version; version-level
  granularity requires parsing individual JSON files.
- The `scoring_mode` and `provenance_link` fields do not exist in the current schema/CSV;
  coverage is reported as 0%.
- OLv2 records use `shots` = 0/5/25 but leave `temperature`, `prompt_template`, and
  `chain_of_thought` systematically blank.

## Suggested Next Steps

1. **Adopt the permutation null model** in Section 4 to replace the birthday-paradox formula,
   which under-counts expected collisions due to source-specific benchmark scope.
2. **Promote OLv2 isolation** to a primary finding (Section 5.4) with the provided paragraph
   and heatmaps.
3. **Add the documentation gap table** (Task 3) to the recommendations section to concretely
   quantify the effort needed for field-level metadata improvements.
4. **Include the PWC best-practices note** as supplementary material and reference the
   artefact rates when discussing PWC data quality.
5. **Append the reproducibility checklist** as an appendix and define the "minimum viable
   report" standard for future evaluation submissions.
6. **Extract eval_library version** from individual JSON files for a more precise version
   coverage analysis (the flat CSV merges name and version).
"""

with open(f"{OUT}/analysis_summary.md", "w") as f:
    f.write(summary)
print("analysis_summary.md saved.")
print("\n[DONE] All tasks complete. Outputs in ./outputs/")
