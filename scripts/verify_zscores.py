"""Verify birthday-paradox z-scores from the paper."""
import pandas as pd
import numpy as np
import math
import sys
from itertools import combinations

sys.path.insert(0, '.')
from src.analysis.collision_detection import normalise_benchmark

df = pd.read_csv('data/aggregated/all_results.csv', low_memory=False)
df['bn'] = df['benchmark'].apply(normalise_benchmark)

n_models = df['model_id'].nunique()
n_bench_raw = df['benchmark'].nunique()
n_bench_norm = df['bn'].nunique()
print(f"Models: {n_models}, Raw benchmarks: {n_bench_raw}, Normalised benchmarks: {n_bench_norm}")

# Per-source sizes (number of records)
sources = df['source'].unique()
sizes = {}
for s in sources:
    sizes[s] = len(df[df['source'] == s])
    
print("\nPer-source sizes:")
for s in sorted(sizes, key=sizes.get, reverse=True):
    print(f"  {s:35s} {sizes[s]:6d}")
print(f"  {'TOTAL':35s} {sum(sizes.values()):6d}")

# But the birthday formula uses per-source UNIQUE (model, benchmark) PAIRS, not raw record counts
# Let's check what the paper actually counts

# Actually, collisions happen at the (model, benchmark) level
# R_i = number of unique (model_id, benchmark_norm) pairs per source
pair_sizes = {}
for s in sources:
    sub = df[df['source'] == s]
    pair_sizes[s] = sub.groupby(['model_id', 'bn']).ngroups
    
print("\nPer-source unique (model, benchmark_norm) pairs:")
for s in sorted(pair_sizes, key=pair_sizes.get, reverse=True):
    print(f"  {s:35s} {pair_sizes[s]:6d}")

# Birthday formula: E = sum_{i<j} R_i * R_j / U
# where U = total number of possible (model, benchmark) cells

for label, bench_count in [("raw (180)", n_bench_raw), ("norm", n_bench_norm)]:
    U = n_models * bench_count
    print(f"\n{'='*60}")
    print(f"Using {label} benchmarks: U = {n_models} x {bench_count} = {U:,}")
    
    # Using pair counts
    E = sum(pair_sizes[a] * pair_sizes[b] 
            for a, b in combinations(pair_sizes.keys(), 2)) / U
    sd = math.sqrt(E)
    z = (16 - E) / sd
    print(f"ALL sources (pair counts):  E={E:.1f}, SD={sd:.1f}, z={z:.1f}")
    
    # Excl OLv2 - still with TOTAL U (the universe doesn't shrink)
    no_olv = {s: v for s, v in pair_sizes.items() if s != 'open_llm_leaderboard_v2'}
    E_no = sum(no_olv[a] * no_olv[b] 
               for a, b in combinations(no_olv.keys(), 2)) / U
    sd_no = math.sqrt(E_no) if E_no > 0 else 0.01
    z_no = (5 - E_no) / sd_no
    print(f"Excl OLv2 (pair counts):    E={E_no:.1f}, SD={sd_no:.2f}, z={z_no:.1f}")

    # Using raw record counts (not de-duped)
    E2 = sum(sizes[a] * sizes[b] 
             for a, b in combinations(sizes.keys(), 2)) / U
    sd2 = math.sqrt(E2)
    z2 = (16 - E2) / sd2
    print(f"ALL sources (record counts): E={E2:.1f}, SD={sd2:.1f}, z={z2:.1f}")

# Now let's also check: what does run_analysis.py actually compute?
# Let me also check with the ACTUAL per-source pair counts from de-duped data
print("\n\n" + "="*60)
print("PERMUTATION APPROACH (quick verification with 1000 iters)")
print("="*60)

# For the permutation test, we need to simulate random assignment of pairs to sources
# and count how many collisions we get
# Let's do a quick analytical check

# First, compute the actual collision count using normalised benchmarks
# Count collisions manually using normalised columns
df['model_id_norm'] = df['model_id'].apply(lambda x: x.strip().lower())
# benchmark_norm already computed as 'bn' above

# Count unique (model_norm, bench_norm) appearing in 2+ sources
merged = df.groupby(['model_id_norm', 'bn'])['source'].apply(lambda x: frozenset(x.unique()))
collision_cells = merged[merged.apply(len) >= 2]
# Count pairwise: each cell with k sources contributes k*(k-1)/2 pairs
from math import comb
actual_collision_pairs = sum(comb(len(s), 2) for s in collision_cells)
print(f"\nActual collision cells (model_norm, bench_norm in 2+ sources): {len(collision_cells)}")
print(f"Actual collision pairs: {actual_collision_pairs}")

# Excl OLv2
df_no_olv = df[df['source'] != 'open_llm_leaderboard_v2']
merged_no = df_no_olv.groupby(['model_id_norm', 'bn'])['source'].apply(lambda x: frozenset(x.unique()))
collision_no = merged_no[merged_no.apply(len) >= 2]
actual_no_olv_pairs = sum(comb(len(s), 2) for s in collision_no)
print(f"Excl OLv2 collision cells: {len(collision_no)}, pairs: {actual_no_olv_pairs}")

# Now run a quick permutation test
rng = np.random.default_rng(42)
n_perm = 10000

# Get unique pairs per source
source_pairs = {}
for s in sources:
    sub = df[df['source'] == s]
    pairs = set(zip(sub['model_id'], sub['bn']))
    source_pairs[s] = pairs

# All unique (model, bench) pairs in the dataset
all_pairs = list(set.union(*source_pairs.values()))
print(f"\nTotal unique (model, bench_norm) pairs in dataset: {len(all_pairs)}")
print(f"Per-source pair counts: {[len(v) for v in source_pairs.values()]}")

# Permutation: shuffle the assignment of pairs to sources
# maintaining source sizes
source_list = sorted(source_pairs.keys())
source_sz = [len(source_pairs[s]) for s in source_list]

perm_counts = []
for _ in range(n_perm):
    # Approach: sample pairs for each source independently from total pool
    # This is the birthday model
    perm_source_pairs = {}
    for i, s in enumerate(source_list):
        idx = rng.choice(len(all_pairs), size=source_sz[i], replace=True)
        perm_source_pairs[s] = set(idx)  # use indices
    
    # Count collisions: pairs appearing in 2+ sources
    collision_count = 0
    for a, b in combinations(source_list, 2):
        collision_count += len(perm_source_pairs[a] & perm_source_pairs[b])
    perm_counts.append(collision_count)

perm_counts = np.array(perm_counts)
E_perm = perm_counts.mean()
SD_perm = perm_counts.std()
z_perm = (16 - E_perm) / SD_perm
print(f"\nPermutation test (all sources, {n_perm} iters):")
print(f"  E = {E_perm:.1f} ± {SD_perm:.1f}")
print(f"  z = {z_perm:.1f}")
print(f"  Paper claims: E=69, SD≈8, z=-6.4")

# Also excl OLv2
source_list_no = [s for s in source_list if s != 'open_llm_leaderboard_v2']
source_sz_no = [len(source_pairs[s]) for s in source_list_no]
all_pairs_no = list(set.union(*[source_pairs[s] for s in source_list_no]))

perm_counts_no = []
for _ in range(n_perm):
    perm_source_pairs = {}
    for i, s in enumerate(source_list_no):
        idx = rng.choice(len(all_pairs_no), size=source_sz_no[i], replace=True)
        perm_source_pairs[s] = set(idx)
    
    collision_count = 0
    for a, b in combinations(source_list_no, 2):
        collision_count += len(perm_source_pairs[a] & perm_source_pairs[b])
    perm_counts_no.append(collision_count)

perm_counts_no = np.array(perm_counts_no)
E_perm_no = perm_counts_no.mean()
SD_perm_no = perm_counts_no.std()
z_perm_no = (5 - E_perm_no) / SD_perm_no
print(f"\nPermutation test (excl OLv2, {n_perm} iters):")
print(f"  E = {E_perm_no:.1f} ± {SD_perm_no:.1f}")
print(f"  z = {z_perm_no:.1f}")
print(f"  Paper claims: E=2.7, z=1.4")
