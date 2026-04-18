"""Final comprehensive z-score verification against paper claims."""
import pandas as pd
import numpy as np
import math
import sys
from itertools import combinations

sys.path.insert(0, '.')
from src.analysis.collision_detection import normalise_benchmark

df = pd.read_csv('data/aggregated/all_results.csv', low_memory=False)

# --- Setup ---
n_models = df['model_id'].nunique()
n_bench_raw = df['benchmark'].nunique()
df['bn'] = df['benchmark'].apply(normalise_benchmark)
n_bench_norm = df['bn'].nunique()

print(f"Models: {n_models}, Raw benchmarks: {n_bench_raw}, Normalised benchmarks: {n_bench_norm}")

# Per-source RECORD counts (R_i) — what the paper uses
sources = df['source'].unique()
R = {}
for s in sources:
    R[s] = len(df[df['source'] == s])

# Universe: U = |models| × |benchmarks| (raw)
U = n_models * n_bench_raw  # 5672 × 180
print(f"U = {n_models} × {n_bench_raw} = {U:,}")

# --- Birthday formula for ALL sources ---
E_all = sum(R[a] * R[b] for a, b in combinations(R.keys(), 2)) / U
SD_all = math.sqrt(E_all)

# Observed collisions (default normalisation = 16)
obs_default = 16
z_default = (obs_default - E_all) / SD_all

# Observed collisions (strict = exact match = 6)
obs_strict = 6
z_strict = (obs_strict - E_all) / SD_all

print(f"\n{'='*60}")
print("ALL SOURCES")
print(f"{'='*60}")
print(f"E = {E_all:.1f}, SD = {SD_all:.1f}")
print(f"Default normalisation: Obs=16, z = (16 - {E_all:.1f}) / {SD_all:.1f} = {z_default:.1f}")
print(f"Strict normalisation:  Obs=6,  z = (6 - {E_all:.1f}) / {SD_all:.1f} = {z_strict:.1f}")
print(f"Paper claims: E=69±8, z_default=-6.4, z_strict=-7.6")
print(f"Match E=69? {abs(round(E_all) - 69) == 0}")
print(f"Match SD=8? {abs(round(SD_all) - 8) == 0}")
print(f"Match z_default=-6.4? {abs(round(z_default, 1) - (-6.4)) < 0.01}")
print(f"Match z_strict=-7.6? {abs(round(z_strict, 1) - (-7.6)) < 0.01}")

# --- Excl OLv2 ---
R_no = {s: v for s, v in R.items() if s != 'open_llm_leaderboard_v2'}
E_no = sum(R_no[a] * R_no[b] for a, b in combinations(R_no.keys(), 2)) / U  # Same U!
SD_no = math.sqrt(E_no)
obs_no = 5
z_no = (obs_no - E_no) / SD_no

print(f"\n{'='*60}")
print("EXCL OLv2 (2,541 records)")
print(f"{'='*60}")
print(f"Sum of excl sources: {sum(R_no.values())}")
print(f"E = {E_no:.2f}, SD = {SD_no:.2f}")
print(f"z = (5 - {E_no:.2f}) / {SD_no:.2f} = {z_no:.1f}")
print(f"Paper claims: E=2.7, z=1.4")
print(f"Match E=2.7? {abs(round(E_no, 1) - 2.7) < 0.01}")
print(f"Match z=1.4? {abs(round(z_no, 1) - 1.4) < 0.01}")

# --- Verify observed collision counts ---
print(f"\n{'='*60}")
print("OBSERVED COLLISION COUNTS")
print(f"{'='*60}")

# Default (normalised benchmarks): model_id × benchmark_norm in 2+ sources
df['model_lower'] = df['model_id'].str.strip().str.lower()
default_groups = df.groupby(['model_lower', 'bn'])['source'].nunique()
default_collisions = (default_groups >= 2).sum()
print(f"Default normalisation collisions: {default_collisions} (paper: 16)")

# Strict (exact benchmark match): model_id × benchmark in 2+ sources  
strict_groups = df.groupby(['model_lower', 'benchmark'])['source'].nunique()
strict_collisions = (strict_groups >= 2).sum()
print(f"Strict (exact match) collisions: {strict_collisions} (paper: 6)")

# Excl OLv2
df_no = df[df['source'] != 'open_llm_leaderboard_v2']
no_groups = df_no.groupby(['model_lower', 'bn'])['source'].nunique()
no_collisions = (no_groups >= 2).sum()
print(f"Excl OLv2 (default norm) collisions: {no_collisions} (paper: 5)")

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
all_pass = True
checks = [
    ("E_all ≈ 69", abs(round(E_all) - 69) == 0),
    ("SD_all ≈ 8", abs(round(SD_all) - 8) == 0),
    ("z_default ≈ -6.4", abs(round(z_default, 1) - (-6.4)) < 0.05),
    ("z_strict ≈ -7.6", abs(round(z_strict, 1) - (-7.6)) < 0.05),
    ("E_excl ≈ 2.7", abs(round(E_no, 1) - 2.7) < 0.05),
    ("z_excl ≈ 1.4", abs(round(z_no, 1) - 1.4) < 0.05),
    ("Obs default = 16", default_collisions == 16),
    ("Obs strict = 6", strict_collisions == 6),
    ("Obs excl OLv2 = 5", no_collisions == 5),
]
for label, ok in checks:
    status = "PASS" if ok else "FAIL"
    if not ok:
        all_pass = False
    print(f"  [{status}] {label}")

if all_pass:
    print("\nAll z-score checks PASS!")
else:
    print("\nSome checks FAILED — investigate!")
