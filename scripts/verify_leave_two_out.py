"""Verify leave-two-out analysis table from paper appendix."""
import pandas as pd
import math
import sys
from itertools import combinations

sys.path.insert(0, '.')
from src.analysis.collision_detection import normalise_benchmark

df = pd.read_csv('data/aggregated/all_results.csv', low_memory=False)
df['bn'] = df['benchmark'].apply(normalise_benchmark)
df['model_lower'] = df['model_id'].str.strip().str.lower()

n_models = df['model_id'].nunique()
n_bench_raw = df['benchmark'].nunique()
U = n_models * n_bench_raw

# Per-source record counts
sources = sorted(df['source'].unique())
R = {s: len(df[df['source'] == s]) for s in sources}

# Paper's leave-two-out table: excl OLv2 + each other source
paper_table = [
    ("PWC",          "papers_with_code",    5, 1.62, 2.65, "above"),
    ("AlpacaEval2",  "alpacaeval2",         5, 1.81, 2.36, "above"),
    ("BFCL",         "bfcl",                5, 2.02, 2.10, "above"),
    ("BigCodeBench", "bigcodebench",        5, 2.11, 1.99, "above"),
    ("ChatbotArena", "chatbot_arena",       5, 2.23, 1.85, "above"),
    ("EvalPlus",     "evalplus",            0, 2.24, -1.50, "below"),
    ("HFModelCard",  "hf_model_card",       0, 2.26, -1.50, "below"),
    ("WildBench",    "wildbench",           5, 2.44, 1.64, "above"),
    ("SWEBench",     "swe_bench",           5, 2.45, 1.63, "above"),
    ("MTBench",      "mt_bench",            5, 2.65, 1.45, "above"),
]

olv2 = 'open_llm_leaderboard_v2'
print(f"{'Label':15s} {'Excl':20s} | {'Obs':>4s} {'E_comp':>7s} {'E_paper':>8s} | {'z_comp':>7s} {'z_paper':>8s} | {'Match':>5s}")
print("-" * 90)

all_pass = True
for label, excl_src, obs_paper, e_paper, z_paper, dir_paper in paper_table:
    # Exclude OLv2 + excl_src
    kept = {s: v for s, v in R.items() if s not in (olv2, excl_src)}
    
    E = sum(kept[a] * kept[b] for a, b in combinations(kept.keys(), 2)) / U
    SD = math.sqrt(E) if E > 0 else 0.01
    
    # Also compute observed collisions
    df_sub = df[~df['source'].isin([olv2, excl_src])]
    groups = df_sub.groupby(['model_lower', 'bn'])['source'].nunique()
    obs_computed = (groups >= 2).sum()
    
    z = (obs_computed - E) / SD
    
    e_match = abs(round(E, 2) - e_paper) < 0.02
    z_match = abs(round(z, 2) - z_paper) < 0.05
    obs_match = obs_computed == obs_paper
    ok = e_match and z_match and obs_match
    if not ok:
        all_pass = False
    
    status = "OK" if ok else "FAIL"
    print(f"{label:15s} {excl_src:20s} | {obs_computed:4d} {E:7.2f} {e_paper:8.2f} | {z:7.2f} {z_paper:8.2f} | {status:>5s}")
    if not ok:
        print(f"  --> obs: {obs_computed} vs {obs_paper}, E: {E:.4f} vs {e_paper:.4f}, z: {z:.4f} vs {z_paper:.4f}")

print()
if all_pass:
    print("All leave-two-out checks PASS!")
else:
    print("Some checks FAILED!")
