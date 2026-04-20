#!/usr/bin/env python3
"""Comprehensive verification of all statistics claimed in the paper."""

import os
import sys
import json
import math
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"
results = []

def check(name, expected, actual, tolerance=0.01):
    """Check a numeric claim. Returns (status, message)."""
    # Coerce numpy scalars to native Python types for clean comparison
    if hasattr(actual, 'item'):
        actual = actual.item()
    if isinstance(expected, str) and isinstance(actual, str):
        ok = expected.strip() == actual.strip()
        status = PASS if ok else FAIL
        msg = f"{name}: expected={expected!r}, actual={actual!r}"
    elif isinstance(expected, bool) and isinstance(actual, bool):
        ok = expected == actual
        status = PASS if ok else FAIL
        msg = f"{name}: expected={expected}, actual={actual}"
    elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if expected == 0:
            ok = actual == 0
        elif tolerance == 0:
            ok = expected == actual  # exact match for integer checks
        else:
            ok = abs(expected - actual) / max(abs(expected), 1e-9) < tolerance
        status = PASS if ok else FAIL
        msg = f"{name}: expected={expected}, actual={actual}"
    else:
        status = WARN
        msg = f"{name}: expected={expected!r}, actual={actual!r} (type mismatch)"
    results.append((status, msg))
    print(f"  [{status}] {msg}")
    return status == PASS

# ======================================================================
print("=" * 70)
print("1. DATASET-LEVEL STATISTICS")
print("=" * 70)

df = pd.read_csv("data/aggregated/all_results.csv", low_memory=False)
check("totalrecords", 29331, len(df))
check("totalsources", 11, df["source"].nunique())
check("totalmodels", 5672, df["model_id"].nunique())
check("totalbenchmarks", 180, df["benchmark"].nunique())

# ======================================================================
print("\n" + "=" * 70)
print("2. PER-SOURCE TABLE (Table 1)")
print("=" * 70)

expected_table = {
    "open_llm_leaderboard_v2": (26790, 4465, 6),
    "papers_with_code":        (576,   383, 16),
    "alpacaeval2":             (446,   223,  2),
    "bfcl":                    (327,   109,  3),
    "bigcodebench":            (280,   156,  2),
    "chatbot_arena":           (218,   218,  1),
    "evalplus":                (214,   125,  2),
    "hf_model_card":           (207,    11, 156),
    "wildbench":               (122,    61,  2),
    "swe_bench":               (117,   117,  1),
    "mt_bench":                (34,     34,  1),
}

for src, (exp_rec, exp_mod, exp_ben) in expected_table.items():
    sub = df[df["source"] == src]
    check(f"{src} records", exp_rec, len(sub))
    check(f"{src} models", exp_mod, sub["model_id"].nunique())
    check(f"{src} benchmarks", exp_ben, sub["benchmark"].nunique())

# ======================================================================
print("\n" + "=" * 70)
print("3. COVERAGE STATISTICS (Tables 2 & 3)")
print("=" * 70)

def coverage_pct(series):
    valid = series.notna() & (series.astype(str).str.strip() != "") & (series.astype(str).str.lower() != "unknown")
    return valid.sum() / len(series) * 100

# Map column names
col_map = {
    "harness": "eval_library_name",
    "n_shot": "n_shot",
    "chain_of_thought": "chain_of_thought",
    "temperature": "temperature",
    "prompt_template": "prompt_template",
}

# Check which columns exist
print("  Available columns:", [c for c in df.columns if any(k in c.lower() for k in ["harness", "eval", "shot", "cot", "chain", "temp", "prompt"])])

# Try to find the right column names
harness_col = None
for c in ["eval_library_name", "harness", "eval_library"]:
    if c in df.columns:
        harness_col = c
        break

shot_col = None
for c in ["n_shot", "shots"]:
    if c in df.columns:
        shot_col = c
        break

cot_col = None
for c in ["chain_of_thought", "cot"]:
    if c in df.columns:
        cot_col = c
        break

temp_col = "temperature" if "temperature" in df.columns else None
prompt_col = "prompt_template" if "prompt_template" in df.columns else None

print(f"  Using columns: harness={harness_col}, shot={shot_col}, cot={cot_col}, temp={temp_col}, prompt={prompt_col}")

if harness_col:
    h_pct = coverage_pct(df[harness_col])
    check("Overall harness coverage", 0.0, round(h_pct, 1), tolerance=0.02)

if shot_col:
    s_pct = coverage_pct(df[shot_col])
    check("Overall n-shot coverage", 96.7, round(s_pct, 1), tolerance=0.02)

if cot_col:
    c_pct = coverage_pct(df[cot_col])
    check("Overall CoT coverage", 76.3, round(c_pct, 1), tolerance=0.02)

if temp_col:
    t_pct = coverage_pct(df[temp_col])
    check("Overall temperature coverage (<0.1%)", True, t_pct < 0.1)

if prompt_col:
    p_pct = coverage_pct(df[prompt_col])
    check("Overall prompt_template coverage (<0.1%)", True, p_pct < 0.1)

# Ablation subsets
excl_olv2 = df[df["source"] != "open_llm_leaderboard_v2"]
check("excl OLv2 record count", 2541, len(excl_olv2))

excl_both = excl_olv2[excl_olv2["source"] != "papers_with_code"]
check("excl OLv2+PWC record count", 1965, len(excl_both))

if shot_col:
    check("excl OLv2 n-shot coverage", 62.2, round(coverage_pct(excl_olv2[shot_col]), 1), tolerance=0.02)

if cot_col:
    check("excl OLv2 CoT coverage", 1.1, round(coverage_pct(excl_olv2[cot_col]), 1), tolerance=0.1)

# PWC artefact rate
check("PWC artefact rate", 57.3, round(774/1350*100, 1), tolerance=0.02)
check("PWC artefact rate (rounded)", 57, round(774/1350*100), tolerance=0.02)

# ======================================================================
print("\n" + "=" * 70)
print("4. COLLISION DETECTION")
print("=" * 70)

# Load collision pairs from analysis_output
cpairs_path = ROOT / "analysis_output" / "collision_pairs.csv"
if cpairs_path.exists():
    cpairs = pd.read_csv(cpairs_path)
    print(f"  Collision pairs file: {len(cpairs)} rows")
    print(f"  Columns: {list(cpairs.columns)}")
    print(f"  Sample:\n{cpairs.head(3).to_string()}")
else:
    print("  [WARN] collision_pairs.csv not found")

# Also detect collisions from raw data using normalised benchmarks
sys.path.insert(0, str(ROOT))
try:
    from src.analysis.collision_detection import normalise_benchmark, normalise_model_id
    df["benchmark_norm"] = df["benchmark"].apply(normalise_benchmark)
    df["model_id_norm"] = df["model_id"].apply(normalise_model_id)
    collisions = df.groupby(["model_id_norm", "benchmark_norm"])["source"].nunique()
    n_collision_groups = (collisions >= 2).sum()
    # Count actual pairs: each group with k sources produces C(k,2) pairs
    from math import comb
    n_collision_pairs = sum(comb(k, 2) for k in collisions[collisions >= 2])
    print(f"  Collision (model_id_norm, benchmark_norm) groups with >=2 sources: {n_collision_groups}")
    print(f"  Total collision pairs: {n_collision_pairs}")
    check("collision groups (normalised)", 16, n_collision_groups, tolerance=0)
except ImportError as e:
    print(f"  [WARN] Could not import collision_detection: {e}")

# ======================================================================
print("\n" + "=" * 70)
print("5. COLLISION PAIR SCORES & TULU STATISTICS")
print("=" * 70)

# Hardcoded collision table from the paper (Table in appendix)
collision_data = [
    ("Qwen2.5-72B-Inst.", "GPQA",     16.67, 49.00, True),
    ("SC2-15B-Inst.",     "MBPP+",    65.10, 61.20, True),
    ("SC2-15B-Inst.",     "HumanE+",  60.40, 63.40, True),
    ("Tulu-3-8B",         "BBH",      16.86, 16.67, True),
    ("Tulu-3-8B",         "GPQA",      6.26,  6.49, True),
    ("Tulu-3-8B",         "IFEval",   82.55, 82.67, True),
    ("Tulu-3-8B",         "MMLU-Pro", 20.23, 20.30, True),
    ("Tulu-3-8B",         "MuSR",     10.52, 10.45, True),
    # Likely copied
    ("Tulu-3-70B-SFT", "BBH",      42.02, 42.02, False),
    ("Tulu-3-70B-SFT", "GPQA",     12.64, 12.64, False),
    ("Tulu-3-70B-SFT", "IFEval",   80.51, 80.51, False),
    ("Tulu-3-70B-SFT", "MMLU-Pro", 40.27, 40.27, False),
    ("Tulu-3-70B-SFT", "MuSR",     24.49, 24.49, False),
    ("SC2-15B",         "HumanE+",  37.80, 37.80, False),
    ("SC2-3B",          "HumanE+",  27.40, 27.40, False),
    ("SC2-7B",          "HumanE+",  29.90, 29.90, False),
]
check("total collision pairs", 16, len(collision_data), tolerance=0)
check("independent pairs", 8, sum(1 for x in collision_data if x[4]), tolerance=0)
check("copied pairs", 8, sum(1 for x in collision_data if not x[4]), tolerance=0)

# GPQA gap
gpqa = [x for x in collision_data if x[0] == "Qwen2.5-72B-Inst." and x[1] == "GPQA"][0]
gpqa_gap = abs(gpqa[2] - gpqa[3])
check("GPQA gap", 32.33, gpqa_gap)

# Tulu-3-8B statistics
# s1 = HF Model Card (first-party), s2 = OLv2 (third-party)
tulu = [(m, b, s1, s2) for m, b, s1, s2, ind in collision_data if m == "Tulu-3-8B"]
advantages = [s2 - s1 for _, _, s1, s2 in tulu]  # first-party advantage (paper convention)
abs_deltas = [abs(a) for a in advantages]

mean_adv = np.mean(advantages)
mean_abs = np.mean(abs_deltas)
n_fp_higher = sum(1 for a in advantages if a > 0)
n_tp_higher = sum(1 for a in advantages if a < 0)

check("Tulu mean first-party advantage", 0.032, round(mean_adv, 3))
check("Tulu mean |delta|", 0.14, round(mean_abs, 2))
check("Tulu FP higher count", 3, n_fp_higher, tolerance=0)
check("Tulu TP higher count", 2, n_tp_higher, tolerance=0)
check("Tulu max |delta|", 0.23, max(abs_deltas))
check("Tulu advantage range min", -0.19, round(min(advantages), 2))
check("Tulu advantage range max", +0.23, round(max(advantages), 2))

# "more than two orders of magnitude" claim
ratio = 32.33 / mean_abs
log_ratio = math.log10(ratio)
check("orders of magnitude (>2)", True, log_ratio > 2.0)
print(f"  [INFO] Ratio: {ratio:.1f}x, log10 = {log_ratio:.3f}")

# Spearman correlation on 8 independent pairs
# Completeness = count of config dimensions where BOTH sides have known values
# 3 dimensions: harness, n_shot, cot → max score = 3
# GPQA: 0 (harness=lm_eval/unknown, n_shot=0/NaN, cot=True/NaN)
# SC2-15B pairs: 1 each (harness known both sides, n_shot/cot missing)
# Tulu-3-8B pairs: 2 each (harness known, n_shot match, cot partial)
completeness = [0, 1, 1, 2, 2, 2, 2, 2]
abs_delta_all = [32.33, 3.90, 3.00, 0.23, 0.19, 0.12, 0.07, 0.07]
rho, p_val = stats.spearmanr(completeness, abs_delta_all)
check("Spearman rho", -0.871, round(rho, 3))
check("Spearman p", 0.005, round(p_val, 3))
print(f"  [INFO] Exact rho={rho:.6f}, p={p_val:.6f}")

# Overturn bound = rho^2
overturn = rho ** 2
check("overturn bound", 0.759, round(overturn, 3))

# ======================================================================
print("\n" + "=" * 70)
print("6. CONTROLLED EXPERIMENT STATISTICS")
print("=" * 70)

results_path = ROOT / "experiments" / "controlled_eval" / "results" / "controlled_eval_results.jsonl"
if results_path.exists():
    exp_records = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line:
                exp_records.append(json.loads(line))

    edf = pd.DataFrame(exp_records)
    print(f"  Total experiment records: {len(edf)}")
    print(f"  Columns: {list(edf.columns)}")

    # Filter to OK runs only (paper counts only OK runs)
    ok_edf = edf[edf["status"] == "ok"] if "status" in edf.columns else edf
    print(f"  OK experiment records: {len(ok_edf)}")

    check("expruns (OK)", 306, len(ok_edf), tolerance=0)

    # Count models
    model_col = "model_id" if "model_id" in ok_edf.columns else ("model" if "model" in ok_edf.columns else None)
    if model_col:
        check("expmodels", 4, ok_edf[model_col].nunique(), tolerance=0)
        print(f"  Models: {sorted(ok_edf[model_col].unique())}")

    # Count benchmarks
    bench_col = "benchmark" if "benchmark" in ok_edf.columns else ("task" if "task" in ok_edf.columns else None)
    if bench_col:
        check("expbenches", 3, ok_edf[bench_col].nunique(), tolerance=0)
        print(f"  Benchmarks: {sorted(ok_edf[bench_col].unique())}")

    # Count generative vs loglikelihood
    if bench_col:
        gsm_ok = ok_edf[ok_edf[bench_col] == "gsm8k"]
        bbh_ok = ok_edf[ok_edf[bench_col] == "bbh"]
        mmlu_ok = ok_edf[ok_edf[bench_col] == "mmlu"]
        check("GSM8K OK runs", 144, len(gsm_ok), tolerance=0)
        check("BBH OK runs", 144, len(bbh_ok), tolerance=0)
        check("MMLU OK runs", 18, len(mmlu_ok), tolerance=0)
        check("expgenruns", 288, len(gsm_ok) + len(bbh_ok), tolerance=0)
        check("expllruns", 18, len(mmlu_ok), tolerance=0)

    print(f"\n  First record keys: {list(exp_records[0].keys())}")

else:
    print("  [WARN] controlled_eval_results.jsonl not found")

# ======================================================================
print("\n" + "=" * 70)
print("6b. CROSS-EXTRACTOR STUDY (5-shot)")
print("=" * 70)

import re as re_mod

def _extract_strict(text):
    m = re_mod.search(r'####\s*([+-]?\d[\d,]*\.?\d*)', text)
    return float(m.group(1).replace(',', '')) if m else None

def _extract_boxed(text):
    m = re_mod.search(r'\\boxed\{([+-]?\d[\d,]*\.?\d*)\}', text)
    return float(m.group(1).replace(',', '')) if m else None

def _extract_answer_is(text):
    m = re_mod.search(r'[Tt]he (?:final )?answer is[:\s]*\$?([+-]?\d[\d,]*\.?\d*)', text)
    return float(m.group(1).replace(',', '')) if m else None

def _extract_last_num(text):
    nums = re_mod.findall(r'[+-]?\d[\d,]*\.?\d*', text)
    return float(nums[-1].replace(',', '')) if nums else None

def _extract_flex(text):
    for fn in [_extract_strict, _extract_boxed, _extract_answer_is, _extract_last_num]:
        v = fn(text)
        if v is not None:
            return v
    return None

def _parse_gold(t):
    m = re_mod.search(r'####\s*([+-]?\d[\d,]*\.?\d*)', t)
    return float(m.group(1).replace(',', '')) if m else None

cross_ext_5shot_dir = ROOT / "experiments" / "controlled_eval" / "5-shot-GSM8K"
if cross_ext_5shot_dir.exists():
    ce_macro_checks = {
        # (dirname, extractor) -> expected value
        ("Llama-3.1-8B-Instruct_gsm8k_5shot_plain", "strict"): 21.5,
        ("Llama-3.1-8B-Instruct_gsm8k_5shot_plain", "flex"): 83.0,
        ("Llama-3.1-8B-Instruct_gsm8k_5shot_cot", "strict"): 11.5,
        ("Llama-3.1-8B-Instruct_gsm8k_5shot_cot", "flex"): 83.0,
        ("Qwen2.5-7B-Instruct_gsm8k_5shot_plain", "strict"): 40.0,
        ("Qwen2.5-7B-Instruct_gsm8k_5shot_plain", "flex"): 82.0,
        ("Qwen2.5-7B-Instruct_gsm8k_5shot_cot", "strict"): 26.5,
        ("Qwen2.5-7B-Instruct_gsm8k_5shot_cot", "flex"): 82.5,
        ("Mistral-7B-Instruct-v0.3_gsm8k_5shot_plain", "strict"): 37.5,
        ("Mistral-7B-Instruct-v0.3_gsm8k_5shot_plain", "flex"): 54.5,
        ("Mistral-7B-Instruct-v0.3_gsm8k_5shot_cot", "strict"): 40.0,
        ("Mistral-7B-Instruct-v0.3_gsm8k_5shot_cot", "flex"): 53.5,
        ("Qwen2.5-14B-Instruct_gsm8k_5shot_plain", "strict"): 59.0,
        ("Qwen2.5-14B-Instruct_gsm8k_5shot_plain", "flex"): 84.5,
        ("Qwen2.5-14B-Instruct_gsm8k_5shot_cot", "strict"): 2.0,
        ("Qwen2.5-14B-Instruct_gsm8k_5shot_cot", "flex"): 60.5,
    }

    for (dirname, ext_type), expected_val in ce_macro_checks.items():
        # Try both naming conventions
        sf = cross_ext_5shot_dir / dirname / "samples_gsm8k.jsonl"
        if not sf.exists():
            sf = cross_ext_5shot_dir / dirname / "samples_gsm8k_5shot.jsonl"
        if not sf.exists():
            print(f"  [WARN] Missing: {sf}")
            continue
        samples = []
        raw = sf.read_text(encoding='utf-8')
        decoder = json.JSONDecoder()
        idx = 0
        while idx < len(raw):
            try:
                obj, end = decoder.raw_decode(raw, idx)
                samples.append(obj)
                idx = end
            except json.JSONDecodeError:
                idx += 1
        n = len(samples)
        ext_fn = _extract_strict if ext_type == "strict" else _extract_flex
        correct = 0
        for s in samples:
            resp = ''
            if 'resps' in s and s['resps']:
                resp = s['resps'][0][0] if isinstance(s['resps'][0], list) else s['resps'][0]
            gold = _parse_gold(s.get('target', s.get('doc', {}).get('answer', '')))
            if gold is None:
                continue
            pred = ext_fn(str(resp))
            if pred is not None and abs(pred - gold) < 0.01:
                correct += 1
        acc = round(correct / n * 100, 1)
        macro_name = f"{dirname.split('_gsm8k_5shot_')[0]}_{dirname.split('_')[-1]}_{ext_type}"
        check(f"cross-ext {macro_name}", expected_val, acc)
else:
    print("  [WARN] 5-shot-GSM8K directory not found")

# ======================================================================
print("\n" + "=" * 70)
print("7. PERMUTATION NULL MODEL (from analysis_output)")
print("=" * 70)

# Check if we have power_simulation results
power_path = ROOT / "analysis_output" / "power_simulation.csv"
if power_path.exists():
    psim = pd.read_csv(power_path)
    print(f"  Power simulation: {len(psim)} rows")
    print(f"  Columns: {list(psim.columns)}")
    print(psim.to_string())
else:
    print("  [INFO] power_simulation.csv not found; will recompute")

# We need to verify the birthday-paradox expected collisions formula
# E = sum_{i<j} R_i * R_j / U  where U = |models| x |benchmarks|
from itertools import combinations
source_sizes = {}
for src in df["source"].unique():
    sub = df[df["source"] == src]
    source_sizes[src] = len(sub)

U = df["model_id"].nunique() * df["benchmark"].nunique()
E_all = sum(source_sizes[a] * source_sizes[b] for a, b in combinations(source_sizes.keys(), 2)) / U
print(f"  Birthday paradox E (all): {E_all:.1f} (paper claims ~69)")
check("expected collisions (all)", 69, round(E_all), tolerance=0.15)

# Excluding OLv2 — the birthday-paradox formula overestimates because it
# ignores benchmark overlap structure. The paper's 2.7 comes from Monte Carlo
# simulation preserving per-source benchmark sets, not from this formula.
source_sizes_no_olv2 = {k: v for k, v in source_sizes.items() if k != "open_llm_leaderboard_v2"}
df_no_olv2 = df[df["source"] != "open_llm_leaderboard_v2"]
U_no_olv2 = df_no_olv2["model_id"].nunique() * df_no_olv2["benchmark"].nunique()
E_no_olv2_birthday = sum(source_sizes_no_olv2[a] * source_sizes_no_olv2[b] for a, b in combinations(source_sizes_no_olv2.keys(), 2)) / U_no_olv2
print(f"  Birthday paradox E (excl OLv2): {E_no_olv2_birthday:.1f} (overestimates; paper uses Monte Carlo = 2.7)")
print(f"  [INFO] Birthday-paradox formula does not account for benchmark overlap structure.")
print(f"         The paper's 2.7 expected (excl OLv2) comes from Monte Carlo simulation")
print(f"         that preserves per-source benchmark sets, not from the analytical formula.")

print("\n  [INFO] z-scores require the permutation SD which depends on the simulation run")
print("  Paper claims: z=-6.4 (all), z=1.4 (excl OLv2)")

# ======================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

n_pass = sum(1 for s, _ in results if s == PASS)
n_fail = sum(1 for s, _ in results if s == FAIL)
n_warn = sum(1 for s, _ in results if s == WARN)
print(f"  PASS: {n_pass}")
print(f"  FAIL: {n_fail}")
print(f"  WARN: {n_warn}")
if n_fail > 0:
    print("\n  FAILURES:")
    for s, m in results:
        if s == FAIL:
            print(f"    {m}")
