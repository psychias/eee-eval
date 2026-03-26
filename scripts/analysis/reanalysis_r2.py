#!/usr/bin/env python3
"""
reanalysis_r2.py — Reviewer Round 2 reanalysis
================================================
Runs four new analyses requested by reviewers:

1. τ_b with bootstrap CIs (replacing τ_a with asymptotic CIs)
2. Mixed-effects model attempt (random intercepts for model)
3. Signed-delta analysis (directional bias per source)
4. Projection ablation (without documentation bonus)

All outputs are printed as structured blocks for easy copy into the paper.
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
DATA_DIR = _ROOT / "data"
OUT_DIR = _ROOT / "analysis_output"
RNG = np.random.default_rng(42)

# ── Helpers ──────────────────────────────────────────────────────────
def load_scores() -> pd.DataFrame:
    rows = []
    for source_dir in sorted(DATA_DIR.iterdir()):
        if not source_dir.is_dir():
            continue
        source = source_dir.name
        for fpath in source_dir.rglob("*.json"):
            try:
                rec = json.loads(fpath.read_text())
            except Exception:
                continue
            model_id = rec.get("model_info", {}).get("id", "")
            harness_top = rec.get("eval_library", {}).get("name", "")
            for result in rec.get("evaluation_results", []):
                bench = result.get("evaluation_name", "")
                score = result.get("score_details", {}).get("score")
                gen_cfg = result.get("generation_config") or {}
                details = gen_cfg.get("additional_details") or {}
                n_shot = str(details.get("n_shot", ""))
                pt = str(details.get("prompt_template", ""))
                harness_r = str(details.get("harness", ""))
                harness = harness_r if harness_r else harness_top
                if score is not None:
                    rows.append({
                        "source": source, "model_id": model_id,
                        "benchmark": bench, "score": float(score),
                        "n_shot": n_shot, "harness": harness,
                        "prompt_template": pt,
                    })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# 1. τ_b with bootstrap CIs
# ══════════════════════════════════════════════════════════════════════
def tau_b_bootstrap(scores_a, scores_b, n_boot=2000, alpha=0.05):
    """Compute τ_b with bootstrap CI."""
    tau_obs, p_obs = kendalltau(scores_a, scores_b, variant='b')
    n = len(scores_a)
    taus = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        t, _ = kendalltau(scores_a[idx], scores_b[idx], variant='b')
        if not np.isnan(t):
            taus.append(t)
    if len(taus) < 50:
        return tau_obs, p_obs, np.nan, np.nan
    taus = np.array(taus)
    ci_lo = float(np.percentile(taus, 100 * alpha / 2))
    ci_hi = float(np.percentile(taus, 100 * (1 - alpha / 2)))
    return float(tau_obs), float(p_obs), ci_lo, ci_hi


def analysis_1_tau_b(df: pd.DataFrame):
    """Compute τ_b for all source pairs with ≥3 shared models."""
    print("\n" + "="*70)
    print("ANALYSIS 1: τ_b with bootstrap 95% CIs")
    print("="*70)
    sources = sorted(df["source"].unique())
    rows = []
    for src_a, src_b in combinations(sources, 2):
        df_a = df[df["source"] == src_a]
        df_b = df[df["source"] == src_b]
        shared_bench = set(df_a["benchmark"].unique()) & set(df_b["benchmark"].unique())
        for bench in sorted(shared_bench):
            sub_a = df_a[df_a["benchmark"] == bench].groupby("model_id")["score"].mean()
            sub_b = df_b[df_b["benchmark"] == bench].groupby("model_id")["score"].mean()
            shared = sub_a.index.intersection(sub_b.index)
            if len(shared) < 3:
                continue
            sa = sub_a.loc[shared].values
            sb = sub_b.loc[shared].values
            tau, pval, ci_lo, ci_hi = tau_b_bootstrap(sa, sb)
            rows.append({
                "source_a": src_a, "source_b": src_b, "benchmark": bench,
                "n": len(shared), "tau_b": round(tau, 3),
                "p": round(pval, 4), "ci_lo": round(ci_lo, 3),
                "ci_hi": round(ci_hi, 3),
            })

    result = pd.DataFrame(rows).sort_values("tau_b")
    print(f"\n{len(result)} source-pair/benchmark combinations with n≥3:")
    print(result.to_string(index=False))
    result.to_csv(OUT_DIR / "rank_instability_tau_b.csv", index=False)
    return result


# ══════════════════════════════════════════════════════════════════════
# 2. Mixed-effects model attempt
# ══════════════════════════════════════════════════════════════════════
def analysis_2_mixed_effects():
    """Attempt mixed-effects model on collision pairs."""
    print("\n" + "="*70)
    print("ANALYSIS 2: Mixed-effects model attempt")
    print("="*70)

    cp_path = OUT_DIR / "collision_pairs.csv"
    if not cp_path.exists():
        print("  collision_pairs.csv not found; skipping.")
        return None

    df = pd.read_csv(cp_path)
    df["abs_delta"] = df["delta"].abs()

    def ne(a, b):
        a, b = str(a).strip(), str(b).strip()
        if a in ("", "nan") or b in ("", "nan"):
            return 0
        return int(a != b)

    df["harness_differs"] = [ne(r.harness_a, r.harness_b) for _, r in df.iterrows()]
    df["n_shot_differs"] = [ne(r.n_shot_a, r.n_shot_b) for _, r in df.iterrows()]
    df["prompt_template_differs"] = [ne(r.prompt_template_a, r.prompt_template_b) for _, r in df.iterrows()]

    # Per-benchmark mixed effects
    benchmarks = df["benchmark"].unique()

    try:
        import statsmodels.formula.api as smf
        from statsmodels.regression.mixed_linear_model import MixedLM
    except ImportError:
        print("  statsmodels not available; skipping.")
        return None

    results = []
    for bench in sorted(benchmarks):
        sub = df[df["benchmark"] == bench].copy()
        if len(sub) < 5:
            continue
        n_models = sub["model_id"].nunique()

        print(f"\n  Benchmark: {bench} (n={len(sub)}, unique models={n_models})")

        # OLS baseline
        formula = "abs_delta ~ harness_differs + n_shot_differs + prompt_template_differs"
        ols_model = smf.ols(formula, data=sub).fit()
        print(f"    OLS R²={ols_model.rsquared:.3f}, AIC={ols_model.aic:.1f}")

        # Mixed-effects: random intercept for model_id
        if n_models >= 3:
            try:
                me_model = smf.mixedlm(
                    formula, data=sub, groups=sub["model_id"],
                    re_formula="1"
                ).fit(reml=True, maxiter=200)
                print(f"    Mixed-effects converged: AIC={me_model.aic:.1f}")
                print(f"    Random intercept variance: {me_model.cov_re.iloc[0,0]:.6f}")
                print(f"    Fixed effects:")
                for param in ["harness_differs", "n_shot_differs", "prompt_template_differs"]:
                    if param in me_model.fe_params:
                        coef = me_model.fe_params[param]
                        pval = me_model.pvalues[param]
                        print(f"      {param}: β={coef:.4f}, p={pval:.4f}")
                results.append({
                    "benchmark": bench, "n": len(sub),
                    "n_models": n_models,
                    "ols_r2": round(ols_model.rsquared, 3),
                    "ols_aic": round(ols_model.aic, 1),
                    "me_aic": round(me_model.aic, 1),
                    "re_var": round(me_model.cov_re.iloc[0,0], 6),
                    "me_converged": True,
                })
            except Exception as e:
                print(f"    Mixed-effects FAILED: {e}")
                results.append({
                    "benchmark": bench, "n": len(sub),
                    "n_models": n_models,
                    "ols_r2": round(ols_model.rsquared, 3),
                    "ols_aic": round(ols_model.aic, 1),
                    "me_aic": np.nan,
                    "re_var": np.nan,
                    "me_converged": False,
                })
        else:
            print(f"    Too few unique models ({n_models}) for random intercept; skipping ME.")
            results.append({
                "benchmark": bench, "n": len(sub),
                "n_models": n_models,
                "ols_r2": round(ols_model.rsquared, 3),
                "ols_aic": round(ols_model.aic, 1),
                "me_aic": np.nan,
                "re_var": np.nan,
                "me_converged": False,
            })

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUT_DIR / "mixed_effects_comparison.csv", index=False)
    print(f"\nMixed-effects comparison saved → {OUT_DIR / 'mixed_effects_comparison.csv'}")
    print(result_df.to_string(index=False))
    return result_df


# ══════════════════════════════════════════════════════════════════════
# 3. Signed-delta analysis
# ══════════════════════════════════════════════════════════════════════
def analysis_3_signed_deltas():
    """Analyse signed deltas per source pair for directional bias."""
    print("\n" + "="*70)
    print("ANALYSIS 3: Signed-delta directional bias")
    print("="*70)

    cp_path = OUT_DIR / "collision_pairs.csv"
    if not cp_path.exists():
        print("  collision_pairs.csv not found; skipping.")
        return None

    df = pd.read_csv(cp_path)
    # Only pairs with non-zero delta for directional analysis
    nz = df[df["delta"].abs() > 1e-9].copy()
    print(f"  Non-zero delta pairs: {len(nz)}")

    # Per source_a: mean signed delta
    print("\n  Per-source mean signed delta (source_a perspective):")
    for src in sorted(nz["source_a"].unique()):
        sub = nz[nz["source_a"] == src]
        mean_d = sub["delta"].mean()
        std_d = sub["delta"].std()
        n = len(sub)
        print(f"    {src:30s}  n={n:2d}  mean_Δ={mean_d:+.4f}  std={std_d:.4f}")

    # Per benchmark signed deltas
    print("\n  Per-benchmark signed delta summary:")
    for bench in sorted(nz["benchmark"].unique()):
        sub = nz[nz["benchmark"] == bench]
        mean_d = sub["delta"].mean()
        median_d = sub["delta"].median()
        n = len(sub)
        # One-sample t-test: is mean delta != 0?
        from scipy.stats import ttest_1samp
        if n >= 3:
            t_stat, p_val = ttest_1samp(sub["delta"].values, 0)
        else:
            t_stat, p_val = np.nan, np.nan
        print(f"    {bench:12s}  n={n:2d}  mean={mean_d:+.4f}  "
              f"median={median_d:+.4f}  t={t_stat:+.3f}  p={p_val:.4f}")

    # GSM8K Gemma source consistently positive?
    print("\n  GSM8K: Gemma (2403.08295) as source_a — signed deltas:")
    gsm_gemma = nz[
        (nz["benchmark"] == "GSM8K") &
        ((nz["source_a"] == "papers_2403.08295") | (nz["source_b"] == "papers_2403.08295"))
    ].copy()
    # Normalise so Gemma is always "source_a"
    for idx, row in gsm_gemma.iterrows():
        if row["source_b"] == "papers_2403.08295":
            gsm_gemma.loc[idx, "delta"] = -row["delta"]
            gsm_gemma.loc[idx, "source_a"] = row["source_b"]
            gsm_gemma.loc[idx, "source_b"] = row["source_a"]
    for _, row in gsm_gemma.iterrows():
        print(f"    vs {row['source_b']:25s}  Δ={row['delta']:+.4f}")
    if len(gsm_gemma) >= 2:
        mean_d = gsm_gemma["delta"].mean()
        print(f"    mean Δ from Gemma perspective: {mean_d:+.4f} (n={len(gsm_gemma)})")

    nz.to_csv(OUT_DIR / "signed_deltas.csv", index=False)
    return nz


# ══════════════════════════════════════════════════════════════════════
# 4. Projection ablation (no documentation bonus)
# ══════════════════════════════════════════════════════════════════════
def analysis_4_projection_ablation():
    """Coverage-power projection: compare with vs without documentation bonus."""
    print("\n" + "="*70)
    print("ANALYSIS 4: Projection ablation (with vs without doc bonus)")
    print("="*70)

    cp_path = OUT_DIR / "collision_pairs.csv"
    if not cp_path.exists():
        print("  collision_pairs.csv not found; skipping.")
        return None

    df = pd.read_csv(cp_path)

    # Observed overlap rate: collision pairs / total possible pairs
    # total_sources = 17, current_documented = 3
    total_sources = 17
    documented_sources = 3
    # Current informative pairs per benchmark
    bench_pairs = {}
    for bench in ["GSM8K", "MMLU", "HellaSwag", "HumanEval"]:
        sub = df[df["benchmark"] == bench]
        bench_pairs[bench] = len(sub)
    print(f"  Current informative pairs: {bench_pairs}")

    # Observed overlap rate per source pair
    n_source_pairs = total_sources * (total_sources - 1) / 2
    total_pairs = len(df)
    base_overlap_rate = total_pairs / n_source_pairs if n_source_pairs > 0 else 0
    print(f"  Base overlap rate: {base_overlap_rate:.3f} pairs per source pair")

    # Simulate adding s = 1..30 new documented sources
    s_range = list(range(1, 31))
    results = []

    for bench in ["GSM8K", "MMLU", "HellaSwag", "HumanEval"]:
        sub = df[df["benchmark"] == bench]
        deltas = sub["delta"].values if len(sub) > 0 else np.array([0.01])

        for s in s_range:
            new_total = total_sources + s
            new_documented = documented_sources + s

            # With documentation bonus (1.5x overlap for documented sources)
            pairs_with_bonus = bench_pairs.get(bench, 0) + int(
                s * (new_total - 1) * base_overlap_rate * 1.5
            )
            # Without documentation bonus (flat overlap rate)
            pairs_no_bonus = bench_pairs.get(bench, 0) + int(
                s * (new_total - 1) * base_overlap_rate
            )

            # Bootstrap power for both
            for label, n_pairs in [("with_bonus", pairs_with_bonus),
                                    ("no_bonus", pairs_no_bonus)]:
                n_eff = max(n_pairs, 2)
                reject = 0
                for _ in range(1000):
                    sample = RNG.choice(deltas, size=min(n_eff, 200), replace=True)
                    boot = RNG.choice(sample, size=(100, len(sample)), replace=True)
                    means = boot.mean(axis=1)
                    lo, hi = np.percentile(means, [2.5, 97.5])
                    if lo > 0 or hi < 0:
                        reject += 1
                power = reject / 1000

                results.append({
                    "benchmark": bench, "new_sources": s,
                    "total_sources": new_total,
                    "coverage_pct": round(new_documented / new_total * 100, 1),
                    "variant": label,
                    "est_pairs": n_pairs,
                    "power": round(power, 3),
                })

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUT_DIR / "projection_ablation.csv", index=False)

    # Print summary: how many sources for 80% power, with vs without bonus
    print("\n  Sources needed for 80% power:")
    print(f"  {'Benchmark':12s} {'With bonus':>12s} {'No bonus':>12s} {'Difference':>12s}")
    for bench in ["GSM8K", "MMLU", "HellaSwag", "HumanEval"]:
        for label in ["with_bonus", "no_bonus"]:
            sub = result_df[(result_df["benchmark"] == bench) &
                           (result_df["variant"] == label)]
            hit = sub[sub["power"] >= 0.80]
            if not hit.empty:
                k = hit.iloc[0]["new_sources"]
            else:
                k = ">30"
            if label == "with_bonus":
                k_bonus = k
            else:
                k_no = k
        print(f"  {bench:12s} {str(k_bonus):>12s} {str(k_no):>12s} "
              f"{'—' if isinstance(k_bonus, str) or isinstance(k_no, str) else f'{k_no - k_bonus:+d}':>12s}")

    return result_df


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading all records...")
    df = load_scores()
    print(f"  {len(df)} rows from {df['source'].nunique()} sources")

    # 1. τ_b
    tau_results = analysis_1_tau_b(df)

    # 2. Mixed-effects
    me_results = analysis_2_mixed_effects()

    # 3. Signed deltas
    signed_results = analysis_3_signed_deltas()

    # 4. Projection ablation
    proj_results = analysis_4_projection_ablation()

    print("\n" + "="*70)
    print("ALL REANALYSES COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
