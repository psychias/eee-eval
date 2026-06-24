#!/usr/bin/env python3
"""Generate all experiment figures locally from Colab result files.

Usage:
    python scripts/generate_experiment_figures.py --results-dir /path/to/eee_eval_results_v4

Expects the results directory to contain:
  - controlled_eval_results.jsonl   (main experiment)
  - sample_outputs/                 (cross-extractor study)
  - negative_control_version.jsonl  (lm-eval version comparison)

Outputs figures to submission/latex/figures/ and updated macros to submission/latex/macros.tex.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = Path(__file__).resolve().parent.parent.parent
FIG_DIR = ROOT / "submission" / "latex" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Paul Tol's vibrant palette (colourblind-safe)
COLORS = ["#CC3311", "#0077BB", "#EE7733", "#009988", "#33BBEE",
           "#EE3377", "#BBBBBB"]

GENERATIVE_BENCHMARKS = {"gsm8k", "bbh", "humaneval"}

CROSS_SOURCE_PAIRS = [
    ("Qwen2.5-72B-Inst.", "GPQA",      16.67, 49.00, 32.33),
    ("SC2-15B-Inst.",     "MBPP+",      65.10, 61.20,  3.90),
    ("SC2-15B-Inst.",     "HumanEval+", 60.40, 63.40,  3.00),
    ("Tulu-3-8B",         "BBH",        16.86, 16.67,  0.19),
    ("Tulu-3-8B",         "GPQA",        6.26,  6.49,  0.23),
    ("Tulu-3-8B",         "IFEval",     82.55, 82.67,  0.12),
    ("Tulu-3-8B",         "MMLU-Pro",   20.23, 20.30,  0.07),
    ("Tulu-3-8B",         "MuSR",       10.52, 10.45,  0.07),
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                r = json.loads(line)
                if r.get("status") == "ok" and r.get("score") is not None:
                    records.append(r)
            except Exception:
                pass
    return records


def axis_effect(axis: str, pool: list[dict]) -> float:
    """Mean |max-min| within matched cells, varying only `axis`."""
    other_keys = {
        "temperature":   ("model_id", "benchmark", "prompt_format", "n_shot", "random_seed"),
        "prompt_format": ("model_id", "benchmark", "temperature", "n_shot", "random_seed"),
        "n_shot":        ("model_id", "benchmark", "temperature", "prompt_format", "random_seed"),
    }
    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in pool:
        key = tuple(r[k] for k in other_keys[axis])
        groups[key].append(r["score"])
    spreads = [max(v) - min(v) for v in groups.values() if len(v) >= 2]
    return float(np.mean(spreads)) if spreads else 0.0


def save(fig, name):
    for ext in (".pdf", ".png"):
        fig.savefig(FIG_DIR / f"{name}{ext}", bbox_inches="tight",
                    dpi=150 if ext == ".png" else None)
    plt.close(fig)
    print(f"  Saved {name}")


# ---------------------------------------------------------------------------
# Figure 1: Config variance boxplots (generative benchmarks)
# ---------------------------------------------------------------------------

def fig_config_variance(records):
    gen = [r for r in records if r["benchmark"] in GENERATIVE_BENCHMARKS]
    if not gen:
        print("  SKIP fig_config_variance_generative: no generative records")
        return
    benches = sorted({r["benchmark"] for r in gen})
    n_b = len(benches)
    fig, axes = plt.subplots(n_b, 3, figsize=(13, 4.5 * n_b), squeeze=False)
    for row, bench in enumerate(benches):
        pool = [r for r in gen if r["benchmark"] == bench]
        for col, (field, label) in enumerate([
            ("temperature", "Temperature"),
            ("prompt_format", "Prompt format"),
            ("n_shot", "N-shot"),
        ]):
            ax = axes[row][col]
            vals = sorted({r[field] for r in pool}, key=str)
            data = [[r["score"] for r in pool if r[field] == v] for v in vals]
            if data and any(data):
                bp = ax.boxplot(data, tick_labels=[str(v) for v in vals],
                                patch_artist=True)
                for patch, c in zip(bp["boxes"], COLORS):
                    patch.set_facecolor(c)
                    patch.set_alpha(0.6)
            ax.set_xlabel(label)
            ax.set_ylabel("Score (pp)")
            ax.set_title(f"{bench.upper()} by {label}")
    plt.tight_layout()
    save(fig, "fig_config_variance_generative")


# ---------------------------------------------------------------------------
# Figure 2: Sensitivity — partial R² bar chart vs overturn bound
# ---------------------------------------------------------------------------

def fig_sensitivity(records):
    gen = [r for r in records if r["benchmark"] in GENERATIVE_BENCHMARKS]
    if len(gen) < 6:
        print("  SKIP fig_sensitivity: too few records")
        return {}, {}

    import pandas as pd
    import statsmodels.api as sm

    # Compute for 5-shot and all-shot
    sensitivity = {}
    for label, pool in [("5shot", [r for r in gen if r["n_shot"] == 5]),
                        ("all",   gen)]:
        if len(pool) < 6:
            continue
        df = pd.DataFrame(pool)
        dm = pd.get_dummies(df["model_id"], prefix="m", drop_first=True)
        db = pd.get_dummies(df["benchmark"], prefix="b", drop_first=True)
        X_base = pd.concat([pd.Series(1.0, index=df.index, name="const"),
                            dm, db], axis=1).astype(float)
        y = df["score"].astype(float)
        base_fit = sm.OLS(y, X_base).fit()
        ssr_base = float(np.sum(base_fit.resid ** 2))
        for var, col in [
            ("temperature",  df["temperature"].astype(float)),
            ("prompt_format", pd.Categorical(df["prompt_format"]).codes.astype(float)),
            ("n_shot",       df["n_shot"].astype(float)),
        ]:
            Xf = pd.concat([X_base, col.rename(var)], axis=1)
            fm = sm.OLS(y, Xf).fit()
            pr2 = max(0.0, (ssr_base - float(np.sum(fm.resid ** 2))) / ssr_base)
            sensitivity[f"{var}_{label}"] = round(pr2, 4)

    overturn = round(0.850 ** 2, 4)

    # Bar chart: 5-shot partial R²
    vars_5 = ["temperature", "prompt_format", "n_shot"]
    vals_5 = [sensitivity.get(f"{v}_5shot", 0) for v in vars_5]
    labels_5 = ["Temperature", "Prompt format", "N-shot"]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(vars_5))
    bars = ax.bar(x, vals_5, color=[COLORS[0], COLORS[1], COLORS[2]], width=0.5)
    ax.axhline(overturn, ls="--", color="black", alpha=0.7, lw=1.5,
               label=f"Overturn bound ({overturn})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_5)
    ax.set_ylabel("Partial R²")
    ax.set_title("Sensitivity: 5-shot GSM8K")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(max(vals_5) * 1.3, overturn * 1.2))
    for bar, val in zip(bars, vals_5):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    save(fig, "fig_sensitivity")

    # Also do all-generative version for appendix
    vars_a = ["temperature", "prompt_format", "n_shot"]
    vals_a = [sensitivity.get(f"{v}_all", 0) for v in vars_a]

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    bars2 = ax2.bar(x, vals_a, color=[COLORS[0], COLORS[1], COLORS[2]], width=0.5)
    ax2.axhline(overturn, ls="--", color="black", alpha=0.7, lw=1.5,
                label=f"Overturn bound ({overturn})")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_5)
    ax2.set_ylabel("Partial R²")
    ax2.set_title("Sensitivity: All generative (incl. 0-shot)")
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, max(max(vals_a) * 1.15, overturn * 1.2))
    for bar, val in zip(bars2, vals_a):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    save(fig2, "fig_sensitivity_all")

    return sensitivity, {"overturn": overturn}


# ---------------------------------------------------------------------------
# Figure 3: Config effects vs cross-source divergence
# ---------------------------------------------------------------------------

def fig_config_vs_crosssource(records):
    gen = [r for r in records if r["benchmark"] in GENERATIVE_BENCHMARKS]
    if not gen:
        print("  SKIP fig_config_vs_crosssource: no generative records")
        return

    by_mb: dict[tuple, list[float]] = defaultdict(list)
    for r in gen:
        by_mb[(r["model_id"], r["benchmark"])].append(r["score"])
    all_effs = [max(v) - min(v) for v in by_mb.values() if len(v) >= 2]
    mean_ctrl = float(np.mean(all_effs)) if all_effs else 0.0

    cs_deltas = [d for *_, d in CROSS_SOURCE_PAIRS]
    cs_labels = [f"{m}\n{b}" for m, b, *_ in CROSS_SOURCE_PAIRS]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(range(len(cs_deltas)), cs_deltas, marker="s",
               color=COLORS[0], s=60, zorder=3,
               label="Cross-source divergence (Table 5)")
    ax.axhline(mean_ctrl, ls="--", color=COLORS[1], alpha=0.8, lw=1.5,
               label=f"Mean controlled config effect ({mean_ctrl:.1f} pp)")
    ax.set_xticks(range(len(cs_labels)))
    ax.set_xticklabels(cs_labels, fontsize=7, ha="center")
    ax.set_ylabel("|Δ| (pp)")
    ax.legend(fontsize=9)
    ax.set_title("Cross-source divergence vs controlled config effect")
    plt.tight_layout()
    save(fig, "fig_config_vs_crosssource")


# ---------------------------------------------------------------------------
# Figure 4 (NEW): Cross-extractor study — grouped bar chart
# ---------------------------------------------------------------------------

def fig_cross_extractor(sample_dir: Path | None):
    if sample_dir is None or not sample_dir.exists():
        print("  SKIP fig_cross_extractor: sample_outputs not found")
        return

    # Import extractors from the analysis script
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from run_cross_extractor_analysis import (
            EXTRACTORS, load_samples, score_samples
        )
    except ImportError:
        print("  SKIP fig_cross_extractor: could not import run_cross_extractor_analysis")
        return

    by_cell: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for run_dir in sorted(sample_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        sample_files = list(run_dir.rglob("samples_*.jsonl"))
        if not sample_files:
            sample_files = list(run_dir.rglob("*gsm8k*.jsonl"))
        if not sample_files:
            continue

        parts = run_dir.name.rsplit("_gsm8k_0shot_", 1)
        model_short = parts[0] if len(parts) == 2 else run_dir.name
        fmt = parts[1] if len(parts) == 2 else "unknown"

        for sf in sample_files:
            samples = load_samples(sf)
            if not samples:
                continue
            for ext_name, ext_fn in EXTRACTORS.items():
                n_correct, n_total = score_samples(samples, ext_fn)
                acc = n_correct / n_total * 100 if n_total > 0 else 0.0
                by_cell[(model_short, fmt)][ext_name] = round(acc, 2)

    if not by_cell:
        print("  SKIP fig_cross_extractor: no sample data found")
        return

    ext_names = list(EXTRACTORS.keys())
    cells = sorted(by_cell.keys())
    x = np.arange(len(cells))
    width = 0.15

    fig, ax = plt.subplots(figsize=(max(10, len(cells) * 1.5), 5))
    for i, (ext, color) in enumerate(zip(ext_names, COLORS)):
        vals = [by_cell[c].get(ext, 0) for c in cells]
        ax.bar(x + i * width, vals, width, label=ext, color=color)

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Cross-Extractor Study: 0-shot GSM8K")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([f"{m}\n{f}" for m, f in cells], fontsize=7)
    ax.legend(fontsize=8, ncol=2)
    all_vals = [v for scores in by_cell.values() for v in scores.values()]
    ax.set_ylim(0, max(max(all_vals) * 1.15, 5) if all_vals else 100)
    plt.tight_layout()
    save(fig, "fig_cross_extractor")

    # Summary: gap between strict and flexible
    gaps = []
    for cell in cells:
        strict = by_cell[cell].get("strict_####", 0)
        flex = by_cell[cell].get("flexible_chain", 0)
        gaps.append(flex - strict)

    if gaps:
        print(f"  Cross-extractor gap (flexible - strict): "
              f"mean={np.mean(gaps):.2f} pp, max={max(gaps):.2f} pp")


# ---------------------------------------------------------------------------
# Figure 5 (NEW): Negative control — lm-eval version comparison
# ---------------------------------------------------------------------------

def fig_version_control(neg_ctrl_path: Path | None):
    if neg_ctrl_path is None or not neg_ctrl_path.exists():
        print("  SKIP fig_version_control: negative_control_version.jsonl not found")
        return

    recs = load_jsonl(neg_ctrl_path)
    if len(recs) < 4:
        print(f"  SKIP fig_version_control: only {len(recs)} records")
        return

    # Compute per-cell version gap
    by_cell: dict[tuple, dict[str, float]] = defaultdict(dict)
    for r in recs:
        model = r["model_id"].split("/")[-1]
        key = (model, r["n_shot"], r["prompt_format"])
        by_cell[key][r["lm_eval_version"]] = r["score"]

    versions = sorted({r["lm_eval_version"] for r in recs})
    cells = sorted(by_cell.keys())

    # Panel A: paired dot plot (version gap per cell)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                    gridspec_kw={"width_ratios": [2, 1]})

    gaps = []
    y_labels = []
    for i, cell in enumerate(cells):
        scores = by_cell[cell]
        if len(scores) == 2:
            v_sorted = sorted(scores.keys())
            s0, s1 = scores[v_sorted[0]], scores[v_sorted[1]]
            ax1.plot([s0, s1], [i, i], "o-", color=COLORS[6], alpha=0.5, ms=6)
            ax1.plot(s0, i, "o", color=COLORS[0], ms=8, zorder=3)
            ax1.plot(s1, i, "s", color=COLORS[1], ms=8, zorder=3)
            gaps.append(abs(s1 - s0))
        model, ns, fmt = cell
        y_labels.append(f"{model} n={ns} {fmt}")

    ax1.set_yticks(range(len(cells)))
    ax1.set_yticklabels(y_labels, fontsize=7)
    ax1.set_xlabel("Score (pp)")
    ax1.set_title(f"lm-eval version comparison ({versions[0]} vs {versions[-1]})")
    # Manual legend
    ax1.plot([], [], "o", color=COLORS[0], ms=8, label=versions[0])
    ax1.plot([], [], "s", color=COLORS[1], ms=8,
             label=versions[-1] if len(versions) > 1 else "v2")
    ax1.legend(fontsize=9)

    # Panel B: gap distribution
    if gaps:
        ax2.hist(gaps, bins=max(5, len(gaps) // 2), color=COLORS[2],
                 edgecolor="white", alpha=0.8)
        ax2.axvline(np.mean(gaps), ls="--", color="black", lw=1.5,
                    label=f"Mean gap: {np.mean(gaps):.2f} pp")
        ax2.set_xlabel("|Δ| between versions (pp)")
        ax2.set_ylabel("Count")
        ax2.set_title("Version gap distribution")
        ax2.legend(fontsize=9)

    plt.tight_layout()
    save(fig, "fig_version_control")

    print(f"  Version gaps: mean={np.mean(gaps):.2f}, max={np.max(gaps):.2f} pp")

    # If statsmodels available, compute partial R² for version
    try:
        import pandas as pd
        import statsmodels.api as sm

        df = pd.DataFrame(recs)
        dm = pd.get_dummies(df["model_id"], prefix="m", drop_first=True)
        X_base = pd.concat([pd.Series(1.0, index=df.index, name="const"),
                            dm], axis=1).astype(float)
        y = df["score"].astype(float)
        base_fit = sm.OLS(y, X_base).fit()
        ssr_base = float(np.sum(base_fit.resid ** 2))

        for var_name, col in [
            ("lm_eval_version", pd.Categorical(df["lm_eval_version"]).codes.astype(float)),
            ("n_shot",          df["n_shot"].astype(float)),
            ("prompt_format",   pd.Categorical(df["prompt_format"]).codes.astype(float)),
        ]:
            Xf = pd.concat([X_base, col.rename(var_name)], axis=1)
            fit = sm.OLS(y, Xf).fit()
            pr2 = max(0.0, (ssr_base - float(np.sum(fit.resid ** 2))) / ssr_base)
            print(f"  Partial R² [{var_name}]: {pr2:.4f}")

    except ImportError:
        print("  statsmodels not available, skipping partial R²")


# ---------------------------------------------------------------------------
# Macro update
# ---------------------------------------------------------------------------

def update_macros(records, sensitivity):
    gen = [r for r in records if r["benchmark"] in GENERATIVE_BENCHMARKS]
    ll = [r for r in records if r["benchmark"] not in GENERATIVE_BENCHMARKS]

    n_models = len({r["model_id"] for r in records})
    n_benches = len({r["benchmark"] for r in records})
    ok_count = len(records)

    gen_benches = sorted({r["benchmark"] for r in gen})

    # GSM8K OK counts
    gsm_ok = len([r for r in records if r["benchmark"] == "gsm8k"])
    mmlu_ok = len([r for r in records if r["benchmark"] == "mmlu"])
    bbh_ok = len([r for r in records if r["benchmark"] == "bbh"])

    # Effect sizes on 5-shot
    gen5 = [r for r in gen if r["n_shot"] == 5]
    temp_eff_5 = axis_effect("temperature", gen5) if gen5 else 0
    fmt_eff_5 = axis_effect("prompt_format", gen5) if gen5 else 0

    # Effect sizes on all
    temp_eff_all = axis_effect("temperature", gen) if gen else 0
    fmt_eff_all = axis_effect("prompt_format", gen) if gen else 0
    nshot_eff_all = axis_effect("n_shot", gen) if gen else 0

    # Noise floor
    by_cell: dict[tuple, list[float]] = defaultdict(list)
    for r in gen:
        k = (r["model_id"], r["benchmark"], r["temperature"],
             r["prompt_format"], r["n_shot"])
        by_cell[k].append(r["score"])
    seed_spreads = [max(v) - min(v) for v in by_cell.values() if len(v) >= 2]
    noise_mean = float(np.mean(seed_spreads)) if seed_spreads else 0
    noise_max = float(np.max(seed_spreads)) if seed_spreads else 0

    pr2_t5 = sensitivity.get("temperature_5shot", 0)
    pr2_f5 = sensitivity.get("prompt_format_5shot", 0)
    pr2_t_all = sensitivity.get("temperature_all", 0)
    pr2_f_all = sensitivity.get("prompt_format_all", 0)
    pr2_n_all = sensitivity.get("n_shot_all", 0)

    overturn = round(0.850 ** 2, 4)

    macros = f"""\
% macros.tex — single-source-of-truth for all key statistics
% Auto-generated by scripts/generate_experiment_figures.py
% DO NOT EDIT MANUALLY — re-run the script to update.

% Dataset scale
\\newcommand{{\\totalrecords}}{{29{{,}}331}}
\\newcommand{{\\totalsources}}{{11}}
\\newcommand{{\\totalmodels}}{{5{{,}}672}}
\\newcommand{{\\totalbenchmarks}}{{180}}

% Collision analysis
\\newcommand{{\\collisionpairs}}{{16}}
\\newcommand{{\\expectedcollisions}}{{69}}
\\newcommand{{\\collisionn}}{{8}}

% Primary correlation statistic (always qualified in text)
\\newcommand{{\\spearmanrho}}{{-0.850}}
\\newcommand{{\\spearmanp}}{{0.007}}
\\newcommand{{\\effectiven}}{{3}}  % effective DoF ≈ 2-3

% Fragmentation test
\\newcommand{{\\spearmanzstat}}{{-6.4}}
\\newcommand{{\\zexcludingolv}}{{1.4}}

% Coverage statistics
\\newcommand{{\\pwcartifacts}}{{57\\%}}
\\newcommand{{\\promptcoverage}}{{0\\%}}
\\newcommand{{\\tempcoverage}}{{0\\%}}

% Collision magnitudes
\\newcommand{{\\gpqagap}}{{32.3}}
\\newcommand{{\\tulumean}}{{0.14}}
\\newcommand{{\\tuluadvantage}}{{$+$0.032}}

% ─── Controlled experiment results ─────────────────────────────────
\\newcommand{{\\expmodels}}{{{n_models}}}
\\newcommand{{\\expbenches}}{{{n_benches}}}
\\newcommand{{\\expruns}}{{{ok_count}}}
\\newcommand{{\\expgenruns}}{{{len(gen)}}}
\\newcommand{{\\expllruns}}{{{len(ll)}}}

% Partial R² on 5-shot generative (the primary specification)
\\newcommand{{\\prtwotempfive}}{{{pr2_t5:.3f}}}
\\newcommand{{\\prtwofmtfive}}{{{pr2_f5:.3f}}}

% Partial R² on all generative data (for appendix / robustness)
\\newcommand{{\\prtwotempall}}{{{pr2_t_all:.3f}}}
\\newcommand{{\\prtwofmtall}}{{{pr2_f_all:.3f}}}
\\newcommand{{\\prtwonshotall}}{{{pr2_n_all:.3f}}}

% Overturn bound from Cinelli & Hazlett
\\newcommand{{\\overturnbound}}{{{overturn}}}  % = rho² = 0.850²

% Mean |max-min| effect per axis (5-shot generative)
\\newcommand{{\\tempeffectfive}}{{{temp_eff_5:.2f}}}
\\newcommand{{\\fmteffectfive}}{{{fmt_eff_5:.2f}}}

% Mean |max-min| effect per axis (all generative)
\\newcommand{{\\tempeffectall}}{{{temp_eff_all:.2f}}}
\\newcommand{{\\fmteffectall}}{{{fmt_eff_all:.2f}}}
\\newcommand{{\\nshoteffectall}}{{{nshot_eff_all:.2f}}}

% Seed-only noise floor (generative, 3 seeds per cell)
\\newcommand{{\\noisefloor}}{{{noise_mean:.2f}}}
\\newcommand{{\\noisefloormax}}{{{noise_max:.2f}}}
"""

    macros_path = ROOT / "submission" / "latex" / "macros.tex"
    macros_path.write_text(macros, encoding="utf-8")
    print(f"\n  Updated {macros_path}")
    print(f"  Models={n_models} Benches={n_benches} Runs={ok_count} "
          f"(gen={len(gen)}, ll={len(ll)})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Directory containing Colab result files")
    parser.add_argument("--skip-macros", action="store_true",
                        help="Skip updating macros.tex")
    args = parser.parse_args()

    rd = args.results_dir

    # 1. Main experiment
    print("=" * 60)
    print("1. MAIN EXPERIMENT FIGURES")
    print("=" * 60)
    main_results = load_jsonl(rd / "controlled_eval_results.jsonl")
    print(f"  Loaded {len(main_results)} OK records")

    if main_results:
        fig_config_variance(main_results)
        sensitivity, meta = fig_sensitivity(main_results)
        fig_config_vs_crosssource(main_results)
    else:
        print("  No main experiment results found!")
        sensitivity = {}

    # 2. Cross-extractor study
    print("\n" + "=" * 60)
    print("2. CROSS-EXTRACTOR STUDY")
    print("=" * 60)
    sample_dir = rd / "sample_outputs"
    fig_cross_extractor(sample_dir)

    # 3. Negative control
    print("\n" + "=" * 60)
    print("3. NEGATIVE CONTROL (lm-eval version)")
    print("=" * 60)
    fig_version_control(rd / "negative_control_version.jsonl")

    # 4. Update macros
    if not args.skip_macros and main_results:
        print("\n" + "=" * 60)
        print("4. UPDATING MACROS")
        print("=" * 60)
        update_macros(main_results, sensitivity)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
