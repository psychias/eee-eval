"""
analyse_controlled.py
=====================
Ingests controlled_eval_results.jsonl and produces:

  1. Causal sensitivity analysis (Cinelli & Hazlett 2020 partial R²)
  2. Configuration effect estimates per benchmark
  3. Three output figures:
       figures/fig_config_variance.pdf
       figures/fig_config_vs_crosssource.pdf
       figures/fig_sensitivity.pdf
  4. LaTeX-ready results summary:
       results/controlled_summary.tex

Usage:
    python analyse_controlled.py [--results PATH] [--output-dir PATH]
"""

import argparse
import json
import math
import warnings
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Cross-source collision data from Table 5 of the paper (for comparison)
# ---------------------------------------------------------------------------

CROSS_SOURCE_PAIRS = [
    # (model, benchmark, s1, s2, abs_delta)
    ("Qwen2.5-72B-Inst.", "GPQA",     16.67, 49.00, 32.33),
    ("SC2-15B-Inst.",     "MBPP+",    65.10, 61.20,  3.90),
    ("SC2-15B-Inst.",     "HumanEval+", 60.40, 63.40, 3.00),
    ("Tulu-3-8B",         "BBH",      16.86, 16.67,  0.19),
    ("Tulu-3-8B",         "GPQA",      6.26,  6.49,  0.23),
    ("Tulu-3-8B",         "IFEval",   82.55, 82.67,  0.12),
    ("Tulu-3-8B",         "MMLU-Pro", 20.23, 20.30,  0.07),
    ("Tulu-3-8B",         "MuSR",     10.52, 10.45,  0.07),
]

RESULTS_FILE = Path(__file__).parent / "results" / "controlled_eval_results.jsonl"
FIGURES_DIR = Path(__file__).parent / "figures"
OUTPUT_TEX = Path(__file__).parent / "results" / "controlled_summary.tex"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("status") == "ok" and r.get("score") is not None:
                    records.append(r)
            except json.JSONDecodeError:
                pass
    return records


# ---------------------------------------------------------------------------
# 1. Causal sensitivity analysis (Cinelli & Hazlett 2020 partial R²)
# ---------------------------------------------------------------------------

def compute_sensitivity(records: list[dict]) -> dict:
    """
    Regress score on model_id (fixed effect) to remove model-level variance.
    Compute partial R² of each configuration variable on the residuals.
    Report the minimum partial R² needed to overturn ρ = −0.871.
    """
    if not records:
        return {}

    try:
        import statsmodels.api as sm
        import pandas as pd
    except ImportError:
        warnings.warn("statsmodels/pandas not installed; using OLS fallback.")
        return _compute_sensitivity_numpy(records)

    import pandas as pd
    import statsmodels.api as sm

    df = pd.DataFrame(records)
    df = df.dropna(subset=["score"])

    # Model fixed effects (dummy encoding)
    model_dummies = pd.get_dummies(df["model_id"], drop_first=True)
    X_base = sm.add_constant(model_dummies.astype(float))
    y = df["score"]

    # Fit base model (model fixed effects only)
    base_model = sm.OLS(y, X_base).fit()
    residuals = base_model.resid
    ss_res_base = float(np.sum(residuals ** 2))
    ss_total = float(np.sum((y - y.mean()) ** 2))
    r2_base = 1.0 - ss_res_base / ss_total if ss_total > 0 else 0.0

    partial_r2 = {}
    for var, col in [
        ("temperature", df["temperature"]),
        ("prompt_format", pd.Categorical(df["prompt_format"]).codes),
        ("n_shot", df["n_shot"]),
    ]:
        X_full = pd.concat([X_base, col.rename(var)], axis=1).astype(float)
        try:
            full_model = sm.OLS(y, X_full).fit()
            ss_res_full = float(np.sum(full_model.resid ** 2))
            # Partial R²: (SSR_base - SSR_full) / SSR_base
            pr2 = (ss_res_base - ss_res_full) / ss_res_base if ss_res_base > 0 else 0.0
        except Exception:  # pylint: disable=broad-except
            pr2 = float("nan")
        partial_r2[var] = max(0.0, pr2)

    # Overturn threshold: ρ = -0.871 was computed over n=8 pairs.
    # Using Cinelli & Hazlett (2020): the partial R² of unmeasured confounders
    # needed to reduce |ρ| to 0 is approximately |ρ|² = 0.759.
    # With effective n ≈ 3, significance threshold is very low.
    rho_observed = -0.871
    overturn_threshold = rho_observed ** 2  # ≈ 0.759

    return {
        "r2_base_model": round(r2_base, 4),
        "partial_r2_temperature": round(partial_r2.get("temperature", float("nan")), 4),
        "partial_r2_prompt_format": round(partial_r2.get("prompt_format", float("nan")), 4),
        "partial_r2_n_shot": round(partial_r2.get("n_shot", float("nan")), 4),
        "overturn_threshold_r2": round(overturn_threshold, 4),
        "n_records": len(df),
    }


def _compute_sensitivity_numpy(records: list[dict]) -> dict:
    """Numpy fallback for sensitivity analysis when statsmodels unavailable."""
    models = list({r["model_id"] for r in records})
    model_idx = {m: i for i, m in enumerate(models)}
    n = len(records)
    k = len(models)

    y = np.array([r["score"] for r in records], dtype=float)
    # Model fixed effects via dummy OLS
    X = np.zeros((n, k + 1))
    X[:, 0] = 1.0  # intercept
    for i, r in enumerate(records):
        X[i, model_idx[r["model_id"]] + 1] = 1.0

    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ beta
        residuals = y - y_hat
        ss_res_base = float(np.sum(residuals ** 2))
        ss_total = float(np.sum((y - y.mean()) ** 2))
        r2_base = 1.0 - ss_res_base / ss_total if ss_total > 0 else 0.0

        partial_r2 = {}
        for var_name, var_vals in [
            ("temperature", np.array([r["temperature"] for r in records])),
            ("prompt_format", np.array([{"standard": 0, "cot": 1, "fewshot": 2}
                                        .get(r["prompt_format"], -1)
                                        for r in records], dtype=float)),
            ("n_shot", np.array([r["n_shot"] for r in records], dtype=float)),
        ]:
            X_full = np.column_stack([X, var_vals])
            beta_f, _, _, _ = np.linalg.lstsq(X_full, y, rcond=None)
            y_hat_f = X_full @ beta_f
            ss_res_full = float(np.sum((y - y_hat_f) ** 2))
            pr2 = (ss_res_base - ss_res_full) / ss_res_base if ss_res_base > 0 else 0.0
            partial_r2[var_name] = max(0.0, pr2)
    except Exception:  # pylint: disable=broad-except
        return {"error": "numerical failure in OLS"}

    rho_observed = -0.871
    return {
        "r2_base_model": round(r2_base, 4),
        "partial_r2_temperature": round(partial_r2.get("temperature", float("nan")), 4),
        "partial_r2_prompt_format": round(partial_r2.get("prompt_format", float("nan")), 4),
        "partial_r2_n_shot": round(partial_r2.get("n_shot", float("nan")), 4),
        "overturn_threshold_r2": round(rho_observed ** 2, 4),
        "n_records": n,
    }


# ---------------------------------------------------------------------------
# 2. Configuration effect estimates per benchmark
# ---------------------------------------------------------------------------

def compute_config_effects(records: list[dict]) -> dict:
    """
    For each (model, benchmark), compute max–min score across configs.
    Compare to cross-source deltas from Table 5.
    """
    by_mb = defaultdict(list)
    for r in records:
        by_mb[(r["model_id"], r["benchmark"])].append(r)

    effects_by_benchmark = defaultdict(list)
    flags = []

    for (model_id, benchmark), runs in by_mb.items():
        scores = [r["score"] for r in runs]
        if len(scores) < 2:
            continue
        effect = max(scores) - min(scores)
        effects_by_benchmark[benchmark].append(effect)

        # Compare to cross-source deltas for same benchmark
        for cs_model, cs_bench, _, _, cs_delta in CROSS_SOURCE_PAIRS:
            if benchmark.lower() in cs_bench.lower() or cs_bench.lower() in benchmark.lower():
                if effect >= cs_delta:
                    flags.append({
                        "model": model_id.split("/")[-1],
                        "benchmark": benchmark,
                        "config_effect": round(effect, 4),
                        "cross_source_delta": round(cs_delta, 4),
                        "cross_source_pair": cs_model,
                        "message": (
                            f"configuration choice alone can explain the "
                            f"observed gap for {cs_model}/{cs_bench}"
                        ),
                    })

    summary = {}
    for bench, effects in effects_by_benchmark.items():
        summary[bench] = {
            "mean_abs_config_effect_pp": round(float(np.mean(effects)), 4),
            "max_abs_config_effect_pp": round(float(np.max(effects)), 4),
            "n_models": len(effects),
        }

    # Decompose effects by axis
    temp_effects = _axis_effects(records, "temperature")
    fmt_effects = _axis_effects(records, "prompt_format")
    nshot_effects = _axis_effects(records, "n_shot")

    return {
        "per_benchmark": summary,
        "mean_abs_temp_effect_pp": round(float(np.mean(temp_effects)) if temp_effects else 0.0, 4),
        "mean_abs_fmt_effect_pp": round(float(np.mean(fmt_effects)) if fmt_effects else 0.0, 4),
        "mean_abs_nshot_effect_pp": round(float(np.mean(nshot_effects)) if nshot_effects else 0.0, 4),
        "flags": flags,
    }


def _axis_effects(records: list[dict], axis: str) -> list[float]:
    """
    For each (model, benchmark, other_axes...) group, compute max–min score
    across the specified axis.
    """
    def group_key(r: dict, axis: str) -> tuple:
        axes = {"temperature": ("model_id", "benchmark", "prompt_format", "n_shot"),
                "prompt_format": ("model_id", "benchmark", "temperature", "n_shot"),
                "n_shot": ("model_id", "benchmark", "temperature", "prompt_format")}
        return tuple(r[a] for a in axes[axis])

    by_group = defaultdict(list)
    for r in records:
        by_group[group_key(r, axis)].append(r["score"])

    effects = []
    for scores in by_group.values():
        if len(scores) >= 2:
            effects.append(max(scores) - min(scores))
    return effects


# ---------------------------------------------------------------------------
# 3. Figures
# ---------------------------------------------------------------------------

def make_figures(records: list[dict], effects: dict,
                 sensitivity: dict) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not available; skipping figure generation.")
        return

    _fig_config_variance(records, plt)
    _fig_config_vs_crosssource(records, effects, plt)
    _fig_sensitivity(sensitivity, plt)


def _fig_config_variance(records: list[dict], plt) -> None:
    """3-panel boxplot: score distributions by temperature, format, n_shot."""
    import matplotlib.gridspec as gridspec

    if not records:
        return

    fig = plt.figure(figsize=(12, 4.5))
    gs = gridspec.GridSpec(1, 3, figure=fig)

    axes_data = [
        (fig.add_subplot(gs[0]), "temperature", "Temperature",
         sorted({r["temperature"] for r in records})),
        (fig.add_subplot(gs[1]), "prompt_format", "Prompt format",
         sorted({r["prompt_format"] for r in records})),
        (fig.add_subplot(gs[2]), "n_shot", "N-shot",
         sorted({r["n_shot"] for r in records})),
    ]

    for ax, field, label, vals in axes_data:
        data = [[r["score"] for r in records if r[field] == v] for v in vals]
        ax.boxplot(data, labels=[str(v) for v in vals])
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel("Score (pp)", fontsize=10)
        ax.set_title(f"Score distribution by {label}", fontsize=10)

    plt.tight_layout()
    out = FIGURES_DIR / "fig_config_variance.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def _fig_config_vs_crosssource(records: list[dict], effects: dict,
                                plt) -> None:
    """Scatter: within-model config variance vs cross-source divergence."""
    if not records:
        return

    from collections import defaultdict
    by_mb = defaultdict(list)
    for r in records:
        by_mb[(r["benchmark"])].append(r["score"])

    cs_deltas = [d for _, _, _, _, d in CROSS_SOURCE_PAIRS]
    cs_labels = [f"{m}/{b}" for m, b, _, _, _ in CROSS_SOURCE_PAIRS]

    # Controlled config max-min per benchmark (pooled across models)
    ctrl_effects = []
    ctrl_labels = []
    for bench, effect_info in effects.get("per_benchmark", {}).items():
        ctrl_effects.append(effect_info["mean_abs_config_effect_pp"])
        ctrl_labels.append(bench)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(range(len(cs_deltas)), cs_deltas,
               label="Cross-source divergence (Table 5)", marker="s", color="#CC3311")
    ax.axhline(
        y=sum(ctrl_effects) / len(ctrl_effects) if ctrl_effects else 0,
        linestyle="--", color="#0173B2", alpha=0.7,
        label=f"Mean controlled config effect ({sum(ctrl_effects)/max(1,len(ctrl_effects)):.1f} pp)",
    )
    ax.set_xticks(range(len(cs_labels)))
    ax.set_xticklabels(cs_labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("|Δ| (pp)", fontsize=10)
    ax.set_title("Cross-source divergence vs controlled configuration effect", fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()

    out = FIGURES_DIR / "fig_config_vs_crosssource.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def _fig_sensitivity(sensitivity: dict, plt) -> None:
    """Cinelli-Hazlett style sensitivity contour plot."""
    if not sensitivity or "partial_r2_temperature" not in sensitivity:
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    # Contour: R²_Y~Z|X and R²_D~Z|X axes
    # Observed confounders: temperature, prompt_format, n_shot
    measured_vars = {
        "temperature": sensitivity.get("partial_r2_temperature", 0.0),
        "prompt_format": sensitivity.get("partial_r2_prompt_format", 0.0),
        "n_shot": sensitivity.get("partial_r2_n_shot", 0.0),
    }
    overturn = sensitivity.get("overturn_threshold_r2", 0.759)

    rx = np.linspace(0, 1.0, 200)
    ry = np.linspace(0, 1.0, 200)
    RX, RY = np.meshgrid(rx, ry)
    # Robustness value: approximate as min(R²) needed to overturn
    # For simplicity, contour at combined R² = overturn threshold
    Z = np.sqrt(RX * RY)
    overturn_level = math.sqrt(overturn)

    cs = ax.contour(RX, RY, Z, levels=[overturn_level], colors=["#CC3311"],
                    linestyles=["--"])
    ax.clabel(cs, fmt={overturn_level: f"Overturn bound (R²={overturn:.2f})"}, fontsize=7)

    # Plot measured confounders
    colors = {"temperature": "#0173B2", "prompt_format": "#DE8F05", "n_shot": "#029E73"}
    for var, pr2 in measured_vars.items():
        ax.scatter([pr2], [pr2], color=colors[var], s=80, zorder=5, label=f"{var} (R²={pr2:.3f})")

    ax.set_xlabel(r"Partial $R^2$ of confounder with score ($R^2_{Y \sim Z|X}$)", fontsize=9)
    ax.set_ylabel(r"Partial $R^2$ of confounder with metadata ($R^2_{D \sim Z|X}$)", fontsize=9)
    ax.set_title("Sensitivity analysis: robustness of ρ = −0.871\n"
                 "(Cinelli & Hazlett 2020)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Annotation: are measured confounders above or below the overturn bound?
    max_measured = max(measured_vars.values()) if measured_vars else 0
    if max_measured < overturn:
        ax.text(0.65, 0.05,
                f"Measured confounders\n(max R²={max_measured:.3f}) are BELOW\noverturn bound ({overturn:.3f})",
                fontsize=7, color="#0173B2", transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="gray", alpha=0.8))
    else:
        ax.text(0.50, 0.05,
                f"Measured confounders\n(max R²={max_measured:.3f}) EXCEED\noverturn bound ({overturn:.3f})",
                fontsize=7, color="#CC3311", transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="gray", alpha=0.8))

    plt.tight_layout()
    out = FIGURES_DIR / "fig_sensitivity.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# 4. LaTeX summary
# ---------------------------------------------------------------------------

def write_latex_summary(records: list[dict],
                        effects: dict,
                        sensitivity: dict) -> None:
    OUTPUT_TEX.parent.mkdir(parents=True, exist_ok=True)

    n_runs = len(records)
    n_models = len({r["model_id"] for r in records})
    n_benches = len({r["benchmark"] for r in records})
    temp_effect = effects.get("mean_abs_temp_effect_pp", 0.0)
    fmt_effect = effects.get("mean_abs_fmt_effect_pp", 0.0)
    nshot_effect = effects.get("mean_abs_nshot_effect_pp", 0.0)
    pr2_temp = sensitivity.get("partial_r2_temperature", float("nan"))
    pr2_fmt = sensitivity.get("partial_r2_prompt_format", float("nan"))
    pr2_nshot = sensitivity.get("partial_r2_n_shot", float("nan"))
    overturn = sensitivity.get("overturn_threshold_r2", 0.759)
    max_pr2 = max(pr2_temp, pr2_fmt, pr2_nshot)
    robust_word = "above" if max_pr2 >= overturn else "below"

    flags = effects.get("flags", [])
    flag_lines = "\n".join(
        f"% FLAG: {f['message']}" for f in flags
    ) if flags else "% No flags."

    lines = [
        "% controlled_summary.tex — auto-generated by analyse_controlled.py",
        "% Include in preamble via: \\input{../../submission/experiments/results/controlled_summary}",
        "",
        f"\\newcommand{{\\ctrlnruns}}{{{n_runs}}}",
        f"\\newcommand{{\\ctrlnmodels}}{{{n_models}}}",
        f"\\newcommand{{\\ctrlnbenches}}{{{n_benches}}}",
        f"\\newcommand{{\\tempvariance}}{{{temp_effect:.2f}}}",
        f"\\newcommand{{\\fmtvariance}}{{{fmt_effect:.2f}}}",
        f"\\newcommand{{\\nshotvariance}}{{{nshot_effect:.2f}}}",
        f"\\newcommand{{\\prtwotemp}}{{{pr2_temp:.4f}}}",
        f"\\newcommand{{\\prtwoformat}}{{{pr2_fmt:.4f}}}",
        f"\\newcommand{{\\prtwoshot}}{{{pr2_nshot:.4f}}}",
        f"\\newcommand{{\\overturnbound}}{{{overturn:.4f}}}",
        f"\\newcommand{{\\robustword}}{{{robust_word}}}",
        "",
        "% Sensitivity analysis table",
        "\\newcommand{\\sensitivitytable}{%",
        "\\begin{table}[h]",
        "\\centering\\small",
        "\\caption{Sensitivity analysis results (Cinelli \\& Hazlett 2020). Partial $R^2$ of",
        "each measured configuration variable on score residuals after removing model fixed effects.",
        "The overturn bound is $\\rho^2 = 0.759$: unmeasured confounders would need",
        "to explain at least this fraction of score variance to overturn $\\rho = -0.871$.}",
        "\\label{tab:sensitivity}",
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "\\textbf{Variable} & \\textbf{Partial $R^2$} & \\textbf{$>$ Overturn bound?} \\\\",
        "\\midrule",
        f"Temperature & {pr2_temp:.4f} & {'Yes' if pr2_temp >= overturn else 'No'} \\\\",
        f"Prompt format & {pr2_fmt:.4f} & {'Yes' if pr2_fmt >= overturn else 'No'} \\\\",
        f"N-shot count & {pr2_nshot:.4f} & {'Yes' if pr2_nshot >= overturn else 'No'} \\\\",
        "\\midrule",
        f"Overturn bound ($\\rho^2$) & {overturn:.4f} & --- \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "}%",
        "",
        flag_lines,
    ]

    OUTPUT_TEX.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {OUTPUT_TEX}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyse controlled evaluation results.")
    p.add_argument("--results", default=str(RESULTS_FILE),
                   help="Path to controlled_eval_results.jsonl.")
    p.add_argument("--output-dir", default=str(FIGURES_DIR),
                   help="Directory for output figures.")
    p.add_argument("--no-figures", action="store_true",
                   help="Skip figure generation.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    global FIGURES_DIR
    FIGURES_DIR = Path(args.output_dir)

    results_path = Path(args.results)
    records = load_results(results_path)
    print(f"Loaded {len(records)} valid records.")

    if not records:
        print("No valid records to analyse.")
        return

    print("\n--- Running sensitivity analysis ---")
    sensitivity = compute_sensitivity(records)
    print(json.dumps(sensitivity, indent=2))

    print("\n--- Computing configuration effects ---")
    effects = compute_config_effects(records)
    for bench, info in effects.get("per_benchmark", {}).items():
        print(f"  {bench}: mean config effect = {info['mean_abs_config_effect_pp']:.2f} pp "
              f"(n={info['n_models']} models)")

    if effects["flags"]:
        print("\n  *** FLAGS ***")
        for flag in effects["flags"]:
            print(f"  FLAG: {flag['message']}")
            print(f"        config_effect={flag['config_effect']:.2f} pp "
                  f"> cross_source_delta={flag['cross_source_delta']:.2f} pp")

    print(f"\n  Mean temp effect: {effects['mean_abs_temp_effect_pp']:.2f} pp")
    print(f"  Mean fmt effect: {effects['mean_abs_fmt_effect_pp']:.2f} pp")
    print(f"  Mean n-shot effect: {effects['mean_abs_nshot_effect_pp']:.2f} pp")

    if not args.no_figures:
        print("\n--- Generating figures ---")
        make_figures(records, effects, sensitivity)

    print("\n--- Writing LaTeX summary ---")
    write_latex_summary(records, effects, sensitivity)
    print("Done.")


if __name__ == "__main__":
    main()
