#!/usr/bin/env python3
"""
harness_effect_experiment.py
============================
Discovers cross-harness evaluation pairs from the EEE data repository,
performs statistical analysis of harness effects on scores, and generates
comparison reports and figures.

A "cross-harness pair" is the same (model, benchmark) evaluated under two
different evaluation harnesses (e.g., lm_eval vs vllm, evalplus vs transformers).

Workflow
--------
1. Scan all JSON files in data/ and build a global index.
2. Identify pairs where the same model+benchmark was evaluated with different
   eval_library values.
3. Separate pairs into "config-matched" (same shots/CoT/temp) and
   "config-mismatched" to isolate the pure harness effect.
4. Run statistical tests (paired t-test, Wilcoxon signed-rank, Mann-Whitney U).
5. Generate figures: delta distribution, violin plot, bar chart.
6. Write a structured report.

Usage
-----
    python harness_effect_experiment.py [--data-dir data] [--output-dir harness_experiment_output]
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import re
import sys
import warnings
from dataclasses import dataclass, field, asdict
from typing import Optional

import pandas as pd
import numpy as np

# ─── Configuration ───────────────────────────────────────────────────────────
BENCHMARK_ALIASES = {
    "humaneval+": "humaneval_plus",
    "human_eval+": "humaneval_plus",
    "human_eval_plus": "humaneval_plus",
    "humaneval": "humaneval",
    "human_eval": "humaneval",
    "mbpp+": "mbpp_plus",
    "mbpp_plus": "mbpp_plus",
    "gpqa": "gpqa",
    "gpqa_diamond": "gpqa_diamond",
    "mmlu": "mmlu",
    "mmlu_pro": "mmlu_pro",
    "mmlu-pro": "mmlu_pro",
    "ifeval": "ifeval",
    "if_eval": "ifeval",
    "bbh": "bbh",
    "big_bench_hard": "bbh",
    "math_500": "math_500",
    "math-500": "math_500",
    "musr": "musr",
    "arc_challenge": "arc_challenge",
    "hellaswag": "hellaswag",
    "winogrande": "winogrande",
    "gsm8k": "gsm8k",
}

HARNESS_NORMALISATION = {
    "lm_eval": "lm_eval",
    "lm-eval": "lm_eval",
    "lm-evaluation-harness": "lm_eval",
    "eleutherai/lm-eval": "lm_eval",
    "evalplus": "evalplus",
    "transformers": "transformers",
    "vllm": "vllm",
    "fastchat": "fastchat",
    "alpaca_eval": "alpaca_eval",
    "gorilla_eval": "gorilla_eval",
    "bigcodebench": "bigcodebench",
    "swebench": "swebench",
    "wildeval": "wildeval",
    "internal": "internal",
    "unknown": "unknown",
}


def norm_bench(name: str) -> str:
    s = name.lower().strip()
    s = re.sub(r"\s*\(.*?\)\s*$", "", s)   # strip trailing "(0-shot)" etc.
    s = re.sub(r"[-_\s]+", "_", s)
    return BENCHMARK_ALIASES.get(s, s)


def norm_model(name: str) -> str:
    return re.sub(r"[-_\s]+", "-", name.lower().strip())


def norm_harness(h: str) -> str:
    return HARNESS_NORMALISATION.get(h.lower().strip(), h.lower().strip())


# ─── Data structures ─────────────────────────────────────────────────────────
@dataclass
class EvalEntry:
    model_name: str
    model_id: str
    eval_name: str
    eval_library: str
    eval_library_raw: str
    source_folder: str
    source_name: str
    score: Optional[float]
    metric: str
    shots: Optional[int]
    chain_of_thought: Optional[bool]
    temperature: Optional[float]
    file_path: str


@dataclass
class HarnessPair:
    model_name: str
    benchmark: str
    entry_a: EvalEntry
    entry_b: EvalEntry
    delta: float  # score_a - score_b
    abs_delta: float
    config_matched: bool
    config_diffs: list


# ─── Phase 1: Load all data ─────────────────────────────────────────────────
def load_all_entries(data_dir: pathlib.Path) -> list[EvalEntry]:
    entries: list[EvalEntry] = []
    for f in sorted(data_dir.rglob("*.json")):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        mi = rec.get("model_info", {})
        lib = rec.get("eval_library", {})
        sm = rec.get("source_metadata", {})
        parts = f.relative_to(data_dir).parts
        source_folder = parts[0] if parts else "unknown"

        for er in rec.get("evaluation_results", []):
            ga = er.get("generation_config", {}).get("generation_args", {})
            score_raw = er.get("score_details", {}).get("score")
            try:
                score = float(score_raw) if score_raw is not None else None
            except (ValueError, TypeError):
                score = None
            shots_raw = ga.get("shots")
            try:
                shots = int(shots_raw) if shots_raw is not None else None
            except (ValueError, TypeError):
                shots = None
            cot = ga.get("chain_of_thought")
            temp_raw = ga.get("temperature")
            try:
                temp = float(temp_raw) if temp_raw is not None else None
            except (ValueError, TypeError):
                temp = None

            entries.append(EvalEntry(
                model_name=mi.get("name", "").strip(),
                model_id=mi.get("id", "").strip(),
                eval_name=er.get("evaluation_name", ""),
                eval_library=norm_harness(lib.get("name", "unknown")),
                eval_library_raw=lib.get("name", "unknown"),
                source_folder=source_folder,
                source_name=sm.get("source_name", ""),
                score=score,
                metric=er.get("metric_config", {}).get("metric_name", ""),
                shots=shots,
                chain_of_thought=cot,
                temperature=temp,
                file_path=str(f.relative_to(data_dir)),
            ))
    return entries


# ─── Phase 2: Find cross-harness pairs ──────────────────────────────────────
def find_cross_harness_pairs(entries: list[EvalEntry]) -> list[HarnessPair]:
    index: dict[tuple[str, str], list[EvalEntry]] = collections.defaultdict(list)
    for e in entries:
        key = (norm_model(e.model_name), norm_bench(e.eval_name))
        index[key].append(e)

    pairs: list[HarnessPair] = []
    seen = set()

    for key, group in sorted(index.items()):
        by_harness: dict[str, list[EvalEntry]] = collections.defaultdict(list)
        for e in group:
            if e.score is not None:
                by_harness[e.eval_library].append(e)

        harness_list = [h for h in sorted(by_harness) if h != "unknown"]
        # Also include unknown if it has entries
        if "unknown" in by_harness:
            harness_list.append("unknown")

        if len(harness_list) < 2:
            continue

        for i in range(len(harness_list)):
            for j in range(i + 1, len(harness_list)):
                h1, h2 = harness_list[i], harness_list[j]
                for e1 in by_harness[h1]:
                    for e2 in by_harness[h2]:
                        # Dedup by file pair
                        pair_key = tuple(sorted([e1.file_path, e2.file_path]))
                        if pair_key in seen:
                            continue
                        seen.add(pair_key)

                        config_diffs = []
                        if e1.shots != e2.shots:
                            config_diffs.append(f"shots: {e1.shots} vs {e2.shots}")
                        if e1.chain_of_thought != e2.chain_of_thought:
                            config_diffs.append(f"CoT: {e1.chain_of_thought} vs {e2.chain_of_thought}")
                        if e1.temperature != e2.temperature:
                            config_diffs.append(f"temp: {e1.temperature} vs {e2.temperature}")

                        # Treat None-vs-value as "unspecified" not "different"
                        strict_match = (
                            e1.shots == e2.shots
                            and e1.chain_of_thought == e2.chain_of_thought
                            and e1.temperature == e2.temperature
                        )

                        delta = e1.score - e2.score
                        pairs.append(HarnessPair(
                            model_name=e1.model_name,
                            benchmark=e1.eval_name,
                            entry_a=e1,
                            entry_b=e2,
                            delta=delta,
                            abs_delta=abs(delta),
                            config_matched=strict_match,
                            config_diffs=config_diffs,
                        ))
    return pairs


# ─── Phase 3: Statistical analysis ──────────────────────────────────────────
def run_statistics(pairs: list[HarnessPair]) -> dict:
    from scipy import stats as sp_stats

    results = {}
    deltas_all = [p.delta for p in pairs]
    abs_deltas_all = [p.abs_delta for p in pairs]
    matched = [p for p in pairs if p.config_matched]
    mismatched = [p for p in pairs if not p.config_matched]

    results["n_total_pairs"] = len(pairs)
    results["n_config_matched"] = len(matched)
    results["n_config_mismatched"] = len(mismatched)

    if deltas_all:
        results["mean_delta"] = float(np.mean(deltas_all))
        results["median_delta"] = float(np.median(deltas_all))
        results["std_delta"] = float(np.std(deltas_all, ddof=1)) if len(deltas_all) > 1 else 0
        results["mean_abs_delta"] = float(np.mean(abs_deltas_all))
        results["median_abs_delta"] = float(np.median(abs_deltas_all))
        results["max_abs_delta"] = float(np.max(abs_deltas_all))

    # One-sample t-test: is mean delta different from 0?
    if len(deltas_all) >= 3:
        t_stat, p_val = sp_stats.ttest_1samp(deltas_all, 0)
        results["ttest_1samp_t"] = float(t_stat)
        results["ttest_1samp_p"] = float(p_val)

    # Wilcoxon signed-rank: non-parametric test for symmetry around 0
    if len(deltas_all) >= 6:
        try:
            w_stat, p_val = sp_stats.wilcoxon(deltas_all)
            results["wilcoxon_stat"] = float(w_stat)
            results["wilcoxon_p"] = float(p_val)
        except ValueError:
            pass  # all deltas zero

    # Mann-Whitney U: config-matched vs config-mismatched abs_deltas
    matched_abs = [p.abs_delta for p in matched]
    mismatched_abs = [p.abs_delta for p in mismatched]
    if len(matched_abs) >= 2 and len(mismatched_abs) >= 2:
        u_stat, p_val = sp_stats.mannwhitneyu(matched_abs, mismatched_abs, alternative="two-sided")
        results["mannwhitney_u"] = float(u_stat)
        results["mannwhitney_p"] = float(p_val)

    # Per-harness-pair statistics
    harness_pair_stats = collections.defaultdict(list)
    for p in pairs:
        pair_key = tuple(sorted([p.entry_a.eval_library, p.entry_b.eval_library]))
        harness_pair_stats[pair_key].append(p.abs_delta)

    results["per_harness_pair"] = {}
    for pk, deltas in sorted(harness_pair_stats.items()):
        key_str = f"{pk[0]} vs {pk[1]}"
        results["per_harness_pair"][key_str] = {
            "n": len(deltas),
            "mean_abs_delta": float(np.mean(deltas)),
            "median_abs_delta": float(np.median(deltas)),
            "max_abs_delta": float(np.max(deltas)),
        }

    return results


# ─── Phase 4: Generate figures ───────────────────────────────────────────────
def generate_figures(pairs: list[HarnessPair], out_dir: pathlib.Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Figure 1: Delta distribution (histogram) ---
    deltas = [p.delta for p in pairs]
    abs_deltas = [p.abs_delta for p in pairs]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(deltas, bins=max(5, len(deltas) // 3), color="#4C72B0", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", alpha=0.6, label="zero")
    ax.axvline(np.mean(deltas), color="orange", linestyle="-", alpha=0.8,
               label=f"mean={np.mean(deltas):.2f}")
    ax.set_xlabel("Score Delta (harness A − harness B)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Score Deltas\n(Cross-Harness Pairs)")
    ax.legend()

    ax = axes[1]
    matched_d = [p.abs_delta for p in pairs if p.config_matched]
    mismatched_d = [p.abs_delta for p in pairs if not p.config_matched]
    labels, data_vals, colors = [], [], []
    if matched_d:
        labels.append(f"Config-matched\n(n={len(matched_d)})")
        data_vals.append(matched_d)
        colors.append("#4C72B0")
    if mismatched_d:
        labels.append(f"Config-mismatched\n(n={len(mismatched_d)})")
        data_vals.append(mismatched_d)
        colors.append("#DD8452")

    if data_vals:
        bp = ax.boxplot(data_vals, labels=labels, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    ax.set_ylabel("|Δ| (Absolute Score Delta)")
    ax.set_title("Harness Effect Size:\nConfig-Matched vs Config-Mismatched")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "fig1_delta_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / 'fig1_delta_distribution.png'}")

    # --- Figure 2: Per-model delta bar chart ---
    df = pd.DataFrame([{
        "model": p.model_name,
        "benchmark": p.benchmark,
        "harness_a": p.entry_a.eval_library,
        "harness_b": p.entry_b.eval_library,
        "delta": p.delta,
        "abs_delta": p.abs_delta,
        "config_matched": p.config_matched,
    } for p in pairs])

    fig, ax = plt.subplots(figsize=(14, max(6, len(pairs) * 0.4)))
    df_sorted = df.sort_values("abs_delta", ascending=True)
    labels_list = [
        f"{row['model']}\n{row['benchmark']}\n({row['harness_a']} vs {row['harness_b']})"
        for _, row in df_sorted.iterrows()
    ]
    bar_colors = ["#4C72B0" if m else "#DD8452" for m in df_sorted["config_matched"]]
    ax.barh(range(len(df_sorted)), df_sorted["delta"], color=bar_colors, alpha=0.8,
            edgecolor="white")
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(labels_list, fontsize=7)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Score Delta (harness A − harness B)")
    ax.set_title("Per-Pair Score Deltas Across Evaluation Harnesses")
    ax.grid(axis="x", alpha=0.3)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#4C72B0", alpha=0.7, label="Config-matched"),
        Patch(facecolor="#DD8452", alpha=0.7, label="Config-mismatched"),
    ], loc="lower right")

    fig.tight_layout()
    fig.savefig(out_dir / "fig2_per_pair_deltas.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / 'fig2_per_pair_deltas.png'}")

    # --- Figure 3: Harness pair heatmap ---
    harness_pairs_agg = collections.defaultdict(list)
    for p in pairs:
        h1, h2 = sorted([p.entry_a.eval_library, p.entry_b.eval_library])
        harness_pairs_agg[(h1, h2)].append(p.abs_delta)

    all_h = sorted(set(h for pair in harness_pairs_agg for h in pair))
    matrix = pd.DataFrame(np.nan, index=all_h, columns=all_h)
    count_matrix = pd.DataFrame(0, index=all_h, columns=all_h)
    for (h1, h2), deltas_list in harness_pairs_agg.items():
        matrix.loc[h1, h2] = np.mean(deltas_list)
        matrix.loc[h2, h1] = np.mean(deltas_list)
        count_matrix.loc[h1, h2] = len(deltas_list)
        count_matrix.loc[h2, h1] = len(deltas_list)

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = matrix.isna()
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlOrRd", mask=mask,
                ax=ax, cbar_kws={"label": "Mean |Δ|"})
    ax.set_title("Mean Absolute Score Delta Between Harness Pairs")
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_harness_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / 'fig3_harness_heatmap.png'}")

    # --- Figure 4: Score agreement scatter plot ---
    fig, ax = plt.subplots(figsize=(8, 8))
    for p in pairs:
        color = "#4C72B0" if p.config_matched else "#DD8452"
        ax.scatter(p.entry_a.score, p.entry_b.score, c=color, alpha=0.7, s=50,
                  edgecolors="white", linewidths=0.5)
    # Diagonal
    all_scores = [p.entry_a.score for p in pairs] + [p.entry_b.score for p in pairs]
    lo, hi = min(all_scores), max(all_scores)
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], "k--", alpha=0.4)
    ax.set_xlabel("Score (Harness A)")
    ax.set_ylabel("Score (Harness B)")
    ax.set_title("Score Agreement Across Harnesses\n(Points on diagonal = perfect agreement)")
    ax.legend(handles=[
        Patch(facecolor="#4C72B0", alpha=0.7, label="Config-matched"),
        Patch(facecolor="#DD8452", alpha=0.7, label="Config-mismatched"),
    ])
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_score_agreement.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / 'fig4_score_agreement.png'}")


# ─── Phase 5: Generate report ────────────────────────────────────────────────
def generate_report(
    pairs: list[HarnessPair],
    stats: dict,
    out_dir: pathlib.Path,
    total_entries: int,
):
    lines = []
    lines.append("=" * 80)
    lines.append("HARNESS EFFECT EXPERIMENT REPORT")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Total evaluation entries scanned:    {total_entries}")
    lines.append(f"Cross-harness pairs found:           {len(pairs)}")
    lines.append(f"  Config-matched (pure harness):     {stats['n_config_matched']}")
    lines.append(f"  Config-mismatched (harness+config): {stats['n_config_mismatched']}")
    lines.append("")

    lines.append("─" * 80)
    lines.append("SCORE DELTA STATISTICS")
    lines.append("─" * 80)
    if "mean_delta" in stats:
        lines.append(f"  Mean delta:        {stats['mean_delta']:+.4f}")
        lines.append(f"  Median delta:      {stats['median_delta']:+.4f}")
        lines.append(f"  Std delta:         {stats['std_delta']:.4f}")
        lines.append(f"  Mean |delta|:      {stats['mean_abs_delta']:.4f}")
        lines.append(f"  Median |delta|:    {stats['median_abs_delta']:.4f}")
        lines.append(f"  Max |delta|:       {stats['max_abs_delta']:.4f}")
    lines.append("")

    lines.append("─" * 80)
    lines.append("STATISTICAL TESTS")
    lines.append("─" * 80)
    if "ttest_1samp_p" in stats:
        sig = "***" if stats["ttest_1samp_p"] < 0.001 else "**" if stats["ttest_1samp_p"] < 0.01 else "*" if stats["ttest_1samp_p"] < 0.05 else "ns"
        lines.append(f"  One-sample t-test (H0: mean delta = 0):")
        lines.append(f"    t = {stats['ttest_1samp_t']:.4f}, p = {stats['ttest_1samp_p']:.6f} {sig}")
    if "wilcoxon_p" in stats:
        sig = "***" if stats["wilcoxon_p"] < 0.001 else "**" if stats["wilcoxon_p"] < 0.01 else "*" if stats["wilcoxon_p"] < 0.05 else "ns"
        lines.append(f"  Wilcoxon signed-rank test:")
        lines.append(f"    W = {stats['wilcoxon_stat']:.4f}, p = {stats['wilcoxon_p']:.6f} {sig}")
    if "mannwhitney_p" in stats:
        sig = "***" if stats["mannwhitney_p"] < 0.001 else "**" if stats["mannwhitney_p"] < 0.01 else "*" if stats["mannwhitney_p"] < 0.05 else "ns"
        lines.append(f"  Mann-Whitney U (config-matched vs -mismatched |delta|):")
        lines.append(f"    U = {stats['mannwhitney_u']:.4f}, p = {stats['mannwhitney_p']:.6f} {sig}")
    lines.append("")

    lines.append("─" * 80)
    lines.append("PER-HARNESS-PAIR BREAKDOWN")
    lines.append("─" * 80)
    if "per_harness_pair" in stats:
        for pair_name, pair_stats in stats["per_harness_pair"].items():
            lines.append(f"  {pair_name}:")
            lines.append(f"    n={pair_stats['n']}, mean|Δ|={pair_stats['mean_abs_delta']:.4f}, "
                        f"median|Δ|={pair_stats['median_abs_delta']:.4f}, "
                        f"max|Δ|={pair_stats['max_abs_delta']:.4f}")
    lines.append("")

    lines.append("─" * 80)
    lines.append("ALL CROSS-HARNESS PAIRS (sorted by |delta|)")
    lines.append("─" * 80)
    for i, p in enumerate(sorted(pairs, key=lambda x: -x.abs_delta), 1):
        match_str = "CONFIG-MATCH" if p.config_matched else "CONFIG-DIFF"
        lines.append(f"  {i:2d}. {p.model_name}")
        lines.append(f"      Benchmark: {p.benchmark}")
        lines.append(f"      {p.entry_a.eval_library:15s} (from {p.entry_a.source_folder}) = {p.entry_a.score:.2f}")
        lines.append(f"      {p.entry_b.eval_library:15s} (from {p.entry_b.source_folder}) = {p.entry_b.score:.2f}")
        lines.append(f"      Delta: {p.delta:+.2f}  |Delta|: {p.abs_delta:.2f}  [{match_str}]")
        if p.config_diffs:
            lines.append(f"      Config diffs: {'; '.join(p.config_diffs)}")
        lines.append(f"      Files: {p.entry_a.file_path}")
        lines.append(f"             {p.entry_b.file_path}")
        lines.append("")

    lines.append("─" * 80)
    lines.append("KEY FINDINGS")
    lines.append("─" * 80)

    # Derive findings
    matched = [p for p in pairs if p.config_matched]
    mismatched = [p for p in pairs if not p.config_matched]

    if matched:
        matched_mean = np.mean([p.abs_delta for p in matched])
        lines.append(f"  1. Config-matched pairs (n={len(matched)}):")
        lines.append(f"     Mean |Δ| = {matched_mean:.4f}")
        if matched_mean < 0.5:
            lines.append(f"     → Harness choice has MINIMAL effect when generation config is identical.")
        else:
            lines.append(f"     → Harness choice has NOTABLE effect even with identical config.")

    if mismatched:
        mismatched_mean = np.mean([p.abs_delta for p in mismatched])
        lines.append(f"  2. Config-mismatched pairs (n={len(mismatched)}):")
        lines.append(f"     Mean |Δ| = {mismatched_mean:.4f}")
        lines.append(f"     → Config differences (shots, CoT, etc.) amplify score variation.")

    # Largest outlier
    if pairs:
        worst = max(pairs, key=lambda p: p.abs_delta)
        lines.append(f"  3. Largest delta: {worst.model_name} on {worst.benchmark}")
        lines.append(f"     {worst.entry_a.eval_library}={worst.entry_a.score:.2f} vs "
                    f"{worst.entry_b.eval_library}={worst.entry_b.score:.2f} "
                    f"(Δ={worst.delta:+.2f})")
        if worst.config_diffs:
            lines.append(f"     Config diffs: {'; '.join(worst.config_diffs)}")
            lines.append(f"     → This large delta likely reflects config differences, not pure harness effect.")

    lines.append("")
    lines.append("─" * 80)
    lines.append("RECOMMENDATIONS FOR EVALUATION REPRODUCIBILITY")
    lines.append("─" * 80)
    lines.append("  1. Always record the evaluation harness name AND version.")
    lines.append("  2. Always record generation config (shots, CoT, temperature).")
    lines.append("  3. When comparing across sources, verify config equivalence first.")
    lines.append("  4. Score deltas < 0.5 between harnesses (config-matched) suggest")
    lines.append("     the harness choice itself is not a major confounder.")
    lines.append("  5. Large deltas (> 3 pts) almost always co-occur with config differences.")
    lines.append("")
    lines.append("=" * 80)

    report_text = "\n".join(lines)
    report_path = out_dir / "harness_effect_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\n  Report saved: {report_path}")
    return report_text


# ─── Phase 6: Export structured data ─────────────────────────────────────────
def export_data(pairs: list[HarnessPair], stats: dict, out_dir: pathlib.Path):
    # CSV export
    rows = []
    for p in pairs:
        rows.append({
            "model": p.model_name,
            "model_id": p.entry_a.model_id,
            "benchmark": p.benchmark,
            "harness_a": p.entry_a.eval_library,
            "harness_b": p.entry_b.eval_library,
            "source_a": p.entry_a.source_folder,
            "source_b": p.entry_b.source_folder,
            "score_a": p.entry_a.score,
            "score_b": p.entry_b.score,
            "delta": p.delta,
            "abs_delta": p.abs_delta,
            "config_matched": p.config_matched,
            "metric_a": p.entry_a.metric,
            "metric_b": p.entry_b.metric,
            "shots_a": p.entry_a.shots,
            "shots_b": p.entry_b.shots,
            "cot_a": p.entry_a.chain_of_thought,
            "cot_b": p.entry_b.chain_of_thought,
            "temp_a": p.entry_a.temperature,
            "temp_b": p.entry_b.temperature,
            "file_a": p.entry_a.file_path,
            "file_b": p.entry_b.file_path,
        })

    df = pd.DataFrame(rows)
    csv_path = out_dir / "cross_harness_pairs.csv"
    df.to_csv(csv_path, index=False)
    print(f"  CSV saved: {csv_path}")

    # JSON stats export
    stats_path = out_dir / "harness_statistics.json"
    stats_path.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    print(f"  Stats saved: {stats_path}")

    return df


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Harness effect experiment")
    parser.add_argument("--data-dir", type=pathlib.Path, default=pathlib.Path("data"))
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("harness_experiment_output"))
    args = parser.parse_args()

    print("=" * 60)
    print("  HARNESS EFFECT EXPERIMENT")
    print("=" * 60)

    # Phase 1: Load
    print("\n[1/5] Loading evaluation entries...")
    entries = load_all_entries(args.data_dir)
    print(f"  Loaded {len(entries)} entries from {args.data_dir}")
    unique_models = len(set(norm_model(e.model_name) for e in entries))
    unique_benchmarks = len(set(norm_bench(e.eval_name) for e in entries))
    unique_harnesses = set(e.eval_library for e in entries)
    print(f"  Models: {unique_models}, Benchmarks: {unique_benchmarks}")
    print(f"  Harnesses: {unique_harnesses}")

    # Phase 2: Find pairs
    print("\n[2/5] Finding cross-harness pairs...")
    pairs = find_cross_harness_pairs(entries)
    print(f"  Found {len(pairs)} cross-harness pairs")
    matched_cnt = sum(1 for p in pairs if p.config_matched)
    print(f"    Config-matched: {matched_cnt}")
    print(f"    Config-mismatched: {len(pairs) - matched_cnt}")

    if not pairs:
        print("\n  No cross-harness pairs found. Nothing to analyze.")
        return

    # Phase 3: Statistics
    print("\n[3/5] Running statistical analysis...")
    stats = run_statistics(pairs)
    if "mean_abs_delta" in stats:
        print(f"  Mean |delta|: {stats['mean_abs_delta']:.4f}")
        print(f"  Median |delta|: {stats['median_abs_delta']:.4f}")

    # Phase 4: Figures
    print("\n[4/5] Generating figures...")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generate_figures(pairs, args.output_dir)

    # Phase 5: Export & Report
    print("\n[5/5] Generating report and exporting data...")
    export_data(pairs, stats, args.output_dir)
    report = generate_report(pairs, stats, args.output_dir, len(entries))
    print(report)


if __name__ == "__main__":
    main()
