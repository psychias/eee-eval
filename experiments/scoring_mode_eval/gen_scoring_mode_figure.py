#!/usr/bin/env python3
"""
Generate scoring-mode comparison figure for the paper.

Reads scoring_mode_results.jsonl and produces a grouped bar chart
showing log-likelihood vs generation accuracy per model.

Usage:
    python gen_scoring_mode_figure.py [--results PATH] [--output PATH]

Defaults:
    --results experiments/scoring_mode_eval/scoring_mode_results.jsonl
             (or from Google Drive path if available)
    --output  submission/latex/figures/fig_scoring_mode
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_results(path):
    records = []
    for line in Path(path).read_text().splitlines():
        try:
            r = json.loads(line)
            if r.get('status') == 'ok':
                records.append(r)
        except Exception:
            pass
    return records


def make_figure(records, output_base):
    # Organize data
    models = []
    ll_scores = {}
    gen_scores = {}
    for r in records:
        short = r['model_id'].split('/')[-1]
        if short not in ll_scores and short not in gen_scores:
            models.append(short)
        if r['scoring_mode'] == 'log_likelihood':
            ll_scores[short] = r['score']
        elif r['scoring_mode'] == 'generation':
            gen_scores[short] = r['score']

    # Sort by model name for consistency
    models = sorted(set(models))

    ll_vals = [ll_scores.get(m, 0) for m in models]
    gen_vals = [gen_scores.get(m, 0) for m in models]
    deltas = [gen_scores.get(m, 0) - ll_scores.get(m, 0) for m in models]

    # Shorten model names for display
    display_names = []
    for m in models:
        if 'Mistral' in m:
            display_names.append('Mistral-7B')
        elif 'Qwen2.5-7B' in m:
            display_names.append('Qwen2.5-7B')
        elif 'Qwen2.5-14B' in m:
            display_names.append('Qwen2.5-14B')
        elif 'Llama' in m:
            display_names.append('Llama-3.1-8B')
        else:
            display_names.append(m[:15])

    # Figure
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(models))
    width = 0.35

    bars_ll = ax.bar(x - width/2, ll_vals, width, label='Log-likelihood',
                     color='#4C72B0', edgecolor='white', linewidth=0.5)
    bars_gen = ax.bar(x + width/2, gen_vals, width, label='Generation',
                      color='#DD8452', edgecolor='white', linewidth=0.5)

    # Delta annotations
    for i, (ll, gen, d) in enumerate(zip(ll_vals, gen_vals, deltas)):
        y_top = max(ll, gen) + 1.5
        ax.annotate(f'{d:+.1f} pp', xy=(x[i], y_top),
                    ha='center', va='bottom', fontsize=9,
                    fontweight='bold',
                    color='#C44E52' if abs(d) >= 5 else '#555555')

    ax.set_ylabel('MMLU Accuracy (%)', fontsize=11)
    ax.set_title('Scoring Mode Effect: Log-Likelihood vs Generation\n'
                 '(MMLU, 5-shot, temp=0.0, seed=42)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, max(max(ll_vals), max(gen_vals)) + 10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save
    for ext in ('pdf', 'png'):
        outpath = f'{output_base}.{ext}'
        fig.savefig(outpath, dpi=300, bbox_inches='tight')
        print(f'Saved: {outpath}')
    plt.close(fig)

    # Print summary
    print('\n=== Scoring-Mode Summary ===')
    for m, ll, gen, d in zip(display_names, ll_vals, gen_vals, deltas):
        print(f'  {m:20s}  LL={ll:5.1f}  Gen={gen:5.1f}  delta={d:+.1f} pp')
    mean_abs = np.mean(np.abs(deltas))
    max_abs = np.max(np.abs(deltas))
    print(f'\n  Mean |delta|: {mean_abs:.1f} pp')
    print(f'  Max  |delta|: {max_abs:.1f} pp')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    # Find results file
    candidates = [
        args.results,
        'experiments/scoring_mode_eval/scoring_mode_results.jsonl',
        '/content/drive/MyDrive/eee_eval_results_scoring_mode/scoring_mode_results.jsonl',
    ]
    results_path = None
    for c in candidates:
        if c and Path(c).exists():
            results_path = c
            break

    if results_path is None:
        print('ERROR: No scoring_mode_results.jsonl found.', file=sys.stderr)
        print('Run the scoring_mode_eval notebook first.', file=sys.stderr)
        sys.exit(1)

    output_base = args.output or 'submission/latex/figures/fig_scoring_mode'
    Path(output_base).parent.mkdir(parents=True, exist_ok=True)

    records = load_results(results_path)
    if not records:
        print('ERROR: No OK records in results file.', file=sys.stderr)
        sys.exit(1)

    print(f'Loaded {len(records)} OK records from {results_path}')
    make_figure(records, output_base)


if __name__ == '__main__':
    main()
