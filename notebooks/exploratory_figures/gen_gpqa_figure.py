"""Generate GPQA configuration sensitivity figure for the paper."""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 10, 'font.family': 'serif'})

RESULTS = Path(__file__).resolve().parent.parent.parent / 'experiments' / 'controlled_eval' / 'gpqa_results.jsonl'
OUT = Path(__file__).resolve().parent.parent.parent / 'submission' / 'latex' / 'figures' / 'fig_gpqa_sensitivity.png'

# Load and deduplicate
data = []
seen = set()
with open(RESULTS) as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        key = (r['model_id'], r['prompt_format'], r['n_shot'])
        if key not in seen:
            seen.add(key)
            data.append(r)

# Organize by model
models = ['Mistral-7B-Instruct-v0.3', 'Qwen2.5-7B-Instruct',
          'Llama-3.1-8B-Instruct', 'Qwen2.5-14B-Instruct']
short_names = ['Mistral-7B', 'Qwen2.5-7B', 'Llama-3.1-8B', 'Qwen2.5-14B']
formats = ['plain', 'instruct', 'cot']
format_colors = {'plain': '#2196F3', 'instruct': '#FF9800', 'cot': '#4CAF50'}
nshots = [0, 5]

by_model = defaultdict(list)
for r in data:
    by_model[r['model_id'].split('/')[-1]].append(r)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

for ax_idx, ns in enumerate(nshots):
    ax = axes[ax_idx]
    x_positions = range(len(models))
    width = 0.25

    for i, fmt in enumerate(formats):
        scores = []
        for model in models:
            s = [r['score'] for r in by_model[model]
                 if r['n_shot'] == ns and r['prompt_format'] == fmt]
            scores.append(s[0] if s else 0)
        offset = (i - 1) * width
        bars = ax.bar([x + offset for x in x_positions], scores, width,
                      label=fmt, color=format_colors[fmt], alpha=0.85,
                      edgecolor='white', linewidth=0.5)
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{score:.0f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(short_names, rotation=15, ha='right', fontsize=9)
    ax.set_title(f'{ns}-shot GPQA-Main', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy (%)' if ax_idx == 0 else '')
    ax.set_ylim(0, 50)
    ax.axhline(y=25, color='gray', linestyle='--', alpha=0.5, label='Random (25%)')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('GPQA Configuration Sensitivity: Prompt Format × N-shot\n'
             '(Log-likelihood scoring, temperature invariant)',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'Saved: {OUT}')
