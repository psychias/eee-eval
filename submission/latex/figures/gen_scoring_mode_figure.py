"""Generate scoring-mode comparison figure for the paper."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

models = ['Mistral-7B', 'Qwen2.5-7B', 'Llama-3.1-8B']
ll_scores = [61.92, 74.26, 68.30]
gen_scores = [61.64, 74.43, 68.16]
deltas = [ll - gen for ll, gen in zip(ll_scores, gen_scores)]

x = np.arange(len(models))
width = 0.32

fig, ax = plt.subplots(figsize=(5.5, 3.2))
bars_ll = ax.bar(x - width/2, ll_scores, width, label='Log-likelihood',
                 color='#4878CF', edgecolor='white', linewidth=0.5)
bars_gen = ax.bar(x + width/2, gen_scores, width, label='Generation',
                  color='#D65F5F', edgecolor='white', linewidth=0.5)

# Annotate deltas
for i, (ll, gen) in enumerate(zip(ll_scores, gen_scores)):
    mid = max(ll, gen) + 0.8
    delta = ll - gen
    sign = '+' if delta >= 0 else ''  # '-' is automatic
    ax.annotate(f'$\\Delta$={sign}{delta:.2f}',
                xy=(x[i], mid), ha='center', va='bottom',
                fontsize=8, fontstyle='italic', color='#333333')

ax.set_ylabel('Accuracy (%)')
ax.set_title('Scoring-Mode Effect on MMLU (5-shot, T=0)')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=9)
ax.set_ylim(55, 82)
ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
ax.axhline(y=0, color='grey', linewidth=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

out = Path(__file__).parent
fig.savefig(out / 'fig_scoring_mode.pdf', bbox_inches='tight')
fig.savefig(out / 'fig_scoring_mode.png', bbox_inches='tight', dpi=200)
print('Saved fig_scoring_mode.pdf and .png')
