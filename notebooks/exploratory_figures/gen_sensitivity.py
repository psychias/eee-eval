"""
Generate fig_sensitivity.pdf — Cinelli-Hazlett sensitivity contour for
the EMNLP paper §5.7 Controlled Configuration Sensitivity.

Shows partial R² of temperature, prompt_format, and n_shot (5-shot GSM8K
analysis) against the overturn bound ρ² = 0.723 that would flip the
observational ρ = -0.850 finding.

Inputs: hardcoded from the analysis of controlled_eval_results.jsonl
        (3 models × 54 runs, Mistral-7B + Qwen2.5-7B + Llama-3.1-8B)
Output: figures/fig_sensitivity.pdf and .png
"""
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# ── Inputs from Cell 8 of controlled_eval_vllm_v3.ipynb ────────────
# Primary specification: 5-shot GSM8K, n=54 records, 3 models
PARTIAL_R2 = {
    'temperature':   0.030,   # \prtwotempfive
    'prompt_format': 0.285,   # \prtwofmtfive
}
# Full-data n_shot value is shown separately (it's a 0-shot
# extraction artefact; see paper §5.7 "Caveat" paragraph)
NSHOT_FULL = 0.873

# Overturn bound: ρ² where ρ = -0.850 from Table 6 observational finding
SPEARMAN_RHO = 0.850
OVERTURN = SPEARMAN_RHO ** 2   # = 0.7225

# Output location
FIGURES_DIR = ROOT / 'submission' / 'latex' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Figure ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.2, 5.0))

# Contour of the overturn bound on the R²_{D~Z|X} × R²_{Y~Z|X} plane.
# In Cinelli-Hazlett, the bound is reached when the product of the two
# partial R² equals ρ²; we draw the contour √(R²_D * R²_Y) = √ρ²
# so points with partial R² ≥ √ρ² on both axes would overturn.
rx = np.linspace(0.001, 1.0, 300)
RX, RY = np.meshgrid(rx, rx)
ov_level = math.sqrt(OVERTURN)
cs = ax.contour(RX, RY, np.sqrt(RX * RY), levels=[ov_level],
                colors=['#CC3311'], linestyles=['--'], linewidths=1.8)
ax.clabel(cs, fmt={ov_level: f'overturn: $\\rho^2 = {OVERTURN:.3f}$'},
          fontsize=8, inline_spacing=6)

# Shade the "would overturn" region (upper-right of the contour)
ax.contourf(RX, RY, np.sqrt(RX * RY), levels=[ov_level, 1.5],
            colors=['#CC3311'], alpha=0.08)

# Plot each measured axis as a point at (partial_R², partial_R²) —
# the diagonal is the conservative case where the confounder is
# equally associated with treatment and outcome.
COLORS = {
    'temperature':   '#0173B2',
    'prompt_format': '#DE8F05',
    'n_shot':        '#029E73',
}
MARKERS = {
    'temperature':   'o',
    'prompt_format': 's',
    'n_shot':        '^',
}
LABELS = {
    'temperature':   f'temperature ($R^2={PARTIAL_R2["temperature"]:.3f}$)',
    'prompt_format': f'prompt format ($R^2={PARTIAL_R2["prompt_format"]:.3f}$)',
    'n_shot':        f'n-shot, full data ($R^2={NSHOT_FULL:.3f}$)',
}

for axis, pr2 in PARTIAL_R2.items():
    ax.scatter([pr2], [pr2], color=COLORS[axis], marker=MARKERS[axis],
               s=110, zorder=5, edgecolor='white', linewidth=1.2,
               label=LABELS[axis])

# n_shot goes above the bound — annotate it specifically to flag the
# caveat discussed in paper §5.7
ax.scatter([NSHOT_FULL], [NSHOT_FULL], color=COLORS['n_shot'],
           marker=MARKERS['n_shot'], s=110, zorder=5,
           edgecolor='white', linewidth=1.2, label=LABELS['n_shot'])
ax.annotate('threshold\nartefact\n(see §5.7)',
            xy=(NSHOT_FULL, NSHOT_FULL),
            xytext=(NSHOT_FULL - 0.22, NSHOT_FULL - 0.18),
            fontsize=8, ha='left',
            arrowprops=dict(arrowstyle='-', color='#666666', lw=0.8))

# Reference guide lines at zero
ax.axhline(0, color='#cccccc', lw=0.5, zorder=1)
ax.axvline(0, color='#cccccc', lw=0.5, zorder=1)

ax.set_xlabel(r'Partial $R^2$: confounder $\to$ treatment  ($R^2_{D \sim Z | X}$)',
              fontsize=10)
ax.set_ylabel(r'Partial $R^2$: confounder $\to$ outcome  ($R^2_{Y \sim Z | X}$)',
              fontsize=10)
ax.set_title('Cinelli-Hazlett sensitivity for $\\rho = -0.850$\n'
             '(5-shot GSM8K, 3 models, $n = 54$ runs)',
             fontsize=11)
ax.set_xlim(0, 1.0)
ax.set_ylim(0, 1.0)
ax.set_aspect('equal')
ax.legend(loc='lower right', fontsize=8.5, frameon=True,
          framealpha=0.95, edgecolor='#cccccc')
ax.grid(True, alpha=0.3, linewidth=0.4)

plt.tight_layout()
for ext in ('pdf', 'png'):
    out = FIGURES_DIR / f'fig_sensitivity.{ext}'
    fig.savefig(out, bbox_inches='tight', dpi=300 if ext == 'png' else None)
    print(f'Saved: {out}')
plt.close(fig)
