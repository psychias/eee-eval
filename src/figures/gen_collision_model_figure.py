"""Generate collision model figure: Tulu-3-8B across 4 benchmarks.

Grouped bar chart showing scores by format and n-shot for each benchmark.
Output: submission/latex/figures/fig_collision_model.png
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "experiments" / "controlled_eval" / "collision_model_results.jsonl"
OUT = ROOT / "submission" / "latex" / "figures" / "fig_collision_model.png"

rows = [json.loads(l) for l in open(DATA) if l.strip()]

BENCHMARKS = ["gsm8k", "bbh", "mmlu", "gpqa_main"]
BENCH_LABELS = ["GSM8K", "BBH", "MMLU", "GPQA"]
FORMATS = ["plain", "instruct", "cot"]
FORMAT_COLORS = {"plain": "#2196F3", "instruct": "#FF9800", "cot": "#4CAF50"}

fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharey=False)

for ax, bench, blabel in zip(axes, BENCHMARKS, BENCH_LABELS):
    bench_rows = [r for r in rows if r["benchmark"] == bench]
    nshots = sorted(set(r["n_shot"] for r in bench_rows))

    x = np.arange(len(nshots))
    width = 0.25

    for j, fmt in enumerate(FORMATS):
        scores = []
        for ns in nshots:
            match = [r for r in bench_rows if r["prompt_format"] == fmt and r["n_shot"] == ns]
            scores.append(match[0]["score"] if match else 0)
        bars = ax.bar(x + j * width, scores, width, label=fmt,
                       color=FORMAT_COLORS[fmt], edgecolor="white", linewidth=0.5)
        for bar, s in zip(bars, scores):
            if s > 3:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                        f"{s:.0f}", ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{ns}-shot" for ns in nshots])
    ax.set_title(blabel, fontsize=11, fontweight="bold")
    ax.set_ylabel("Score (%)" if bench == "gsm8k" else "")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].legend(fontsize=8, loc="upper left", framealpha=0.9)

fig.suptitle("Tulu-3-8B: Configuration Sensitivity Across Benchmarks",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved -> {OUT}")
