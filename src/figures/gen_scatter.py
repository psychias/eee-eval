import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
import os

pairs = [
    {"model": "Tulu-3-8B",    "bench": "GPQA",     "completeness": 2, "delta": 0.23},
    {"model": "Tulu-3-8B",    "bench": "BBH",      "completeness": 2, "delta": 0.19},
    {"model": "Tulu-3-8B",    "bench": "IFEval",   "completeness": 2, "delta": 0.12},
    {"model": "Tulu-3-8B",    "bench": "MMLU-Pro", "completeness": 2, "delta": 0.07},
    {"model": "Tulu-3-8B",    "bench": "MuSR",     "completeness": 2, "delta": 0.07},
    {"model": "SC2-15B",      "bench": "MBPP+",    "completeness": 1, "delta": 3.90},
    {"model": "SC2-15B",      "bench": "HumanE+",  "completeness": 1, "delta": 3.00},
    {"model": "Qwen2.5-72B",  "bench": "GPQA",     "completeness": 0, "delta": 32.33},
]

x = np.array([p["completeness"] for p in pairs])
y = np.array([p["delta"] for p in pairs])

rho, pval = spearmanr(x, y)
print(f"Spearman rho = {rho:.3f}, p = {pval:.3f}")

plt.style.use("ggplot")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

fig, ax = plt.subplots(figsize=(3.5, 3.2))

# All points
ax.scatter(x, y, s=60, c="#2166ac", edgecolors="white", linewidths=0.5, zorder=3)

# Outlier ring
qwen = pairs[7]
ax.scatter(qwen["completeness"], qwen["delta"], s=60, c="#2166ac",
           edgecolors="#d6604d", linewidths=1.5, zorder=4)

ax.set_yscale("log")
ax.set_xlim(-0.3, 2.7)
ax.set_xticks([0, 1, 2])
ax.set_xlabel("Metadata completeness (fields documented out of 5)", fontsize=8)
ax.set_ylabel(r"Cross-source score divergence |Δ| (pp, log$_{10}$ scale)", fontsize=8)
ax.tick_params(labelsize=7)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

# Labels — manual placement
try:
    from adjustText import adjust_text
    HAS_AT = True
except ImportError:
    HAS_AT = False

labels_data = []
for p in pairs:
    labels_data.append((p["completeness"], p["delta"], f"{p['model']} / {p['bench']}"))

if HAS_AT:
    texts = []
    for cx, cy, lab in labels_data:
        texts.append(ax.text(cx, cy, lab, fontsize=7, color="#444444"))
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="#999999", lw=0.5))
else:
    # Manual: Tulu points at x=2, sorted by delta desc, stacked left
    tulu = sorted([p for p in pairs if p["completeness"] == 2], key=lambda p: -p["delta"])
    # Place them stacked vertically from top
    y_positions = []
    for i, p in enumerate(tulu):
        log_y = np.log10(p["delta"])
        # stack with offset in log space
        nudged = log_y + 0.12 - i * 0.16
        y_positions.append(10**nudged)

    for i, p in enumerate(tulu):
        lab = f"{p['model']} / {p['bench']}"
        ax.annotate(lab, (p["completeness"], p["delta"]),
                    xytext=(p["completeness"] - 0.15, y_positions[i]),
                    fontsize=7, color="#444444", ha="right", va="center")

    # SC2 points at x=1
    for p in pairs:
        if p["completeness"] == 1:
            ax.annotate(f"{p['model']} / {p['bench']}", (p["completeness"], p["delta"]),
                        xytext=(p["completeness"] + 0.08, p["delta"]),
                        fontsize=7, color="#444444", ha="left", va="center")

    # Qwen at x=0
    ax.annotate(f"{qwen['model']} / {qwen['bench']}", (qwen["completeness"], qwen["delta"]),
                xytext=(qwen["completeness"] + 0.08, qwen["delta"]),
                fontsize=7, color="#444444", ha="left", va="center")

# Spearman annotation upper right
ax.text(0.97, 0.97, f"Spearman \u03c1 = {rho:.3f}\np = {pval:.3f}  (n = 8)",
        transform=ax.transAxes, fontsize=8, color="#333333",
        ha="right", va="top")

fig.tight_layout()
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
out_dir = os.path.join(ROOT, "submission", "latex", "figures")
os.makedirs(out_dir, exist_ok=True)
fig.savefig(os.path.join(out_dir, "fig_completeness_vs_delta.pdf"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(out_dir, "fig_completeness_vs_delta.png"), dpi=300, bbox_inches="tight")

for ext in ["pdf", "png"]:
    path = os.path.join(out_dir, f"fig_completeness_vs_delta.{ext}")
    sz = os.path.getsize(path)
    print(f"Saved {path} ({sz:,} bytes)")
