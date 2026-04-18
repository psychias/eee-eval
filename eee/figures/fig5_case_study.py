"""
fig5_case_study.py — three case-study collision boxes.

Shows 3 collision pairs as annotated FancyBboxPatch boxes illustrating how
the same (model, benchmark) yields different scores depending on methodology:
  - blue boxes = Source A
  - orange boxes = Source B
  - grey boxes summarise the mechanism (n-shot diff, harness diff, etc.)

Output: figures/fig5_case_study.pdf  +  submission/fig5_case_study.png
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

OUT_DIR = _ROOT / "figures"
OUT_DIR.mkdir(exist_ok=True)
SUB_DIR = _ROOT / "submission"
SUB_DIR.mkdir(exist_ok=True)


def draw_case(ax, x, y, w, h,
              title: str,
              source_a: str, score_a: float, nshot_a, harness_a: str,
              source_b: str, score_b: float, nshot_b, harness_b: str,
              delta: float,
              mechanism: str,
              color_a: str = "#4C72B0", color_b: str = "#DD8452"):
    """Draw one collision-pair case with source boxes + grey mechanism box."""
    box_style = "round,pad=0.04"

    # Main bounding box
    main_box = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.02",
                              linewidth=2, edgecolor="#333333",
                              facecolor="#F8F9FA", zorder=1)
    ax.add_patch(main_box)

    # Title
    ax.text(x + w / 2, y + h - 0.025, title,
            ha="center", va="top", fontsize=9, fontweight="bold",
            color="#222222")

    # Layout: two source boxes side by side + mechanism box below
    mid_y = y + h * 0.55
    box_w = w * 0.38
    box_h = h * 0.38
    gap = w * 0.06
    pad = 0.03

    # Source A box (blue)
    ax_a = FancyBboxPatch((x + pad, mid_y - box_h / 2), box_w, box_h,
                           facecolor=color_a, alpha=0.18,
                           edgecolor=color_a, linewidth=1.5,
                           boxstyle=box_style, zorder=2)
    ax.add_patch(ax_a)
    ax.text(x + pad + box_w / 2, mid_y + box_h * 0.28,
            source_a.replace("papers_", "arXiv:"),
            ha="center", va="center", fontsize=7.5,
            color=color_a, fontweight="bold")
    ax.text(x + pad + box_w / 2, mid_y,
            f"score = {score_a:.3f}", ha="center", va="center", fontsize=8.5)
    nshot_str_a = f"{nshot_a}-shot" if nshot_a is not None else "n/a"
    ax.text(x + pad + box_w / 2, mid_y - box_h * 0.28,
            f"{nshot_str_a} | {harness_a}", ha="center", va="center",
            fontsize=6.5, color="#555555", style="italic")

    # Source B box (orange)
    bx = x + pad + box_w + gap
    ax_b = FancyBboxPatch((bx, mid_y - box_h / 2), box_w, box_h,
                           facecolor=color_b, alpha=0.18,
                           edgecolor=color_b, linewidth=1.5,
                           boxstyle=box_style, zorder=2)
    ax.add_patch(ax_b)
    ax.text(bx + box_w / 2, mid_y + box_h * 0.28,
            source_b.replace("papers_", "arXiv:"),
            ha="center", va="center", fontsize=7.5,
            color=color_b, fontweight="bold")
    ax.text(bx + box_w / 2, mid_y,
            f"score = {score_b:.3f}", ha="center", va="center", fontsize=8.5)
    nshot_str_b = f"{nshot_b}-shot" if nshot_b is not None else "n/a"
    ax.text(bx + box_w / 2, mid_y - box_h * 0.28,
            f"{nshot_str_b} | {harness_b}", ha="center", va="center",
            fontsize=6.5, color="#555555", style="italic")

    # Grey mechanism summary box (right side)
    mech_x = bx + box_w + gap * 0.5
    mech_w = w - (mech_x - x) - pad
    mech_h = box_h * 0.7
    mech_y = mid_y - mech_h / 2
    mech_box = FancyBboxPatch(
        (mech_x, mech_y), mech_w, mech_h,
        boxstyle="round,pad=0.02",
        linewidth=1.2, edgecolor="#888888",
        facecolor="#E8E8E8", alpha=0.9, zorder=2)
    ax.add_patch(mech_box)
    ax.text(mech_x + mech_w / 2, mech_y + mech_h / 2,
            mechanism, ha="center", va="center",
            fontsize=6.5, color="#444444", fontweight="bold",
            wrap=True,
            bbox=dict(boxstyle="round,pad=0.1", facecolor="none",
                      edgecolor="none"))

    # Delta annotation
    delta_color = "#C44E52" if abs(delta) > 0.01 else "#2CA02C"
    ax.text(x + w / 2, y + 0.02,
            f"Δ = {delta:+.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
            color=delta_color,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=delta_color, linewidth=1.2))


def main():
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.suptitle(
        "Score Collision Case Studies: Same (Model, Benchmark) — Different Methodology",
        fontsize=11, fontweight="bold", y=0.97
    )

    # Case 1: GSM8K / Mistral-7B-v0.1 — n-shot difference
    draw_case(
        ax, x=0.02, y=0.66, w=0.96, h=0.28,
        title="Case 1 — GSM8K / Mistral-7B-v0.1   (Δ = −0.225)",
        source_a="papers_2401.04088", score_a=0.352, nshot_a=5,
        harness_a="lm-eval-harness",
        source_b="papers_2403.08295", score_b=0.577, nshot_b=11,
        harness_b="lm-eval-harness",
        delta=-0.225,
        mechanism="n-shot differs (5 vs 11)\ngeneration_config.\nadditional_details.n_shot\nchain-of-thought variant",
    )

    # Case 2: HumanEval / Llama-2-7B — different harness, score matches
    draw_case(
        ax, x=0.02, y=0.35, w=0.96, h=0.28,
        title="Case 2 — HumanEval / Llama-2-7B   (Δ = 0.000)",
        source_a="papers_2307.09288", score_a=0.122, nshot_a=0,
        harness_a="meta-internal",
        source_b="papers_2310.06825", score_b=0.122, nshot_b=0,
        harness_b="lm-eval-harness",
        delta=0.000,
        mechanism="eval_library.name differs\n(meta-internal vs\nlm-eval-harness)\nbut Δ = 0",
        color_b="#55A868",
    )

    # Case 3: HellaSwag / Mistral-7B-v0.1 — undocumented normalisation
    draw_case(
        ax, x=0.02, y=0.04, w=0.96, h=0.28,
        title="Case 3 — HellaSwag / Mistral-7B-v0.1   (Δ = −0.021)",
        source_a="papers_2310.06825", score_a=0.812, nshot_a=10,
        harness_a="lm-eval-harness",
        source_b="papers_2401.04088", score_b=0.833, nshot_b=10,
        harness_b="lm-eval-harness",
        delta=-0.021,
        mechanism="same eval_library &\nn-shot but Δ ≠ 0\n→ undocumented field in\ngeneration_config missing",
    )

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4C72B0", alpha=0.3, edgecolor="#4C72B0",
              label="Source A"),
        Patch(facecolor="#DD8452", alpha=0.3, edgecolor="#DD8452",
              label="Source B"),
        Patch(facecolor="#E8E8E8", edgecolor="#888888", label="Mechanism"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower left",
              bbox_to_anchor=(0.01, -0.02), ncol=3, framealpha=0.9)

    out_pdf = OUT_DIR / "fig5_case_study.pdf"
    out_png = SUB_DIR / "fig5_case_study.png"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=150)
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved → {out_pdf}")
    print(f"Saved → {out_png}")


if __name__ == "__main__":
    main()
