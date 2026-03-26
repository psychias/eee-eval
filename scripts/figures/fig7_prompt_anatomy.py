#!/usr/bin/env python3
"""
fig7_prompt_anatomy.py — Unified four-column prompt anatomy figure.

One figure* with four columns (Cases 1-4), each showing:
  - Source A box (blue header)  — structural prompt excerpt
  - Source B box (orange header) — structural prompt excerpt
  - Delta arrow + mechanism label
  - Score annotations

Cases:
  1. GSM8K   — n-shot + CoT  (Δ = −0.225)
  2. HumanEval — positive control, identical methodology (Δ = 0.000)
  3. HellaSwag — scoring paradigm  (Δ = −0.021)
  4. MMLU    — prompt template + answer format  (Δ = −0.018)

Uses [...] ellipsis and **bold** to highlight structural differences
(CoT trigger, n-shot count, answer format) rather than full prompt text.

Output: figures/fig7_prompt_anatomy.pdf  +  submission/fig7_prompt_anatomy.png
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

OUT_DIR = _ROOT / "figures"
OUT_DIR.mkdir(exist_ok=True)
SUB_DIR = _ROOT / "submission"
SUB_DIR.mkdir(exist_ok=True)

# ── Colours ──────────────────────────────────────────────────────────
COL_A      = "#4C72B0"    # blue  = Source A
COL_B      = "#DD8452"    # orange = Source B
BG_CODE    = "#F7F7F3"    # light code background
COL_TEXT   = "#2D2D2D"    # code text colour
COL_BOLD   = "#C0392B"    # red for structural highlights

# ── Structural prompt excerpts (abbreviated) ─────────────────────────
# Each (text, is_highlight) pair.  is_highlight=True → bold red.

CASES = [
    # ── Case 1: GSM8K ────────────────────────────────────────────────
    {
        "title": "Case 1 — GSM8K",
        "model": "Mistral-7B-v0.1",
        "delta": -0.225,
        "score_a": 0.352,
        "score_b": 0.577,
        "label_a": "Mixtral paper (2401.04088)",
        "sub_a":   "5-shot  ·  standard",
        "prompt_a": [
            ("Q: Natalia sold clips to 48 of her", False),
            ("   friends in April [...]", False),
            ("A: 72", False),
            ("", False),
            ("[exemplars 2–5 same format]", False),
            ("", False),
            ("Q: Janet's ducks lay 16 eggs [...]", False),
            ("A:", True),
        ],
        "label_b": "Gemma paper (2403.08295)",
        "sub_b":   "11-shot  ·  chain-of-thought",
        "prompt_b": [
            ("Q: Natalia sold clips [...]", False),
            ("A: Let's think step by step.", True),
            ("   Natalia sold 48/2 = 24 in May.", True),
            ("   48 + 24 = 72. The answer is 72.", True),
            ("", False),
            ("[exemplars 2–11 with CoT]", True),
            ("", False),
            ("Q: Janet's ducks lay 16 [...]", False),
            ("A: Let's think step by step.", True),
        ],
        "mechanism": "n-shot 5→11 + CoT scaffold\n→  +22.5 pp",
        "mech_color": "#D55E00",
    },
    # ── Case 2: HumanEval (positive control) ─────────────────────────
    {
        "title": "Case 2 — HumanEval",
        "model": "Llama-2-7B",
        "delta": 0.000,
        "score_a": 0.122,
        "score_b": 0.122,
        "label_a": "Llama-2 paper (2307.09288)",
        "sub_a":   "0-shot  ·  pass@1  ·  lm-eval",
        "prompt_a": [
            ("def has_close_elements(", False),
            ("    numbers: List[float],", False),
            ("    threshold: float", False),
            (") -> bool:", False),
            ('    """Check if any two nums', False),
            ("    are closer than threshold.", False),
            ("    >>> has_close_elements(", False),
            ("    ...   [1.0, 2.0, 3.9], 0.3)", False),
            ("    True", False),
            ('    """', False),
        ],
        "label_b": "Mistral paper (2310.06825)",
        "sub_b":   "0-shot  ·  pass@1  ·  lm-eval",
        "prompt_b": [
            ("def has_close_elements(", False),
            ("    numbers: List[float],", False),
            ("    threshold: float", False),
            (") -> bool:", False),
            ('    """Check if any two nums', False),
            ("    are closer than threshold.", False),
            ("    >>> has_close_elements(", False),
            ("    ...   [1.0, 2.0, 3.9], 0.3)", False),
            ("    True", False),
            ('    """', False),
        ],
        "mechanism": "Identical methodology\n→  Δ = 0.000 (positive control)",
        "mech_color": "#56B4E9",
    },
    # ── Case 3: HellaSwag ────────────────────────────────────────────
    {
        "title": "Case 3 — HellaSwag",
        "model": "Mistral-7B-v0.1",
        "delta": -0.021,
        "score_a": 0.812,
        "score_b": 0.833,
        "label_a": "Mistral paper (2310.06825)",
        "sub_a":   "10-shot  ·  MC selection",
        "prompt_a": [
            ("Context: A woman is outside", False),
            ("with a bucket and a dog [...]", False),
            ("", False),
            ("Which ending is best?", False),
            ("A) She puts a leash on …", False),
            ("B) She throws a ball …", False),
            ("C) She dumps the bucket …", False),
            ("D) She begins to water …", False),
            ("", False),
            ("Answer:", True),
        ],
        "label_b": "Mixtral paper (2401.04088)",
        "sub_b":   "10-shot  ·  continuation likelihood",
        "prompt_b": [
            ("Context: A woman is outside", False),
            ("with a bucket and a dog [...]", False),
            ("", False),
            ("Continuation: She throws a", True),
            ("ball for the dog.", True),
            ("", False),
            ("→ score = log P(cont|ctx)", True),
            ("", False),
            ("Each candidate scored", True),
            ("independently; highest wins.", True),
            ("Length-normalised = True", True),
        ],
        "mechanism": "MC selection vs. continuation\nlikelihood  →  +2.1 pp",
        "mech_color": "#0072B2",
    },
    # ── Case 4: MMLU ─────────────────────────────────────────────────
    {
        "title": "Case 4 — MMLU",
        "model": "Mistral-7B-v0.1",
        "delta": -0.018,
        "score_a": 0.601,
        "score_b": 0.619,
        "label_a": "Mistral paper (2310.06825)",
        "sub_a":   "5-shot  ·  lm-eval v1",
        "prompt_a": [
            ("The following are multiple", False),
            ("choice questions (with answers)", False),
            ("about abstract algebra.", False),
            ("", False),
            ("Q: Find the degree of [...]", False),
            ("(A) 1  (B) 2  (C) 3  (D) 4", True),
            ("A: (D)", True),
            ("", False),
            ("[4 more exemplars]", False),
            ("", False),
            ("Q: Find all c in Z_3 [...]", False),
            ("(A) 0  (B) 1  (C) 2  (D) 3", True),
            ("A:", True),
        ],
        "label_b": "Gemma paper (2403.08295)",
        "sub_b":   "5-shot  ·  lm-eval v2",
        "prompt_b": [
            ("Answer the following multiple", False),
            ("choice question about abstract", False),
            ("algebra.", False),
            ("", False),
            ("Question: Find the degree [...]", False),
            ("Choices:", True),
            ("  A. 1", True),
            ("  B. 2", True),
            ("  C. 3    D. 4", True),
            ("Answer: D", True),
            ("", False),
            ("[4 more exemplars]", False),
            ("", False),
            ("Question: Find all c [...]", False),
            ("Answer:", True),
        ],
        "mechanism": "Same harness & n-shot →\nprompt template + answer\ntoken format differ  →  +1.8 pp",
        "mech_color": "#009E73",
    },
]


def draw_prompt_box(ax, x, y, w, h, lines, title, subtitle,
                    border_color, fontsize=5.8):
    """Draw a prompt box with coloured title bar and structural text."""
    title_h = 0.055
    sub_h   = 0.030

    # Title bar
    tb = FancyBboxPatch(
        (x, y + h - title_h), w, title_h,
        boxstyle="round,pad=0.006",
        facecolor=border_color, edgecolor="none", alpha=0.88, zorder=3)
    ax.add_patch(tb)
    ax.text(x + w / 2, y + h - title_h / 2, title,
            ha="center", va="center", fontsize=6, fontweight="bold",
            color="white", zorder=4)

    # Subtitle bar (lighter)
    sb = FancyBboxPatch(
        (x, y + h - title_h - sub_h), w, sub_h,
        boxstyle="square,pad=0",
        facecolor=border_color, edgecolor="none", alpha=0.18, zorder=3)
    ax.add_patch(sb)
    ax.text(x + w / 2, y + h - title_h - sub_h / 2, subtitle,
            ha="center", va="center", fontsize=5.2, fontstyle="italic",
            color=border_color, zorder=4)

    # Code body
    body_top = y + h - title_h - sub_h
    body = FancyBboxPatch(
        (x, y), w, body_top - y,
        boxstyle="round,pad=0.006",
        facecolor=BG_CODE, edgecolor=border_color, linewidth=1.0, zorder=2)
    ax.add_patch(body)

    # Render lines
    n_content = sum(1 for t, _ in lines if t) + sum(0.5 for t, _ in lines if not t)
    line_h = (body_top - y - 0.018) / max(n_content, 1)
    line_h = min(line_h, 0.024)
    cur_y = body_top - 0.012
    for text, is_highlight in lines:
        if not text:
            cur_y -= line_h * 0.5
            continue
        fc = COL_BOLD if is_highlight else COL_TEXT
        fw = "bold" if is_highlight else "normal"
        ax.text(x + 0.008, cur_y, text,
                ha="left", va="top", fontsize=fontsize,
                fontfamily="monospace", color=fc, fontweight=fw,
                zorder=4)
        cur_y -= line_h


def main():
    fig_w, fig_h = 11.5, 5.4
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")

    # Column x-positions and widths (4 columns)
    gap = 0.012
    w1 = 0.235   # GSM8K
    w2 = 0.210   # HumanEval (simpler prompts)
    w3 = 0.235   # HellaSwag
    w4 = 0.270   # MMLU (widest, most subtle)
    total = w1 + w2 + w3 + w4 + 3 * gap
    offset = (1 - total) / 2
    col_specs = [
        (offset, w1),
        (offset + w1 + gap, w2),
        (offset + w1 + gap + w2 + gap, w3),
        (offset + w1 + gap + w2 + gap + w3 + gap, w4),
    ]

    fontsizes = [5.0, 5.0, 5.0, 5.0]

    for idx, (case, (cx, cw), fs) in enumerate(
            zip(CASES, col_specs, fontsizes)):
        # ── Column title ──
        ax.text(cx + cw / 2, 0.97, case["title"],
                ha="center", va="top", fontsize=9, fontweight="bold")
        ax.text(cx + cw / 2, 0.935, case["model"],
                ha="center", va="top", fontsize=7.5, fontstyle="italic",
                color="#555555")

        # ── Source A box (blue, top) ──
        box_h = 0.35
        y_a = 0.535
        draw_prompt_box(ax, cx, y_a, cw, box_h,
                        case["prompt_a"], case["label_a"], case["sub_a"],
                        COL_A, fontsize=fs)
        # Score label
        ax.text(cx + cw - 0.006, y_a + 0.004,
                f"score = {case['score_a']:.3f}",
                ha="right", va="bottom", fontsize=5.8, fontweight="bold",
                color=COL_A, zorder=5,
                bbox=dict(boxstyle="round,pad=0.10", fc="white",
                          ec=COL_A, lw=0.5, alpha=0.85))

        # ── Delta arrow ──
        arrow_y = 0.50
        ax.annotate("",
                    xy=(cx + cw / 2 + 0.04, arrow_y - 0.045),
                    xytext=(cx + cw / 2 + 0.04, arrow_y + 0.015),
                    arrowprops=dict(arrowstyle="-|>",
                                    color=case["mech_color"], lw=1.5),
                    zorder=5)
        ax.text(cx + cw / 2 - 0.02, arrow_y - 0.015,
                f"\u0394 = {case['delta']:+.3f}",
                ha="center", va="center", fontsize=7, fontweight="bold",
                color=case["mech_color"], zorder=5)

        # ── Source B box (orange, bottom) ──
        y_b = 0.10
        draw_prompt_box(ax, cx, y_b, cw, box_h,
                        case["prompt_b"], case["label_b"], case["sub_b"],
                        COL_B, fontsize=fs)
        # Score label
        ax.text(cx + cw - 0.006, y_b + 0.004,
                f"score = {case['score_b']:.3f}",
                ha="right", va="bottom", fontsize=5.8, fontweight="bold",
                color=COL_B, zorder=5,
                bbox=dict(boxstyle="round,pad=0.10", fc="white",
                          ec=COL_B, lw=0.5, alpha=0.85))

        # ── Mechanism label at bottom ──
        ax.text(cx + cw / 2, 0.025, case["mechanism"],
                ha="center", va="bottom", fontsize=5.8, fontstyle="italic",
                color=case["mech_color"], fontweight="bold",
                multialignment="center",
                bbox=dict(boxstyle="round,pad=0.16", fc="white",
                          ec=case["mech_color"], lw=0.8, alpha=0.85),
                zorder=5)

    # Column dividers
    for cx, cw in col_specs[:-1]:
        div_x = cx + cw + gap / 2
        ax.axvline(div_x, color="#CCCCCC", linewidth=0.8, linestyle=":",
                   ymin=0.02, ymax=0.95, zorder=1)

    # Legend
    ax.text(0.5, 0.005,
            "Bold red text = structural difference driving the score gap",
            ha="center", va="bottom", fontsize=6.5, color=COL_BOLD,
            fontstyle="italic", zorder=5)

    out_pdf = OUT_DIR / "fig7_prompt_anatomy.pdf"
    out_png = SUB_DIR / "fig7_prompt_anatomy.png"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=150)
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved \u2192 {out_pdf}")
    print(f"Saved \u2192 {out_png}")


if __name__ == "__main__":
    main()
