"""
Parser-flow diagram for §5.1 (Case Study 1) — CoT visual register.
Generates 3 variants with different score display styles.

Outputs (LLM_Evaluation_Report/figures/):
  case1_parser_flow_{S1,S2,S3}.pdf/png

Score display styles:
  S1 — no scores in figure body; scores appear in LaTeX caption only
  S2 — score badge (white box, coloured border) beside parser output
  S3 — solid coloured footer strip inside each score region

Shows: same Qwen2.5-14B-Instruct output → two parser branches → different scores.
Aggregate: Plain 5-shot = 58.5%, CoT 5-shot = 2.5%
           (lm-evaluation-harness 0.4.11, limit=200)
Prompts: real run-log data from
  experiments/controlled_eval/5-shot-GSM8K/Qwen2.5-14B-Instruct_gsm8k_5shot_{plain,cot}/
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, "..")

# ── Palette ───────────────────────────────────────────────────────────────────
BLUE_HDR  = "#4A90D9"
BLUE_FILL = "#E8F1FB"
ORNG_HDR  = "#E89A4F"
ORNG_FILL = "#FBEFE2"
HI_BG     = "#FFF3B0"
HI_RED    = "#FDECEA"
GREEN_DRK = "#2A7A3B"
RED       = "#C0392B"
DGREY     = "#444444"
MGREY     = "#CCCCCC"
WHITE     = "#FFFFFF"
BLACK     = "#111111"
MONO      = "DejaVu Sans Mono"
SANS      = "DejaVu Sans"

FW, FH = 12.5, 7.5

# Scores
PLAIN_SCORE = 58.5
COT_SCORE   = 2.5
DELTA       = COT_SCORE - PLAIN_SCORE   # −56.0


def _rbox(ax, xL, xR, yB, yT, fc, ec=MGREY, lw=0.8, zorder=1, r=0.012):
    patch = mpatches.FancyBboxPatch(
        (xL, yB), xR - xL, yT - yB,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        transform=ax.transAxes, zorder=zorder, clip_on=False,
    )
    ax.add_patch(patch)


def _pill(ax, x, y, label, color, fs=8.0):
    ax.text(
        x, y, label,
        ha="left", va="top",
        fontsize=fs, fontfamily=SANS, color=WHITE, fontweight="bold",
        transform=ax.transAxes, zorder=6,
        bbox=dict(facecolor=color, edgecolor="none",
                  boxstyle="round,pad=0.30", alpha=1.0),
    )


def _txt(ax, x, y, s, **kw):
    kw.setdefault("transform", ax.transAxes)
    kw.setdefault("zorder", 5)
    ax.text(x, y, s, **kw)


def _arrow(ax, x0, y0, x1, y1, color=DGREY, lw=1.6, style="->"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle=style, color=color, lw=lw),
                zorder=8)


def _score_badge(ax, xC, yC, score, color):
    """Score badge centred at (xC, yC). Used for S2."""
    label = f"{score:.1f}%"
    bw, bh = 0.090, 0.048
    bx, by = xC - bw / 2, yC - bh / 2
    _rbox(ax, bx, bx + bw, by, by + bh, fc=WHITE, ec=color, lw=1.8, zorder=7, r=0.008)
    ax.text(xC, yC, label,
            ha="center", va="center",
            fontsize=13, fontfamily=SANS, color=color, fontweight="bold",
            transform=ax.transAxes, zorder=8)


def _score_strip(ax, xL, xR, yB, yT, score, color):
    """Solid coloured strip at yB..yT. Used for S3."""
    label = f"{score:.1f}%"
    _rbox(ax, xL, xR, yB, yT, fc=color, ec=color, lw=0, zorder=5, r=0.008)
    ax.text((xL + xR) / 2, (yB + yT) / 2, label,
            ha="center", va="center",
            fontsize=13, fontfamily=SANS, color=WHITE, fontweight="bold",
            transform=ax.transAxes, zorder=6)


def generate_variant(score_style):
    fig = plt.figure(figsize=(FW, FH), facecolor=WHITE)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── Title ─────────────────────────────────────────────────────────────────
    _txt(ax, 0.5, 0.967,
         "Case Study 1 · GSM8K: Answer-Parser Failure from Prompt Format",
         ha="center", va="top", fontsize=12, fontfamily=SANS,
         fontweight="bold", color=BLACK)
    _txt(ax, 0.5, 0.942,
         "Qwen2.5-14B-Instruct · lm-evaluation-harness 0.4.11 · 5-shot · temperature 0.0",
         ha="center", va="top", fontsize=8.5, fontfamily=SANS, color=DGREY)

    # ── LEFT: prompt boxes ────────────────────────────────────────────────────
    # Plain prompt (top, blue)
    _rbox(ax, 0.010, 0.370, 0.548, 0.915, fc=BLUE_FILL, ec=BLUE_HDR, lw=1.3)
    _pill(ax, 0.022, 0.908, "Plain few-shot  (direct-answer)", BLUE_HDR, fs=8.0)

    plain_lines = [
        "[sys] You are Qwen, created by Alibaba Cloud.",
        "Q: Jen and Tyler are gymnasts …",
        "A: … #### 12",
        "Q: Four people in a law firm …",
        "A: … #### 1",
        "     ...  [3 more few-shot examples]  ...",
        "Q: Janet's ducks lay 16 eggs/day.",
        "   She eats 3 for breakfast; bakes 4 into muffins.",
        "   She sells the rest at $2 each. How much?",
        "A: 16−3−4 = 9 eggs.  9 × $2 = $18.  #### 18",
    ]
    hi_plain = "A: 16−3−4 = 9 eggs.  9 × $2 = $18.  #### 18"
    yT_plain, yB_plain = 0.880, 0.555
    n = len(plain_lines)
    usable = (yT_plain - yB_plain) - 0.016
    lh = usable / n
    for i, line in enumerate(plain_lines):
        y     = yT_plain - 0.008 - (i + 0.5) * lh
        is_hi = line.strip() == hi_plain.strip()
        kw = {}
        if is_hi:
            kw["bbox"] = dict(facecolor=HI_BG, edgecolor="none",
                              boxstyle="square,pad=0.13", alpha=0.95)
        _txt(ax, 0.022, y, line,
             ha="left", va="center", fontsize=6.5, fontfamily=MONO,
             color=DGREY, fontweight="bold" if is_hi else "normal", **kw)

    # CoT prompt (bottom, orange)
    _rbox(ax, 0.010, 0.370, 0.075, 0.505, fc=ORNG_FILL, ec=ORNG_HDR, lw=1.3)
    _pill(ax, 0.022, 0.498, "Chain-of-thought few-shot", ORNG_HDR, fs=8.0)

    cot_lines = [
        "[sys] You are an expert problem solver.",
        "   Think step by step, then state the answer.",
        "Q: Jen and Tyler …  A: … #### 12",
        "     ...  [same 5 examples, answers end with ####]  ...",
        "Q: Janet's ducks lay 16 eggs/day. …",
        "A: Janet uses 3 + 4 = 7 eggs each day.",
        "   16 − 7 = 9 eggs remain.  9 × $2 = $18.",
        "   The answer is $18 per day.",
    ]
    hi_cot = "   The answer is $18 per day."
    yT_cot, yB_cot = 0.466, 0.082
    n2 = len(cot_lines)
    usable2 = (yT_cot - yB_cot) - 0.016
    lh2 = usable2 / n2
    for i, line in enumerate(cot_lines):
        y     = yT_cot - 0.008 - (i + 0.5) * lh2
        is_hi = line.strip() == hi_cot.strip()
        kw = {}
        if is_hi:
            kw["bbox"] = dict(facecolor=HI_RED, edgecolor="none",
                              boxstyle="square,pad=0.13", alpha=0.95)
        _txt(ax, 0.022, y, line,
             ha="left", va="center", fontsize=6.5, fontfamily=MONO,
             color=DGREY, fontweight="bold" if is_hi else "normal", **kw)

    # ── CENTRE: parser boxes ──────────────────────────────────────────────────
    _arrow(ax, 0.370, 0.730, 0.440, 0.730, color=BLUE_HDR, lw=2.0)
    _arrow(ax, 0.370, 0.290, 0.440, 0.290, color=ORNG_HDR, lw=2.0)

    # Parser 1 — plain (blue tint)
    _rbox(ax, 0.440, 0.720, 0.630, 0.825, fc=BLUE_FILL, ec=BLUE_HDR, lw=1.1)
    _txt(ax, 0.580, 0.818, "Parser: #### delimiter regex",
         ha="center", va="top", fontsize=8, fontfamily=SANS,
         fontweight="bold", color=BLUE_HDR)
    _txt(ax, 0.580, 0.792, r"re.search(r'####\s+(\S+)', output)",
         ha="center", va="top", fontsize=7, fontfamily=MONO, color=DGREY)
    _txt(ax, 0.580, 0.766,
         "Match found: '18'  →  exact_match = 1",
         ha="center", va="top", fontsize=7.5, fontfamily=SANS,
         color=GREEN_DRK, fontweight="bold",
         bbox=dict(facecolor=HI_BG, edgecolor="none",
                   boxstyle="square,pad=0.15", alpha=0.9))

    # Parser 2 — CoT (orange tint)
    _rbox(ax, 0.440, 0.720, 0.185, 0.385, fc=ORNG_FILL, ec=ORNG_HDR, lw=1.1)
    _txt(ax, 0.580, 0.378, "Parser: same #### delimiter regex",
         ha="center", va="top", fontsize=8, fontfamily=SANS,
         fontweight="bold", color=ORNG_HDR)
    _txt(ax, 0.580, 0.352, r"re.search(r'####\s+(\S+)', output)",
         ha="center", va="top", fontsize=7, fontfamily=MONO, color=DGREY)
    _txt(ax, 0.580, 0.326,
         "No match  →  filtered_resps = ['[invalid]']",
         ha="center", va="top", fontsize=7.5, fontfamily=SANS,
         color=RED, fontweight="bold",
         bbox=dict(facecolor=HI_RED, edgecolor="none",
                   boxstyle="square,pad=0.15", alpha=0.9))
    _txt(ax, 0.580, 0.298, "exact_match = 0",
         ha="center", va="top", fontsize=7.5, fontfamily=SANS,
         color=RED, fontweight="bold")

    # ── RIGHT: score display ──────────────────────────────────────────────────
    _arrow(ax, 0.720, 0.730, 0.800, 0.730, color=BLUE_HDR, lw=2.0)
    _arrow(ax, 0.720, 0.290, 0.800, 0.290, color=ORNG_HDR, lw=2.0)

    PLAIN_XL, PLAIN_XR = 0.800, 0.990
    PLAIN_YB, PLAIN_YT = 0.630, 0.825
    COT_XL,   COT_XR   = 0.800, 0.990
    COT_YB,   COT_YT   = 0.185, 0.385
    STRIP_H = 0.040

    if score_style == "S1":
        # Just labelled boxes, no score number inside figure
        _rbox(ax, PLAIN_XL, PLAIN_XR, PLAIN_YB, PLAIN_YT,
              fc=BLUE_FILL, ec=BLUE_HDR, lw=1.3)
        _txt(ax, (PLAIN_XL + PLAIN_XR) / 2, (PLAIN_YB + PLAIN_YT) / 2,
             "plain · 5-shot\n(score in caption)",
             ha="center", va="center", fontsize=8.5, fontfamily=SANS,
             color=BLUE_HDR, fontweight="bold")

        _rbox(ax, COT_XL, COT_XR, COT_YB, COT_YT,
              fc=ORNG_FILL, ec=ORNG_HDR, lw=1.3)
        _txt(ax, (COT_XL + COT_XR) / 2, (COT_YB + COT_YT) / 2,
             "CoT · 5-shot\n(score in caption)",
             ha="center", va="center", fontsize=8.5, fontfamily=SANS,
             color=ORNG_HDR, fontweight="bold")

    elif score_style == "S2":
        _rbox(ax, PLAIN_XL, PLAIN_XR, PLAIN_YB, PLAIN_YT,
              fc=BLUE_FILL, ec=BLUE_HDR, lw=1.3)
        _score_badge(ax, (PLAIN_XL + PLAIN_XR) / 2,
                     (PLAIN_YB + PLAIN_YT) / 2,
                     PLAIN_SCORE, BLUE_HDR)
        _txt(ax, (PLAIN_XL + PLAIN_XR) / 2,
             (PLAIN_YB + PLAIN_YT) / 2 - 0.055,
             "plain · 5-shot\nn=200, limit=200",
             ha="center", va="top", fontsize=7, fontfamily=SANS, color=DGREY)

        _rbox(ax, COT_XL, COT_XR, COT_YB, COT_YT,
              fc=ORNG_FILL, ec=ORNG_HDR, lw=1.3)
        _score_badge(ax, (COT_XL + COT_XR) / 2,
                     (COT_YB + COT_YT) / 2,
                     COT_SCORE, ORNG_HDR)
        _txt(ax, (COT_XL + COT_XR) / 2,
             (COT_YB + COT_YT) / 2 - 0.055,
             "CoT · 5-shot\nn=200, limit=200",
             ha="center", va="top", fontsize=7, fontfamily=SANS, color=DGREY)

    else:  # S3
        _rbox(ax, PLAIN_XL, PLAIN_XR, PLAIN_YB, PLAIN_YT,
              fc=BLUE_FILL, ec=BLUE_HDR, lw=1.3)
        _score_strip(ax, PLAIN_XL, PLAIN_XR, PLAIN_YB, PLAIN_YB + STRIP_H,
                     PLAIN_SCORE, BLUE_HDR)
        _txt(ax, (PLAIN_XL + PLAIN_XR) / 2,
             (PLAIN_YB + STRIP_H + PLAIN_YT) / 2,
             "plain · 5-shot\nn=200, limit=200",
             ha="center", va="center", fontsize=7.5, fontfamily=SANS,
             color=BLUE_HDR, fontweight="bold")

        _rbox(ax, COT_XL, COT_XR, COT_YB, COT_YT,
              fc=ORNG_FILL, ec=ORNG_HDR, lw=1.3)
        _score_strip(ax, COT_XL, COT_XR, COT_YB, COT_YB + STRIP_H,
                     COT_SCORE, ORNG_HDR)
        _txt(ax, (COT_XL + COT_XR) / 2,
             (COT_YB + STRIP_H + COT_YT) / 2,
             "CoT · 5-shot\nn=200, limit=200",
             ha="center", va="center", fontsize=7.5, fontfamily=SANS,
             color=ORNG_HDR, fontweight="bold")

    # ── Δ annotation ──────────────────────────────────────────────────────────
    mid_x = (PLAIN_XL + PLAIN_XR) / 2
    ax.annotate("",
                xy=(mid_x, COT_YT + 0.010),
                xytext=(mid_x, PLAIN_YB - 0.010),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.2),
                zorder=9)
    _txt(ax, mid_x + 0.012, (PLAIN_YB + COT_YT) / 2,
         "Δ = −56.0 pp\n(CoT − plain)",
         ha="left", va="center", fontsize=9, fontfamily=SANS,
         fontweight="bold", color=RED)

    # ── Section labels ────────────────────────────────────────────────────────
    _txt(ax, 0.190, 0.058, "Prompt format (input)",
         ha="center", va="top", fontsize=8, fontfamily=SANS,
         color=DGREY, style="italic")
    _txt(ax, 0.580, 0.058, "Answer parser (harness)",
         ha="center", va="top", fontsize=8, fontfamily=SANS,
         color=DGREY, style="italic")
    _txt(ax, 0.895, 0.058, "Aggregate accuracy",
         ha="center", va="top", fontsize=8, fontfamily=SANS,
         color=DGREY, style="italic")

    # ── Source note ───────────────────────────────────────────────────────────
    _txt(ax, 0.010, 0.020,
         "Prompts: real run-log data · experiments/controlled_eval/5-shot-GSM8K/"
         "Qwen2.5-14B-Instruct_gsm8k_5shot_{plain,cot}/samples_gsm8k_5shot.jsonl",
         ha="left", va="bottom", fontsize=5.5, fontfamily=MONO, color="#999999")

    # variant tag
    _txt(ax, 0.997, 0.005, f"[{score_style}]",
         ha="right", va="bottom", fontsize=6, fontfamily=MONO, color="#BBBBBB")

    stem     = f"case1_parser_flow_{score_style}"
    pdf_path = os.path.join(OUT_DIR, f"{stem}.pdf")
    png_path = os.path.join(OUT_DIR, f"{stem}.png")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.savefig(png_path, format="png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved: {pdf_path}")


def main():
    for style in ("S1", "S2", "S3"):
        generate_variant(style)


if __name__ == "__main__":
    main()
