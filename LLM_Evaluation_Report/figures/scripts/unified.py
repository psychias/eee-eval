"""
Unified three-column case-study comparison — CoT visual register.
Generates 6 variants: 2 middle columns × 3 score display styles.

Outputs (LLM_Evaluation_Report/figures/):
  case_studies_unified_{hellaswag,belebele}_{S1,S2,S3}.pdf/png

Score display styles:
  S1 — no scores in figure body; scores appear in LaTeX caption only
  S2 — score badge (white box, coloured border) at upper-right of each panel
  S3 — solid coloured footer strip at bottom of each panel

Sign convention: Δ = bottom_score − top_score
  GSM8K:     2.5  − 58.5  = −56.0 pp
  HellaSwag: 81.20 − 49.80 = +31.4 pp
  Belebele:  79.42 − 32.80 = +46.6 pp
  BBH:       68.73 − 34.10 = +34.6 pp
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, "..")

# ── Palette (CoT register, Wei et al. 2022) ──────────────────────────────────
BLUE_HDR  = "#4A90D9"
BLUE_FILL = "#E8F1FB"
ORNG_HDR  = "#E89A4F"
ORNG_FILL = "#FBEFE2"
HI_BG     = "#FFF3B0"
RED       = "#C0392B"
GREEN_DRK = "#2A6E3F"
DGREY     = "#444444"
MGREY     = "#CCCCCC"
WHITE     = "#FFFFFF"
BLACK     = "#111111"
MONO      = "DejaVu Sans Mono"
SANS      = "DejaVu Sans"

# ── Column data ───────────────────────────────────────────────────────────────
COLS = {
    "gsm8k": dict(
        title     = "GSM8K  ·  §5.1",
        subtitle  = "Qwen2.5-14B-Instruct · lm-evaluation-harness 0.4.11 · 5-shot",
        top_hdr   = "Plain few-shot  (direct-answer)",
        top_score = 58.5,
        top_unit  = "%",
        top_text  = (
            "[sys] You are Qwen, created by Alibaba Cloud.\n"
            "Q: Jen and Tyler are gymnasts …\n"
            "A: … #### 12\n"
            "Q: Four people in a law firm …\n"
            "A: … #### 1\n"
            "     ...  [3 more few-shot examples]  ...\n"
            "Q: Janet's ducks lay 16 eggs/day.\n"
            "   She eats 3 for breakfast; bakes 4 into muffins.\n"
            "   She sells the rest at $2 each. How much?\n"
            "A: 16−3−4 = 9 eggs.  9 × $2 = $18.  #### 18"
        ),
        top_hi    = "A: 16−3−4 = 9 eggs.  9 × $2 = $18.  #### 18",
        bot_hdr   = "Chain-of-thought few-shot",
        bot_score = 2.5,
        bot_unit  = "%",
        bot_text  = (
            "[sys] You are an expert problem solver.\n"
            "   Think step by step, then state the answer.\n"
            "Q: Jen and Tyler are gymnasts …\n"
            "A: … #### 12\n"
            "     ...  [same 5 few-shot examples]  ...\n"
            "Q: Janet's ducks lay 16 eggs/day. …\n"
            "A: Janet uses 3 + 4 = 7 eggs each day.\n"
            "   16 − 7 = 9 eggs remain.\n"
            "   9 × $2 = $18 per day.\n"
            "   The answer is $18 per day."
        ),
        bot_hi    = "   The answer is $18 per day.",
        mech      = (
            "Same model · same harness · same 5-shot context.\n"
            "CoT prompt elicits step-by-step prose;\n"
            "the #### delimiter regex finds nothing.\n"
            "Parser records 0 for correctly-answered items."
        ),
        source    = (
            "Prompts: run logs · experiments/controlled_eval/5-shot-GSM8K/\n"
            "Scores verified: 58.5% plain, 2.5% CoT (limit=200, seed=42/123/7)"
        ),
    ),

    "hellaswag": dict(
        title     = "HellaSwag  ·  §5.2",
        subtitle  = "Gemma-7B · two papers, two protocols",
        top_hdr   = "Phi-3 paper · 5-shot · generation-based",
        top_score = 49.80,
        top_unit  = "",
        top_text  = (
            "# 5-shot generation template\n"
            "#  (canonical; verbatim not published)\n"
            "Context: She bopped along, but the music\n"
            "  was really loud, so she turned it down.\n"
            "  Now, how does the next sentence go?\n"
            "A) She turned it up more.\n"
            "B) She left the store.\n"
            "C) She decreased the volume.\n"
            "D) She plugged her ears.\n"
            "Answer: C   ← free-text generation"
        ),
        top_hi    = "Answer: C   ← free-text generation",
        bot_hdr   = "Gemma paper · 0-shot · log-likelihood",
        bot_score = 81.20,
        bot_unit  = "",
        bot_text  = (
            "# 0-shot log-likelihood template\n"
            "#  (canonical; verbatim not published)\n"
            "Context: She bopped along, but the music\n"
            "  was really loud, so she turned it down.\n"
            "Completion A: She turned it up more.\n"
            "Completion B: She left the store.\n"
            "Completion C: She decreased the volume.\n"
            "Completion D: She plugged her ears.\n"
            "\n"
            "← rank by log P(completion | context)"
        ),
        bot_hi    = "← rank by log P(completion | context)",
        mech      = (
            "Same benchmark label · two dimensions differ.\n"
            "Scoring mode: generation vs. log-likelihood.\n"
            "N-shot: 5-shot vs. 0-shot.\n"
            "Primary driver: scoring mode, not n-shot."
        ),
        source    = (
            "Scores: arXiv:2404.14219 Table 3 (Phi-3, 49.80)\n"
            "        arXiv:2403.08295 Table 6 (Gemma, 81.20)"
        ),
    ),

    "belebele": dict(
        title     = "Belebele  ·  §5.2",
        subtitle  = "Mistral-7B · two papers · two evaluation regimes",
        top_hdr   = "Reka paper · 0-shot · multilingual avg (150 langs)",
        top_score = 32.80,
        top_unit  = "",
        top_text  = (
            "# 0-shot multilingual template (Reka)\n"
            "#  (canonical; verbatim not published)\n"
            "Passage: [reading comprehension context]\n"
            "Question: What does the passage say about …?\n"
            "A) [option]   B) [option]\n"
            "C) [option]   D) [option]\n"
            "Answer:\n"
            "← 0-shot; no few-shot examples in context\n"
            "← average over 150 language variants\n"
            "Mistral-7B score: 32.80"
        ),
        top_hi    = "← 0-shot; no few-shot examples in context",
        bot_hdr   = "Falcon2 report · 5-shot · English only",
        bot_score = 79.42,
        bot_unit  = "",
        bot_text  = (
            "# 5-shot English-only template (Falcon2)\n"
            "#  (canonical; verbatim not published)\n"
            "Example 1: [English passage + question]\n"
            "Answer: B\n"
            "   ...  [4 more few-shot examples]  ...\n"
            "Passage: [English reading comprehension]\n"
            "Question: What does the passage say about …?\n"
            "A) [option]   B) [option]\n"
            "Answer:   ← 5-shot; English only\n"
            "Mistral-7B score: 79.42"
        ),
        bot_hi    = "Mistral-7B score: 79.42",
        mech      = (
            "Same benchmark label · two dimensions differ.\n"
            "Language scope: multilingual avg vs. English only.\n"
            "N-shot: 0-shot vs. 5-shot.\n"
            "Falcon2 internal 0-shot English = 32.48 ≈ Reka 32.80;\n"
            "n-shot, not language scope, drives the 46.6 pp gap."
        ),
        source    = (
            "Scores: arXiv:2404.12387 (Reka, 32.80)\n"
            "        arXiv:2407.14885 (Falcon2, 79.42 · internal 0-shot = 32.48)"
        ),
    ),

    "bbh": dict(
        title     = "BBH · Gemma-2-9B  ·  §5.3",
        subtitle  = "Five sources · two score clusters · two protocols",
        top_hdr   = "OLv2 & SEA-LION · lighteval · 3-shot · log-likelihood",
        top_score = 34.10,
        top_unit  = "",
        top_text  = (
            "# lighteval BBH protocol (OLv2 / SEA-LION)\n"
            "Q: Which statement is logically false?\n"
            "   given: all mammals breathe air;\n"
            "          whales are mammals.\n"
            "(A) Whales breathe air.  (B) Whales swim.\n"
            "(C) Whales are reptiles. (D) Whales have lungs.\n"
            "rank: argmax log P(choice | context)\n"
            "      ← 3-shot; log-likelihood ranking\n"
            "OLv2     = 34.10  ┐ exact 2-decimal match\n"
            "SEA-LION = 34.10  ┘ → shared pipeline origin"
        ),
        top_hi    = "OLv2     = 34.10  ┐ exact 2-decimal match",
        bot_hdr   = "OLMoE / SmolLM2 / RecurrentGemma · 0-shot CoT",
        bot_score = 68.73,
        bot_unit  = "",
        bot_text  = (
            "# 0-shot CoT protocol (Suzgun et al. 2023)\n"
            "Q: Which statement is logically false? …\n"
            "Let's think step by step.\n"
            "All mammals breathe air; whales are mammals\n"
            "therefore whales breathe air. That is true.\n"
            "Whales are not reptiles — that is false.\n"
            "The answer is (C).\n"
            "──────────────────────────────────────\n"
            "OLMoE = 68.20  SmolLM2 = 69.00\n"
            "RecurrentGemma = 69.00  (independent)"
        ),
        bot_hi    = "OLMoE = 68.20  SmolLM2 = 69.00",
        mech      = (
            "Two score clusters — two protocols.\n"
            "OLv2 & SEA-LION: exact 2-decimal match;\n"
            "shared pipeline, not independent evidence.\n"
            "OLMoE/SmolLM2/RecurrentGemma: 0-shot CoT."
        ),
        source    = (
            "Scores: analysis_output/validation_pairs.txt\n"
            "        arXiv:2504.05747 (SEA-LION, Table 3)\n"
            "BBH cluster 2 mean: (68.20+69.00+69.00)/3 = 68.73"
        ),
    ),
}

# ── Delta verification ────────────────────────────────────────────────────────
_CHECKS = [
    ("gsm8k",     COLS["gsm8k"]["bot_score"]     - COLS["gsm8k"]["top_score"],     -56.0),
    ("hellaswag", COLS["hellaswag"]["bot_score"]  - COLS["hellaswag"]["top_score"],  +31.4),
    ("belebele",  COLS["belebele"]["bot_score"]   - COLS["belebele"]["top_score"],   +46.6),
    ("bbh",       COLS["bbh"]["bot_score"]        - COLS["bbh"]["top_score"],        +34.6),
]
for _name, _got, _exp in _CHECKS:
    assert abs(_got - _exp) < 0.15, f"Delta FAILED {_name}: got {_got:.2f} expected {_exp}"

# ── Layout constants ───────────────────────────────────────────────────────────
FW, FH    = 18.0, 11.0
COL_W     = 0.305
COL_GAP   = 0.040
COL_LEFT  = [0.020, 0.020 + COL_W + COL_GAP, 0.020 + 2 * (COL_W + COL_GAP)]
HDRBAND   = 0.028   # height reserved below panel top for pill label
STRIPBAND = 0.034   # height of S3 footer strip

Y_TITLE    = 0.975
Y_SUBTITLE = 0.953
Y_TOPBOX_T = 0.930
Y_TOPBOX_B = 0.622
Y_DELTA    = 0.596
Y_BOTBOX_T = 0.570
Y_BOTBOX_B = 0.262
Y_MECH_T   = 0.242
Y_SRC_T    = 0.058


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _rbox(ax, xL, xR, yB, yT, fc, ec=MGREY, lw=0.7, zorder=1, r=0.012):
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
                  boxstyle="round,pad=0.32", alpha=1.0),
    )


def _prompt_lines(ax, xL, xR, yT, yB, text_str, hi_line, fs=6.0):
    lines = text_str.split("\n")
    n = len(lines)
    mv, mh = 0.010, 0.014
    usable  = (yT - yB) - 2 * mv
    line_h  = usable / max(n, 1)
    for i, line in enumerate(lines):
        y     = yT - mv - (i + 0.5) * line_h
        is_hi = (hi_line is not None) and (line.strip() == hi_line.strip())
        kw = {}
        if is_hi:
            kw["bbox"] = dict(facecolor=HI_BG, edgecolor="none",
                              boxstyle="square,pad=0.15", alpha=0.95)
        ax.text(
            xL + mh, y, line,
            ha="left", va="center",
            fontsize=fs, fontfamily=MONO, color=DGREY,
            fontweight="bold" if is_hi else "normal",
            transform=ax.transAxes, zorder=4, clip_on=True, **kw,
        )


def _score_badge_S2(ax, xL, xR, yT, score, unit, color):
    """Small badge at upper-right inside panel text area."""
    label = f"{score:.1f}{unit}" if unit == "%" else f"{score:.2f}"
    bw, bh = 0.072, 0.040
    bx = xR - bw - 0.010
    by = yT - bh - 0.010
    _rbox(ax, bx, bx + bw, by, by + bh, fc=WHITE, ec=color, lw=1.5, zorder=7, r=0.007)
    ax.text(bx + bw / 2, by + bh / 2, label,
            ha="center", va="center",
            fontsize=10, fontfamily=SANS, color=color, fontweight="bold",
            transform=ax.transAxes, zorder=8)


def _score_strip_S3(ax, xL, xR, yB, score, unit, color):
    """Solid coloured strip at bottom of panel."""
    label = f"{score:.1f}{unit}" if unit == "%" else f"{score:.2f}"
    strip_t = yB + STRIPBAND
    _rbox(ax, xL, xR, yB, strip_t, fc=color, ec=color, lw=0, zorder=5, r=0.008)
    ax.text((xL + xR) / 2, yB + STRIPBAND / 2, label,
            ha="center", va="center",
            fontsize=10, fontfamily=SANS, color=WHITE, fontweight="bold",
            transform=ax.transAxes, zorder=6)


def _delta_arrow(ax, xL, xR, y_center, delta):
    xm     = (xL + xR) / 2
    sign   = "+" if delta >= 0 else ""
    label  = f"Δ = {sign}{delta:.1f} pp"
    color  = GREEN_DRK if delta >= 0 else RED
    dy     = 0.028
    if delta < 0:
        ay_start, ay_end = y_center + dy / 2, y_center - dy / 2
    else:
        ay_start, ay_end = y_center - dy / 2, y_center + dy / 2
    ax.annotate(
        "", xy=(xm, ay_end), xytext=(xm, ay_start),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color=color, lw=2.0),
        zorder=7,
    )
    ax.text(xm + 0.020, y_center, label,
            ha="left", va="center",
            fontsize=8.5, fontfamily=SANS, color=color, fontweight="bold",
            transform=ax.transAxes, zorder=7)


# ── Column drawing ────────────────────────────────────────────────────────────

def draw_column(ax, xL, col, score_style):
    xR    = xL + COL_W
    delta = col["bot_score"] - col["top_score"]

    # titles
    ax.text((xL + xR) / 2, Y_TITLE, col["title"],
            ha="center", va="center", fontsize=10, fontfamily=SANS,
            color=BLACK, fontweight="bold", transform=ax.transAxes)
    ax.text((xL + xR) / 2, Y_SUBTITLE, col["subtitle"],
            ha="center", va="center", fontsize=7, fontfamily=SANS,
            color=DGREY, transform=ax.transAxes)

    # ── Top panel ─────────────────────────────────────────────────────────────
    top_text_yB = Y_TOPBOX_B + (STRIPBAND if score_style == "S3" else 0)
    _rbox(ax, xL, xR, Y_TOPBOX_B, Y_TOPBOX_T, fc=BLUE_FILL, ec=BLUE_HDR, lw=1.1)
    _pill(ax, xL + 0.012, Y_TOPBOX_T - 0.010, col["top_hdr"], BLUE_HDR)
    _prompt_lines(ax, xL, xR, Y_TOPBOX_T - HDRBAND, top_text_yB,
                  col["top_text"], col.get("top_hi"))
    if score_style == "S2":
        _score_badge_S2(ax, xL, xR, Y_TOPBOX_T - HDRBAND,
                        col["top_score"], col["top_unit"], BLUE_HDR)
    elif score_style == "S3":
        _score_strip_S3(ax, xL, xR, Y_TOPBOX_B,
                        col["top_score"], col["top_unit"], BLUE_HDR)

    # ── Delta ─────────────────────────────────────────────────────────────────
    _delta_arrow(ax, xL, xR, Y_DELTA, delta)

    # ── Bottom panel ──────────────────────────────────────────────────────────
    bot_text_yB = Y_BOTBOX_B + (STRIPBAND if score_style == "S3" else 0)
    _rbox(ax, xL, xR, Y_BOTBOX_B, Y_BOTBOX_T, fc=ORNG_FILL, ec=ORNG_HDR, lw=1.1)
    _pill(ax, xL + 0.012, Y_BOTBOX_T - 0.010, col["bot_hdr"], ORNG_HDR)
    _prompt_lines(ax, xL, xR, Y_BOTBOX_T - HDRBAND, bot_text_yB,
                  col["bot_text"], col.get("bot_hi"))
    if score_style == "S2":
        _score_badge_S2(ax, xL, xR, Y_BOTBOX_T - HDRBAND,
                        col["bot_score"], col["bot_unit"], ORNG_HDR)
    elif score_style == "S3":
        _score_strip_S3(ax, xL, xR, Y_BOTBOX_B,
                        col["bot_score"], col["bot_unit"], ORNG_HDR)

    # mechanism + source
    ax.text((xL + xR) / 2, Y_MECH_T, col["mech"],
            ha="center", va="top", fontsize=7, fontfamily=SANS,
            color=DGREY, transform=ax.transAxes, linespacing=1.5)
    ax.text((xL + xR) / 2, Y_SRC_T, col["source"],
            ha="center", va="top", fontsize=5.5, fontfamily=MONO,
            color="#888888", transform=ax.transAxes, linespacing=1.4)


# ── Variant generator ─────────────────────────────────────────────────────────

def generate_variant(middle, score_style):
    col_order = [COLS["gsm8k"], COLS[middle], COLS["bbh"]]

    fig = plt.figure(figsize=(FW, FH), facecolor=WHITE)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for ci, (col, xL) in enumerate(zip(col_order, COL_LEFT)):
        draw_column(ax, xL, col, score_style)
        if ci < 2:
            sep_x = xL + COL_W + COL_GAP / 2
            ax.plot([sep_x, sep_x], [0.04, 0.96],
                    color=MGREY, lw=0.7, transform=ax.transAxes, zorder=1)

    # variant tag
    ax.text(0.997, 0.005, f"[{score_style}]",
            ha="right", va="bottom", fontsize=6, fontfamily=MONO,
            color="#BBBBBB", transform=ax.transAxes)

    stem     = f"case_studies_unified_{middle}_{score_style}"
    pdf_path = os.path.join(OUT_DIR, f"{stem}.pdf")
    png_path = os.path.join(OUT_DIR, f"{stem}.png")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.savefig(png_path, format="png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved: {pdf_path}")


def main():
    for middle in ("hellaswag", "belebele"):
        for style in ("S1", "S2", "S3"):
            generate_variant(middle, style)


if __name__ == "__main__":
    main()
