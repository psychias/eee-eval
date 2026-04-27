#!/usr/bin/env python3
"""Analyze cross-extractor results from sample_outputs and generate figure.

Handles lm-eval's concatenated-JSON format (no newlines between records).
"""
import json, re, csv, statistics
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_DIR = ROOT / "experiments" / "sample_outputs"
FIG_DIR = ROOT / "submission" / "latex" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Extractors ────────────────────────────────────────────────────────────
def extract_strict(text):
    m = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    return m.group(1).replace(",", "") if m else ""

def extract_last_number(text):
    matches = re.findall(r"-?[\d,]+\.?\d*", text)
    return matches[-1].replace(",", "") if matches else ""

def extract_boxed(text):
    m = re.search(r"\\boxed\{(-?[\d,]+\.?\d*)\}", text)
    return m.group(1).replace(",", "") if m else ""

def extract_answer_is(text):
    m = re.search(r"(?:the\s+)?answer\s+is\s*:?\s*(-?[\d,]+\.?\d*)", text, re.I)
    return m.group(1).replace(",", "") if m else ""

def extract_flexible(text):
    for fn in [extract_strict, extract_boxed, extract_answer_is, extract_last_number]:
        r = fn(text)
        if r:
            return r
    return ""

EXTRACTORS = {
    "strict (####)": extract_strict,
    "last number": extract_last_number,
    "\\boxed{}": extract_boxed,
    "answer is": extract_answer_is,
    "flexible chain": extract_flexible,
}

def normalize_answer(ans):
    try:
        return str(int(float(ans)))
    except (ValueError, OverflowError):
        return ans.strip()


def load_concat_json(path):
    """Load concatenated JSON objects (no newline separators)."""
    with open(path, encoding="utf-8") as f:
        data = f.read()
    if not data.strip():
        return []
    if data.strip().startswith("["):
        return json.loads(data)
    # Split on }{ boundary (handles lm-eval's concat format)
    parts = re.split(r"(?<=\})(?=\{)", data.strip())
    samples = []
    for part in parts:
        try:
            samples.append(json.loads(part))
        except json.JSONDecodeError:
            continue
    return samples


def score_samples(samples, extractor_fn):
    nc = nt = 0
    for s in samples:
        # For cross-extractor study, always use raw resps (not filtered_resps
        # which contains lm-eval's own extraction result like '[invalid]')
        output = ""
        resps = s.get("resps")
        if resps:
            val = resps[0] if isinstance(resps, list) and resps else resps
            if isinstance(val, list):
                val = val[0] if val else ""
            output = str(val)
        if not output or output == "[invalid]":
            for key in ("response", "output", "completion"):
                val = s.get(key)
                if val:
                    if isinstance(val, list):
                        val = val[0] if val else ""
                        if isinstance(val, list):
                            val = val[0] if val else ""
                    output = str(val)
                    break
        if not output:
            continue
        # Extract gold number from target string (e.g. "... #### 18")
        gold_src = str(s.get("target", "") or s.get("doc", {}).get("answer", ""))
        gold_m = re.search(r"####\s*(-?[\d,]+\.?\d*)", gold_src)
        if not gold_m:
            continue
        gold = gold_m.group(1).replace(",", "")
        nt += 1
        predicted = extractor_fn(output)
        if predicted and normalize_answer(predicted) == normalize_answer(gold):
            nc += 1
    return nc, nt


# ── Run analysis ──────────────────────────────────────────────────────────
rows = []
for run_dir in sorted(SAMPLE_DIR.iterdir()):
    if not run_dir.is_dir():
        continue
    sample_files = list(run_dir.glob("samples_*.jsonl"))
    if not sample_files:
        continue
    parts = run_dir.name.rsplit("_gsm8k_0shot_", 1)
    model_short = parts[0] if len(parts) == 2 else run_dir.name
    fmt = parts[1] if len(parts) == 2 else "unknown"

    for sf in sample_files:
        samples = load_concat_json(sf)
        if not samples:
            continue
        for ext_name, ext_fn in EXTRACTORS.items():
            nc, nt = score_samples(samples, ext_fn)
            acc = nc / nt * 100 if nt > 0 else 0.0
            rows.append({
                "model": model_short, "format": fmt,
                "extractor": ext_name, "accuracy": round(acc, 2),
                "n_correct": nc, "n_total": nt,
            })

if not rows:
    print("ERROR: No results found!")
    exit(1)

# ── Print table ───────────────────────────────────────────────────────────
by_cell = defaultdict(dict)
for r in rows:
    by_cell[(r["model"], r["format"])][r["extractor"]] = r["accuracy"]

ext_names = list(EXTRACTORS.keys())

print(f"\n{'Model':35s} {'Fmt':10s}", end="")
for e in ext_names:
    print(f" {e:>15s}", end="")
print()
print("-" * 115)
for (model, fmt), scores in sorted(by_cell.items()):
    print(f"{model:35s} {fmt:10s}", end="")
    for e in ext_names:
        print(f" {scores.get(e, 0):15.2f}", end="")
    print()

# ── Gaps ──────────────────────────────────────────────────────────────────
print("\n=== STRICT vs FLEXIBLE GAP ===")
gaps = []
for (model, fmt), scores in sorted(by_cell.items()):
    strict = scores.get("strict (####)", 0)
    flex = scores.get("flexible chain", 0)
    gap = flex - strict
    gaps.append(gap)
    print(f"  {model:35s} {fmt:10s}  strict={strict:6.2f}  flex={flex:6.2f}  gap={gap:+.2f} pp")

print(f"\n  Mean gap: {np.mean(gaps):.2f} pp")
print(f"  Max gap:  {max(gaps):.2f} pp")
print(f"  Cells where gap > 0: {sum(1 for g in gaps if g > 0)}/{len(gaps)}")

# ── Per-extractor summary ────────────────────────────────────────────────
print("\n=== PER-EXTRACTOR SUMMARY ===")
for ext in ext_names:
    vals = [r["accuracy"] for r in rows if r["extractor"] == ext]
    print(f"  {ext:20s}: mean={statistics.mean(vals):.2f}  "
          f"min={min(vals):.2f}  max={max(vals):.2f}")

# ── Save CSV ──────────────────────────────────────────────────────────────
csv_path = ROOT / "experiments" / "cross_extractor_results.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print(f"\nCSV saved to {csv_path}")

# ── Generate figure ──────────────────────────────────────────────────────
COLORS = ["#CC3311", "#0077BB", "#EE7733", "#009988", "#33BBEE"]

cells = sorted(by_cell.keys())
x = np.arange(len(cells))
width = 0.15

fig, ax = plt.subplots(figsize=(max(10, len(cells) * 1.2), 5))
for i, (ext, color) in enumerate(zip(ext_names, COLORS)):
    vals = [by_cell[c].get(ext, 0) for c in cells]
    ax.bar(x + i * width, vals, width, label=ext, color=color)

ax.set_ylabel("Accuracy (%)")
ax.set_title("Cross-Extractor Study: 0-shot GSM8K (3 models × 3 formats)")
ax.set_xticks(x + width * 2)
ax.set_xticklabels([f"{m}\n{f}" for m, f in cells], fontsize=7)
ax.legend(fontsize=7, ncol=2, loc="upper right")
all_vals = [v for scores in by_cell.values() for v in scores.values()]
ax.set_ylim(0, max(max(all_vals) * 1.15, 5) if all_vals else 100)
plt.tight_layout()

for ext_str in ("pdf", "png"):
    fig.savefig(FIG_DIR / f"fig_cross_extractor.{ext_str}",
                bbox_inches="tight", dpi=150 if ext_str == "png" else None)
plt.close(fig)
print(f"\nFigure saved to {FIG_DIR}/fig_cross_extractor.{{pdf,png}}")
