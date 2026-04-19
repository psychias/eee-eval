#!/usr/bin/env python3
"""Cross-extractor analysis for 0-shot GSM8K sample outputs.

Applies 5 answer-extraction strategies to saved sample-level JSONL files
and reports accuracy per (model, format, extractor). Produces a CSV and
a grouped bar-chart figure.

Usage:
    python scripts/run_cross_extractor_analysis.py --sample-dir /path/to/sample_outputs

The --sample-dir should contain subdirectories named like:
    <model>_gsm8k_0shot_<format>/
each containing lm-eval sample JSONL files (samples_*.jsonl).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

def extract_strict(text: str) -> str:
    """lm-eval default: looks for `#### <number>`."""
    m = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    return m.group(1).replace(",", "") if m else ""

def extract_last_number(text: str) -> str:
    """Take the last number in the generation."""
    matches = re.findall(r"-?[\d,]+\.?\d*", text)
    return matches[-1].replace(",", "") if matches else ""

def extract_boxed(text: str) -> str:
    r"""Look for \boxed{<number>}."""
    m = re.search(r"\\boxed\{(-?[\d,]+\.?\d*)\}", text)
    return m.group(1).replace(",", "") if m else ""

def extract_answer_is(text: str) -> str:
    """Look for 'the answer is <number>' pattern."""
    m = re.search(r"(?:the\s+)?answer\s+is\s*:?\s*(-?[\d,]+\.?\d*)", text, re.I)
    return m.group(1).replace(",", "") if m else ""

def extract_flexible(text: str) -> str:
    """Cascading fallback: strict -> boxed -> answer_is -> last_number."""
    for fn in [extract_strict, extract_boxed, extract_answer_is, extract_last_number]:
        result = fn(text)
        if result:
            return result
    return ""

EXTRACTORS = {
    "strict_####":    extract_strict,
    "last_number":    extract_last_number,
    "boxed":          extract_boxed,
    "answer_is":      extract_answer_is,
    "flexible_chain": extract_flexible,
}

def normalize_answer(ans: str) -> str:
    try:
        return str(int(float(ans)))
    except (ValueError, OverflowError):
        return ans.strip()


# ---------------------------------------------------------------------------
# Sample loading and scoring
# ---------------------------------------------------------------------------

def load_samples(path: Path) -> list[dict]:
    """Load samples from JSONL files.

    Handles both newline-delimited JSONL and concatenated JSON objects
    (multiple {...}{...} on a single line, as produced by lm-eval's
    log_samples mode on Colab).
    """
    samples = []
    with open(path, encoding="utf-8") as fh:
        content = fh.read().strip()
    if not content:
        return samples

    # First try standard newline-delimited JSONL
    lines = content.split("\n")
    if len(lines) > 1:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if samples:
            return samples

    # Fallback: concatenated JSON objects on a single line
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(content):
        # Skip whitespace
        while idx < len(content) and content[idx] in " \t\r\n":
            idx += 1
        if idx >= len(content):
            break
        try:
            obj, end_idx = decoder.raw_decode(content, idx)
            samples.append(obj)
            idx = end_idx
        except json.JSONDecodeError:
            break

    return samples


def _unwrap(val):
    """Unwrap nested lists [[str]] -> str."""
    while isinstance(val, list):
        val = val[0] if val else ""
    return str(val) if val else ""


def _extract_gold_number(text: str) -> str:
    """Extract the numeric answer from GSM8K gold text (after ####)."""
    m = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if m:
        return m.group(1).replace(",", "")
    # Fallback: last number in text
    matches = re.findall(r"-?[\d,]+\.?\d*", text)
    return matches[-1].replace(",", "") if matches else text.strip()


def score_samples(samples: list[dict], extractor_fn) -> tuple[int, int]:
    n_correct = 0
    n_total = 0
    for s in samples:
        # Get model output — prefer resps (raw generation) over
        # filtered_resps which may contain "[invalid]" from lm-eval's
        # strict extractor.
        output = ""
        for key in ("resps", "filtered_resps", "response", "output", "completion"):
            val = s.get(key)
            if val:
                text = _unwrap(val)
                if text and text != "[invalid]":
                    output = text
                    break
        if not output:
            continue

        # Get gold answer number.  GSM8K stores the full solution in
        # doc.answer with the numeric answer after "####".
        gold_raw = ""
        doc = s.get("doc", {})
        if doc.get("answer"):
            gold_raw = str(doc["answer"])
        else:
            for key in ("target", "answer"):
                if key in s and s[key]:
                    gold_raw = str(s[key])
                    break
        if not gold_raw:
            continue

        gold_num = _extract_gold_number(gold_raw)
        if not gold_num:
            continue

        n_total += 1
        predicted = extractor_fn(output)
        if predicted and normalize_answer(predicted) == normalize_answer(gold_num):
            n_correct += 1

    return n_correct, n_total


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(sample_dir: Path, output_csv: Path | None, output_fig: Path | None):
    rows = []

    for run_dir in sorted(sample_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        # Find sample JSONL files
        sample_files = list(run_dir.rglob("samples_*.jsonl"))
        if not sample_files:
            sample_files = list(run_dir.rglob("*gsm8k*.jsonl"))
        if not sample_files:
            print(f"  No sample file in {run_dir.name}, skipping")
            continue

        # Parse model and format from dir name
        parts = run_dir.name.rsplit("_gsm8k_0shot_", 1)
        model_short = parts[0] if len(parts) == 2 else run_dir.name
        fmt = parts[1] if len(parts) == 2 else "unknown"

        for sf in sample_files:
            samples = load_samples(sf)
            if not samples:
                continue

            for ext_name, ext_fn in EXTRACTORS.items():
                n_correct, n_total = score_samples(samples, ext_fn)
                acc = n_correct / n_total * 100 if n_total > 0 else 0.0
                rows.append({
                    "model": model_short,
                    "format": fmt,
                    "extractor": ext_name,
                    "accuracy": round(acc, 2),
                    "n_correct": n_correct,
                    "n_total": n_total,
                })
                print(
                    f"  {model_short:40s}  {fmt:10s}  {ext_name:20s}  "
                    f"{acc:6.2f}%  ({n_correct}/{n_total})"
                )

    if not rows:
        print("No results. Check that sample files exist in the given directory.")
        return

    # --- Summary table ---
    print("\n" + "=" * 70)
    print("CROSS-EXTRACTOR STUDY: 0-shot GSM8K")
    print("=" * 70)

    # Group by (model, format)
    by_cell: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for r in rows:
        by_cell[(r["model"], r["format"])][r["extractor"]] = r["accuracy"]

    # Print header
    ext_names = list(EXTRACTORS.keys())
    header = f"{'model':30s} {'format':10s} " + " ".join(f"{e:>16s}" for e in ext_names)
    print(header)
    print("-" * len(header))
    for (model, fmt), scores in sorted(by_cell.items()):
        line = f"{model:30s} {fmt:10s} "
        line += " ".join(f"{scores.get(e, 0):16.2f}" for e in ext_names)
        print(line)

    # Summary stats
    print("\n--- Summary across all (model, format) cells ---")
    for ext in ext_names:
        vals = [r["accuracy"] for r in rows if r["extractor"] == ext]
        if vals:
            print(
                f"  {ext:20s}: mean={statistics.mean(vals):.2f}  "
                f"min={min(vals):.2f}  max={max(vals):.2f}"
            )

    # Key finding: gap between strict and flexible
    strict_by_cell = {
        (r["model"], r["format"]): r["accuracy"]
        for r in rows if r["extractor"] == "strict_####"
    }
    flex_by_cell = {
        (r["model"], r["format"]): r["accuracy"]
        for r in rows if r["extractor"] == "flexible_chain"
    }
    gaps = []
    for cell in strict_by_cell:
        if cell in flex_by_cell:
            gaps.append(flex_by_cell[cell] - strict_by_cell[cell])

    if gaps:
        print(f"\n--- Score left on table (flexible - strict) ---")
        print(f"  Mean gap: {statistics.mean(gaps):.2f} pp")
        print(f"  Max gap:  {max(gaps):.2f} pp")
        print(f"  Cells where gap > 0: {sum(1 for g in gaps if g > 0)}/{len(gaps)}")

    # --- Save CSV ---
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV saved to {output_csv}")

    # --- Generate figure ---
    if output_fig:
        try:
            import numpy as np
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            cells = sorted(by_cell.keys())
            x = np.arange(len(cells))
            width = 0.15
            colors = ["#CC3311", "#0077BB", "#EE7733", "#009988", "#33BBEE"]

            fig, ax = plt.subplots(figsize=(12, 5))
            for i, (ext, color) in enumerate(zip(ext_names, colors)):
                vals = [by_cell[c].get(ext, 0) for c in cells]
                ax.bar(x + i * width, vals, width, label=ext, color=color)

            ax.set_ylabel("Accuracy (%)")
            ax.set_title("Cross-Extractor Study: 0-shot GSM8K")
            ax.set_xticks(x + width * 2)
            ax.set_xticklabels(
                [f"{m}\n{f}" for m, f in cells], fontsize=7
            )
            ax.legend(fontsize=8, ncol=2)
            all_vals = [v for scores in by_cell.values() for v in scores.values()]
            ax.set_ylim(0, max(max(all_vals) * 1.15, 5) if all_vals else 100)
            plt.tight_layout()

            output_fig.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_fig, bbox_inches="tight", dpi=150)
            plt.close(fig)
            print(f"Figure saved to {output_fig}")

            # Also save PDF if PNG was requested
            if output_fig.suffix == ".png":
                pdf_path = output_fig.with_suffix(".pdf")
                fig2, ax2 = plt.subplots(figsize=(12, 5))
                for i, (ext, color) in enumerate(zip(ext_names, colors)):
                    vals = [by_cell[c].get(ext, 0) for c in cells]
                    ax2.bar(x + i * width, vals, width, label=ext, color=color)
                ax2.set_ylabel("Accuracy (%)")
                ax2.set_title("Cross-Extractor Study: 0-shot GSM8K")
                ax2.set_xticks(x + width * 2)
                ax2.set_xticklabels(
                    [f"{m}\n{f}" for m, f in cells], fontsize=7
                )
                ax2.legend(fontsize=8, ncol=2)
                ax2.set_ylim(0, max(max(all_vals) * 1.15, 5) if all_vals else 100)
                plt.tight_layout()
                fig2.savefig(pdf_path, bbox_inches="tight")
                plt.close(fig2)
                print(f"PDF saved to {pdf_path}")

        except ImportError:
            print("matplotlib not available -- skipping figure generation")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--sample-dir", type=Path, required=True,
        help="Directory containing per-run sample output subdirectories",
    )
    parser.add_argument(
        "--output-csv", type=Path, default=None,
        help="Path to write CSV results (default: <sample-dir>/../cross_extractor_results.csv)",
    )
    parser.add_argument(
        "--output-fig", type=Path, default=None,
        help="Path to write figure (default: <sample-dir>/../figures/fig_cross_extractor.png)",
    )
    args = parser.parse_args()

    if args.output_csv is None:
        args.output_csv = args.sample_dir.parent / "cross_extractor_results.csv"
    if args.output_fig is None:
        args.output_fig = args.sample_dir.parent / "figures" / "fig_cross_extractor.png"

    run_analysis(args.sample_dir, args.output_csv, args.output_fig)


if __name__ == "__main__":
    main()
