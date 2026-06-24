#!/usr/bin/env python3
"""Cross-extractor study for 0-shot GSM8K.

Compares the default lm-eval `#### <number>` extraction regex against
more flexible alternatives to quantify how much score the strict regex
leaves on the table when 0-shot models omit the `####` prefix.

Usage (on Colab or locally after downloading outputs):
    python cross_extractor_study.py --results-dir /path/to/lm_eval_output_dirs

The script scans for lm-eval sample-level output files (*_results.jsonl or
samples_*.jsonl) in subdirectories of --results-dir, filters for 0-shot
GSM8K runs, and re-scores each sample under multiple extractors.

Output: a CSV with columns [model, extractor, accuracy, n_correct, n_total]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Extractors — each takes the model's raw text output and returns a number
# string (or "" if no number found).
# ---------------------------------------------------------------------------

def extract_strict(text: str) -> str:
    """lm-eval default: looks for `#### <number>`."""
    m = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    return m.group(1).replace(",", "") if m else ""


def extract_last_number(text: str) -> str:
    """Take the last number in the generation (the most common heuristic)."""
    matches = re.findall(r"-?[\d,]+\.?\d*", text)
    return matches[-1].replace(",", "") if matches else ""


def extract_boxed(text: str) -> str:
    r"""Look for \\boxed{<number>} (common in CoT models)."""
    m = re.search(r"\\boxed\{(-?[\d,]+\.?\d*)\}", text)
    return m.group(1).replace(",", "") if m else ""


def extract_answer_is(text: str) -> str:
    """Look for 'the answer is <number>' pattern."""
    m = re.search(r"(?:the\s+)?answer\s+is\s*:?\s*(-?[\d,]+\.?\d*)", text, re.I)
    return m.group(1).replace(",", "") if m else ""


def extract_flexible(text: str) -> str:
    """Try strict → boxed → 'answer is' → last-number fallback chain."""
    for fn in [extract_strict, extract_boxed, extract_answer_is, extract_last_number]:
        result = fn(text)
        if result:
            return result
    return ""


EXTRACTORS = {
    "strict_####": extract_strict,
    "last_number": extract_last_number,
    "boxed": extract_boxed,
    "answer_is": extract_answer_is,
    "flexible_chain": extract_flexible,
}


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

def _normalize_answer(ans: str) -> str:
    """Strip trailing zeros for float comparison."""
    try:
        return str(int(float(ans)))
    except (ValueError, OverflowError):
        return ans.strip()


def find_sample_files(results_dir: Path) -> list[Path]:
    """Find all lm-eval sample output files for GSM8K 0-shot runs."""
    candidates = []
    for root, _dirs, files in os.walk(results_dir):
        for f in files:
            fp = Path(root) / f
            name_lower = f.lower()
            # lm-eval saves samples as samples_<task>_*.jsonl
            if "gsm8k" in name_lower and name_lower.endswith(".jsonl"):
                candidates.append(fp)
    return candidates


def is_zero_shot_dir(path: Path) -> bool:
    """Heuristic: directory name or parent contains '0shot' or 'n0'."""
    parts = str(path).lower()
    return "0shot" in parts or "0-shot" in parts or "_n0_" in parts or "/n0/" in parts


def load_samples(path: Path) -> list[dict]:
    """Load sample-level results from a JSONL file."""
    samples = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return samples


def score_samples(samples: list[dict], extractor_fn) -> tuple[int, int]:
    """Return (n_correct, n_total) using the given extractor.

    Expects each sample dict to have:
      - 'resps' or 'response' or 'filtered_resps': the model generation
      - 'target' or 'answer' or 'doc.answer': the gold answer
    Adapts to multiple lm-eval output formats.
    """
    n_correct = 0
    n_total = 0
    for s in samples:
        # Get model output text
        output = ""
        for key in ("filtered_resps", "resps", "response", "output", "completion"):
            val = s.get(key)
            if val:
                if isinstance(val, list):
                    # lm-eval nests as [[text]]
                    val = val[0] if val else ""
                    if isinstance(val, list):
                        val = val[0] if val else ""
                output = str(val)
                break
        if not output:
            continue

        # Get gold answer
        gold = ""
        for key in ("target", "answer"):
            if key in s:
                gold = str(s[key])
                break
        if not gold:
            doc = s.get("doc", {})
            gold = str(doc.get("answer", ""))
        if not gold:
            continue

        n_total += 1
        predicted = extractor_fn(output)
        if predicted and _normalize_answer(predicted) == _normalize_answer(gold):
            n_correct += 1

    return n_correct, n_total


def run_study(results_dir: Path) -> list[dict]:
    """Run all extractors over all 0-shot GSM8K sample files."""
    sample_files = find_sample_files(results_dir)
    if not sample_files:
        print(f"No GSM8K sample files found under {results_dir}", file=sys.stderr)
        return []

    rows = []
    for sf in sample_files:
        # Try to infer model name from path
        model = sf.parent.name or "unknown"
        samples = load_samples(sf)
        if not samples:
            continue

        for name, fn in EXTRACTORS.items():
            n_correct, n_total = score_samples(samples, fn)
            acc = n_correct / n_total if n_total > 0 else 0.0
            rows.append({
                "file": str(sf),
                "model": model,
                "extractor": name,
                "accuracy": round(acc * 100, 2),
                "n_correct": n_correct,
                "n_total": n_total,
            })
            print(f"  {model:40s}  {name:20s}  {acc*100:6.2f}%  ({n_correct}/{n_total})")

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, required=True,
        help="Root directory containing lm-eval output subdirectories",
    )
    parser.add_argument(
        "--output-csv", type=Path, default=None,
        help="Optional path to write CSV summary",
    )
    args = parser.parse_args()

    rows = run_study(args.results_dir)
    if not rows:
        sys.exit(1)

    if args.output_csv:
        import csv
        with open(args.output_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved to {args.output_csv}")


if __name__ == "__main__":
    main()
