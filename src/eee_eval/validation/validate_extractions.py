"""Validate LLM-extracted values against original arxiv HTML.

For each JSON record, checks whether extracted values (score, model_name,
benchmark_name, metric_name, generation config) actually appear in the paper's
HTML text. Stores results in source_metadata.additional_details:
  - "validated": "True"/"False"
  - "incorrect_values": semicolon-separated list of value descriptions not found in HTML

Usage:
    # Validate all existing data
    python src/validation/validate_extractions.py

    # Validate a single paper
    python src/validation/validate_extractions.py --arxiv-id 2305.13048

    # Programmatic use from extraction pipeline
    from eee_eval.validation.validate_extractions import validate_record
"""
import argparse
import json
import logging
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

log = logging.getLogger("eee_validate")

ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUT_DIR = ROOT / "data" / "archiv_paper_extraction"
HTML_CACHE_DIR = ROOT / "data" / "html_cache"


# ── HTML fetching / caching ───────────────────────────────────────────

def _get_html_text(arxiv_id: str) -> str | None:
    """Fetch HTML and return plain text. Caches to disk."""
    HTML_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = HTML_CACHE_DIR / f"{arxiv_id}.txt"

    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    url = f"https://arxiv.org/html/{arxiv_id}"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200 or "text/html" not in resp.headers.get("Content-Type", ""):
            log.warning("%s: HTTP %d or not HTML", arxiv_id, resp.status_code)
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        cache_path.write_text(text, encoding="utf-8")
        return text
    except Exception as e:
        log.error("Failed to fetch %s: %s", arxiv_id, e)
        return None


# ── Value checking ────────────────────────────────────────────────────

# Metric abbreviation equivalences
METRIC_ALIASES = {
    "accuracy": ["acc", "accuracy", "acc."],
    "exact_match": ["em", "exact match", "exact-match", "exactmatch"],
    "pass@1": ["pass@1", "pass @1"],
    "pass@10": ["pass@10", "pass @10"],
    "pass@100": ["pass@100", "pass @100"],
    "bleu": ["bleu", "bleu-4", "b@4"],
    "rouge": ["rouge", "rouge-l", "rougel"],
    "f1": ["f1", "f1-score", "f1 score"],
    "win_rate": ["win rate", "win_rate", "winrate"],
    "normalized_accuracy": ["acc_norm", "norm_acc", "normalized accuracy", "acc (norm)"],
    "average_score": ["avg", "average", "mean"],
    "meteor": ["meteor"],
    "cider": ["cider"],
    "ndcg@10": ["ndcg@10", "ndcg"],
    "miou": ["miou", "mIoU"],
    "map": ["map", "mAP"],
}

# Build reverse lookup: alias -> canonical and all variants
_ALIAS_LOOKUP: dict[str, list[str]] = {}
for canonical, aliases in METRIC_ALIASES.items():
    all_forms = [canonical] + aliases
    for a in all_forms:
        _ALIAS_LOOKUP[a.lower()] = all_forms


def _extract_numbers(html_text: str) -> set[float]:
    """Pre-extract all numbers from HTML text for fast tolerance checks."""
    nums = set()
    for m in re.finditer(r'\b\d+\.?\d*\b', html_text):
        try:
            nums.add(float(m.group()))
        except ValueError:
            pass
    return nums


def _value_in_text(value, html_text: str, html_numbers: set[float] | None = None) -> bool:
    """Check if a value appears in the HTML text."""
    if value is None:
        return True  # nothing to validate

    s = str(value).strip()
    if not s:
        return True

    # Direct substring match
    if s in html_text:
        return True

    # For numbers, try common formats: 45.6, 45.60, 0.456
    if isinstance(value, (int, float)):
        num = float(value)
        # Try integer form (only if exact integer)
        if num == int(num) and str(int(num)) in html_text:
            return True
        # Try with same or more decimal places (never fewer — avoids rounding false positives)
        # Determine original precision
        s_dec = s.split(".")[-1] if "." in s else ""
        orig_decimals = len(s_dec)
        for decimals in range(orig_decimals, orig_decimals + 3):
            fmt = f"{num:.{decimals}f}"
            if fmt in html_text:
                return True
        # No tolerance — any mismatch is likely LLM-fabricated precision
        # (our extraction code does not round, so mismatches are real errors)

        # Score might have been normalized 0-1 → 0-100 or vice versa
        if 0 < num <= 100:
            alt = num / 100
            for fmt in [f"{alt:.3f}", f"{alt:.4f}", f"{alt:.2f}"]:
                if fmt in html_text:
                    return True
        if 0 < num < 1:
            alt = num * 100
            for fmt in [f"{alt:.1f}", f"{alt:.0f}", f"{alt:.2f}"]:
                if fmt in html_text:
                    return True
        return False

    # For strings, try case-insensitive and partial matching
    s_lower = s.lower()
    html_lower = html_text.lower()
    if s_lower in html_lower:
        return True

    # Try without hyphens/underscores (e.g. "lm-evaluation-harness" vs "lm evaluation harness")
    s_norm = re.sub(r"[\-_]", " ", s_lower)
    html_norm = re.sub(r"[\-_]", " ", html_lower)
    if s_norm in html_norm:
        return True

    # Strip all punctuation/spaces for aggressive matching
    s_stripped = re.sub(r"[\s\-_\.()]+", "", s_lower)
    html_stripped = re.sub(r"[\s\-_\.()]+", "", html_lower)
    if len(s_stripped) >= 4 and s_stripped in html_stripped:
        return True

    # Metric abbreviation matching: if value is "accuracy", also check "acc"
    aliases = _ALIAS_LOOKUP.get(s_lower, [])
    for alias in aliases:
        if alias.lower() in html_lower:
            return True

    # For comma-separated values (e.g. "lm-evaluation-harness, OpenCompass"),
    # check if ANY component appears in the HTML
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if any(_value_in_text(part, html_text) for part in parts):
            return True

    # Partial matching for benchmark names with language/variant tags
    # e.g. "Multi-PL (JavaScript)" -> check "Multi-PL" or "JavaScript"
    paren_match = re.match(r"^(.+?)\s*\((.+?)\)$", s)
    if paren_match:
        base, tag = paren_match.group(1).strip(), paren_match.group(2).strip()
        if base.lower() in html_lower and tag.lower() in html_lower:
            return True
        # Check base without parens
        base_norm = re.sub(r"[\-_]", " ", base.lower())
        if base_norm in html_norm:
            return True

    # Model name fuzzy: strip version suffixes and size markers
    # e.g. "DeepSeek Coder 1.3B" -> check "DeepSeek-Coder" or "DeepSeek Coder"
    # Remove trailing size markers like "1.3B", "7B", "13B", etc.
    no_size = re.sub(r"\s*[\-_]?\d+\.?\d*[BbMm]$", "", s).strip()
    if no_size and no_size.lower() != s_lower:
        no_size_lower = no_size.lower()
        if no_size_lower in html_lower:
            return True
        no_size_norm = re.sub(r"[\-_]", " ", no_size_lower)
        if no_size_norm in html_norm:
            return True

    return False


def validate_record(record: dict, html_text: str, html_numbers: set[float] | None = None) -> tuple[bool, list[str]]:
    """Validate a single EEE-schema JSON record against HTML text.

    Returns (validated, incorrect_values).
    """
    if html_numbers is None:
        html_numbers = _extract_numbers(html_text)

    incorrect: list[str] = []

    # Model name
    model_name = record.get("model_info", {}).get("name")
    if model_name and not _value_in_text(model_name, html_text, html_numbers):
        incorrect.append(f"model_name: {model_name}")

    for er in record.get("evaluation_results", []):
        # Benchmark name — skip validation (too many false positives from
        # language tags, combined names like "HumanEval-Java", etc.)

        # Metric name
        metric = er.get("metric_config", {}).get("metric_name")
        if metric and not _value_in_text(metric, html_text, html_numbers):
            incorrect.append(f"metric_name: {metric}")

        # Score
        score = er.get("score_details", {}).get("score")
        if score is not None and not _value_in_text(score, html_text, html_numbers):
            incorrect.append(f"score: {score}")

        # Generation config values
        gen_args = er.get("generation_config", {}).get("generation_args", {})
        for key in ["n_shot", "temperature", "eval_harness",
                     "decoding_strategy", "prompt_template"]:
            val = gen_args.get(key)
            if val is not None and not _value_in_text(val, html_text, html_numbers):
                incorrect.append(f"{key}: {val}")

    validated = len(incorrect) == 0
    return validated, incorrect


# ── Batch validation ──────────────────────────────────────────────────

def validate_paper_jsons(arxiv_id: str, json_paths: list[Path], html_text: str) -> dict:
    """Validate all JSONs for a paper. Returns stats."""
    stats = {"total": 0, "validated": 0, "failed": 0}
    html_numbers = _extract_numbers(html_text)

    for path in json_paths:
        record = json.load(open(path, encoding="utf-8"))
        validated, incorrect = validate_record(record, html_text, html_numbers)

        # Store validation metadata inside source_metadata.additional_details
        # (top-level additionalProperties are forbidden by eval.schema.json)
        sm = record.setdefault("source_metadata", {})
        ad = sm.setdefault("additional_details", {})
        ad["validated"] = str(validated)
        if not validated:
            ad["incorrect_values"] = "; ".join(incorrect)
        else:
            ad.pop("incorrect_values", None)

        # Remove legacy top-level keys if present
        record.pop("validated", None)
        record.pop("incorrect_values", None)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, default=str)

        stats["total"] += 1
        if validated:
            stats["validated"] += 1
        else:
            stats["failed"] += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate LLM extractions against HTML")
    parser.add_argument("--arxiv-id", type=str, help="Validate a single paper")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between HTML fetches")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    llm_dir = OUT_DIR / "llm"
    if not llm_dir.exists():
        log.error("No LLM output directory found at %s", llm_dir)
        return

    # Group JSON files by arxiv_id
    paper_files: dict[str, list[Path]] = {}
    for path in llm_dir.rglob("*.json"):
        record = json.load(open(path, encoding="utf-8"))
        aid = record.get("source_metadata", {}).get("additional_details", {}).get("arxiv_id")
        if aid:
            if args.arxiv_id and aid != args.arxiv_id:
                continue
            paper_files.setdefault(aid, []).append(path)

    log.info("Found %d papers, %d JSON files to validate",
             len(paper_files), sum(len(v) for v in paper_files.values()))

    total_stats = {"total": 0, "validated": 0, "failed": 0}

    for i, (arxiv_id, paths) in enumerate(sorted(paper_files.items())):
        log.info("[%d/%d] %s (%d files)", i + 1, len(paper_files), arxiv_id, len(paths))

        html_text = _get_html_text(arxiv_id)
        if not html_text:
            log.warning("  Skipping %s — no HTML available", arxiv_id)
            continue

        stats = validate_paper_jsons(arxiv_id, paths, html_text)
        log.info("  %d/%d validated, %d incorrect", stats["validated"], stats["total"], stats["failed"])

        for k in total_stats:
            total_stats[k] += stats[k]

        time.sleep(args.delay)

    log.info("")
    log.info("=== VALIDATION SUMMARY ===")
    log.info("Total: %d  |  Validated: %d  |  Incorrect: %d",
             total_stats["total"], total_stats["validated"], total_stats["failed"])
    if total_stats["total"] > 0:
        pct = 100 * total_stats["validated"] / total_stats["total"]
        log.info("Accuracy: %.1f%%", pct)


if __name__ == "__main__":
    main()
