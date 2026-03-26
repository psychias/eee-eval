"""Benchmark and model name normalization functions."""
from __future__ import annotations

import re

from .constants import (
    _BENCHMARK_CANONICAL,
    _BENCHMARK_KEYWORDS_STRONG,
    _MODEL_ALIASES,
    _NON_BENCHMARK_NAMES,
    _TABLE_CATEGORY_PREFIXES,
)

def _normalize_benchmark(name: str) -> str | None:
    """Canonicalize a benchmark name.

    Returns the canonical form, or None if the name should be filtered out.
    """
    # Normalize unicode apostrophes / primes to ASCII
    name = name.replace("\u2018", "'").replace("\u2019", "'").replace("\u2032", "'")
    # Strip control characters (Docling artifacts)
    name = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', ' ', name)
    stripped = name.strip()

    # Strip leading '& ' — LaTeX table separator artifact from Docling
    stripped = re.sub(r'^[&]\s*', '', stripped).strip()

    # Strip trailing variance/CI artifacts from concatenated headers:
    # "API-Bank 82.6 ±3.8 56.5 ±4.9" → "API-Bank"
    # "BFCL 76.1 ±2.0 –" → "BFCL"
    stripped = re.sub(r'\s+\d+\.?\d*\s*[±]\s*\d+\.?\d*(?:\s+\d+\.?\d*\s*[±]\s*\d+\.?\d*)*(?:\s*[–—-])*$', '', stripped).strip()
    # Also handle: "Nexus 38.5 ±4.1 –" pattern
    stripped = re.sub(r'\s+\d+\.?\d*\s*[±]\s*\d+\.?\d*\s*[–—-]*$', '', stripped).strip()

    # Strip leading score-like prefix: "33.1 INCLUDE" → "INCLUDE"
    leading_score = re.match(r'^(\d+\.\d+)\s+([A-Za-z].+)$', stripped)
    if leading_score:
        candidate = leading_score.group(2).strip()
        cand_lower = candidate.lower()
        if cand_lower in _BENCHMARK_CANONICAL or cand_lower in {
            bm.lower() for bm in _BENCHMARK_CANONICAL.values()
        }:
            stripped = candidate

    # Strip trailing score-like suffixes: "GPQA-Diamond 62.1" → "GPQA-Diamond"
    # Also catches small decimals like 0.9, 1.8, 5.3 that leak from table cells.
    # Excludes version strings (preceded by 'v') and 4-digit years.
    stripped_score = re.sub(r'\s+(?<!v)(\d{1,3}\.\d{1,4})$', '', stripped)
    if stripped_score != stripped:
        test_lower = stripped_score.strip().lower()
        # If prefix is a non-benchmark, let it through to be rejected below
        if test_lower in _NON_BENCHMARK_NAMES:
            stripped = stripped_score.strip()
        elif test_lower in _BENCHMARK_CANONICAL or test_lower in {
            bm.lower() for bm in _BENCHMARK_CANONICAL.values()
        }:
            stripped = stripped_score.strip()
        else:
            # Also accept if it looks like a known canonical value
            for canon in _BENCHMARK_CANONICAL.values():
                if test_lower == canon.lower():
                    stripped = stripped_score.strip()
                    break

    # Strip Docling category prefixes: "Commonsense Understanding.Winogrande" → "Winogrande"
    if "." in stripped:
        prefix, _, suffix = stripped.rpartition(".")
        if prefix.replace("_", " ").strip().lower() in _TABLE_CATEGORY_PREFIXES:
            stripped = suffix.strip()

    # Also strip space-separated (or underscore-separated) category prefixes:
    # "Alignment Tasks Arena-Hard" → "Arena-Hard"
    # "Text_MATH-500" → "MATH-500"
    lowered_tmp = stripped.lower()
    for cat in sorted(_TABLE_CATEGORY_PREFIXES, key=len, reverse=True):
        # Check "cat & remainder" FIRST to avoid "math " matching "math & text ..."
        if lowered_tmp.startswith(cat + " & "):
            stripped = stripped[len(cat) + 3:].strip()
            # After stripping "Math & ", check if the remainder starts with another category
            lowered_tmp2 = stripped.lower()
            for cat2 in sorted(_TABLE_CATEGORY_PREFIXES, key=len, reverse=True):
                if lowered_tmp2.startswith(cat2 + " ") or lowered_tmp2.startswith(cat2 + "_"):
                    stripped = stripped[len(cat2):].lstrip("_ ").strip()
                    break
            break
        if lowered_tmp.startswith(cat + " "):
            stripped = stripped[len(cat):].strip()
            break
        if lowered_tmp.startswith(cat + "_"):
            stripped = stripped[len(cat):].lstrip("_").strip()
            break

    # Strip leading '& ' again (may appear after category prefix stripping)
    stripped = re.sub(r'^[&]\s*', '', stripped).strip()

    # Second pass: strip trailing score suffix AFTER category prefix removal.
    # "MATH-500 83.9" → "MATH-500" (category prefix "Text " was stripped above)
    # Also catches small decimals: "WritingBench 5.18" → "WritingBench"
    stripped_pass2 = re.sub(r'\s+(\d{1,3}(?:\.\d{1,4})?)$', '', stripped)
    if stripped_pass2 != stripped:
        # Don't strip if what remains is empty or if suffix looks like a version
        # Check: is the stripped suffix preceded by 'v'? If so, keep it.
        suffix_match = re.search(r'\s+(\d{1,3}(?:\.\d{1,4})?)$', stripped)
        prefix_char = stripped[suffix_match.start() - 1] if suffix_match and suffix_match.start() > 0 else ''
        if prefix_char.lower() != 'v' and len(stripped_pass2.strip()) >= 2:
            stripped = stripped_pass2.strip()
        else:
            stripped = stripped.strip()
    else:
        stripped = stripped.strip()

    lowered = stripped.lower()

    if lowered in _NON_BENCHMARK_NAMES:
        return None

    # Reject concatenated multi-benchmark names:
    # "MMLU-Redux GPQA-Diamond" → should be filtered
    # "MMMLU 14 languages MT-AIME2024" → concatenated
    # "C-Eval 82.2 LiveBench" → concatenated
    # Heuristic: if the string contains 2+ known benchmark names, it's corrupt
    canonical_names_lower = {v.lower() for v in _BENCHMARK_CANONICAL.values()}
    found_benchmarks = []
    for cn in canonical_names_lower:
        if len(cn) >= 3 and cn in lowered:
            found_benchmarks.append(cn)
    if len(found_benchmarks) >= 2:
        # Sort by position - use the first (leftmost) benchmark
        found_benchmarks.sort(key=lambda b: lowered.index(b))
        canonical = _BENCHMARK_CANONICAL.get(found_benchmarks[0])
        if canonical:
            return canonical

    # Handle "IINCLUDE" → "INCLUDE" (doubled first letter)
    if lowered == "iinclude":
        return "INCLUDE"

    # Strip date suffixes from benchmarks: "LiveBench 2024-11-25" → "LiveBench"
    date_stripped = re.sub(r'\s+\d{4}[-/]\d{2}[-/]\d{2}$', '', stripped)
    if date_stripped != stripped:
        # Re-lookup with the date stripped
        canonical = _BENCHMARK_CANONICAL.get(date_stripped.lower())
        if canonical:
            return canonical
        return date_stripped

    canonical = _BENCHMARK_CANONICAL.get(lowered)
    if canonical:
        return canonical

    return stripped


def _normalize_model_name(name: str) -> str:
    """Normalize model name casing for consistency.

    - Uppercases parameter-size suffixes: "27b" → "27B"
    - Normalizes known casing: "gemma-3-27b-it" → "Gemma-3-27B-IT"
    - Strips score-like data that got concatenated from merged table cells
    """
    # Strip control characters
    name = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', ' ', name)
    # Clean up model names with score data concatenated from merged cells:
    # "Qwen3-1.7B (Non-thinking)  53.0 (Thinking)" → "Qwen3-1.7B"
    # Pattern: model_name + mode_label + score + mode_label
    cleaned = re.sub(
        r'\s*\((?:Non-[Tt]hinking|[Tt]hinking)\)\s+\d+\.?\d*\s*\((?:Non-[Tt]hinking|[Tt]hinking)\)\s*$',
        '', name
    ).strip()
    if cleaned:
        name = cleaned
    # Also handle: "ModelName (Non-thinking) (Thinking)" without score
    # but keep single "(Thinking)" or "(Non-Thinking)" as valid suffixes
    # Standardize date formats in model names: GPT-4o-2024-1120 → GPT-4o-2024-11-20
    name = re.sub(
        r'(\d{4})-(\d{4})$',
        lambda m: f"{m.group(1)}-{m.group(2)[:2]}-{m.group(2)[2:]}" if len(m.group(2)) == 4 and int(m.group(2)[:2]) <= 12 else m.group(0),
        name,
    )
    # Normalize unicode multiplication sign → ASCII 'x' (Mixtral 8×22B → 8x22B)
    name = name.replace('\u00d7', 'x')
    # Uppercase parameter-size suffixes: digit(s) + 'b' at word boundary
    name = re.sub(r'(\d+)([bB])\b', lambda m: m.group(1) + 'B', name)
    # Normalize LLaMA → Llama (Meta renamed the family in v2+)
    name = re.sub(r'\bLLaMA\b', 'Llama', name)
    # Apply canonical model aliases for cross-paper consistency
    alias = _MODEL_ALIASES.get(name.lower().strip())
    if alias:
        # Preserve any suffix (e.g. date) that was already attached
        name = alias
    return name


# Model families known to support thinking/non-thinking mode.  Models
# from other families appearing in a "Non-Thinking" section of a table
# are baselines — don't tag them with reasoning_mode.
_THINKING_MODE_FAMILIES: set[str] = {
    "qwen3", "qwen2.5", "qwq", "deepseek", "deepseek-r1", "o1", "o3", "o4",
    "gemini-2.5", "gemini2.5", "claude-3.7",
}


def _should_tag_thinking_mode(model_name: str) -> bool:
    """Return True if *model_name* is from a family that supports
    thinking/non-thinking mode selection."""
    m = model_name.lower().strip()
    return any(fam in m for fam in _THINKING_MODE_FAMILIES)


def _model_name_matches(model_name: str, override_pattern: str) -> bool:
    """Return True if *model_name* belongs to the family described by
    *override_pattern*.

    Handles both directions:
      - "Qwen3-32B" matches override pattern "Qwen3"
      - "Qwen3" matches model name "Qwen3-235B-A22B"

    Also handles thinking-mode suffixes by stripping them before comparison.
    """
    if not model_name or not override_pattern:
        return False

    _THINKING_QUALIFIERS = ("(thinking)", "(non-thinking)", "thinking", "non-thinking")
    m = model_name.lower().strip()
    for sfx in _THINKING_QUALIFIERS:
        if m.endswith(sfx):
            m = m[: -len(sfx)].strip()
            break

    p = override_pattern.lower().strip()
    # Strip thinking qualifiers from the override pattern too —
    # e.g. an LLM might emit "Qwen3 thinking" as a model_name override.
    for sfx in _THINKING_QUALIFIERS:
        if p.endswith(sfx):
            p = p[: -len(sfx)].strip()
            break

    return (
        m == p
        or m.startswith(p)
        or p.startswith(m)
        or p in m
        or m in p
    )


def _merge_key(model: str, benchmark: str) -> tuple[str, str]:
    """Build a normalized (model, benchmark) key for Phase 5 merge.

    Applies the same normalization that ``_convert_model`` does so that
    LLM-emitted names like "GPQA-Diamond" and table-parsed names like
    "GPQA Diamond" resolve to the same merge key.
    """
    # Normalise stray spaces before hyphens
    m = re.sub(r'\s+-', '-', model).strip()
    m = _normalize_model_name(m)
    b = re.sub(r'\s+-', '-', benchmark).strip()
    b = _normalize_aime(b)
    nb = _normalize_benchmark(b)
    if nb is not None:
        b = nb
    return (m.lower(), b.lower())


# AIME variants: normalize "AIME'24" → "AIME 2024", "AIME'25" → "AIME 2025"
def _normalize_aime(name: str) -> str:
    """Expand AIME short-year references."""
    def _expand(m: re.Match) -> str:
        prefix = m.group(1)
        yy = int(m.group(2))
        yyyy = 2000 + yy if yy < 100 else yy
        return f"{prefix} {yyyy}"
    return re.sub(r"(AIME)[''\u2019]?(\d{2})\b", _expand, name)


# ---------------------------------------------------------------------------
# Expected score ranges — used to detect 0-1 vs 0-100 scale ambiguity.
# When a score <= 1.0 is found for a benchmark whose typical range is 40-100,
# the score is likely already a fraction and should NOT be divided by 100.
# ---------------------------------------------------------------------------
_EXPECTED_PERCENTAGE_RANGES: dict[str, tuple[float, float]] = {
    "mmlu": (25.0, 100.0),
    "gsm8k": (0.0, 100.0),
    "humaneval": (0.0, 100.0),
    "mbpp": (0.0, 100.0),
    "bbh": (0.0, 100.0),
    "hellaswag": (25.0, 100.0),
    "arc-c": (25.0, 100.0),
    "arc-e": (25.0, 100.0),
    "winogrande": (50.0, 100.0),
    "piqa": (50.0, 100.0),
    "truthfulqa": (0.0, 100.0),
    "ifeval": (0.0, 100.0),
    "gpqa": (25.0, 100.0),
    "mmlu-pro": (0.0, 100.0),
    "math": (0.0, 100.0),
    "math-500": (0.0, 100.0),
    "drop": (0.0, 100.0),
}


def _is_likely_fraction(bench: str, score: float) -> bool:
    """Return True if *score* looks like a 0-1 fraction for a %-scale benchmark."""
    if score > 1.0 or score < 0.0:
        return False
    bench_lower = bench.lower()
    for pattern, (lo, _hi) in _EXPECTED_PERCENTAGE_RANGES.items():
        if pattern in bench_lower and lo >= 1.0:
            return True
    return False


