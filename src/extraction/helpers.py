"""Pure helper functions — parsing, inference, coercion utilities.

No dependency on Docling or any heavy extraction class.
"""
from __future__ import annotations

import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests

# Ensure repo root + utils/ are on sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / 'utils') not in sys.path:
    sys.path.insert(0, str(_ROOT / 'utils'))

from helpers import get_developer  # utils/helpers.py
from eval_types import (
    EvalLibrary, EvaluatorRelationship, SourceMetadata, SourceDataUrl,
)
from src.converters import SCHEMA_VERSION as _SCHEMA_VERSION

from .constants import (
    _BENCHMARK_KEYWORDS_STRONG,
    _BENCHMARK_KEYWORDS_WEAK,
    _DEVELOPER_PATTERNS,
    _AMBIGUOUS_DEVELOPER_PATTERNS,
    _EVAL_FRAMEWORK_SIGNATURES,
    _FOOTNOTE_RE,
    _HF_AUTHOR_CACHE,
    _MODEL_ALIGNMENT_PATTERNS,
    _QUANTIZATION_PATTERNS,
)

# --- cell parsing (needed by docling_parser and table_parser) ---

def _clean_cell(value: str) -> str:
    """strip footnote/superscript markers and surrounding whitespace from *value*.

    Handles common decoration found in academic paper tables:
      - trailing markers: ``85.2†``, ``92*``, ``77.3‡``
      - leading markers:  ``*73.1``, ``†85.2``
      - inline superscript digits that pdfplumber sometimes preserves
        (e.g. ``LLaMA1 65B`` → ``LLaMA 65B`` is intentionally NOT stripped
         because superscripts on model names are less predictable; we only
         strip from pure numeric cells inside _parse_numeric).
    """
    value = _FOOTNOTE_RE.sub("", value).strip()
    value = re.sub(r'\s+-', '-', value)
    return value


# Regex to extract the primary base float before a variance symbol
# Matches: '6.84 +/-0.07', '6.38 \pm 0.07', '6.58±0.05', '85.2 +/- 1.3'
_VARIANCE_RE = re.compile(
    r"^\s*([+-]?\d+\.?\d*)\s*(?:[±]|\+/?-|\\pm)\s*\d+\.?\d*"
)


def _parse_numeric(value: str) -> float | None:
    """parse *value* as a float; return None if not parseable.

    Handles formats commonly found in paper tables:
      - plain floats/ints:  ``85.2``, ``92``
      - percentage suffix:  ``85.2%``
      - footnote markers:   ``85.2†``, ``*92.0``, ``73.1‡``
      - parenthesised:      ``(85.2)``  (often used for std-dev or N/A rows)
      - dash / em-dash:     ``-``, ``—``  → None  (missing value sentinel)
      - bold LaTeX artefacts that pdfplumber sometimes leaves: ``\\textbf{85}
      - std-dev / CI:       ``6.84 +/-0.07``, ``6.38 \\pm 0.07``, ``6.58±0.05``
    """
    v = value.strip()
    # explicit missing-value sentinels
    if v in ("-", "—", "–", "n/a", "N/A", "na", "NA", ""):
        return None
    # strip footnote / superscript markers
    v = _FOOTNOTE_RE.sub("", v).strip()
    # unwrap parentheses  e.g. "(85.2)"
    if v.startswith("(") and v.endswith(")"):
        v = v[1:-1].strip()
    # detect explicit percentage sign before stripping it
    has_pct = v.endswith("%")
    # strip trailing % and any remaining whitespace
    v = v.rstrip("%").strip()
    # strip common LaTeX bold/italic residue
    v = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", v)
    # extract base float before variance symbols (±, +/-, \pm)
    m = _VARIANCE_RE.match(v)
    if m:
        v = m.group(1)
    try:
        return float(v)
    except ValueError:
        return None


def _parse_numeric_with_pct(value: str) -> tuple[float | None, bool]:
    """Like _parse_numeric, but also returns whether a '%' sign was present.

    Returns (parsed_float, had_percent_sign).  When the input explicitly
    contains '%' (e.g. '1.0%'), the caller knows this is a *percentage*
    that should be normalised to 0-1 even if the numeric value is <= 1.
    """
    v = value.strip()
    if v in ("-", "—", "–", "n/a", "N/A", "na", "NA", ""):
        return None, False
    v = _FOOTNOTE_RE.sub("", v).strip()
    if v.startswith("(") and v.endswith(")"):
        v = v[1:-1].strip()
    has_pct = v.endswith("%")
    v = v.rstrip("%").strip()
    v = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", v)
    m = _VARIANCE_RE.match(v)
    if m:
        v = m.group(1)
    try:
        return float(v), has_pct
    except ValueError:
        return None, False


def _is_separator(text: str) -> bool:
    """return True if *text* looks like a table separator or section header.

    Matches:
      - rows of dashes / equals / underscores used as visual dividers
      - single-dot or ellipsis cells that pdfplumber extracts from ruling lines
    """
    stripped = text.strip()
    if not stripped:
        return True
    return bool(re.match(r"^[-=_.…\s]+$", stripped))


# ---------------------------------------------------------------------------
# helpers — arXiv metadata and eval-library detection
# ---------------------------------------------------------------------------




# --- coercion helpers ---

def _coerce_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    return None


def _infer_model_metadata(model_name: str) -> dict[str, str | None]:
    parameter_count = None
    match = re.search(r"(\d+(?:\.\d+)?)([BM])(?:-A(\d+(?:\.\d+)?)[BM])?", model_name, re.IGNORECASE)
    if match:
        parameter_count = match.group(0).upper().replace(" ", "")

    model_alignment = None
    for pattern, alignment in _MODEL_ALIGNMENT_PATTERNS:
        if pattern.search(model_name):
            model_alignment = alignment
            break

    quantization = None
    for pattern, label in _QUANTIZATION_PATTERNS:
        if pattern.search(model_name):
            quantization = label
            break

    return {
        "parameter_count": parameter_count,
        "model_alignment": model_alignment,
        "quantization": quantization,
    }




# --- inference helpers ---

def _infer_metric_name(benchmark: str) -> str | None:
    lowered = benchmark.lower()
    if "pass@" in lowered:
        match = re.search(r"pass@\d+", lowered)
        return match.group(0) if match else "pass@k"
    if "exact match" in lowered or lowered.endswith(" em"):
        return "exact_match"
    if "f1" in lowered:
        return "f1"
    if "bleu" in lowered:
        return "bleu"
    if "rouge" in lowered:
        return "rouge"
    # Judge-scored benchmarks
    if "arena-hard" in lowered or "arena hard" in lowered:
        return "win_rate"
    if "mt-bench" in lowered or "mt bench" in lowered:
        return "judge_score"
    if "alpacaeval" in lowered or "alpaca_eval" in lowered or "alpaca eval" in lowered:
        return "win_rate"
    if "wildbench" in lowered:
        return "judge_score"
    if "judgebench" in lowered:
        return "judge_score"

    # Elo/rating benchmarks
    if "chatbot arena" in lowered or "lmsys" in lowered:
        return "elo"
    # Generic "arena" that isn't arena-hard/chatbot-arena
    if "arena" in lowered:
        return "win_rate"

    if "accuracy" in lowered or any(keyword in lowered for keyword in _BENCHMARK_KEYWORDS_STRONG):
        return "accuracy"
    # Default to "accuracy" for benchmarks not matching any special pattern.
    # The specific cases above already handle pass@k, f1, bleu, rouge,
    # judge-scored (MT-Bench, AlpacaEval), and Elo (Chatbot Arena).
    # Everything else in eval papers is effectively accuracy / correctness.
    return "accuracy"


def _infer_dataset_split(benchmark: str) -> str | None:
    lowered = benchmark.lower()
    for split in ("test", "validation", "dev", "train"):
        if re.search(rf"\b{split}\b", lowered):
            return split
    return None


def _infer_sample_size(benchmark: str) -> int | None:
    lowered = benchmark.lower()
    if "math-500" in lowered:
        return 500
    match = re.search(r"\b([1-9]\d{2,4})\b", lowered)
    if match and "aime" not in lowered and "2024" not in lowered and "2025" not in lowered:
        return int(match.group(1))
    return None


def _infer_dataset_split_for_items(items: list[dict[str, Any]]) -> str | None:
    for item in items:
        split = _infer_dataset_split(str(item.get("benchmark", "")))
        if split is not None:
            return split
    return None


def _infer_sample_size_for_items(items: list[dict[str, Any]]) -> int | None:
    for item in items:
        sample_size = _infer_sample_size(str(item.get("benchmark", "")))
        if sample_size is not None:
            return sample_size
    return None


def _normalise_arxiv_id(source: str) -> str:
    """extract the arXiv ID from *source* (path or ID string)."""
    p = Path(source)
    if p.exists() and p.suffix == ".pdf":
        return p.stem
    return source.strip()


def _make_eval_name(arxiv_id: str) -> str:
    """create a filesystem-safe eval_name from an arXiv ID."""
    safe = re.sub(r"[^a-zA-Z0-9_\-.]", "_", arxiv_id)
    return f"papers_{safe}"


def _make_source_metadata(arxiv_id: str, title: str = "") -> SourceMetadata:
    """build SourceMetadata for an academic paper."""
    return SourceMetadata(
        source_name=title if title else f"arXiv:{arxiv_id}",
        source_type="documentation",
        source_organization_name="arXiv",
        source_organization_url=f"https://arxiv.org/abs/{arxiv_id}",
        evaluator_relationship=EvaluatorRelationship.third_party,
    )


def _infer_developer(model_name: str, *, use_hf_api: bool = True) -> str:
    """infer the developer org from *model_name*.

    Resolution order:
    1. If the name is already in ``org/model`` HF format, return the org.
    2. Match against the local ``_DEVELOPER_MAP`` (startswith / substring).
    3. Try the HuggingFace Hub ``/api/models/{model_name}`` endpoint to read
       the ``author`` field.  Results are cached in ``_HF_AUTHOR_CACHE`` so
       repeated lookups within the same run incur only one HTTP request each.
    4. Delegate to the shared ``get_developer()`` helper as a final fallback.
    """
    if "/" in model_name:
        return model_name.split("/")[0]

    lower = model_name.lower()
    # Iterate the pre-sorted patterns tuple (longest key first) so that
    # "gpt-j" / "command-r" etc. are matched before the bare "gpt" / "command"
    # prefixes that would otherwise shadow them.
    for pattern, dev in _DEVELOPER_PATTERNS:
        # For short patterns (≤ 3 chars like "yi", "t5", "o1") use word-
        # boundary matching to avoid false positives from substrings
        # (e.g. "City-7B" matching "yi", "rot5" matching "t5").
        if len(pattern) <= 3:
            if re.search(rf"\b{re.escape(pattern)}\b", lower):
                return dev
        elif lower.startswith(pattern) or f"-{pattern}" in lower or f" {pattern}" in lower:
            if pattern in _AMBIGUOUS_DEVELOPER_PATTERNS:
                print(
                    f"  [developer-inference] '{model_name}' matched ambiguous "
                    f"pattern '{pattern}' -> attributed to '{dev}'. "
                    f"Many community fine-tunes share this prefix; verify the "
                    f"attribution and use 'org/model' HF format if possible.",
                    file=sys.stderr,
                )
            return dev

    if use_hf_api:
        cached = _HF_AUTHOR_CACHE.get(model_name)
        if cached is not None:
            return cached
        try:
            resp = requests.get(
                f"https://huggingface.co/api/models/{model_name}",
                timeout=5,
                headers={"User-Agent": "EEE-pipeline/1.0"},
            )
            if resp.status_code == 200:
                author: str = resp.json().get("author", "") or ""
                if author:
                    _HF_AUTHOR_CACHE[model_name] = author
                    return author
        except Exception:  # noqa: BLE001  # network errors must not abort extraction
            pass

    result = get_developer(model_name)
    if result == "unknown":
        print(
            f"  [developer-inference] could not identify developer for {model_name!r} "
            f"\u2014 recorded as 'unknown'. Add a mapping to _DEVELOPER_MAP or use "
            f"HuggingFace 'org/model' format.",
            file=sys.stderr,
        )
    return result




# --- arXiv metadata + evaluator relationship ---

def _fetch_arxiv_title(arxiv_id: str) -> str:
    """fetch the paper title from the arXiv Atom API.

    Returns an empty string on any network or parse error so callers can
    safely fall back to the arXiv ID as a display name.
    """
    meta = _fetch_arxiv_metadata(arxiv_id)
    return meta.get("title", "")


def _fetch_arxiv_metadata(arxiv_id: str) -> dict[str, Any]:
    """Fetch title and author names from the arXiv Atom API.

    Returns ``{"title": str, "authors": list[str]}`` with author names
    as plain strings (e.g. ``["Hugo Touvron", "Louis Martin"]``).
    All fields default to empty on any network or parse error.
    """
    import xml.etree.ElementTree as ET

    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}&max_results=1"
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "EEE-pipeline/1.0"})
        if resp.status_code != 200:
            return {"title": "", "authors": []}
        root = ET.fromstring(resp.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entry = root.find("atom:entry", ns)
        if entry is None:
            return {"title": "", "authors": []}

        title_el = entry.find("atom:title", ns)
        title = " ".join(title_el.text.strip().split()) if title_el is not None and title_el.text else ""

        authors: list[str] = []
        for author_el in entry.findall("atom:author", ns):
            name_el = author_el.find("atom:name", ns)
            if name_el is not None and name_el.text:
                authors.append(name_el.text.strip())

        return {"title": title, "authors": authors}
    except Exception:  # noqa: BLE001 — network errors must not abort extraction
        return {"title": "", "authors": []}


# Map from developer identifiers to known author affiliations / org names.
# Used by _infer_evaluator_relationship to detect first-party evaluations.
_DEVELOPER_AFFILIATION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "meta-llama": ("meta", "facebook", "fair", "llama"),
    "meta": ("meta", "facebook", "fair", "llama"),
    "openai": ("openai",),
    "google": ("google", "deepmind", "gemma", "gemini"),
    "google-deepmind": ("google", "deepmind", "gemma", "gemini"),
    "anthropic": ("anthropic", "claude"),
    "mistralai": ("mistral",),
    "qwen": ("alibaba", "qwen", "qwq", "tongyi"),
    "microsoft": ("microsoft", "phi-"),
    "deepseek-ai": ("deepseek",),
    "allenai": ("allen", "ai2"),
}


def _infer_evaluator_relationship(
    model_developer: str, paper_authors: list[str], paper_title: str = ""
) -> str:
    """Infer whether the evaluation is first_party or third_party.

    Checks whether any paper author's name or affiliation, or the paper
    title itself, matches the model's developer organization.
    """
    if not paper_authors or model_developer == "unknown":
        return "third_party"

    affiliation_keywords = _DEVELOPER_AFFILIATION_KEYWORDS.get(
        model_developer.lower(), _DEVELOPER_AFFILIATION_KEYWORDS.get(
            model_developer, (model_developer.lower(),)
        )
    )
    # Check author names and paper title for known affiliation keywords.
    search_text = " ".join(paper_authors).lower()
    if paper_title:
        search_text += " " + paper_title.lower()
    for kw in affiliation_keywords:
        if kw in search_text:
            return "first_party"

    return "third_party"




# --- sanity filters ---

def _is_plausible_result(item: dict[str, Any], known_benchmarks: set[str] | None = None) -> bool:
    """Return False for extraction artefacts that should be dropped.

    Guards against:
    - Citation years parsed as scores (e.g. 2021.0, 2024.0)
    - Model names that are actually citation strings
    - Scores that are implausibly large for any known benchmark
    - Negative scores on benchmarks where lower_is_better is False
    - Model names that are actually benchmark names (from transposed tables)
    """
    model = str(item.get("model", ""))
    score = item.get("score")
    benchmark = str(item.get("benchmark", ""))

    # reject model names containing citation markers
    if re.search(r"et\s+al\.?", model, re.IGNORECASE):
        return False
    # reject model names longer than 60 chars (citation/description strings)
    if len(model) > 60:
        return False
    # reject model names that contain embedded scores (mis-parsed table rows)
    # e.g. "ZeroSCROLLS.SQuALITY 15.3 ±7.9 16.4 ±8.1 15.4 ±7.9 13.2 ±7.4"
    if re.search(r"±|\\pm|\+/-", model):
        return False
    # reject model names with 3+ numeric tokens (score values, not "Llama 3 70B")
    numeric_tokens = sum(1 for t in model.split() if re.fullmatch(r"-?\d+\.?\d*%?", t))
    if numeric_tokens >= 3:
        return False
    # reject model names ending with trailing score-like decimal numbers
    # from collapsed table reconstruction (e.g. "Qwen3-14B (Thinking) 88.6 87.3")
    model_tokens = model.split()
    if len(model_tokens) >= 2:
        trailing_decimals = 0
        for t in reversed(model_tokens):
            cleaned = t.rstrip("-")
            if re.fullmatch(r"-?\d+\.\d+", cleaned):
                trailing_decimals += 1
            else:
                break
        # 2+ trailing decimals always garbage; 1 trailing decimal >= 10 is a score
        if trailing_decimals >= 2:
            return False
        if trailing_decimals == 1:
            try:
                val = float(model_tokens[-1].rstrip("-"))
                if val >= 10.0:
                    return False
            except ValueError:
                pass
    # reject model names that start with a known benchmark keyword
    # (from transposed tables where benchmark names end up as model names)
    model_lower = model.lower().strip()
    all_bench = _BENCHMARK_KEYWORDS_STRONG | (known_benchmarks or set())
    for kw in all_bench:
        if len(kw) < 3:
            continue
        if model_lower == kw or model_lower.startswith(kw + " ") or model_lower.startswith(kw + "-"):
            return False
    # reject purely numeric model names (e.g. "33.1" from mis-parsed cells)
    if re.fullmatch(r"-?\d+\.?\d*%?", model.strip()):
        return False
    # reject single-word generic labels that aren't model names
    _NON_MODEL_ENTITIES = {
        "base", "model", "average", "avg", "mean", "overall", "total",
        "human", "humans", "human expert", "human performance",
        "random", "random baseline", "majority", "majority vote",
        "expert", "oracle", "upper bound", "lower bound",
        "chance", "random chance", "ceiling", "floor",
    }
    if model.strip().lower() in _NON_MODEL_ENTITIES:
        return False
    # reject model names that are a known model family name without a size
    # indicator (e.g. "Llama 3" without "8B"/"70B"/"405B") — these are
    # typically header or caption artefacts, not specific model evaluations.
    _FAMILY_NAMES_NEEDING_SIZE = {
        "llama 3", "llama 3.1", "llama 3.2", "llama 3.3",
        "llama 2", "llama", "gpt-4", "gpt-3", "gpt",
        "gemma", "phi", "falcon", "bloom", "opt", "pythia",
        "mpt", "olmo", "yi", "internlm", "baichuan",
        "qwen3", "qwen2.5", "mistral",
    }
    _has_size = bool(re.search(r'\d+[bBmM]\b', model)) or bool(re.search(r'\d{2,}[bBkKmM]', model))
    if model_lower in _FAMILY_NAMES_NEEDING_SIZE and not _has_size:
        return False
    # reject prose-like model names (e.g. "the flagship model Qwen3-235B-A22B")
    if model_lower.startswith("the "):
        return False
    # reject training-method / ablation labels that aren't model names
    _NON_MODEL_LABELS = {
        "off-policy distillation", "on-policy distillation",
        "supervised fine-tuning", "reinforcement learning",
        "rejection sampling", "direct preference optimization",
        "self-play", "best-of-n", "baseline",
    }
    if model_lower in _NON_MODEL_LABELS:
        return False
    # reject merged model names: if two space-separated tokens each start
    # with a different known developer prefix, it's likely two models
    # concatenated from a table (e.g. "OpenAI-o1 DeepSeek-R1").
    # BUT "DeepSeek-R1-Distill-Llama-70B" is a single hyphenated name and
    # should NOT be rejected — we only check space-separated word groups.
    _space_parts = model_lower.split()
    if len(_space_parts) >= 2:
        _part_devs: list[str] = []
        for _sp in _space_parts:
            _sp_prefix = _sp.split("-")[0]  # first hyphen-segment
            for _pat, _dev in _DEVELOPER_PATTERNS:
                if len(_pat) <= 3:
                    if _sp == _pat or _sp.startswith(_pat + "-") or _sp_prefix == _pat:
                        _part_devs.append(_dev)
                        break
                elif _sp.startswith(_pat) or _sp_prefix == _pat:
                    _part_devs.append(_dev)
                    break
        if len(set(_part_devs)) >= 2:
            return False
    # reject merged model names from the SAME developer: e.g.
    # "Qwen2.5-32B-Instruct  Qwen3-14B Qwen3-30B-A3B" — multiple distinct
    # models concatenated from a table row. Detect via:
    # (a) double-space artifact in the middle, or
    # (b) 3+ parameter-size indicators (e.g. "32B", "14B", "30B")
    if "  " in model:  # double space = cell merge artifact
        return False
    size_indicators = re.findall(r'\b\d+\.?\d*[BbMm]\b', model)
    if len(size_indicators) >= 3:
        return False
    # reject empty or whitespace-only model/benchmark names
    if not model.strip() or not benchmark.strip():
        return False

    # reject benchmark names that are primarily numeric / confidence intervals
    # e.g. "58.2 ±7.7  54.4 ±5.0" from mis-parsed table cells
    bench_stripped = re.sub(r"[\d\.\+\-±/\s×x]", "", benchmark)
    if len(bench_stripped) < 2:
        return False
    # reject benchmark names that contain programming language lists
    # (from mis-parsed MultiPL-E table headers)
    _PROG_LANG_TOKENS = {"c++", "c#", "java", "php", "javascript", "typescript",
                         "python", "ruby", "go", "rust", "swift", "shell", "bash",
                         "perl", "lua", "scala", "kotlin", "ts", "js", "cpp"}
    bench_words = set(benchmark.lower().replace("+", "plus").replace("#", "sharp").split())
    if len(bench_words & _PROG_LANG_TOKENS) >= 3:
        return False
    # Also catch raw programming-language benchmark names with special chars
    if re.search(r'C\+\+.*Java.*PHP|Java.*C\+\+.*Shell', benchmark):
        return False
    # reject benchmark names with trailing score-like numbers
    # e.g. "Indo European Sino Tibetan 89.2 86.3"
    bench_tokens = benchmark.strip().split()
    if len(bench_tokens) >= 3:
        trailing_nums = 0
        for bt in reversed(bench_tokens):
            if re.fullmatch(r"-?\d+\.?\d*", bt):
                trailing_nums += 1
            else:
                break
        if trailing_nums >= 2:
            return False

    if score is not None:
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            return False
        # scores that look like years (2018–2030) are almost certainly
        # citation years parsed from footnote/reference table rows
        if 1900 <= score_f <= 2100 and score_f == int(score_f):
            return False
        # implausibly large: no standard benchmark reports scores > 3000
        # (Chatbot Arena Elo caps around 1500 in practice)
        if abs(score_f) > 3000:
            return False

    return True


def _sanitize_extracted(
    items: list[dict[str, Any]],
    known_benchmarks: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Filter extracted items, dropping implausible results.

    Returns a new list with only plausible items; logs warnings for dropped
    items so the user can review.
    """
    clean: list[dict[str, Any]] = []
    dropped = 0
    for item in items:
        if _is_plausible_result(item, known_benchmarks=known_benchmarks):
            clean.append(item)
        else:
            dropped += 1

    # Drop models where ALL scores are exactly 0.0 — this almost always
    # means empty cells were parsed as zeros (common for base models in
    # instruct-focused tables, or columns that don't apply).
    from collections import defaultdict
    model_items: dict[str, list[dict]] = defaultdict(list)
    for item in clean:
        model_items[str(item.get("model", ""))].append(item)
    all_zero_models = set()
    for model, mitems in model_items.items():
        if len(mitems) >= 2 and all(float(it.get("score", 1)) == 0.0 for it in mitems):
            all_zero_models.add(model)
            dropped += len(mitems)
    if all_zero_models:
        clean = [it for it in clean if str(it.get("model", "")) not in all_zero_models]
        print(
            f"  [sanity] dropped {len(all_zero_models)} model(s) with all-zero scores "
            f"(likely empty cells): {all_zero_models}",
            file=sys.stderr,
        )
    if dropped:
        print(
            f"  [sanity] dropped {dropped} implausible result(s) "
            f"(citation-as-model, year-as-score, or out-of-range)",
            file=sys.stderr,


# --- small helpers ---

        )
    return clean


def _aggregate_extraction_confidence(items: list[dict]) -> str:
    """Return the lowest confidence tier across all extraction items for one model.

    Using the minimum (most pessimistic) tier ensures the field reflects the
    weakest evidence that contributed to this model's results, not an average
    that could mask a poorly-extracted table.  This matters for Track 1
    reviewers who need to know if *any* score came from a dubious table.
    """
    tier_rank = {"high": 2, "medium": 1, "low": 0, "llm": 1, "prose": 1}
    rank_to_tier = {2: "high", 1: "medium", 0: "low"}
    min_rank = 2
    min_tier = "high"
    for item in items:
        tier = item.get("_extraction_confidence", "low")
        rank = tier_rank.get(tier, 0)
        if rank < min_rank:
            min_rank = rank
            # preserve the original tier label (e.g. 'llm', 'prose')
            min_tier = tier
        elif rank == min_rank and tier not in rank_to_tier.values():
            # prefer specific provenance labels over generic ones
            min_tier = tier
    return min_tier


def _normalise_benchmark_key(benchmark: str) -> str:
    return re.sub(r"\s+", " ", benchmark.strip().lower())


def _lookup_protocol(
    protocol_map: dict[str, dict[str, str | int]], benchmark: str
) -> dict[str, str | int]:
    if not protocol_map:
        return {}

    norm = _normalise_benchmark_key(benchmark)

    # Pass 1: exact match (highest priority)
    for key, value in protocol_map.items():
        if key == "__default__":
            continue
        norm_key = _normalise_benchmark_key(key)
        if norm == norm_key:
            merged = dict(protocol_map.get("__default__", {}))
            merged.update(value)
            return merged

    # Pass 2: best substring match (prefer longest matching key to avoid
    # "mmlu" accidentally matching "mmlu-pro" before "mmlu")
    best_key: str | None = None
    best_value: dict[str, str | int] | None = None
    best_len = 0
    for key, value in protocol_map.items():
        if key == "__default__":
            continue
        norm_key = _normalise_benchmark_key(key)
        if norm in norm_key or norm_key in norm:
            # Prefer the key that is closest in length to the query
            # (i.e. shortest difference), which avoids "mmlu" matching
            # "mmlu-pro" when an exact "mmlu" entry exists.
            key_len = len(norm_key)
            if best_key is None or abs(len(norm) - key_len) < abs(len(norm) - best_len):
                best_key = key
                best_value = value
                best_len = key_len

    if best_value is not None:
        merged = dict(protocol_map.get("__default__", {}))
        merged.update(best_value)
        return merged

    return dict(protocol_map.get("__default__", {}))


def _stringify_detail_values(details: dict[str, Any]) -> dict[str, str]:
    return {key: str(value) for key, value in details.items() if value is not None}


def _extract_table_id(context: str) -> str | None:
    match = re.search(r"\btable\s+([A-Za-z0-9.-]+)", context, re.IGNORECASE)
    if match:
        return f"Table {match.group(1)}"
    return None


def _clean_caption(context: str) -> str | None:
    cleaned = " ".join(context.strip().split())
    return cleaned[:500] if cleaned else None

