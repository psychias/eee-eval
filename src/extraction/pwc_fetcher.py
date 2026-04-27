"""
pwc_fetcher.py — fetch Papers With Code evaluation results and write EEE records.

Data source
-----------
Papers With Code (PWC) published a structured CSV of benchmark results at:
    https://huggingface.co/datasets/felixleungsc/paperswithcode-data-evaluation-tables

Each row has:
    task_path, dataset, model_name, paper_url, metric_name, metric_value

This module:
  1. Downloads results.csv (cached locally for 7 days)
  2. Filters to rows that match our canonical benchmark list
  3. Requires an arxiv paper_url (verified paper-linked results only)
  4. Requires a valid numeric metric_value
  5. Writes EEE-format JSON records under data/papers_with_code/

Usage
-----
  python pwc_fetcher.py                   # fetch + write all
  python pwc_fetcher.py --refresh         # bypass cache
  python pwc_fetcher.py --dry-run         # fetch but don't write (show stats)
  python pwc_fetcher.py --list-benchmarks # list matched benchmarks

Requirements
------------
  pip install requests
"""
from __future__ import annotations

import csv
import io
import json
import os
import pathlib
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import requests

try:
    import openai as _openai
except ImportError:
    _openai = None  # type: ignore[assignment]

_ROOT     = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

DATA_DIR  = _ROOT / "data"
CACHE_DIR = pathlib.Path(".cache/pwc")
CACHE_TTL = 7 * 24 * 3600  # 7 days — PWC archive updates rarely

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
_HAIKU_MODEL = "anthropic/claude-haiku-4.5"

_RESULTS_CSV_URL = (
    "https://huggingface.co/datasets/felixleungsc/"
    "paperswithcode-data-evaluation-tables/resolve/main/results.csv"
)

_TIMEOUT = 120  # longer timeout for the 45 MB CSV

# ---------------------------------------------------------------------------
# Canonical benchmark mapping
#
# Maps PWC dataset names (case-insensitive) to our canonical benchmark names.
# Only results matching these benchmarks are imported.
# ---------------------------------------------------------------------------

# Exact matches (PWC dataset name → canonical name)
_BENCH_EXACT: dict[str, str] = {
    "mmlu":                   "MMLU",
    "mmlu (5-shot)":          "MMLU",
    "mmlu (5-Shot)":          "MMLU",
    "gsm8k":                  "GSM8K",
    "gsm8k (5-shot)":        "GSM8K",
    "gsm8k (8-shot)":        "GSM8K",
    "gsm8k (8-Shot)":        "GSM8K",
    "hellaswag":              "HellaSwag",
    "hellaswag (10-shot)":   "HellaSwag",
    "hellaswag (10-Shot)":   "HellaSwag",
    "truthfulqa":             "TruthfulQA",
    "truthfulqa (0-shot)":   "TruthfulQA",
    "arc (challenge)":        "ARC-Challenge",
    "arc-challenge":          "ARC-Challenge",
    "winogrande":             "WinoGrande",
    "humaneval":              "HumanEval",
    "mbpp":                   "MBPP",
    "ifeval":                 "IFEval",
    "bbh":                    "BBH",
    "gpqa":                   "GPQA",
    "math":                   "MATH",
    "math (4-shot)":         "MATH",
    "math500":                "MATH-500",
    "math 500":               "MATH-500",
    "mmlu-pro":               "MMLU-Pro",
    "mmlu-Pro":               "MMLU-Pro",
    "musr":                   "MuSR",
    "swe-bench":              "SWE-bench",
    "swe-bench verified":     "SWE-bench",
    "drop":                   "DROP",
    "natural questions":      "NaturalQuestions",
    "alpaca_eval":            "AlpacaEval 2.0",
    "alpacaeval 2.0":         "AlpacaEval 2.0",
    "mt-bench":               "MT-Bench",
    "humaneval+":             "HumanEval+",
    "mbpp+":                  "MBPP+",
    "bigcodebench":           "BigCodeBench",
}

# Build lower-case lookup
_BENCH_LOOKUP: dict[str, str] = {k.lower(): v for k, v in _BENCH_EXACT.items()}


def _normalize_benchmark(pwc_dataset: str) -> str | None:
    """Map a PWC dataset name to canonical benchmark name, or None if not canonical."""
    return _BENCH_LOOKUP.get(pwc_dataset.strip().lower())


def _extract_arxiv_id(paper_url: str) -> str | None:
    """Extract arxiv ID (without version suffix) from a paper URL."""
    m = re.search(r"arxiv\.org/abs/(\d{4}\.\d{4,5})", paper_url)
    if m:
        return m.group(1)
    return None


def _arxiv_id_to_date(arxiv_id: str) -> str:
    """Derive an approximate ISO date from an arxiv ID (YYMM.NNNNN → YYYY-MM-01)."""
    m = re.match(r"(\d{2})(\d{2})\.", arxiv_id)
    if m:
        yy, mm = int(m.group(1)), int(m.group(2))
        year = 2000 + yy
        if 1 <= mm <= 12:
            return f"{year}-{mm:02d}-01"
    return ""


def _parse_score(raw: str) -> float | None:
    """Parse a metric value string to float. Returns None on failure."""
    raw = raw.strip().rstrip("%")
    try:
        return float(raw)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Eval metadata extracted from model name
# ---------------------------------------------------------------------------

@dataclass
class EvalMeta:
    """Evaluation metadata parsed from a PWC model name suffix."""
    clean_name:        str = ""     # model name with suffix stripped
    shots:             str = ""     # e.g. "0", "5", "few"
    chain_of_thought:  str = ""     # "yes", "no", or ""
    self_consistency:  str = ""     # "yes" or ""
    temperature:       str = ""     # e.g. "0.0", "0.7"
    decoding:          str = ""     # "greedy", "sampling", ""
    samples:           str = ""     # e.g. "8", "40" (for maj@k)
    tool_use:          str = ""     # "code", "PAL", ""
    fine_tuned:        str = ""     # "yes" or ""
    prompt_info:       str = ""     # any extra prompt/eval detail
    source:            str = "regex" # "regex" or "llm"


def _parse_meta_regex(model_name: str) -> EvalMeta:
    """Extract evaluation metadata from model name using regex patterns."""
    meta = EvalMeta()

    # Extract parenthetical suffix
    m_paren = re.search(r"\(([^)]+)\)\s*$", model_name)
    # Also check for non-paren suffixes like "model(COT,Greedy)"
    m_paren2 = re.search(r"\(([^)]+)\)", model_name)
    suffix = ""
    if m_paren:
        suffix = m_paren.group(1)
        meta.clean_name = model_name[: m_paren.start()].strip()
    elif m_paren2:
        suffix = m_paren2.group(1)
        meta.clean_name = re.sub(r"\([^)]+\)", "", model_name).strip()
    else:
        meta.clean_name = model_name.strip()

    text = f"{suffix} {model_name}".lower()

    # --- Shots ---
    shot_m = re.search(r"(\d+)[- ]shot", text)
    if shot_m:
        meta.shots = shot_m.group(1)
    elif "zero-shot" in text or "zero shot" in text or "0-shot" in text:
        meta.shots = "0"
    elif re.search(r"few[- ]shot", text):
        meta.shots = "few"
        # Check for k=N
        k_m = re.search(r"k\s*=\s*(\d+)", text)
        if k_m:
            meta.shots = k_m.group(1)
    elif "one-shot" in text or "one shot" in text or "1-shot" in text:
        meta.shots = "1"

    # --- Chain-of-thought ---
    if re.search(r"\bcot\b|chain[- ]of[- ]thought|\bCoT\b", text):
        meta.chain_of_thought = "yes"
    elif re.search(r"w/o\s+code|without\s+code", text):
        meta.chain_of_thought = "yes"  # CoT without code typically means pure CoT

    # --- Self-consistency ---
    if re.search(r"self[- ]consist|\bsc\b|\bSC\b", text):
        meta.self_consistency = "yes"

    # --- Majority voting (maj@k) ---
    maj_m = re.search(r"maj(?:1)?@(\d+)", text)
    if maj_m:
        meta.samples = maj_m.group(1)
        meta.self_consistency = "yes"

    # --- Samples ---
    samp_m = re.search(r"(\d+)\s*samples?", text)
    if samp_m and not meta.samples:
        meta.samples = samp_m.group(1)

    # --- Decoding strategy ---
    if "greedy" in text:
        meta.decoding = "greedy"
    elif "sampling" in text or "sample" in text:
        meta.decoding = "sampling"

    # --- Temperature ---
    temp_m = re.search(r"temp(?:erature)?\s*=?\s*([0-9.]+)", text)
    if temp_m:
        meta.temperature = temp_m.group(1)
    elif meta.decoding == "greedy":
        meta.temperature = "0.0"  # greedy implies T=0

    # --- Tool/code use ---
    if re.search(r"\bPAL\b|\bpal\b", text):
        meta.tool_use = "PAL"
    elif re.search(r"\bTIR\b|tool[- ]integrated", text):
        meta.tool_use = "TIR"
    elif re.search(r"w/\s*code|with\s+code", text) and "w/o" not in text:
        meta.tool_use = "code"
    elif re.search(r"\bLEVER\b", text):
        meta.tool_use = "LEVER"

    # --- Fine-tuned ---
    if re.search(r"fine[- ]?tun", text):
        meta.fine_tuned = "yes"

    # --- Prompt info (catch remaining signal) ---
    prompt_parts: list[str] = []
    if re.search(r"auto[- ]optimized", text):
        prompt_parts.append("auto-optimized prompting")
    if re.search(r"mc[ot]\b|\bmCoT\b", text):
        prompt_parts.append("multilingual CoT")
    if re.search(r"QA prompt", text, re.IGNORECASE):
        prompt_parts.append("QA prompt")
    if prompt_parts:
        meta.prompt_info = "; ".join(prompt_parts)

    meta.source = "regex"
    return meta


def _has_useful_meta(meta: EvalMeta) -> bool:
    """Check if regex found any metadata at all."""
    return bool(
        meta.shots or meta.chain_of_thought or meta.self_consistency
        or meta.temperature or meta.decoding or meta.samples
        or meta.tool_use or meta.fine_tuned or meta.prompt_info
    )


# ---------------------------------------------------------------------------
# LLM metadata extraction (Haiku via OpenRouter)
# ---------------------------------------------------------------------------

_LLM_META_CACHE: dict[str, EvalMeta] = {}  # model_name -> EvalMeta

_LLM_PROMPT = """Extract evaluation metadata from this LLM benchmark model name.
Return ONLY a JSON object with these fields (use empty string if unknown):
- clean_name: the model name without evaluation config suffixes
- shots: number of few-shot examples ("0", "5", "few", or "")
- chain_of_thought: "yes" or ""
- self_consistency: "yes" or ""
- temperature: e.g. "0.0", "0.7", or ""
- decoding: "greedy", "sampling", or ""
- samples: number of samples for voting, or ""
- tool_use: "code", "PAL", "TIR", or ""
- fine_tuned: "yes" or ""
- prompt_info: any other eval details, or ""

Model name: {model_name}

JSON:"""


def _extract_meta_llm_batch(
    model_names: list[str],
    batch_size: int = 20,
) -> dict[str, EvalMeta]:
    """Use Haiku to extract metadata for a batch of model names.

    Groups names into batches to minimize API calls.
    Returns {model_name: EvalMeta}.
    """
    if not OPENROUTER_API_KEY or _openai is None:
        return {}

    # Filter to names not already cached
    todo = [n for n in model_names if n not in _LLM_META_CACHE]
    if not todo:
        return {n: _LLM_META_CACHE[n] for n in model_names if n in _LLM_META_CACHE}

    client = _openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    # Process in batches
    for i in range(0, len(todo), batch_size):
        batch = todo[i : i + batch_size]
        # Build a single prompt with all names in the batch
        batch_prompt = (
            "Extract evaluation metadata from these LLM benchmark model names.\n"
            "Return a JSON array where each element corresponds to a model name.\n"
            "Each element should have these fields (use empty string if not found):\n"
            "- model_name: the original name (for matching)\n"
            "- clean_name: the base model name without evaluation config suffixes\n"
            "- shots: number of few-shot examples (\"0\", \"5\", \"few\", etc.)\n"
            "- chain_of_thought: \"yes\" or \"\"\n"
            "- self_consistency: \"yes\" or \"\"\n"
            "- temperature: e.g. \"0.0\", \"0.7\", or \"\"\n"
            "- decoding: \"greedy\", \"sampling\", or \"\"\n"
            "- samples: number of samples for majority voting, or \"\"\n"
            "- tool_use: \"code\", \"PAL\", \"TIR\", or \"\"\n"
            "- fine_tuned: \"yes\" or \"\"\n"
            "- prompt_info: any other evaluation details, or \"\"\n\n"
            "Model names:\n"
        )
        for j, name in enumerate(batch, 1):
            batch_prompt += f"{j}. {name}\n"
        batch_prompt += "\nJSON array:"

        try:
            resp = client.chat.completions.create(
                model=_HAIKU_MODEL,
                messages=[{"role": "user", "content": batch_prompt}],
                max_tokens=2048,
                temperature=0.0,
            )
            raw = resp.choices[0].message.content or ""
            # Extract JSON from response
            json_m = re.search(r"\[.*\]", raw, re.DOTALL)
            if json_m:
                parsed = json.loads(json_m.group(0))
                for entry in parsed:
                    if not isinstance(entry, dict):
                        continue
                    orig_name = entry.get("model_name", "")
                    # Match back to a batch name
                    matched_name = None
                    for bn in batch:
                        if bn == orig_name or bn.lower() == orig_name.lower():
                            matched_name = bn
                            break
                    if not matched_name:
                        # Try partial match
                        for bn in batch:
                            if orig_name.lower() in bn.lower() or bn.lower() in orig_name.lower():
                                matched_name = bn
                                break
                    if matched_name:
                        meta = EvalMeta(
                            clean_name=str(entry.get("clean_name", matched_name)),
                            shots=str(entry.get("shots", "")),
                            chain_of_thought=str(entry.get("chain_of_thought", "")),
                            self_consistency=str(entry.get("self_consistency", "")),
                            temperature=str(entry.get("temperature", "")),
                            decoding=str(entry.get("decoding", "")),
                            samples=str(entry.get("samples", "")),
                            tool_use=str(entry.get("tool_use", "")),
                            fine_tuned=str(entry.get("fine_tuned", "")),
                            prompt_info=str(entry.get("prompt_info", "")),
                            source="llm",
                        )
                        _LLM_META_CACHE[matched_name] = meta
        except Exception as e:
            print(f"    [pwc-llm] batch {i // batch_size + 1} error: {e}")
            continue

        # Brief pause between batches
        if i + batch_size < len(todo):
            time.sleep(0.5)

    # Return results for requested names
    return {n: _LLM_META_CACHE[n] for n in model_names if n in _LLM_META_CACHE}


def _extract_metadata(
    model_names: list[str],
    use_llm: bool = True,
) -> dict[str, EvalMeta]:
    """Extract metadata for model names. Regex first, LLM for ambiguous ones."""
    result: dict[str, EvalMeta] = {}
    llm_candidates: list[str] = []

    for name in model_names:
        meta = _parse_meta_regex(name)
        if _has_useful_meta(meta):
            result[name] = meta
        else:
            # Check if it looks like it MIGHT have metadata (has parentheses or
            # special suffixes) but regex couldn't parse it
            if re.search(r"\([^)]+\)", name):
                llm_candidates.append(name)
            else:
                result[name] = meta  # empty meta, no suffix to parse

    if llm_candidates and use_llm and OPENROUTER_API_KEY and _openai is not None:
        print(f"  [pwc] extracting metadata via LLM for {len(llm_candidates)} ambiguous names...")
        llm_results = _extract_meta_llm_batch(llm_candidates)
        for name in llm_candidates:
            if name in llm_results and _has_useful_meta(llm_results[name]):
                result[name] = llm_results[name]
            else:
                result[name] = _parse_meta_regex(name)  # fallback to empty regex

    return result


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PWCResult:
    """A single verified benchmark result from Papers With Code."""
    model_name:     str
    benchmark:      str     # canonical name
    metric_name:    str
    score:          float
    paper_url:      str
    arxiv_id:       str
    paper_title:    str = ""
    pwc_dataset:    str = ""   # original PWC dataset name
    task_path:      str = ""   # PWC task path
    meta:           EvalMeta | None = None  # extracted eval metadata


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------

def _download_results_csv(force_refresh: bool = False) -> str:
    """Download (or load from cache) the PWC results CSV. Returns file path."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "results.csv"

    if not force_refresh and cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age < CACHE_TTL:
            print(f"  [pwc] using cached results.csv ({age / 3600:.1f}h old)")
            return str(cache_path)

    print("  [pwc] downloading results.csv ...")
    token = os.environ.get("HF_TOKEN", "")
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    r = requests.get(_RESULTS_CSV_URL, headers=headers, timeout=_TIMEOUT)
    r.raise_for_status()

    cache_path.write_text(r.text, encoding="utf-8")
    print(f"  [pwc] downloaded {len(r.text):,} bytes")
    return str(cache_path)


def fetch_pwc_results(force_refresh: bool = False) -> list[PWCResult]:
    """
    Fetch PWC results, filtered to:
      1. Canonical benchmarks only
      2. Rows with arxiv paper_url (verified paper-linked)
      3. Valid numeric scores
    """
    csv_path = _download_results_csv(force_refresh=force_refresh)

    results: list[PWCResult] = []
    skipped_no_bench = 0
    skipped_no_arxiv = 0
    skipped_no_score = 0
    total = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            dataset = row.get("dataset", "")
            paper_url = row.get("paper_url", "")
            metric_value = row.get("metric_value", "")

            # 1. Must be a canonical benchmark
            bench = _normalize_benchmark(dataset)
            if bench is None:
                skipped_no_bench += 1
                continue

            # 2. Must have an arxiv paper URL
            arxiv_id = _extract_arxiv_id(paper_url)
            if arxiv_id is None:
                skipped_no_arxiv += 1
                continue

            # 3. Must have a valid numeric score
            score = _parse_score(metric_value)
            if score is None:
                skipped_no_score += 1
                continue

            # 4. Filter out metadata metrics (parameter counts, FLOPs, etc.)
            metric_name_raw = row.get("metric_name", "").strip()
            if re.search(r"param|size|flop|latency|throughput|memory",
                         metric_name_raw, re.IGNORECASE):
                skipped_no_score += 1
                continue

            # 5. Reject scores impossibly above 100 on percentage benchmarks
            if score > 100 and bench not in ("MT-Bench",):
                skipped_no_score += 1
                continue

            results.append(PWCResult(
                model_name=row.get("model_name", "").strip(),
                benchmark=bench,
                metric_name=row.get("metric_name", "").strip(),
                score=score,
                paper_url=paper_url.strip(),
                arxiv_id=arxiv_id,
                pwc_dataset=dataset.strip(),
                task_path=row.get("task_path", "").strip(),
            ))

    print(f"  [pwc] {total:,} total rows -> {len(results)} matched")
    print(f"         skipped: {skipped_no_bench} no-bench, "
          f"{skipped_no_arxiv} no-arxiv, {skipped_no_score} no-score")

    # --- Extract metadata from model names ---
    unique_names = list(set(r.model_name for r in results))
    use_llm = bool(OPENROUTER_API_KEY and _openai is not None)
    meta_map = _extract_metadata(unique_names, use_llm=use_llm)

    regex_hit = sum(1 for m in meta_map.values() if _has_useful_meta(m) and m.source == "regex")
    llm_hit = sum(1 for m in meta_map.values() if _has_useful_meta(m) and m.source == "llm")
    print(f"  [pwc] metadata extracted: {regex_hit} regex, {llm_hit} llm, "
          f"{len(unique_names) - regex_hit - llm_hit} no-meta")

    # Attach metadata to results (model name + dataset name merged)
    for r in results:
        model_meta = meta_map.get(r.model_name)
        # Also extract metadata from the original PWC dataset name
        dataset_meta = _parse_meta_regex(r.pwc_dataset) if r.pwc_dataset else None
        if model_meta and dataset_meta and _has_useful_meta(dataset_meta):
            # Merge: dataset metadata fills gaps in model metadata
            if not model_meta.shots and dataset_meta.shots:
                model_meta.shots = dataset_meta.shots
            if not model_meta.chain_of_thought and dataset_meta.chain_of_thought:
                model_meta.chain_of_thought = dataset_meta.chain_of_thought
            if not model_meta.temperature and dataset_meta.temperature:
                model_meta.temperature = dataset_meta.temperature
        elif dataset_meta and _has_useful_meta(dataset_meta) and not model_meta:
            model_meta = dataset_meta
        r.meta = model_meta

    return results


# ---------------------------------------------------------------------------
# Record writing (same schema as add_leaderboard_records.py)
# ---------------------------------------------------------------------------

_SOURCE_DIR  = "papers_with_code"
_SOURCE_NAME = "Papers With Code"
_SOURCE_ORG  = "Papers With Code"
_SOURCE_URL  = "https://paperswithcode.com"
_EVAL_LIB    = "unknown"


def _already_exists_cache() -> set[str]:
    """Build a set of existing (model_name, benchmark) keys in pwc data dir."""
    d = DATA_DIR / _SOURCE_DIR
    if not d.exists():
        return set()
    keys: set[str] = set()
    for f in d.rglob("*.json"):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
            mid = rec.get("model_info", {}).get("id", "")
            for er in rec.get("evaluation_results", []):
                bench = er.get("evaluation_name", "")
                if mid and bench:
                    keys.add(f"{mid}|{bench}")
        except Exception:
            pass
    return keys


def _infer_developer_from_model(model_name: str) -> str:
    """Best-effort developer inference from PWC model names."""
    name_lower = model_name.lower()
    patterns = [
        ("gpt", "openai"), ("o1", "openai"), ("o3", "openai"),
        ("claude", "anthropic"), ("gemini", "google"), ("gemma", "google"),
        ("llama", "meta-llama"), ("codellama", "meta-llama"),
        ("mistral", "mistralai"), ("mixtral", "mistralai"),
        ("qwen", "Qwen"), ("phi", "microsoft"),
        ("falcon", "tiiuae"), ("deepseek", "deepseek-ai"),
        ("yi-", "01-ai"), ("olmo", "allenai"),
        ("palm", "google"), ("command", "CohereForAI"),
        ("starcoder", "bigcode"), ("vicuna", "lmsys"),
        ("grok", "xai"), ("internlm", "internlm"),
    ]
    for pat, dev in patterns:
        if pat in name_lower:
            return dev
    return "unknown"


def _make_model_id(model_name: str) -> str:
    """Create a reasonable model_id from a PWC model name."""
    # Clean common suffixes
    clean = re.sub(r"\s*\(.*?\)\s*$", "", model_name).strip()
    dev = _infer_developer_from_model(model_name)
    slug = re.sub(r"[^a-zA-Z0-9._-]", "-", clean).strip("-")
    if dev != "unknown":
        return f"{dev}/{slug}"
    return f"pwc/{slug}"


def _record(
    model_name: str,
    results: list[dict],
    paper_url: str,
    arxiv_id: str,
) -> dict:
    """Build an EEE evaluation record from PWC data."""
    dev = _infer_developer_from_model(model_name)
    model_id = _make_model_id(model_name)
    ts = str(int(time.time()))
    eval_date = _arxiv_id_to_date(arxiv_id)

    return {
        "schema_version": "0.2.1",
        "evaluation_id": f"{_SOURCE_DIR}/{model_id.replace('/', '_')}/{ts}",
        **({"evaluation_timestamp": eval_date} if eval_date else {}),
        "retrieved_timestamp": ts,
        "source_metadata": {
            "source_name": _SOURCE_NAME,
            "source_type": "documentation",
            "source_organization_name": _SOURCE_ORG,
            "source_organization_url": _SOURCE_URL,
            "evaluator_relationship": "third_party",
            "additional_details": {
                "paper_url": paper_url,
                "arxiv_id": arxiv_id,
            },
        },
        "eval_library": {"name": _EVAL_LIB, "version": "unknown"},
        "model_info": {
            "name": model_name,
            "id": model_id,
            "developer": dev if dev != "unknown" else "",
        },
        "evaluation_results": [
            {
                "evaluation_name": r["bench"],
                **({"evaluation_timestamp": eval_date} if eval_date else {}),
                "source_data": {
                    "dataset_name": r["bench"],
                    "source_type": "url",
                    "url": [paper_url, _SOURCE_URL],
                },
                "metric_config": {
                    "metric_name": r.get("metric_name", "accuracy"),
                    "lower_is_better": False,
                    "score_type": "continuous",
                    "min_score": r.get("min_score", 0.0),
                    "max_score": r.get("max_score", 100.0),
                },
                "score_details": {
                    "score": r["score"],
                },
                "generation_config": {
                    "generation_args": {
                        **({
                            "shots": int(r["meta_shots"]) if isinstance(r["meta_shots"], str) and r["meta_shots"].isdigit() else (r["meta_shots"] if isinstance(r["meta_shots"], int) else None),
                        } if r.get("meta_shots") and str(r.get("meta_shots")) != "few" else {}),
                        **({
                            "temperature": float(r["meta_temperature"]),
                        } if r.get("meta_temperature") else {}),
                        **({
                            "chain_of_thought": r["meta_cot"] == "yes",
                        } if r.get("meta_cot") else {}),
                        **({
                            "prompt_template": r["meta_prompt_info"],
                        } if r.get("meta_prompt_info") else {}),
                    },
                    "additional_details": {
                        "source": _SOURCE_DIR,
                        "pwc_dataset": r.get("pwc_dataset", ""),
                        "pwc_task_path": r.get("task_path", ""),
                        **({
                            "shots_raw": r["meta_shots"],
                        } if r.get("meta_shots") else {}),
                        **({
                            "self_consistency": r["meta_sc"],
                        } if r.get("meta_sc") else {}),
                        **({
                            "decoding": r["meta_decoding"],
                        } if r.get("meta_decoding") else {}),
                        **({
                            "samples": r["meta_samples"],
                        } if r.get("meta_samples") else {}),
                        **({
                            "tool_use": r["meta_tool_use"],
                        } if r.get("meta_tool_use") else {}),
                        **({
                            "fine_tuned": r["meta_fine_tuned"],
                        } if r.get("meta_fine_tuned") else {}),
                        **({
                            "meta_source": r["meta_source"],
                        } if r.get("meta_source") else {}),
                    },
                },
            }
            for r in results
        ],
    }


def _guess_scale(benchmark: str, score: float) -> tuple[float, float]:
    """Guess the score scale based on benchmark and value."""
    # Percentage-based benchmarks (0-100)
    pct_benchmarks = {
        "MMLU", "GSM8K", "HellaSwag", "TruthfulQA", "ARC-Challenge",
        "WinoGrande", "HumanEval", "MBPP", "IFEval", "BBH", "GPQA",
        "MATH", "MATH-500", "MMLU-Pro", "MuSR", "HumanEval+", "MBPP+",
        "BigCodeBench", "SWE-bench", "DROP",
    }
    if benchmark in pct_benchmarks:
        if score <= 1.0:
            return (0.0, 1.0)
        return (0.0, 100.0)
    # MT-Bench: 1-10
    if benchmark == "MT-Bench":
        return (1.0, 10.0)
    # AlpacaEval: 0-100
    if "AlpacaEval" in benchmark:
        return (0.0, 100.0)
    # Default
    return (0.0, 100.0)


def write_pwc_records(results: list[PWCResult]) -> int:
    """Group results by (model, paper) and write EEE records. Returns count written."""
    existing = _already_exists_cache()

    # Group by (model_name, arxiv_id)
    groups: dict[tuple[str, str], list[PWCResult]] = {}
    for r in results:
        key = (r.model_name, r.arxiv_id)
        groups.setdefault(key, []).append(r)

    written = 0
    skipped = 0

    for (model_name, arxiv_id), group in groups.items():
        model_id = _make_model_id(model_name)

        # Check if any of the benchmarks already exist for this model
        all_exist = all(
            f"{model_id}|{r.benchmark}" in existing for r in group
        )
        if all_exist:
            skipped += 1
            continue

        result_dicts = []
        for r in group:
            min_s, max_s = _guess_scale(r.benchmark, r.score)
            rd: dict[str, Any] = {
                "bench": r.benchmark,
                "score": r.score,
                "min_score": min_s,
                "max_score": max_s,
                "metric_name": r.metric_name or "accuracy",
                "pwc_dataset": r.pwc_dataset,
                "task_path": r.task_path,
            }
            # Attach eval metadata if available
            if r.meta and _has_useful_meta(r.meta):
                if r.meta.shots:
                    rd["meta_shots"] = r.meta.shots
                if r.meta.chain_of_thought:
                    rd["meta_cot"] = r.meta.chain_of_thought
                if r.meta.self_consistency:
                    rd["meta_sc"] = r.meta.self_consistency
                if r.meta.temperature:
                    rd["meta_temperature"] = r.meta.temperature
                if r.meta.decoding:
                    rd["meta_decoding"] = r.meta.decoding
                if r.meta.samples:
                    rd["meta_samples"] = r.meta.samples
                if r.meta.tool_use:
                    rd["meta_tool_use"] = r.meta.tool_use
                if r.meta.fine_tuned:
                    rd["meta_fine_tuned"] = r.meta.fine_tuned
                if r.meta.prompt_info:
                    rd["meta_prompt_info"] = r.meta.prompt_info
                rd["meta_source"] = r.meta.source
            result_dicts.append(rd)

        rec = _record(
            model_name=model_name,
            results=result_dicts,
            paper_url=group[0].paper_url,
            arxiv_id=arxiv_id,
        )

        # Save
        dev = _infer_developer_from_model(model_name)
        slug = re.sub(r"[^a-zA-Z0-9._-]", "-", model_name).strip("-")
        out_dir = DATA_DIR / _SOURCE_DIR / (dev if dev != "unknown" else "other") / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"{uuid.uuid4()}.json"
        p.write_text(json.dumps(rec, indent=2), encoding="utf-8")

        # Update existing cache
        for r in group:
            existing.add(f"{model_id}|{r.benchmark}")

        written += 1

    return written


# ---------------------------------------------------------------------------
# High-level entry points
# ---------------------------------------------------------------------------

def run_pwc(force_refresh: bool = False) -> dict[str, int]:
    """Fetch and write all PWC results. Returns {source: count}."""
    results = fetch_pwc_results(force_refresh=force_refresh)
    n = write_pwc_records(results)

    with_shots = sum(1 for r in results if r.meta and r.meta.shots)
    with_cot = sum(1 for r in results if r.meta and r.meta.chain_of_thought)
    with_temp = sum(1 for r in results if r.meta and r.meta.temperature)

    print(f"\n  [pwc] {n} new records written "
          f"({len(results)} results from {len(set(r.arxiv_id for r in results))} papers)")
    if with_shots or with_cot or with_temp:
        print(f"  [pwc] metadata coverage: "
              f"shots={with_shots}/{len(results)}, "
              f"cot={with_cot}/{len(results)}, "
              f"temperature={with_temp}/{len(results)}")
    return {_SOURCE_DIR: n}


def list_benchmarks(force_refresh: bool = False) -> None:
    """Print matched benchmarks and their counts."""
    results = fetch_pwc_results(force_refresh=force_refresh)

    from collections import Counter
    bench_counts = Counter(r.benchmark for r in results)
    paper_counts = Counter()
    for r in results:
        paper_counts[r.benchmark] += 0  # init
    for r in results:
        paper_counts[r.benchmark] = len(set(
            rr.arxiv_id for rr in results if rr.benchmark == r.benchmark
        ))

    print(f"\n  Matched benchmarks ({len(bench_counts)}):")
    for bench, cnt in sorted(bench_counts.items(), key=lambda x: -x[1]):
        print(f"    {cnt:5d} results  {paper_counts[bench]:3d} papers  {bench}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch Papers With Code results and write EEE records.",
    )
    parser.add_argument("--refresh", action="store_true", help="Bypass cache.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch and show stats, but don't write records.")
    parser.add_argument("--list-benchmarks", action="store_true",
                        help="List matched benchmarks and their counts.")
    args = parser.parse_args()

    if args.list_benchmarks:
        list_benchmarks(force_refresh=args.refresh)
        return

    results = fetch_pwc_results(force_refresh=args.refresh)

    if args.dry_run:
        from collections import Counter
        bench_counts = Counter(r.benchmark for r in results)
        unique_papers = len(set(r.arxiv_id for r in results))
        unique_models = len(set(r.model_name for r in results))
        print(f"\n  DRY RUN — would write records for:")
        print(f"    {len(results)} results")
        print(f"    {unique_papers} unique papers")
        print(f"    {unique_models} unique models")
        print(f"    Benchmarks:")
        for bench, cnt in sorted(bench_counts.items(), key=lambda x: -x[1]):
            print(f"      {cnt:5d}  {bench}")
        return

    n = write_pwc_records(results)
    unique_papers = len(set(r.arxiv_id for r in results))
    print(f"\n  Written: {n} records ({len(results)} results, {unique_papers} papers)")


if __name__ == "__main__":
    main()
