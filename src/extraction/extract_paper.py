"""Extract evaluation metadata from general LLM papers (arxiv HTML).

Mitigations applied:
  1. Smart section filtering: relevant sections only, fallback = first+last 10k
  2. Expanded section keywords: benchmark, performance, comparison, ablation, training details
  3. Chunked extraction for long papers: split into chunks, extract from each, merge
  4. Title validation: check fetched paper title matches expected model name

Output: data/arxiv_extraction_general/{naive,llm}/<paper>/<model>/<benchmark>/<uuid>.json
        data/arxiv_extraction_general/summary.{json,csv}

Usage:
    python scripts/extract_arxiv_metadata_general.py [--clean] [--yes] [--verbose]
           [--paper-delay 0.5] [--chunk-delay 0.2]
"""
import argparse
import csv
import hashlib
import json
import logging
import os
import re
import shutil
import time
import uuid as _uuid
from datetime import date as _date
from pathlib import Path
from typing import Any, Optional

import requests
from pydantic import BaseModel, field_validator
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type,
)
from bs4 import BeautifulSoup

# ── Inline validation import ──────────────────────────────────────────
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.validation.validate_extractions import validate_record

# ── Logging ────────────────────────────────────────────────────────────
log = logging.getLogger("eee_extract")

# ── CLI defaults (overridden by argparse) ─────────────────────────────
PAPER_DELAY: float = 0.5

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
PAPER_LIST = ROOT / "general_llm_papers.txt"
OUT_DIR = ROOT / "data" / "arxiv_extraction_general"

# ── OpenRouter config ──────────────────────────────────────────────────
# API key must be set via OPENROUTER_API_KEY env var (or in .env).
# To override the CA bundle, set REQUESTS_CA_BUNDLE=/path/to/cert.pem
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
if not OPENROUTER_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY env var is not set. "
        "See .env.example for required configuration."
    )
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HAIKU_MODEL = "anthropic/claude-haiku-4.5"

FIELDS = ["n_shot", "temperature", "prompt_template", "eval_harness",
          "decoding_strategy", "seed"]

# Expanded section keywords
SECTION_KEYWORDS = [
    "experiment", "result", "evaluat", "setup", "setting",
    "method", "implementation", "detail", "appendix", "hyperparameter",
    "benchmark", "performance", "comparison", "ablation",
    "training detail", "inference", "generation", "decoding",
    "configuration", "reproduce",
]

# Max chars to send to Haiku per chunk
MAX_INPUT_CHARS = 100_000  # max chars to send to LLM in a single call

# ── Metric configuration lookup (2.1) ────────────────────────────────
# Maps normalized metric name → correct bounds and direction.
METRIC_CONFIG = {
    "accuracy":    {"lower_is_better": False, "min": 0.0,  "max": 100.0},
    "pass@1":      {"lower_is_better": False, "min": 0.0,  "max": 100.0},
    "pass@k":      {"lower_is_better": False, "min": 0.0,  "max": 100.0},
    "pass@10":     {"lower_is_better": False, "min": 0.0,  "max": 100.0},
    "pass@32":     {"lower_is_better": False, "min": 0.0,  "max": 100.0},
    "pass@100":    {"lower_is_better": False, "min": 0.0,  "max": 100.0},
    "em":          {"lower_is_better": False, "min": 0.0,  "max": 100.0},
    "exactmatch":  {"lower_is_better": False, "min": 0.0,  "max": 100.0},
    "exact_match": {"lower_is_better": False, "min": 0.0,  "max": 100.0},
    "f1":          {"lower_is_better": False, "min": 0.0,  "max": 100.0},
    "bleu":        {"lower_is_better": False, "min": 0.0,  "max": 100.0},
    "rouge":       {"lower_is_better": False, "min": 0.0,  "max": 100.0},
    "rougel":      {"lower_is_better": False, "min": 0.0,  "max": 100.0},
    "rouge1":      {"lower_is_better": False, "min": 0.0,  "max": 100.0},
    "average":     {"lower_is_better": False, "min": 0.0,  "max": 100.0},
    "perplexity":  {"lower_is_better": True,  "min": 1.0,  "max": None},
    "ppl":         {"lower_is_better": True,  "min": 1.0,  "max": None},
    "loss":        {"lower_is_better": True,  "min": 0.0,  "max": None},
    "bitsperbyte": {"lower_is_better": True,  "min": 0.0,  "max": None},
}

# Metrics where scores should be on 0-100 scale (2.5)
PERCENT_SCALE_METRICS = {
    "accuracy", "pass@1", "pass@k", "pass@10", "pass@32", "pass@100",
    "em", "exactmatch", "exact_match", "f1", "bleu", "rouge", "rougel",
    "rouge1", "average",
}


def _norm_metric(name: str) -> str:
    """Normalize metric name for lookup: lowercase, strip hyphens/spaces/underscores."""
    if not isinstance(name, str):
        return ""
    return re.sub(r"[\s\-_]+", "", name).lower().strip()


def _parse_score_range(score_range: str | None) -> tuple[float | None, float | None]:
    """Parse LLM-provided score_range like '0-100', '0-1', '1-inf' into (min, max)."""
    if not score_range or not isinstance(score_range, str):
        return None, None
    parts = re.split(r"[\-–—to ]+", score_range.strip())
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) != 2:
        return None, None
    try:
        lo = float(parts[0]) if parts[0].lower() not in ("inf", "-inf", "none") else None
        hi = float(parts[1]) if parts[1].lower() not in ("inf", "none") else None
        return lo, hi
    except ValueError:
        return None, None


def _get_metric_config(metric_name: str, llm_lower_is_better=None, llm_score_range=None) -> dict:
    """Return metric_config dict with correct bounds/direction.

    For known metrics, use hardcoded bounds. For unknown metrics, fall back
    to the LLM-provided lower_is_better and score_range values.
    """
    normed = _norm_metric(metric_name)
    # Try exact match, then prefix match for pass@N variants
    cfg = METRIC_CONFIG.get(normed)
    if not cfg and normed.startswith("pass@"):
        cfg = METRIC_CONFIG.get("pass@k")
    if not cfg and normed.startswith("rouge"):
        cfg = METRIC_CONFIG.get("rouge")
    if not cfg:
        # Fall back to LLM-provided values
        llm_min, llm_max = _parse_score_range(llm_score_range)
        lib = llm_lower_is_better if isinstance(llm_lower_is_better, bool) else False
        log.warning("Unknown metric '%s', using LLM-provided bounds (range=%s, lower_is_better=%s)",
                    metric_name, llm_score_range, lib)
        cfg = {"lower_is_better": lib, "min": llm_min or 0.0, "max": llm_max}
    return {
        "metric_name": metric_name,
        "lower_is_better": cfg["lower_is_better"],
        "score_type": "continuous",
        "min_score": cfg["min"],
        "max_score": cfg["max"],
    }


def _normalize_score(score, metric_name: str) -> float | None:
    """Normalize score to 0-100 scale for percentage metrics (2.5)."""
    if score is None:
        return None
    score = float(score)
    normed = _norm_metric(metric_name)
    if normed in PERCENT_SCALE_METRICS and 0 < score <= 1.0:
        log.warning("Score %s for '%s' appears 0-1 scaled, converting to 0-100", score, metric_name)
        score *= 100.0
    return score


def _same_org(paper_authors: str, model_family: str | None) -> bool:
    """Check if a model family matches the paper's org (2.2)."""
    if not model_family:
        return False
    a = paper_authors.lower().replace("-", "").replace(" ", "")
    f = model_family.lower().replace("-", "").replace(" ", "")
    return f in a or a.startswith(f)


# ── 3.3: Pydantic validation models for LLM output ───────────────────

class BenchmarkEntry(BaseModel):
    benchmark_name: str
    score: Optional[float] = None
    metric_name: Optional[str] = None
    lower_is_better: Optional[bool] = None
    score_range: Optional[str] = None
    n_shot: Optional[int] = None
    temperature: Optional[float] = None
    prompt_template: Optional[str] = None
    decoding_strategy: Optional[str] = None
    # Per-benchmark harness assignment. Only set if the paper explicitly
    # attributes THIS benchmark to a named harness. Generative tasks,
    # LLM-as-judge evals, proof assistants, code-execution sandboxes, and
    # internal pipelines should be left as None — even if the paper names a
    # harness for other benchmarks.
    eval_harness: Optional[str] = None

    @field_validator("n_shot", mode="before")
    @classmethod
    def coerce_nshot(cls, v):
        if isinstance(v, float):
            return int(v)
        return v

    @field_validator("score", mode="before")
    @classmethod
    def coerce_score(cls, v):
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                return None
        return v


class ModelEntry(BaseModel):
    model_name: str
    model_family: Optional[str] = None
    developer: Optional[str] = None
    is_baseline: bool = False
    benchmarks: list[BenchmarkEntry] = []


class GlobalConfig(BaseModel):
    eval_harness: Optional[str] = None
    eval_library_version: Optional[str] = None
    seed: Optional[int | list[int]] = None
    # Only set this to True if the paper explicitly says the named harness
    # was used for EVERY benchmark in the paper (e.g., "we evaluate all
    # benchmarks with lm-eval-harness"). If the paper names the harness only
    # for some benchmarks (a "we used X for the commonsense reasoning suite"
    # style attribution), set this to False and rely on per-benchmark
    # eval_harness fields. Default False — never blanket-apply a harness.
    eval_harness_applies_to_all: Optional[bool] = False


class LLMExtractionResult(BaseModel):
    global_config: GlobalConfig = GlobalConfig()
    models: list[ModelEntry] = []


def _validate_llm_result(cleaned: dict, arxiv_id: str) -> dict:
    """Validate cleaned LLM output against pydantic schema. Returns dict."""
    try:
        validated = LLMExtractionResult(**cleaned)
        return validated.model_dump()
    except Exception as e:
        log.warning("Validation error for %s: %s", arxiv_id, e)
        # Return as-is if validation fails — better than losing data
        return cleaned


# ── 3.4: Name normalization for merge dedup ───────────────────────────

def _norm_name(s: str) -> str:
    """Normalize model/benchmark name for dedup: lowercase, strip separators."""
    return re.sub(r"[\s\-_.]+", "", s).lower().strip()


# ═══════════════════════════════════════════════════════════════════════
# Step 1: Load papers
# ═══════════════════════════════════════════════════════════════════════

def load_papers(paper_list_path: Path | None = None) -> list[dict[str, str]]:
    """Parse general_llm_papers.txt and return list of dicts.
    Format: 'arxiv_id   — Title (Org, Authors)'
    """
    src = paper_list_path or PAPER_LIST
    try:
        text = src.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = src.read_text(encoding="cp1252")
    papers = []
    seen = set()
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith(("-", "SEARCH", "#")):
            continue
        m = re.match(r'(\d+\.\d+)\s*[—–-]+\s*(.+?)(?:\s*\(([^)]*)\))?\s*$', line)
        if not m:
            continue
        arxiv_id = m.group(1)
        title = m.group(2).strip()
        authors = m.group(3).strip() if m.group(3) else "unknown"
        if arxiv_id not in seen:
            seen.add(arxiv_id)
            papers.append({
                "title": title,
                "arxiv_id": arxiv_id,
                "authors": authors,
                "year": "20" + arxiv_id[:2],
                "model_name": title,
            })
    return papers


# ═══════════════════════════════════════════════════════════════════════
# Step 2: Fetch + parse HTML with title validation
# ═══════════════════════════════════════════════════════════════════════

def fetch_html(arxiv_id: str) -> str | None:
    base_id = re.sub(r"v\d+$", "", arxiv_id)
    url = f"https://arxiv.org/html/{base_id}"
    try:
        resp = requests.get(url, timeout=30, verify=False,
                            headers={"User-Agent": "EEE-Eval-Research/1.0"})
        if resp.status_code == 200 and "<html" in resp.text[:500].lower():
            return resp.text
        log.warning("%s: HTTP %d or not HTML", arxiv_id, resp.status_code)
    except Exception as e:
        log.error("%s: fetch failed: %s", arxiv_id, e)
    return None


def validate_title(html: str, expected_title: str) -> tuple[bool, str]:
    """Check if the fetched paper title matches the expected model.
    Returns (is_valid, actual_title)."""
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("title")
    actual_title = title_tag.get_text(strip=True) if title_tag else ""

    # Also check <h1> which often has the paper title
    h1 = soup.find("h1")
    h1_text = h1.get_text(strip=True) if h1 else ""

    # Extract key model/family words from expected title
    keywords = re.findall(r"[A-Z][a-zA-Z0-9.]+", expected_title)
    if not keywords:
        keywords = expected_title.lower().split()[:3]

    combined = (actual_title + " " + h1_text).lower()

    # Check if at least one key term appears
    for kw in keywords:
        if kw.lower() in combined:
            return True, actual_title or h1_text

    return False, actual_title or h1_text


def extract_sections(html: str) -> dict[str, str]:
    """Extract sections, using expanded keyword list."""
    soup = BeautifulSoup(html, "html.parser")
    body = soup.get_text(separator="\n", strip=True)

    sections = {}
    headings = soup.find_all(re.compile(r"^h[1-4]$", re.I))
    for h in headings:
        title = h.get_text(strip=True).lower()
        parts = []
        for sib in h.find_next_siblings():
            if sib.name and re.match(r"^h[1-4]$", sib.name, re.I):
                break
            parts.append(sib.get_text(separator="\n", strip=True))
        sections[title] = "\n".join(parts)

    relevant_keys = [
        k for k in sections
        if any(kw in k for kw in SECTION_KEYWORDS)
    ]
    relevant_text = "\n\n".join(
        f"=== {k} ===\n{sections[k]}" for k in relevant_keys
    )
    return {"full": body, "relevant": relevant_text, "sections": sections}


def get_extraction_text(sects: dict[str, str]) -> str:
    """Smart text selection: relevant sections > first+last 10k fallback."""
    if sects["relevant"]:
        return sects["relevant"]
    # Fallback: first 10k (abstract/intro) + last 10k (appendix/details)
    full = sects["full"]
    if len(full) <= 20_000:
        return full
    return full[:10_000] + "\n\n[...MIDDLE OMITTED...]\n\n" + full[-10_000:]


# ═══════════════════════════════════════════════════════════════════════
# Naive extraction (unchanged)
# ═══════════════════════════════════════════════════════════════════════

NSHOT_PAT = re.compile(r"(\d+)[- ]?shot|few[- ]?shot|zero[- ]?shot|(\d+)[- ]?example", re.I)
TEMP_PAT = re.compile(r"temperature\s*(?:of|=|:)?\s*([0-9]*\.?[0-9]+)", re.I)
HARNESS_PAT = re.compile(
    r"(lm[- _]eval(?:uation)?[- _]harness|eleuther|helm|big-?bench"
    r"|open[- ]?llm[- ]?leaderboard|eval-?plus|human-?eval"
    r"|simple-?evals|alpaca-?eval|mt-?bench"
    r"|evalverse|hai[- _]llm|catwalk|bigcode[- _]eval(?:uation)?[- _]harness"
    r"|lighteval|opencompass|vlmevalkit|unitxt"
    r"|internal\s+eval(?:uation)?(?:\s+\w+){0,3}?\s*(?:library|framework|tool|harness|pipeline)"
    r"|(?:internal|in-house|proprietary)\s+(?:evaluation|eval)\s+(?:library|framework|tool)"
    r"|our\s+internal\s+(?:eval|tool|framework|library))", re.I)
PROMPT_PAT = re.compile(
    r"(chain[- ]?of[- ]?thought|cot|zero[- ]?shot[- ]?cot"
    r"|few[- ]?shot|direct|standard prompt|instruction"
    r"|system prompt|prompt template)", re.I)
SCORING_PAT = re.compile(
    r"(log[- ]?likelihood|multiple[- ]?choice|exact[- ]?match"
    r"|pass@k|greedy|beam|sampling|majority[- ]?voting"
    r"|self[- ]?consistency|maj@)", re.I)
SEED_PAT = re.compile(r"(?:random\s+)?seed\s*(?:of|=|:)?\s*(\d+)", re.I)
SAMPLES_PAT = re.compile(r"(?:k\s*=\s*|n\s*=\s*|samples?\s*(?:=|:)\s*)(\d+)", re.I)


def naive_extract(text: str) -> dict:
    result = {f: None for f in FIELDS}
    shots = NSHOT_PAT.findall(text)
    if shots:
        values = set()
        for m in shots:
            if m[0]: values.add(int(m[0]))
            elif m[1]: values.add(int(m[1]))
        if "zero" in text.lower(): values.add(0)
        result["n_shot"] = sorted(values) if values else None
    temps = TEMP_PAT.findall(text)
    if temps:
        result["temperature"] = sorted(set(float(t) for t in temps))
    harnesses = HARNESS_PAT.findall(text)
    if harnesses:
        result["eval_harness"] = list(set(h.lower() for h in harnesses))
    prompts = PROMPT_PAT.findall(text)
    if prompts:
        result["prompt_template"] = list(set(p.lower() for p in prompts))
    seeds = SEED_PAT.findall(text)
    if seeds:
        result["seed"] = sorted(set(int(s) for s in seeds))
    if re.search(r"greedy|temperature\s*(?:=|of)\s*0", text, re.I):
        result["decoding_strategy"] = "greedy"
    elif re.search(r"beam\s*search|num_beams", text, re.I):
        result["decoding_strategy"] = "beam_search"
    elif re.search(r"nucleus|top[- _]?p", text, re.I):
        result["decoding_strategy"] = "nucleus"
    elif re.search(r"top[- _]?k", text, re.I):
        result["decoding_strategy"] = "top_k"
    return result


# ═══════════════════════════════════════════════════════════════════════
# LLM extraction with chunking
# ═══════════════════════════════════════════════════════════════════════

EXTRACTION_PROMPT = """\
You are an expert at reading ML papers. Extract evaluation results and configuration for ALL models evaluated in this paper.

The paper's primary model(s): {own_model}
Mark these (and their variants/sizes) as is_baseline=false.
Mark ALL other models as is_baseline=true.

Return a JSON object with:
1. "global_config" — shared evaluation settings across all models/benchmarks:
   - eval_harness: string or null. The named evaluation framework, ONLY if the paper explicitly names one. Use the EXACT VERBATIM name the paper gives the tool — INCLUDING when the paper renames its own fork or extension. Do NOT canonicalize a custom-named fork to its parent library. Examples: if the paper says "We release our Prompted Language Model Evaluation Harness", the value is "Prompted Language Model Evaluation Harness" (the fork's name), NOT "lm-evaluation-harness" (the parent). If the paper says "we use lm-evaluation-harness package (Gao et al., 2021)", the value is "lm-evaluation-harness" (the paper's actual wording). If the paper says "eval-harness [Gao et al., 2023]", use "eval-harness". Common tools: "lm-evaluation-harness", "Evalverse", "HAI-LLM", "Catwalk", "bigcode-evaluation-harness", "lighteval", "mteb". For internal/proprietary tools without a public name, use "internal". Use null if the paper says nothing about evaluation infrastructure.
   - eval_harness_applies_to_all: boolean. Set to true ONLY when the paper explicitly says the named harness was used for EVERY benchmark (e.g., "all evaluations were run with lm-eval-harness"). Set to false (the default) when the paper attributes the harness to only some benchmarks (e.g., "we used X for the commonsense reasoning suite") or when the paper names a harness in passing without specifying coverage. WHEN IN DOUBT, USE FALSE.
   - eval_library_version: string or null (version of the evaluation framework)
   - seed: integer or list (random seeds used)

2. "models" — an array of objects, one per model evaluated. Each model object has:
   - model_name: string (e.g., "Llama-3-8B", "GPT-4", "Qwen2-72B-Instruct")
   - model_family: string or null (e.g., "Llama 3", "GPT-4", "Qwen2")
   - developer: string or null (organization that created the model, e.g., "Meta", "OpenAI", "DeepSeek")
   - is_baseline: boolean (true=comparison model, false=paper's own model)
   - benchmarks: array of objects, one per benchmark. Each benchmark has:
     - benchmark_name: string (e.g., "MMLU", "GSM8K", "HumanEval")
     - score: number or null (the reported score/accuracy)
     - metric_name: string (e.g., "accuracy", "pass@1", "F1", "perplexity")
     - lower_is_better: boolean or null (true for perplexity/loss, false for accuracy/F1)
     - score_range: string or null (e.g., "0-100", "0-1", "1-inf")
     - n_shot: integer or null
     - temperature: float or null
     - prompt_template: string or null (e.g., "chain-of-thought", "direct")
     - decoding_strategy: string or null (e.g., "greedy", "nucleus sampling")
     - eval_harness: string or null. Per-benchmark harness assignment. Set to the named harness ONLY if the paper EXPLICITLY attributes THIS specific benchmark to that harness (e.g., "we use lm-eval-harness for MMLU"). DO NOT set this just because the paper names a harness somewhere — many papers use the named harness for some benchmarks but use different pipelines for others. **Use the EXACT VERBATIM fork name from the paper** — if the paper says "Prompted Language Model Evaluation Harness", that is the value (not "lm-evaluation-harness"). The benchmark-level eval_harness MUST match the global_config.eval_harness string when the benchmark is in the group covered by the named tool. **Group attributions count**: if the paper attributes the named harness to a specific group of benchmarks (e.g., "our open-access models", "the commonsense reasoning suite", "for chain-of-thought results", "across all aforementioned tasks"), set this field for every benchmark that fits the described group — not only the benchmarks named in the same sentence as the harness. Leave as null when the benchmark is run via any of:
       * a generative-task pipeline (machine translation like FLORES, summarization like XLSum)
       * an LLM-as-judge evaluation (e.g., GPT-4 win-rates, dolly open-ended)
       * a human-rater evaluation
       * a code-execution sandbox (HumanEval, MBPP) when the paper does not name the harness
       * a proof-assistant evaluation (Isabelle, Lean, Coq)
       * a majority-voting / self-consistency aggregation beyond the harness's default
       * a separate pipeline the paper describes (custom probes, MTEB embedding eval, HELM, etc.) that is distinct from the named harness
       * any benchmark for which the paper does not explicitly say (directly or via a group attribution) which tool produced the score

Include ALL models mentioned in results tables — both the paper's own models AND baselines/competitors.
Focus on EVALUATION SETUP, not training. Only extract what is EXPLICITLY stated.
Use null for fields where no information is found.

Example 1 — paper names a harness for ALL benchmarks (Sheared-LLaMA-style):

```json
{{
  "global_config": {{
    "eval_harness": "lm-evaluation-harness",
    "eval_harness_applies_to_all": true,
    "eval_library_version": null,
    "seed": null
  }},
  "models": [
    {{
      "model_name": "ExampleModel-7B",
      "model_family": "ExampleModel",
      "developer": "ExampleOrg",
      "is_baseline": false,
      "benchmarks": [
        {{
          "benchmark_name": "MMLU",
          "score": 68.4,
          "metric_name": "accuracy",
          "lower_is_better": false,
          "score_range": "0-100",
          "n_shot": 5,
          "temperature": null,
          "prompt_template": "direct",
          "decoding_strategy": "greedy",
          "eval_harness": "lm-evaluation-harness"
        }},
        {{
          "benchmark_name": "HellaSwag",
          "score": 75.2,
          "metric_name": "accuracy",
          "lower_is_better": false,
          "score_range": "0-100",
          "n_shot": 0,
          "temperature": null,
          "prompt_template": "direct",
          "decoding_strategy": "greedy",
          "eval_harness": "lm-evaluation-harness"
        }}
      ]
    }}
  ]
}}
```

Example 2 — paper names a harness for SOME benchmarks but not others (Aya-23-style, Nemotron-4-style):

```json
{{
  "global_config": {{
    "eval_harness": "lm-evaluation-harness",
    "eval_harness_applies_to_all": false,
    "eval_library_version": null,
    "seed": null
  }},
  "models": [
    {{
      "model_name": "ExampleModel-7B",
      "model_family": "ExampleModel",
      "developer": "ExampleOrg",
      "is_baseline": false,
      "benchmarks": [
        {{
          "benchmark_name": "MMLU",
          "score": 68.4,
          "metric_name": "accuracy",
          "n_shot": 5,
          "prompt_template": "direct",
          "decoding_strategy": "greedy",
          "eval_harness": "lm-evaluation-harness"
        }},
        {{
          "benchmark_name": "FLORES-200",
          "score": 32.1,
          "metric_name": "BLEU",
          "n_shot": 0,
          "eval_harness": null
        }},
        {{
          "benchmark_name": "Dolly Open-Ended (GPT-4 judge)",
          "score": 55.0,
          "metric_name": "win-rate",
          "eval_harness": null
        }}
      ]
    }}
  ]
}}
```

Example 3 — paper builds and releases its own renamed fork of a standard harness (BLOOM-style); fork name must be used verbatim:

```json
{{
  "global_config": {{
    "eval_harness": "Prompted Language Model Evaluation Harness",
    "eval_harness_applies_to_all": false,
    "eval_library_version": null,
    "seed": null
  }},
  "models": [
    {{
      "model_name": "ExampleModel-176B",
      "model_family": "ExampleModel",
      "developer": "ExampleOrg",
      "is_baseline": false,
      "benchmarks": [
        {{
          "benchmark_name": "WMT14-en-fr",
          "score": 28.4,
          "metric_name": "BLEU",
          "n_shot": 0,
          "eval_harness": "Prompted Language Model Evaluation Harness"
        }},
        {{
          "benchmark_name": "HumanEval",
          "score": 15.5,
          "metric_name": "pass@1",
          "n_shot": 0,
          "eval_harness": null
        }}
      ]
    }}
  ]
}}
```

Return ONLY valid JSON, no other text.

Paper text:
{text}"""


BENCH_FIELDS = ["n_shot", "temperature", "prompt_template", "decoding_strategy"]


def _clean_value(v: Any) -> Any:
    """Return None for null-like values."""
    if v is None:
        return None
    if isinstance(v, str) and v.strip().lower() in (
        "not found", "null", "n/a", "unknown", "not specified",
        "not mentioned", "not explicitly stated", ""
    ):
        return None
    if isinstance(v, list) and all(_clean_value(x) is None for x in v):
        return None
    return v


def _clean_llm_result(raw: dict) -> dict:
    """Normalize LLM output with global_config + models array."""
    result = {"global_config": {}, "models": []}

    # Handle global_config
    gc = raw.get("global_config", {})
    if isinstance(gc, dict):
        for k in ("eval_harness", "eval_library_version", "seed"):
            v = _clean_value(gc.get(k))
            if v is not None:
                result["global_config"][k] = v
        # eval_harness_applies_to_all is a boolean flag — preserve False
        # (the default) but skip null/missing
        flag = gc.get("eval_harness_applies_to_all")
        if isinstance(flag, bool):
            result["global_config"]["eval_harness_applies_to_all"] = flag

    # Handle models array (new format)
    models = raw.get("models", [])
    if isinstance(models, list):
        for m in models:
            if not isinstance(m, dict):
                continue
            name = _clean_value(m.get("model_name"))
            if not name:
                continue
            model_entry = {
                "model_name": name,
                "model_family": _clean_value(m.get("model_family")),
                "developer": _clean_value(m.get("developer")),
                "is_baseline": m.get("is_baseline", False),
                "benchmarks": [],
            }
            for b in m.get("benchmarks", []):
                if not isinstance(b, dict):
                    continue
                bname = _clean_value(b.get("benchmark_name"))
                if not bname:
                    continue
                cleaned = {"benchmark_name": bname}
                for k in ["score", "metric_name", "lower_is_better", "score_range", "eval_harness"] + BENCH_FIELDS:
                    v = _clean_value(b.get(k))
                    if v is not None:
                        cleaned[k] = v
                model_entry["benchmarks"].append(cleaned)
            if model_entry["benchmarks"]:
                result["models"].append(model_entry)

    # Fallback: old format with flat "benchmarks" array (no models) → single unnamed model
    if not result["models"]:
        benchmarks = raw.get("benchmarks", [])
        if isinstance(benchmarks, list) and benchmarks:
            model_entry = {
                "model_name": "unknown",
                "model_family": None,
                "is_baseline": False,
                "benchmarks": [],
            }
            for b in benchmarks:
                if not isinstance(b, dict):
                    continue
                bname = _clean_value(b.get("benchmark_name"))
                if not bname:
                    continue
                cleaned = {"benchmark_name": bname}
                for k in ["score", "metric_name"] + BENCH_FIELDS:
                    v = _clean_value(b.get(k))
                    if v is not None:
                        cleaned[k] = v
                model_entry["benchmarks"].append(cleaned)
            if model_entry["benchmarks"]:
                result["models"].append(model_entry)

    # Fallback: old flat format (no benchmarks, no models)
    if not result["models"] and not result["global_config"]:
        flat = {}
        for k in FIELDS:
            v = _clean_value(raw.get(k))
            if v is not None:
                flat[k] = v
        if flat:
            result["global_config"] = {k: flat[k] for k in ("eval_harness", "seed") if k in flat}
            bench = {"benchmark_name": "unknown"}
            for k in BENCH_FIELDS:
                if k in flat:
                    bench[k] = flat[k]
            result["models"] = [{
                "model_name": "unknown",
                "model_family": None,
                "is_baseline": False,
                "benchmarks": [bench],
            }]

    return result


def _parse_llm_json(content: str) -> dict | None:
    """Extract first JSON object from LLM response."""
    content = re.sub(r"```json\s*", "", content)
    content = re.sub(r"```\s*", "", content)
    start = content.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(content[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(content[start:i+1])
    return None


class _RetryableHTTPError(Exception):
    """Raised for 429/5xx responses to trigger tenacity retry."""
    pass


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError, _RetryableHTTPError)),
)
def _llm_call(text: str, arxiv_id: str, own_model: str = "unknown") -> dict | None:
    """Single LLM extraction call with retry on transient errors."""
    resp = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": HAIKU_MODEL,
            "messages": [
                {"role": "user",
                 "content": EXTRACTION_PROMPT.format(text=text, own_model=own_model)}
            ],
            "temperature": 0.0,
        },
        timeout=120,
        verify=False,
    )
    # 3.1: Retry on 429 and 5xx, raise immediately on other 4xx
    if resp.status_code == 429 or resp.status_code >= 500:
        log.info("Retrying %s: HTTP %d", arxiv_id, resp.status_code)
        raise _RetryableHTTPError(f"HTTP {resp.status_code}")
    resp.raise_for_status()

    data = resp.json()
    choice = data["choices"][0]
    content = choice["message"]["content"]

    # 3.2: Detect truncation
    if choice.get("finish_reason") == "length":
        log.warning("%s: Response truncated (finish_reason=length), JSON may be incomplete", arxiv_id)

    raw = _parse_llm_json(content)
    if raw:
        cleaned = _clean_llm_result(raw)
        # 3.3: Validate against pydantic schema
        return _validate_llm_result(cleaned, arxiv_id)
    return None



def _truncate_for_llm(text: str, sections: dict[str, str] | None = None) -> str:
    """Truncate text to MAX_INPUT_CHARS, preferring section boundaries."""
    if len(text) <= MAX_INPUT_CHARS:
        return text

    # If we have parsed sections, pick the most relevant ones until budget fills
    if sections:
        selected = ""
        for key in sections:
            section_text = f"=== {key} ===\n{sections[key]}\n\n"
            if len(selected) + len(section_text) > MAX_INPUT_CHARS:
                # Add as much of this section as fits
                remaining = MAX_INPUT_CHARS - len(selected)
                if remaining > 500:
                    selected += section_text[:remaining] + "\n[...TRUNCATED...]"
                break
            selected += section_text
        if selected:
            return selected

    # Fallback: first + last portions
    half = MAX_INPUT_CHARS // 2
    return text[:half] + "\n\n[...MIDDLE OMITTED...]\n\n" + text[-half:]


def llm_extract(text: str, arxiv_id: str, sections: dict[str, str] | None = None,
                own_model: str = "unknown") -> dict | None:
    """Single LLM extraction call per paper with truncated input."""
    truncated = _truncate_for_llm(text, sections)
    if len(truncated) < len(text):
        log.info("Truncated %d → %d chars for LLM", len(text), len(truncated))
    try:
        return _llm_call(truncated, arxiv_id, own_model=own_model)
    except Exception as e:
        log.error("LLM call failed for %s: %s", arxiv_id, e)
        return None


# ═══════════════════════════════════════════════════════════════════════
# Record building / saving
# ═══════════════════════════════════════════════════════════════════════

def make_benchmark_record(paper: dict, model_info: dict, benchmark_dict: dict, global_config: dict, approach: str) -> dict:
    """Build one EEE-schema JSON per model per benchmark."""
    today = _date.today().isoformat()
    model_name = model_info.get("model_name", "unknown")
    # Normalize underscores to hyphens for consistency with HF model IDs
    model_name = model_name.replace("_", "-")
    safe_model = re.sub(r"[^\w\-.]", "_", model_name)[:80]
    bench_name = benchmark_dict.get("benchmark_name", "unknown")
    raw_score = benchmark_dict.get("score")
    metric = benchmark_dict.get("metric_name", "accuracy")
    src_label = f"arxiv_html_{approach}"

    # 2.5: Normalize score to 0-100 for percentage metrics
    score = _normalize_score(raw_score, metric)

    # 2.1: Correct metric bounds/direction — use LLM values for unknown metrics
    metric_cfg = _get_metric_config(
        metric,
        llm_lower_is_better=benchmark_dict.get("lower_is_better"),
        llm_score_range=benchmark_dict.get("score_range"),
    )

    # 2.2: evaluator_relationship — same-org baselines are first_party
    is_baseline = model_info.get("is_baseline", False)
    model_family = model_info.get("model_family")
    if not is_baseline or _same_org(paper["authors"], model_family):
        relationship = "first_party"
    else:
        relationship = "third_party"

    # 2.4 + 4.2: Use LLM-provided developer, fall back to model_family or paper authors
    llm_developer = model_info.get("developer")
    if is_baseline:
        developer = llm_developer or model_family or "unknown"
    else:
        developer = llm_developer or paper["authors"]

    # 2.3: Use validated HTML title as canonical, keep expected as audit trail
    canonical_title = paper.get("actual_title") or paper["title"]

    # Build generation_config from benchmark + global
    gen_args = {}
    gen_extra = {}  # non-schema keys go to generation_config.additional_details

    # Resolve eval_harness per-benchmark first, then fall back to global ONLY
    # when the paper explicitly said the named harness applies to all
    # benchmarks. This avoids blanket-labelling every row with a harness
    # the paper only attributed to some benchmarks.
    effective_harness = benchmark_dict.get("eval_harness")
    if effective_harness is None and global_config:
        if global_config.get("eval_harness_applies_to_all") is True:
            effective_harness = global_config.get("eval_harness")
    if effective_harness is not None:
        gen_extra["eval_harness"] = str(effective_harness)

    if global_config:
        seed = global_config.get("seed")
        if seed is not None:
            gen_extra["seed"] = str(seed)
    for k in BENCH_FIELDS:
        v = benchmark_dict.get(k)
        if v is not None:
            if k == "n_shot":
                gen_args["shots"] = v
            elif k == "decoding_strategy":
                gen_extra["decoding_strategy"] = str(v)
            else:
                gen_args[k] = v

    # 3.6: Collision-proof eval_id with n_shot + hash of gen_args
    nshot_tag = gen_args.get("shots", "x")
    args_hash = hashlib.md5(json.dumps(gen_args, sort_keys=True, default=str).encode()).hexdigest()[:8]

    return {
        "schema_version": "0.2.1",
        "evaluation_id": f"{src_label}/{safe_model}/{paper['arxiv_id']}/{bench_name}/{nshot_tag}_{args_hash}",
        "evaluation_timestamp": today,
        "retrieved_timestamp": today,
        "source_metadata": {
            "source_name": src_label,
            "source_type": "documentation",
            "source_organization_name": "arXiv",
            "source_organization_url": "https://arxiv.org",
            "evaluator_relationship": relationship,
            "additional_details": {
                "paper_url": f"https://arxiv.org/abs/{paper['arxiv_id']}",
                "arxiv_id": paper["arxiv_id"],
                "paper_title": canonical_title,
                "paper_title_expected": paper["title"],
                "paper_authors": paper["authors"],
                "year": paper["year"],
                "is_baseline": str(is_baseline),
                "model_family": str(model_family) if model_family else "",
                "extraction_approach": approach,
            },
        },
        "eval_library": {
            "name": gen_extra.get("eval_harness", "unknown"),
            "version": global_config.get("eval_library_version", "unknown") if global_config else "unknown",
        },
        "model_info": {
            "name": model_name,
            "id": safe_model,
            "developer": developer,
        },
        "evaluation_results": [{
            "evaluation_name": bench_name,
            "evaluation_timestamp": today,
            "source_data": {
                "dataset_name": bench_name,
                "source_type": "url",
                "url": [f"https://arxiv.org/abs/{paper['arxiv_id']}"],
            },
            "metric_config": metric_cfg,
            "score_details": {"score": score},
            "generation_config": {
                "generation_args": gen_args,
                "additional_details": {"source": src_label, **gen_extra},
            },
        }],
    }


def make_fallback_record(paper: dict, extracted_flat: dict | None, approach: str) -> dict:
    """Fallback: one record when no benchmarks detected (naive or empty LLM)."""
    today = _date.today().isoformat()
    safe_name = re.sub(r"[^\w\-.]", "_", paper["model_name"])[:80]
    src_label = f"arxiv_html_{approach}"
    canonical_title = paper.get("actual_title") or paper["title"]
    # Separate schema-compliant gen_args from extras
    gen_args = {}
    gen_extra = {"source": src_label}
    if extracted_flat:
        for f in FIELDS:
            v = extracted_flat.get(f)
            if v is not None:
                if f == "n_shot":
                    gen_args["shots"] = v
                elif f in ("eval_harness", "decoding_strategy", "seed"):
                    gen_extra[f] = str(v)
                else:
                    gen_args[f] = v

    return {
        "schema_version": "0.2.1",
        "evaluation_id": f"{src_label}/{safe_name}/{paper['arxiv_id']}",
        "evaluation_timestamp": today,
        "retrieved_timestamp": today,
        "source_metadata": {
            "source_name": src_label, "source_type": "documentation",
            "source_organization_name": "arXiv",
            "source_organization_url": "https://arxiv.org",
            "evaluator_relationship": "first_party",
            "additional_details": {
                "paper_url": f"https://arxiv.org/abs/{paper['arxiv_id']}",
                "arxiv_id": paper["arxiv_id"],
                "paper_title": canonical_title,
                "paper_title_expected": paper["title"],
                "authors": paper["authors"],
                "year": paper["year"],
                "extraction_approach": approach,
            },
        },
        "eval_library": {"name": gen_extra.get("eval_harness", "unknown"), "version": "unknown"},
        "model_info": {"name": paper["model_name"], "id": safe_name, "developer": paper["authors"]},
        "evaluation_results": [{
            "evaluation_name": "multiple_benchmarks",
            "evaluation_timestamp": today,
            "source_data": {"dataset_name": "multiple_benchmarks", "source_type": "url",
                            "url": [f"https://arxiv.org/abs/{paper['arxiv_id']}"]},
            "metric_config": {"metric_name": "various", "lower_is_better": False,
                              "score_type": "continuous", "min_score": 0.0, "max_score": 100.0},
            "score_details": {"score": 0.0},
            "generation_config": {"generation_args": gen_args, "additional_details": gen_extra},
        }],
    }


def save_records(paper: dict, llm_result: dict | None, naive_result: dict | None, html_text: str | None = None) -> dict[str, int]:
    """Save per-model per-benchmark JSONs for LLM, plus naive fallback."""
    safe_paper = re.sub(r"[^\w\-.]", "_", paper["model_name"])[:80]
    saved = {"llm": 0, "llm_models": 0, "naive": 0, "validated": 0, "invalid": 0}

    # LLM: per-model per-benchmark records
    if llm_result and llm_result.get("models"):
        gc = llm_result.get("global_config", {})
        for model in llm_result["models"]:
            safe_model = re.sub(r"[^\w\-.]", "_", model["model_name"])[:80]
            for b in model.get("benchmarks", []):
                rec = make_benchmark_record(paper, model, b, gc, "llm")
                if html_text:
                    valid, incorrect = validate_record(rec, html_text)
                    sm = rec.setdefault("source_metadata", {})
                    ad = sm.setdefault("additional_details", {})
                    ad["validated"] = str(valid)
                    if not valid:
                        ad["incorrect_values"] = "; ".join(incorrect)
                    saved["validated" if valid else "invalid"] += 1
                # Organize by paper/model/benchmark
                bench_name = b.get("benchmark_name", "unknown")
                safe_bench = re.sub(r"[^\w\-.]", "_", bench_name)[:80]
                d = OUT_DIR / "llm" / safe_paper / safe_model / safe_bench
                d.mkdir(parents=True, exist_ok=True)
                with open(d / f"{_uuid.uuid4()}.json", "w", encoding="utf-8") as f:
                    json.dump(rec, f, indent=2, default=str)
                saved["llm"] += 1
            saved["llm_models"] += 1

    # Naive: single fallback record (paper-level)
    d = OUT_DIR / "naive" / safe_paper
    d.mkdir(parents=True, exist_ok=True)
    rec = make_fallback_record(paper, naive_result, "naive")
    with open(d / f"{_uuid.uuid4()}.json", "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, default=str)
    saved["naive"] = 1

    return saved


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract eval metadata from arxiv HTML papers.")
    parser.add_argument("--paper-list", type=str, default=None, help="Path to paper list file (default: general_llm_papers.txt)")
    parser.add_argument("--clean", action="store_true", help="Remove existing output directory before running")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompts")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG-level logging")
    parser.add_argument("--paper-delay", type=float, default=PAPER_DELAY, help="Delay between papers (seconds)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.clean and OUT_DIR.exists():
        if not args.yes:
            resp = input(f"Remove {OUT_DIR}? [y/N] ").strip().lower()
            if resp != "y":
                log.info("Aborted.")
                return
        shutil.rmtree(OUT_DIR)
        log.info("Cleaned %s", OUT_DIR)

    paper_list_path = Path(args.paper_list) if args.paper_list else None
    papers = load_papers(paper_list_path)
    log.info("Loaded %d unique papers", len(papers))

    for d in ["naive", "llm"]:
        (OUT_DIR / d).mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    for i, paper in enumerate(papers):
        arxiv_id = paper["arxiv_id"]
        log.info("[%d/%d] %s -- %s", i + 1, len(papers), arxiv_id, paper["title"])

        html = fetch_html(arxiv_id)
        if not html:
            results.append({
                **paper, "available": False, "title_valid": None,
                "actual_title": None, "naive": None, "llm": None,
            })
            time.sleep(args.paper_delay)
            continue

        # Title validation
        title_valid, actual_title = validate_title(html, paper["title"])
        if not title_valid:
            safe_actual = actual_title.encode('ascii', 'replace').decode('ascii')
            log.warning("Title mismatch for %s: expected '%s', got '%s' — proceeding anyway",
                        arxiv_id, paper["title"], safe_actual)

        sects = extract_sections(html)
        text = get_extraction_text(sects)
        rel_len = len(sects["relevant"])
        log.info("  Text: %d chars (relevant: %d, sections: %d)  Title valid: %s",
                 len(text), rel_len, len(sects["sections"]), title_valid)

        # Naive extraction
        naive = naive_extract(text)
        found_naive = {k: v for k, v in naive.items() if v}
        log.debug("  Naive: %s", json.dumps(found_naive, default=str) if found_naive else "{}")

        # LLM extraction (with chunking)
        llm_result = llm_extract(text, arxiv_id, sections=sects.get("sections"),
                                  own_model=paper["model_name"])
        if llm_result:
            found_llm = {k: v for k, v in llm_result.items() if v}
            log.debug("  LLM: %s", json.dumps(found_llm, default=str) if found_llm else "{}")

        # 2.3: Inject validated HTML title for canonical use in records
        paper["actual_title"] = actual_title

        # Extract plain text from HTML for validation
        html_plain = BeautifulSoup(html, "html.parser").get_text(separator=" ", strip=True)

        saved = save_records(paper, llm_result, naive, html_text=html_plain)
        n_models = len(llm_result.get("models", [])) if llm_result else 0
        n_bench = sum(len(m.get("benchmarks", [])) for m in llm_result.get("models", [])) if llm_result else 0
        log.info("  Saved: %d LLM JSONs (%d models, %d benchmarks), %d naive | Validated: %d, Invalid: %d",
                 saved["llm"], n_models, n_bench, saved["naive"], saved["validated"], saved["invalid"])

        results.append({
            **paper, "available": True, "title_valid": title_valid,
            "actual_title": actual_title, "naive": naive, "llm": llm_result,
        })
        time.sleep(args.paper_delay)

    # ── Filter to HTML-available only ──────────────────────────────────
    available = [r for r in results if r["available"]]
    unavailable = [r for r in results if not r["available"]]
    mismatched = [r for r in available if not r["title_valid"]]

    # ── Summaries ──────────────────────────────────────────────────────
    today = _date.today().isoformat()

    def _count_naive(field):
        return sum(1 for r in available
                   if r["naive"] and r["naive"].get(field) is not None)

    # Count LLM models and benchmarks
    total_llm_benchmarks = 0
    total_llm_models = 0
    bench_names_all = set()
    model_names_all = set()
    for r in available:
        if r["llm"] and r["llm"].get("models"):
            for m in r["llm"]["models"]:
                total_llm_models += 1
                model_names_all.add(m.get("model_name", "unknown"))
                for b in m.get("benchmarks", []):
                    total_llm_benchmarks += 1
                    bench_names_all.add(b.get("benchmark_name", "unknown"))

    def _count_llm_bench_field(field):
        """Count how many individual model-benchmark entries have this field."""
        n = 0
        for r in available:
            if r["llm"] and r["llm"].get("models"):
                for m in r["llm"]["models"]:
                    for b in m.get("benchmarks", []):
                        if b.get(field) is not None:
                            n += 1
        return n

    summary = {
        "generated": today,
        "total_papers": len(papers),
        "html_available": len(available),
        "html_unavailable": len(unavailable),
        "title_mismatches": len(mismatched),
        "total_llm_model_entries": total_llm_models,
        "total_llm_benchmark_entries": total_llm_benchmarks,
        "unique_models": sorted(model_names_all),
        "unique_benchmarks": sorted(bench_names_all),
        "unavailable_ids": [r["arxiv_id"] for r in unavailable],
        "mismatched_ids": [
            {"arxiv_id": r["arxiv_id"], "expected": r["title"], "actual": r["actual_title"]}
            for r in mismatched
        ],
        "naive_field_coverage": {f: _count_naive(f) for f in FIELDS},
        "llm_benchmark_field_coverage": {
            f: _count_llm_bench_field(f) for f in ["score", "metric_name"] + BENCH_FIELDS
        },
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(OUT_DIR / "extraction_results.json", "w", encoding="utf-8") as f:
        json.dump(available, f, indent=2, default=str)

    # Summary CSV — one row per model per benchmark per paper
    with open(OUT_DIR / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arxiv_id", "paper_title", "authors", "year", "title_valid",
                     "model_name", "model_family", "is_baseline",
                     "benchmark_name", "score", "metric_name",
                     *BENCH_FIELDS,
                     "eval_harness", "seed",
                     *[f"naive_{f}" for f in FIELDS]])
        for r in available:
            base = [r["arxiv_id"], r["title"], r["authors"], r["year"], r["title_valid"]]
            naive_vals = []
            for field in FIELDS:
                v = r["naive"].get(field) if r["naive"] else None
                naive_vals.append(json.dumps(v, default=str) if v is not None else "")

            if r["llm"] and r["llm"].get("models"):
                gc = r["llm"].get("global_config", {})
                for m in r["llm"]["models"]:
                    for b in m.get("benchmarks", []):
                        row_out = base + [
                            m.get("model_name", ""),
                            m.get("model_family", ""),
                            m.get("is_baseline", False),
                            b.get("benchmark_name", ""),
                            b.get("score", ""),
                            b.get("metric_name", ""),
                        ]
                        for bf in BENCH_FIELDS:
                            v = b.get(bf)
                            row_out.append(json.dumps(v, default=str) if v is not None else "")
                        row_out.append(json.dumps(gc.get("eval_harness"), default=str) if gc.get("eval_harness") else "")
                        row_out.append(json.dumps(gc.get("seed"), default=str) if gc.get("seed") else "")
                        row_out.extend(naive_vals)
                        w.writerow(row_out)
            else:
                row_out = base + ["", "", "", "", "", ""] + [""] * len(BENCH_FIELDS) + ["", ""]
                row_out.extend(naive_vals)
                w.writerow(row_out)

    # Console
    log.info("")
    log.info("=" * 70)
    log.info("EXTRACTION COVERAGE")
    log.info("=" * 70)
    log.info("Papers: %d  |  HTML: %d  |  Unavailable: %d  |  Title mismatch: %d",
             len(papers), len(available), len(unavailable), len(mismatched))
    log.info("LLM: %d model entries, %d benchmark entries", total_llm_models, total_llm_benchmarks)
    log.info("     %d unique models, %d unique benchmarks", len(model_names_all), len(bench_names_all))
    if unavailable:
        log.info("Missing IDs: %s", ", ".join(r["arxiv_id"] for r in unavailable))
    if mismatched:
        log.info("Mismatched IDs: %s", ", ".join(r["arxiv_id"] for r in mismatched))
    log.info("")
    log.info("%-30s  %6s", "Naive (paper-level)", "Count")
    log.info("-" * 40)
    for field in FIELDS:
        log.info("  %-28s  %6d", field, summary["naive_field_coverage"][field])
    log.info("")
    log.info("%-30s  %6s  (of %d total)", "LLM (per-model-benchmark)", "Count", total_llm_benchmarks)
    log.info("-" * 50)
    for field in ["score", "metric_name"] + BENCH_FIELDS:
        log.info("  %-28s  %6d", field, summary["llm_benchmark_field_coverage"][field])
    log.info("Output: %s", OUT_DIR)


if __name__ == "__main__":
    main()
