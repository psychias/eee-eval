"""Evaluation protocol extractors (regex-based and LLM-based)."""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import openai

from .constants import (
    _BENCH_ALT_RE,
    _PROMPT_TEMPLATE_PATTERNS,
    OPENROUTER_API_KEY,
    PROTOCOL_CACHE_DIR,
    _PROTOCOL_CACHE_VERSION,
)
from .normalize import _model_name_matches
from .docling_parser import _docling_parser

class EvalProtocolExtractor:
    """Extract benchmark evaluation protocol metadata from paper prose."""

    _BENCH_ALIASES: tuple[tuple[str, str], ...] = (
        ("NaturalQuestions", "NQ"),
        ("NaturalQuestion", "NQ"),
        ("ARC-Challenge", "ARC-c"),
        ("ARC-Easy", "ARC-e"),
        ("Winogrande", "WinoGrande"),
        ("Hellaswag", "HellaSwag"),
        ("GSM-8K", "GSM8K"),
        ("OpenbookQA", "OpenbookQA"),
        ("CommonsenseQA", "CommonsenseQA"),
        ("SIQA", "SIQA"),
    )
    _BENCH_ALT_WITH_ALIASES_RE = "|".join(
        re.escape(alias)
        for alias, _canonical in _BENCH_ALIASES
    ) + "|" + _BENCH_ALT_RE
    _BENCH_ALT_WITH_ALIASES_GROUP_RE = r"(?:" + _BENCH_ALT_WITH_ALIASES_RE + r")"
    _BENCH_CAPTURE = r"(" + _BENCH_ALT_WITH_ALIASES_RE + r")"
    _SHOT_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(_BENCH_CAPTURE + r'\s*(?:\[\d+\])?\s*\((\d+)-shot', re.IGNORECASE),
        re.compile(_BENCH_CAPTURE + r'\s*(?:\[\d+\])?\s*(\d+)-shot', re.IGNORECASE),
        re.compile(_BENCH_CAPTURE + r"[^\n\.]{0,60}?\((\d+)-shot\)", re.IGNORECASE),
        re.compile(_BENCH_CAPTURE + r"[^\n\.]{0,60}?(\d+)-shot", re.IGNORECASE),
        re.compile(r"we use (\d+)-shot[^\n\.]{0,80}?for " + _BENCH_CAPTURE, re.IGNORECASE),
        re.compile(_BENCH_CAPTURE + r"[^\n\.]{0,60}?with (\d+) shots?", re.IGNORECASE),
        re.compile(r'(\d+)-shot[^:\n]{0,30}:\s*((?:' + _BENCH_ALT_WITH_ALIASES_GROUP_RE + r'(?:(?:\s*\[\d+\])?[,\s]+)?)+)', re.IGNORECASE),
    )
    _DEFAULT_SHOT_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"all tasks use (\d+)-shot unless stated otherwise", re.IGNORECASE),
        re.compile(r"unless otherwise stated[^\n\.]{0,40}?(\d+)-shot", re.IGNORECASE),
        re.compile(r"we use (\d+)-shot for all tasks", re.IGNORECASE),
    )
    _METHOD_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
        (re.compile(r"log[ -]?likelihood", re.IGNORECASE), "log_likelihood"),
        (re.compile(r"few-shot prompt", re.IGNORECASE), "generation"),
        (re.compile(r"chain-of-thought", re.IGNORECASE), "generation"),
        (re.compile(r"generation", re.IGNORECASE), "generation"),
    )
    _TEMPERATURE_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"temperature\s*(?:=|of)?\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
    )
    _TOP_P_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"top[_ -]?p\s*(?:=|of)?\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
    )

    def extract(self, pdf_path: Path, arxiv_id: str) -> dict[str, dict[str, str | int]]:
        protocol_map: dict[str, dict[str, str | int]] = {}
        try:
            full_text = _docling_parser.get_full_text(pdf_path)
        except Exception as exc:  # noqa: BLE001
            print(f"  [protocol] could not parse text: {exc}", file=sys.stderr)
            return protocol_map

        # Restrict temperature/top_p search to evaluation/experiment sections
        # to avoid picking up training hyperparameters mentioned earlier.
        eval_section_text = self._extract_eval_section(full_text)

        default_shots: int | None = None
        default_method: str | None = None
        default_temperature: float | None = None
        default_top_p: float | None = None
        default_prompt_template: str | None = None

        for pattern in self._DEFAULT_SHOT_PATTERNS:
            match = pattern.search(full_text)
            if match:
                default_shots = int(match.group(1))
                break

        for method_pattern, method_name in self._METHOD_PATTERNS:
            if method_pattern.search(full_text):
                default_method = method_name
                break

        # Search eval section first for temperature/top_p; fall back to full
        # text only if nothing was found in a clearly-scoped section.
        for search_text in (eval_section_text, full_text):
            if default_temperature is not None:
                break
            for pattern in self._TEMPERATURE_PATTERNS:
                match = pattern.search(search_text)
                if match:
                    default_temperature = float(match.group(1))
                    break

        for search_text in (eval_section_text, full_text):
            if default_top_p is not None:
                break
            for pattern in self._TOP_P_PATTERNS:
                match = pattern.search(search_text)
                if match:
                    default_top_p = float(match.group(1))
                    break

        for pattern, prompt_template in _PROMPT_TEMPLATE_PATTERNS:
            if pattern.search(full_text):
                default_prompt_template = prompt_template
                break

        # Do NOT set a global chain_of_thought flag from any mention in the
        # paper.  CoT is benchmark-specific and should be set per-benchmark
        # by the LLM extractor which understands context.  The regex fallback
        # only provides shots, temperature, and top_p as paper-wide defaults.

        if any(
            value is not None
            for value in (default_shots, default_method, default_temperature, default_top_p, default_prompt_template)
        ):
            default_entry: dict[str, str | int] = {"protocol_source": "paper_default"}
            if default_shots is not None:
                default_entry["shots"] = default_shots
            if default_method is not None:
                default_entry["method"] = default_method
            if default_temperature is not None:
                default_entry["temperature"] = str(default_temperature)
            if default_top_p is not None:
                default_entry["top_p"] = str(default_top_p)
            if default_prompt_template is not None:
                default_entry["prompt_template"] = default_prompt_template
            protocol_map["__default__"] = default_entry

        for pattern in self._SHOT_PATTERNS:
            for match in pattern.finditer(full_text):
                if pattern.pattern.startswith(r'(\d+)-shot') and match.group(1).isdigit() and match.group(2):
                    shots = int(match.group(1))
                    benchmarks_text_clean = re.sub(r'\s*\[\d+\]', '', match.group(2))
                    for benchmark_match in re.finditer(self._BENCH_ALT_WITH_ALIASES_GROUP_RE, benchmarks_text_clean, re.IGNORECASE):
                        benchmark_key = self._canonicalize_benchmark_key(benchmark_match.group(0).strip())
                        entry = protocol_map.setdefault(
                            benchmark_key,
                            {"protocol_source": "inherited_from_category"},
                        )
                        entry["shots"] = shots
                elif match.group(1).isdigit():
                    shots = int(match.group(1))
                    benchmark = match.group(2)
                    benchmark_key = self._canonicalize_benchmark_key(benchmark.strip())
                    entry = protocol_map.setdefault(benchmark_key, {"protocol_source": "explicit"})
                    entry["shots"] = shots
                else:
                    benchmark = match.group(1)
                    shots = int(match.group(2))
                    benchmark_key = self._canonicalize_benchmark_key(benchmark.strip())
                    entry = protocol_map.setdefault(benchmark_key, {"protocol_source": "explicit"})
                    entry["shots"] = shots

        for benchmark in list(protocol_map.keys()):
            if benchmark == "__default__":
                continue
            method = self._find_method_near_benchmark(full_text, benchmark)
            if method is not None:
                protocol_map[benchmark]["method"] = method
                protocol_map[benchmark]["protocol_source"] = "explicit"
            elif default_method is not None and "method" not in protocol_map[benchmark]:
                protocol_map[benchmark]["method"] = default_method

        return protocol_map

    def _canonicalize_benchmark_key(self, benchmark: str) -> str:
        lowered = benchmark.strip().lower()
        for alias, canonical in self._BENCH_ALIASES:
            if lowered == alias.lower():
                return canonical
        return benchmark.strip()

    def _find_method_near_benchmark(self, full_text: str, benchmark: str) -> str | None:
        pattern = re.compile(re.escape(benchmark), re.IGNORECASE)
        for match in pattern.finditer(full_text):
            window = full_text[max(0, match.start() - 120): match.end() + 120]
            for method_pattern, method_name in self._METHOD_PATTERNS:
                if method_pattern.search(window):
                    return method_name
        return None

    @staticmethod
    def _extract_eval_section(full_text: str) -> str:
        """Extract text from evaluation/experiment sections of the paper.

        Returns the substring from the first evaluation-related section header
        to the references section.  Falls back to the full text if no clear
        section boundary is found.
        """
        eval_headers = re.compile(
            r"^#+\s*(?:\d+\.?\s*)?(?:evaluation|experiment|results|benchmark|setup)\b",
            re.IGNORECASE | re.MULTILINE,
        )
        match = eval_headers.search(full_text)
        start = match.start() if match else 0

        end_markers = ("references\n", "bibliography\n", "acknowledgements\n", "acknowledgments\n")
        end = len(full_text)
        for marker in end_markers:
            idx = full_text.lower().find(marker, start)
            if idx != -1 and idx < end:
                end = idx

        section = full_text[start:end]
        return section if section.strip() else full_text


# ---------------------------------------------------------------------------
# S: results table parser — heuristic identification and parsing only
# ---------------------------------------------------------------------------




class GeminiProtocolExtractor:
    """
    LLM-based protocol extractor.
    Falls back to EvalProtocolExtractor if OPENROUTER_API_KEY is not set or
    if the LLM call fails.
    Results are cached to disk so re-runs are free and deterministic.
    """

    MODEL ="anthropic/claude-haiku-4.5"

    PROMPT = """You are a precise data-extraction assistant for ML research papers.
Extract evaluation protocol metadata from the text below.
Return only valid JSON. No markdown fences. No explanation.

IMPORTANT RULES:
- Shot counts may differ per benchmark AND per model. Assign shots to
  each benchmark individually.  If a specific model uses different shots
  for the same benchmark, put those in model_overrides.
- chain_of_thought: set to true ONLY for benchmarks where the paper
  explicitly describes using chain-of-thought / CoT / scratchpad
  prompting as the evaluation method.  If CoT is merely discussed or
  mentioned but not used as the eval method, leave it null.
- temperature / top_p / presence_penalty / top_k: CRITICAL SCOPING RULE.
  If the paper states these for a SPECIFIC model family (e.g. 'For Qwen3
  models...', 'For thinking mode...'), you MUST set default_protocol
  temperature and top_p to NULL and put those values ONLY in the matching
  model_overrides entry. NEVER copy a model-specific temperature into
  default_protocol — this causes other models in the paper (e.g. GPT-4o,
  Phi-4, Gemini) to incorrectly inherit those settings.
  Only put temperature/top_p in default_protocol when the paper says they
  apply to ALL models being evaluated. When in doubt, use null in
  default_protocol and model_overrides for the specific family.
- Pay attention to table captions.  Symbols like ♢ or △ in captions
  often define shot counts or evaluation modes for specific benchmarks.
  For example "♢11-shot" means benchmarks marked with ♢ use 11 shots.
- For eval_library: only return a framework name if the authors explicitly
  state they RAN their evaluations using it. If it is only mentioned in a
  citation or reference list, return "internal".
- Canonical benchmark names: use WinoGrande (not Winogrande),
  NQ (not NaturalQuestions), ARC-e (not ARC-Easy),
  ARC-c (not ARC-Challenge), GSM8K (not GSM-8K),
  HellaSwag (not Hellaswag).

Return exactly this JSON schema:
{
  "eval_library": "internal" | "lm-evaluation-harness" | "helm" |
                  "bigbench" | "eleuther" | null,
  "evaluator_note": "string describing who ran the evals, or null",
  "default_protocol": {
    "scoring_method": "log_likelihood" | "generation" | null,
    "temperature": <float> | null,
    "top_p": <float> | null,
    "top_k": <int> | null,
    "max_tokens": <int> | null,
    "benchmark_configs": {
      "BENCHMARK_NAME": {
        "shots": <int or null>,
        "chain_of_thought": true | false | null,
        "scoring_method": "log_likelihood" | "generation" | null
      }
    }
  },
  "model_overrides": [
    {
      "model_name": "exact model name or family, e.g. Qwen3, Llama 3",
      "note": "reason for override",
      "temperature": <float> | null,
      "top_p": <float> | null,
      "top_k": <int> | null,
      "max_tokens": <int> | null,
      "chain_of_thought": true | false | null,
      "benchmark_configs": {
        "BENCHMARK_NAME": {
          "shots": <int or null>,
          "chain_of_thought": true | false | null,
          "scoring_method": "log_likelihood" | "generation" | null
        }
      }
    }
  ]
}

Paper text (first 60% only, references excluded):
{text}"""

    def __init__(self) -> None:
        PROTOCOL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._model: Any = None
        self._last_call_failed = False

    @property
    def last_call_failed(self) -> bool:
        return self._last_call_failed

    def _get_client(self) -> Any:
        if self._model is None:
            if not OPENROUTER_API_KEY:
                raise RuntimeError("OPENROUTER_API_KEY not set")
            self._model = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
            )
        return self._model

    def _cache_path(self, arxiv_id: str) -> Path:
        return PROTOCOL_CACHE_DIR / f"{arxiv_id}.json"

    def extract_raw(self, pdf_path: Path, arxiv_id: str) -> dict[str, Any]:
        """Call Gemini once per paper (cached). Returns raw Gemini output."""
        cache = self._cache_path(arxiv_id)
        if cache.exists():
            try:
                cached = json.loads(cache.read_text(encoding="utf-8"))
                if isinstance(cached, dict):
                    # Invalidate cache when prompt version changes
                    if cached.get("_cache_version") != _PROTOCOL_CACHE_VERSION:
                        pass  # stale cache — re-extract
                    else:
                        # Only trust cache if it has actual protocol data
                        dp = cached.get("default_protocol") or {}
                        has_data = (
                            dp.get("benchmark_configs")
                            or dp.get("shots_per_benchmark")
                            or dp.get("temperature") is not None
                            or dp.get("chain_of_thought") is not None
                            or cached.get("model_overrides")
                        )
                        if has_data:
                            self._last_call_failed = False
                            return cached
                        # Empty cache — re-extract
            except Exception:
                pass  # corrupt cache — re-extract

        fulltext = _docling_parser.get_full_text(pdf_path)

        # Strip references/bibliography but keep appendices
        # (appendices often contain detailed eval protocol info)
        for marker in ["references\n", "bibliography\n"]:
            idx = fulltext.lower().find(marker)
            if idx != -1:
                appendix_markers = ["appendix", "supplementary", "additional results"]
                post_bib = fulltext[idx:].lower()
                if not any(m in post_bib for m in appendix_markers):
                    fulltext = fulltext[:idx]
                break

        # Truncate to ~120K chars to fit within context window
        # while capturing instruct-model evaluation sections
        _MAX_PROTOCOL_CHARS = 120_000
        text = fulltext[:_MAX_PROTOCOL_CHARS]

        max_retries = 4
        base_delay = 10

        for attempt in range(max_retries):
            try:
                client = self._get_client()
                user_content = self.PROMPT.replace("{text}", text)
                response = client.chat.completions.create(
                    model=self.MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a precise data-extraction assistant for ML research papers. "
                                "Return only valid JSON with no additional text, explanation, or markdown. "
                                "Your entire response must be a single JSON object starting with { "
                                "and ending with }."
                            ),
                        },
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0.0,
                    max_tokens=4096,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                msg = response.choices[0].message
                raw_content = msg.content or "{}"
                # Debug: show what the API returned
                print(
                    f"  [protocol-debug] content length={len(raw_content)}, "
                    f"content[:200]={raw_content[:200]!r}",
                    file=sys.stderr,
                )
                if hasattr(msg, "reasoning_content") and msg.reasoning_content:
                    print(
                        f"  [protocol-debug] reasoning_content length="
                        f"{len(msg.reasoning_content)}",
                        file=sys.stderr,
                    )
                # Extract JSON from response (may contain thinking tags, markdown, etc.)
                raw_json = raw_content
                import re as _re
                # Remove <think>...</think> blocks (Qwen3 thinking mode)
                raw_json = _re.sub(r"<think>.*?</think>", "", raw_json, flags=_re.DOTALL).strip()
                # Extract from ```json ... ``` code block if present
                json_block = _re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_json, flags=_re.DOTALL)
                if json_block:
                    raw_json = json_block.group(1)
                elif not raw_json.startswith("{"):
                    # Find first { and last } to extract JSON object
                    brace_start = raw_json.find("{")
                    brace_end = raw_json.rfind("}")
                    if brace_start >= 0 and brace_end > brace_start:
                        raw_json = raw_json[brace_start:brace_end + 1]
                    else:
                        raw_json = "{}"
                result = json.loads(raw_json)
                if not isinstance(result, dict):
                    result = {}
                result["_cache_version"] = _PROTOCOL_CACHE_VERSION
                self._last_call_failed = False
                cache.write_text(json.dumps(result, indent=2), encoding="utf-8")
                return result
            except Exception as exc:
                is_rate_limit = "429" in str(exc) or "RateLimit" in type(exc).__name__
                is_json_error = isinstance(exc, json.JSONDecodeError)
                if (is_rate_limit or is_json_error) and attempt < max_retries - 1:
                    sleep_time = base_delay * (2 ** attempt)
                    reason = "Rate Limit" if is_rate_limit else "JSON parse error"
                    print(
                        f"  [{reason}] LLM protocol extraction retry. Waiting {sleep_time}s "
                        f"(Attempt {attempt + 1}/{max_retries})...",
                        file=sys.stderr,
                    )
                    time.sleep(sleep_time)
                    continue

                print(
                    f"  LLM protocol extraction failed for {arxiv_id}: {exc}",
                    file=sys.stderr,
                )
                self._last_call_failed = True
                return {
                    "eval_library": None,
                    "evaluator_note": None,
                    "default_protocol": {
                        "shots_per_benchmark": {},
                        "scoring_method": None,
                        "chain_of_thought": None,
                        "temperature": None,
                        "top_p": None,
                    },
                    "model_overrides": [],
                }

        self._last_call_failed = True
        return {
            "eval_library": None,
            "evaluator_note": None,
            "default_protocol": {
                "shots_per_benchmark": {},
                "scoring_method": None,
                "chain_of_thought": None,
                "temperature": None,
                "top_p": None,
            },
            "model_overrides": [],
        }

    def to_protocol_map(
        self,
        raw: dict[str, Any],
        model_name: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Convert LLM output to the protocol_map format expected by
        the rest of the pipeline. Applies model-specific overrides when
        model_name is provided.
        """
        if not isinstance(raw, dict):
            return {}
        protocol_map: dict[str, dict[str, Any]] = {}

        default = raw.get("default_protocol") or {}
        default_method = default.get("scoring_method")
        default_temp = default.get("temperature")
        default_topp = default.get("top_p")
        default_topk = default.get("top_k")
        default_max_tokens = default.get("max_tokens")

        # Guard: suppress default temperature/top_p when they are
        # model-family-specific values that leaked into default_protocol.
        # If model_overrides specify temperature/top_p for OTHER families
        # and the current model doesn't match ANY override, this model
        # has no stated temperature — don't inherit the leaked default.
        if model_name and default_temp is not None:
            override_entries_temp = [
                ov for ov in (raw.get("model_overrides") or [])
                if ov.get("temperature") is not None
            ]
            if override_entries_temp:
                current_model_has_temp_override = any(
                    _model_name_matches(model_name, ov.get("model_name", ""))
                    for ov in override_entries_temp
                )
                if not current_model_has_temp_override:
                    default_temp = None

        if model_name and default_topp is not None:
            override_entries_topp = [
                ov for ov in (raw.get("model_overrides") or [])
                if ov.get("top_p") is not None
            ]
            if override_entries_topp:
                current_model_has_topp_override = any(
                    _model_name_matches(model_name, ov.get("model_name", ""))
                    for ov in override_entries_topp
                )
                if not current_model_has_topp_override:
                    default_topp = None

        # ------------------------------------------------------------------
        # Handle both old format (shots_per_benchmark) and new format
        # (benchmark_configs with per-benchmark chain_of_thought).
        # ------------------------------------------------------------------
        bench_configs = default.get("benchmark_configs") or {}
        shots_per_bench = default.get("shots_per_benchmark") or {}

        # Merge: new format takes precedence, old format is fallback
        all_benchmarks = set(bench_configs.keys()) | set(shots_per_bench.keys())
        for bench in all_benchmarks:
            entry: dict[str, Any] = {"protocol_source": "paper_default"}
            cfg = bench_configs.get(bench) or {}
            shots = cfg.get("shots") if cfg.get("shots") is not None else shots_per_bench.get(bench)
            if shots is not None:
                entry["shots"] = int(shots)
            bench_method = cfg.get("scoring_method") or default_method
            if bench_method:
                entry["method"] = bench_method
            bench_cot = cfg.get("chain_of_thought")
            if bench_cot is not None:
                entry["chain_of_thought"] = bench_cot
            if default_temp is not None:
                entry["temperature"] = str(default_temp)
            if default_topp is not None:
                entry["top_p"] = str(default_topp)
            if default_topk is not None:
                entry["top_k"] = str(default_topk)
            if default_max_tokens is not None:
                entry["max_tokens"] = str(default_max_tokens)
            protocol_map[bench] = entry

        if model_name:
            for override in raw.get("model_overrides") or []:
                override_model = str(override.get("model_name", ""))
                if not override_model:
                    continue
                if not _model_name_matches(model_name, override_model):
                    continue

                # Model-level defaults from this override
                ov_temp = override.get("temperature")
                ov_topp = override.get("top_p")
                ov_topk = override.get("top_k")
                ov_max_tokens = override.get("max_tokens")
                ov_cot = override.get("chain_of_thought")

                # Apply model-level overrides to ALL existing benchmarks
                if any(v is not None for v in (ov_temp, ov_topp, ov_topk, ov_max_tokens, ov_cot)):
                    for bench in list(protocol_map.keys()):
                        if bench == "__default__":
                            continue
                        if ov_temp is not None:
                            protocol_map[bench]["temperature"] = str(ov_temp)
                        if ov_topp is not None:
                            protocol_map[bench]["top_p"] = str(ov_topp)
                        if ov_topk is not None:
                            protocol_map[bench]["top_k"] = str(ov_topk)
                        if ov_max_tokens is not None:
                            protocol_map[bench]["max_tokens"] = str(ov_max_tokens)
                        if ov_cot is not None:
                            protocol_map[bench]["chain_of_thought"] = ov_cot

                # Per-benchmark overrides from model_overrides
                ov_bench_configs = override.get("benchmark_configs") or {}
                ov_shots_per_bench = override.get("shots_per_benchmark") or {}
                ov_all_benches = set(ov_bench_configs.keys()) | set(ov_shots_per_bench.keys())
                for bench in ov_all_benches:
                    cfg = ov_bench_configs.get(bench) or {}
                    base = dict(protocol_map.get(bench, {"protocol_source": "paper_default"}))
                    base["protocol_source"] = "model_specific_override"
                    base["override_note"] = override.get("note", "")
                    shots = cfg.get("shots") if cfg.get("shots") is not None else ov_shots_per_bench.get(bench)
                    if shots is not None:
                        base["shots"] = int(shots)
                    if cfg.get("scoring_method"):
                        base["method"] = cfg["scoring_method"]
                    elif override.get("scoring_method"):
                        base["method"] = override["scoring_method"]
                    if cfg.get("chain_of_thought") is not None:
                        base["chain_of_thought"] = cfg["chain_of_thought"]
                    if ov_temp is not None:
                        base["temperature"] = str(ov_temp)
                    if ov_topp is not None:
                        base["top_p"] = str(ov_topp)
                    if ov_topk is not None:
                        base["top_k"] = str(ov_topk)
                    if ov_max_tokens is not None:
                        base["max_tokens"] = str(ov_max_tokens)
                    protocol_map[bench] = base

        # Provide a __default__ fallback entry for benchmarks not explicitly
        # listed in the protocol, carrying only the paper-level defaults.
        if "__default__" not in protocol_map:
            default_entry: dict[str, Any] = {"protocol_source": "paper_default"}
            if default_method:
                default_entry["method"] = default_method
            if default_temp is not None:
                default_entry["temperature"] = str(default_temp)
            if default_topp is not None:
                default_entry["top_p"] = str(default_topp)
            if default_topk is not None:
                default_entry["top_k"] = str(default_topk)
            if default_max_tokens is not None:
                default_entry["max_tokens"] = str(default_max_tokens)
            protocol_map["__default__"] = default_entry

        return protocol_map

    def get_eval_library(self, raw: dict[str, Any] | Any) -> str:
        if not isinstance(raw, dict):
            return "internal"
        return str(raw.get("eval_library") or "internal")

    def identify_benchmarks(self, pdf_path: Path, arxiv_id: str) -> dict[str, list[str]]:
        """Identify benchmarks and results tables via LLM. Cached to disk."""
        cache = PROTOCOL_CACHE_DIR / f"{arxiv_id}_benchmarks.json"
        if cache.exists():
            try:
                cached = json.loads(cache.read_text(encoding="utf-8"))
                if isinstance(cached, dict) and cached.get("known_benchmarks"):
                    return cached
                # Empty cache — re-extract
            except Exception:
                pass

        try:
            fulltext = _docling_parser.get_full_text(pdf_path)
        except Exception:
            return {"known_benchmarks": [], "results_table_indicators": []}

        # Exclude bibliography section but keep appendix (detailed results).
        # Only strip references/bibliography, NOT acknowledgements (which
        # precede appendices in most papers).
        for marker in ["references\n", "bibliography\n"]:
            idx = fulltext.lower().find(marker)
            if idx != -1:
                # Check if there's an appendix AFTER the bibliography
                appendix_markers = ["appendix", "supplementary", "additional results"]
                post_bib = fulltext[idx:].lower()
                has_appendix = any(m in post_bib for m in appendix_markers)
                if not has_appendix:
                    fulltext = fulltext[:idx]
                break
        # Truncate to avoid exceeding context limits
        text = fulltext[:120_000]

        prompt = (
            "List every benchmark, dataset, or evaluation task name that appears "
            "in this paper as a subject of numeric evaluation. "
            "Include leaderboard names (MT-Bench, Chatbot Arena), coding benchmarks "
            "(HumanEval, MBPP), math benchmarks (MATH, GSM8K, AIME), reasoning "
            "benchmarks (ARC, HellaSwag, WinoGrande), and any domain-specific "
            "evaluations. "
            "Also list which table numbers (Table 1, Table 2...) contain model "
            "comparison results with numeric scores. "
            "Return JSON only:\n"
            '{"known_benchmarks": ["list of canonical benchmark names"], '
            '"results_table_indicators": ["Table 2", "Table 3"]}\n\n'
            f"Paper text:\n{text}"
        )

        max_retries = 4
        base_delay = 10
        for attempt in range(max_retries):
            try:
                client = self._get_client()
                response = client.chat.completions.create(
                    model=self.MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a precise data-extraction assistant. "
                                "Return only valid JSON with no additional text or markdown."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=2048,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                raw_content = response.choices[0].message.content or "{}"
                # Extract JSON from response (may contain thinking tags, markdown)
                import re as _re
                raw_json = _re.sub(r"<think>.*?</think>", "", raw_content, flags=_re.DOTALL).strip()
                json_block = _re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_json, flags=_re.DOTALL)
                if json_block:
                    raw_json = json_block.group(1)
                elif not raw_json.startswith("{"):
                    brace_start = raw_json.find("{")
                    brace_end = raw_json.rfind("}")
                    if brace_start >= 0 and brace_end > brace_start:
                        raw_json = raw_json[brace_start:brace_end + 1]
                    else:
                        raw_json = "{}"
                result = json.loads(raw_json)
                if not isinstance(result, dict):
                    result = {}
                out = {
                    "known_benchmarks": result.get("known_benchmarks", []),
                    "results_table_indicators": result.get("results_table_indicators", []),
                }
                cache.write_text(json.dumps(out, indent=2), encoding="utf-8")
                return out
            except Exception as exc:
                if "429" in str(exc) or "RateLimit" in type(exc).__name__:
                    if attempt < max_retries - 1:
                        time.sleep(base_delay * (2 ** attempt))
                        continue
                print(f"  [identify_benchmarks] failed for {arxiv_id}: {exc}", file=sys.stderr)
                return {"known_benchmarks": [], "results_table_indicators": []}
        return {"known_benchmarks": [], "results_table_indicators": []}


# ---------------------------------------------------------------------------
# D: pipeline orchestrator — depends on abstractions only
# ---------------------------------------------------------------------------


