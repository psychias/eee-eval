"""LLM-based score extraction via OpenRouter API."""
from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import (
    _BENCHMARK_KEYWORDS_STRONG,
    _BENCHMARK_KEYWORDS_WEAK,
    OPENROUTER_API_KEY,
)
from .docling_parser import _docling_parser

class LLMFallbackExtractor:
    """Extract benchmark results from paper text when table parsing yields nothing.

    Handles two gaps that table parsing cannot cover:
      - prose-embedded results (``"achieves 87.3 % on MMLU"``)
      - figure captions describing bar-chart scores

    Uses the OpenAI chat API with ``response_format=json_object`` to get
    structured output.  All responses are cached to disk so re-runs are
    fully deterministic and do not re-incur API costs.

    Requires ``openai`` package (``pip install openai``) and an
    ``OPENAI_API_KEY`` environment variable unless *api_key* is passed.
    """

    _SYSTEM = (
        "You are a precise data-extraction assistant for NLP research papers. "
        "Extract every LLM benchmark evaluation result from the provided text. "
        "Only include results with a clearly stated numeric score. "
        "Do not infer, estimate, or invent values. "
        "IMPORTANT: Do NOT extract contamination analysis results — numbers that "
        "represent estimated performance gain from data contamination/leakage, "
        "percentage of contaminated evaluation sets, or delta improvements from "
        "ablation studies. Only extract actual benchmark scores. "
        "Output only valid JSON, nothing else."
    )

    _USER_TMPL = (
        "Paper arXiv:{arxiv_id} — chunk {chunk_idx}.\n\n"
        "Extract all benchmark result records. "
        "Include scores from tables, sentences, and figure captions. "
        "Do not include results without a numeric value.\n\n"
        "REJECT any result where:\n"
        "- the benchmark name is not a known public dataset or leaderboard "
        "(e.g. reject 'self-reflection', 'reasoning ability', 'instruction following')\n"
        "- the score was not directly stated as a number in a table cell or "
        "explicit sentence like 'achieves X on Y'\n"
        "- the metric is precision/recall/F1 from a single experiment rather "
        "than a standard benchmark\n"
        "- the number represents a contamination delta, performance gain estimate, "
        "or improvement from data leakage analysis (NOT an actual score)\n"
        "- the number comes from an ablation study showing the effect of removing "
        "a component (these are deltas, not scores)\n"
        "Only include results where the benchmark name would appear on a public "
        "leaderboard like MMLU, GSM8K, HumanEval, MT-Bench, etc.\n\n"
        "Respond with exactly this JSON schema "
        "(no extra keys, no markdown fences):\n"
        '{{ "results": [ {{"model": "<str>", "benchmark": "<str>", '
        '"score": <float>, "score_type": "absolute|relative|null", '
        '"relative_to": "<str>|null", "shots": <int|null>, '
        '"scoring_method": "log_likelihood|generation|null"}} ] }}\n\n'
        "TEXT:\n{text}"
    )

    # Maximum characters per LLM request — well within 128 K-token context
    # while keeping per-call cost low on gpt-4o-mini.
    _MAX_CHARS_PER_CHUNK: int = 6_000

    _CACHE_DIR: Path = Path("scripts/scrapers/raw/llm_cache")

    _FALLBACK_MODELS: list[str] = [
        "anthropic/claude-haiku-4.5",
        "google/gemini-2.0-flash-001",
        "meta-llama/llama-3.3-70b-instruct",
        "deepseek/deepseek-chat",
    ]

    def __init__(
        self,
        model: str = "anthropic/claude-haiku-4.5",
        api_key: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        # Use only the specified model — no fallback to other providers.
        self._models = [model]
        self._model = model
        self._api_key = api_key
        self._temperature = temperature
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import openai  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError(
                    "openai package required for LLM fallback: pip install openai"
                ) from exc

            api_key = self._api_key or os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable not set.")

            self._client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
        return self._client

    def _cache_path(self, arxiv_id: str, chunk_idx: int) -> Path:
        # Include model name in cache key so switching models invalidates cache.
        model_tag = self._model.rsplit("/", 1)[-1]  # e.g. "claude-haiku-4.5"
        return self._CACHE_DIR / f"{arxiv_id}_{model_tag}_chunk{chunk_idx:04d}.json"

    def _call_llm(self, arxiv_id: str, text: str, chunk_idx: int) -> list[dict]:
        """Call the LLM for one text chunk, returning from disk cache when available."""
        cache_file = self._cache_path(arxiv_id, chunk_idx)
        if cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                return cached.get("results", [])
            except Exception:
                pass  # corrupted cache — fall through to re-call

        client = self._get_client()
        prompt = self._USER_TMPL.format(
            arxiv_id=arxiv_id, chunk_idx=chunk_idx, text=text
        )

        max_retries = 4
        base_delay = 10

        for model_idx, current_model in enumerate(self._models):
            rate_limited = False
            for attempt in range(max_retries):
                try:
                    response = client.chat.completions.create(
                        model=current_model,
                        messages=[
                            {"role": "system", "content": self._SYSTEM},
                            {"role": "user", "content": prompt},
                        ],
                        response_format={"type": "json_object"},
                        temperature=self._temperature,
                    )
                    raw_json = response.choices[0].message.content or "{}"

                    try:
                        parsed = json.loads(raw_json)
                    except json.JSONDecodeError:
                        parsed = {}
                    if not isinstance(parsed, dict):
                        parsed = {}

                    clean: list[dict] = []
                    for item in parsed.get("results", []):
                        try:
                            score = float(item["score"])
                            if score != score:
                                continue
                            clean.append({
                                "model": str(item["model"]).strip(),
                                "benchmark": str(item["benchmark"]).strip(),
                                "score": score,
                                "score_type": str(item.get("score_type") or "unknown").strip(),
                                "relative_to": (
                                    str(item["relative_to"]).strip()
                                    if item.get("relative_to")
                                    else None
                                ),
                                "shots": int(item["shots"]) if item.get("shots") is not None else None,
                                "scoring_method": (
                                    str(item["scoring_method"]).strip()
                                    if item.get("scoring_method")
                                    else None
                                ),
                            })
                        except (KeyError, TypeError, ValueError):
                            continue

                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    cache_file.write_text(
                        json.dumps(
                            {"arxiv_id": arxiv_id, "chunk": chunk_idx, "results": clean},
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    return clean
                except Exception as exc:
                    exc_str = str(exc)
                    exc_type = type(exc).__name__

                    if "429" in exc_str or "RateLimit" in exc_type:
                        if attempt < max_retries - 1:
                            sleep_time = base_delay * (2 ** attempt)
                            print(
                                f"  [429] chunk {chunk_idx} on {current_model}. "
                                f"Waiting {sleep_time}s (Attempt {attempt + 1}/{max_retries})...",
                                file=sys.stderr,
                            )
                            time.sleep(sleep_time)
                            continue
                        # All retries exhausted for this model — try next
                        rate_limited = True
                        break

                    # Non-rate-limit error — give up entirely
                    print(
                        f"  [llm-fallback] chunk {chunk_idx} failed on {current_model}: {exc}",
                        file=sys.stderr,
                    )
                    return []

            if rate_limited and model_idx < len(self._models) - 1:
                next_model = self._models[model_idx + 1]
                print(
                    f"  [llm-fallback] chunk {chunk_idx}: {current_model} rate-limited, "
                    f"falling back to {next_model}",
                    file=sys.stderr,
                )
                continue
            elif rate_limited:
                print(
                    f"  [llm-fallback] chunk {chunk_idx} failed: all {len(self._models)} models rate-limited",
                    file=sys.stderr,
                )
                return []

        return []

    def _chunk_text(self, text: str) -> list[str]:
        """Split *text* into chunks at paragraph boundaries."""
        chunks: list[str] = []
        remaining = text
        while remaining:
            chunk = remaining[: self._MAX_CHARS_PER_CHUNK]
            # prefer splitting at a paragraph boundary in the second half
            boundary = chunk.rfind("\n\n", self._MAX_CHARS_PER_CHUNK // 2)
            if boundary > 0:
                chunk = chunk[:boundary]
            chunks.append(chunk)
            remaining = remaining[len(chunk):]
        return chunks

    def extract(self, pdf_path: Path, arxiv_id: str) -> list[dict[str, Any]]:
        """Extract benchmark results from prose and captions in *pdf_path*.

        Returns a list in the same format as ``ResultsTableParser.parse``
        (``{model, benchmark, score}``) with ``_extraction_confidence='llm'``
        to clearly distinguish these items from table-parsed results.

        Deduplication is applied: when the same (model, benchmark) pair appears
        in multiple chunks, the first occurrence wins, consistent with the
        table-parsing deduplication strategy.
        """
        pages: list[str] = []
        try:
            # Use the shared Docling parser (cached). Docling handles structured
            # PDFs with high fidelity; scanned PDFs can be enabled by setting
            # do_ocr=True in DoclingParser if needed.
            full_text = _docling_parser.get_full_text(pdf_path)
            if full_text.strip():
                pages = [full_text]
        except Exception as exc:
            print(f"  [llm-fallback] could not open PDF: {exc}", file=sys.stderr)
            return []

        full_text = "\n\n".join(pages)
        chunks = self._chunk_text(full_text)

        # Pre-filter: skip chunks dominated by contamination analysis text.
        # These chunks describe performance *gains from data leakage*, not
        # actual benchmark scores.  Sending them to the LLM risks extracting
        # deltas as if they were real scores.
        _CONTAM_SIGNALS = (
            "contamination", "contaminated", "data contamination",
            "training data overlap", "performance gain est",
            "estimated performance gain", "data leakage",
        )
        filtered_chunks: list[tuple[int, str]] = []
        for i, chunk in enumerate(chunks):
            chunk_lower = chunk.lower()
            contam_hits = sum(1 for sig in _CONTAM_SIGNALS if sig in chunk_lower)
            if contam_hits >= 2:
                print(
                    f"  [llm-fallback] chunk {i} skipped "
                    f"(contamination analysis text, {contam_hits} signals)",
                    file=sys.stderr,
                )
                continue
            filtered_chunks.append((i, chunk))

        print(
            f"  [llm-fallback] {len(chunks)} chunk(s) -> {self._model} "
            f"(cache: {self._CACHE_DIR})"
            + (f" ({len(chunks) - len(filtered_chunks)} contamination chunk(s) skipped)"
               if len(filtered_chunks) < len(chunks) else "")
        )

        all_results: list[dict] = []
        seen: set[tuple[str, str]] = set()

        for idx, chunk in filtered_chunks:
            # Check if this chunk is cached before calling — if not, the call
            # will hit the API and we need to pace subsequent requests.
            was_cached = self._cache_path(arxiv_id, idx).exists()
            try:
                items = self._call_llm(arxiv_id, chunk, idx)
            except Exception as exc:
                print(f"  [llm-fallback] chunk {idx} failed: {exc}", file=sys.stderr)
                continue
            if not was_cached:
                time.sleep(1.5)  # pace API calls to stay under rate limits
            for item in items:
                key = (item["model"].lower(), item["benchmark"].lower())
                if key not in seen:
                    seen.add(key)
                    item["_extraction_confidence"] = "llm"
                    all_results.append(item)

        # Post-filter: drop any benchmark not matching known keywords
        all_results = [
            r for r in all_results
            if any(
                kw in r["benchmark"].lower()
                for kw in _BENCHMARK_KEYWORDS_STRONG | _BENCHMARK_KEYWORDS_WEAK
            )
        ]

        print(
            f"  [llm-fallback] extracted {len(all_results)} data point(s) "
            f"({len({it['model'] for it in all_results})} models, "
            f"{len({it['benchmark'] for it in all_results})} benchmarks)"
        )
        return all_results


# ---------------------------------------------------------------------------
# Coverage reporting — per-paper and batch stats
# ---------------------------------------------------------------------------


@dataclass
class CoverageStats:
    """Extraction quality metrics for a single paper.

    Accumulated inside ``PaperExtractionPipeline.run`` and returned to the
    caller so that ``main`` can aggregate stats across a batch run and write
    a machine-readable coverage report to disk.
    """

    arxiv_id: str
    tables_scanned: int = 0
    results_tables_accepted: int = 0
    density_rejected: int = 0
    table_data_points: int = 0
    prose_data_points: int = 0
    llm_data_points: int = 0
    unique_models: int = 0
    unique_benchmarks: int = 0
    files_written: int = 0
    source: str = "none"   # "table" | "llm" | "both" | "none"
    validation_passed: int = 0
    validation_failed: int = 0
    relative_tables: int = 0
    protocol_found: int = 0
    protocol_missing: int = 0
    section_header_rows_skipped: int = 0

    @property
    def total_data_points(self) -> int:
        return self.table_data_points + self.prose_data_points + self.llm_data_points


