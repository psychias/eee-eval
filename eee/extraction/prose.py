"""Prose extractor — regex-based score extraction from running text."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .constants import _BENCH_ALT_RE
from .docling_parser import _docling_parser

class ProseExtractor:
    """Extract benchmark scores from running prose and figure captions.

    Handles two gaps that ``TableExtractor`` cannot cover without an LLM:

    * Inline sentences: ``"LLaMA-3 achieves 87.3 % on MMLU"``
    * Figure captions: ``"GPT-4 obtains 72.1 on HumanEval (Figure 3)"``

    Uses compiled regex patterns built from ``_BENCHMARK_KEYWORDS_STRONG``
    so there are zero new runtime dependencies.  Always runs before the LLM
    fallback so the LLM is reserved for genuinely unstructured cases.
    Results carry ``_extraction_confidence='prose'`` for clear provenance.

    Note: scores encoded *only* in bar-chart or radar-plot images are not
    accessible from text alone; those require ``--llm-fallback`` with a
    multimodal model.
    """

    # Pattern A: "MODEL achieves/scores/obtains/reaches/gets NUMBER on/for BENCHMARK"
    # MODEL captured as 1-4 hyphen-or-word tokens to avoid full-sentence matches.
    _PATTERN_MODEL_FIRST: re.Pattern[str] = re.compile(
        r"((?:[\w][\w\-\.]*(?:\s[\w\-\.]+){0,3}))\s+"
        r"(?:achieves?|scores?|obtains?|reaches?|reports?|gets?)\s+"
        r"(-?\d+(?:\.\d+)?)\s*%?\s+"
        r"(?:on|for)\s+("
        + _BENCH_ALT_RE
        + r")\b",
        re.IGNORECASE,
    )

    # Pattern B: "on/for BENCHMARK, MODEL achieves/scores NUMBER"
    _PATTERN_BENCH_FIRST: re.Pattern[str] = re.compile(
        r"(?:on|for)\s+("
        + _BENCH_ALT_RE
        + r")\b[^,\n]{0,20},?\s+"
        r"((?:[\w][\w\-\.]*(?:\s[\w\-\.]+){0,3}))\s+"
        r"(?:achieves?|scores?|obtains?|reaches?|gets?)\s+"
        r"(-?\d+(?:\.\d+)?)\s*%?",
        re.IGNORECASE,
    )

    # Stop-words that are never valid model names — filters false positives
    # in sentences like "the method achieves 87.3 on MMLU".
    _STOP_WORDS: frozenset[str] = frozenset({
        "the", "its", "our", "this", "that", "which", "model",
        "system", "method", "approach", "baseline", "both", "all",
        "each", "when", "then", "with", "where",
    })

    def extract(self, pdf_path: Path, arxiv_id: str) -> list[dict[str, Any]]:
        """Return ``{model, benchmark, score, _extraction_confidence}`` dicts.

        Reads the full document text via Docling's markdown export.
        """
        results: list[dict] = []
        seen: set[tuple[str, str]] = set()
        try:
            full_text = _docling_parser.get_full_text(pdf_path)
            if full_text.strip():
                self._scan_page(full_text, results, seen)
        except Exception as exc:  # noqa: BLE001
            print(f"  [prose] could not open PDF: {exc}", file=sys.stderr)
            return []
        if results:
            print(
                f"  [prose] {len(results)} data point(s) extracted from text "
                f"({len({r['model'] for r in results})} models, "
                f"{len({r['benchmark'] for r in results})} benchmarks)"
            )
        return results

    def _scan_page(
        self,
        text: str,
        results: list[dict],
        seen: set[tuple[str, str]],
    ) -> None:
        """Apply all patterns to *text*; append novel hits to *results*."""
        # Pattern A: model comes first
        for m in self._PATTERN_MODEL_FIRST.finditer(text):
            self._record(
                m.group(1), m.group(3), m.group(2), "prose", results, seen
            )
        # Pattern B: benchmark comes first
        for m in self._PATTERN_BENCH_FIRST.finditer(text):
            # group order is (benchmark, model, score)
            self._record(
                m.group(2), m.group(1), m.group(3), "prose", results, seen
            )

    def _record(
        self,
        model: str,
        benchmark: str,
        score_raw: str,
        confidence: str,
        results: list[dict],
        seen: set[tuple[str, str]],
    ) -> None:
        """Validate and append a (model, benchmark, score) hit."""
        model = model.strip()
        benchmark = benchmark.strip()
        # Reject implausibly short strings and pure-numeric captures
        if len(model) < 3 or model.isdigit():
            return
        if model.lower() in self._STOP_WORDS:
            return
        try:
            score = float(score_raw.rstrip("%"))
        except ValueError:
            return
        key = (model.lower(), benchmark.lower())
        if key in seen:
            return
        seen.add(key)
        results.append({
            "model": model,
            "benchmark": benchmark,
            "score": score,
            "score_type": "unknown",
            "relative_to": None,
            "shots": None,
            "scoring_method": None,
            "_extraction_confidence": confidence,
        })


