"""
PDF and arXiv paper extraction pipeline for the EEE schema.

downloads PDFs from arXiv (or uses local files), extracts results tables
using Docling (IBM), and converts found benchmarks to EEE schema JSON files.

design follows SOLID principles:
  - PDFDownloader: fetch only
  - TableExtractor: table detection only
  - ResultsTableParser: parsing only
  - PaperConverter: schema conversion only
  - PaperWriter: file I/O only
  - PaperExtractionPipeline: orchestration only

usage:
    python scripts/extract_paper.py --arxiv_id 2407.21783
    python scripts/extract_paper.py --pdf path/to/paper.pdf
    python scripts/extract_paper.py --batch scripts/arxiv_ids.txt
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

# Raise Python 3.11+ integer-string conversion limit to handle large
# numbers emitted by Docling (e.g. 8192-digit integers in table cells).
if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)

# add repo root and utils/ to sys.path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "utils"))

import requests
import openai

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

PROTOCOL_CACHE_DIR = Path(".cache/llm_protocol")
# Bump this when the LLM prompt changes to invalidate stale cached responses.
_PROTOCOL_CACHE_VERSION = 2

from eval_types import (
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    GenerationArgs,
    GenerationConfig,
    MetricConfig,
    ModelInfo,
    ScoreDetails,
    ScoreType,
    SourceDataUrl,
    SourceMetadata,
)
from helpers import get_developer
from eval_converters import SCHEMA_VERSION as _SCHEMA_VERSION

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
_ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}.pdf"
_ARXIV_ABS_URL = "https://arxiv.org/abs/{arxiv_id}"
_TIMEOUT = 60  # seconds for HTTP
_PDF_DOWNLOAD_DIR = Path("scripts/scrapers/raw/papers")

# Specific benchmark names — a single match is sufficient evidence that a
# table contains evaluation results.
_BENCHMARK_KEYWORDS_STRONG: set[str] = {
    "mmlu", "humaneval", "human eval", "gsm8k", "gsm-8k",
    "bbh", "big-bench", "big bench", "hellaswag", "hella swag",
    "arc", "truthfulqa", "truthful qa", "winogrande", "wino grande",
    "piqa", "lambada", "pass@k", "pass@1", "pass@10", "pass@100",
    "mbpp", "human-eval",
    "math", "gpqa", "musr", "mmlu-pro", "mmlu pro", "ifeval", "if eval",
    "drop", "nq", "triviaqa", "trivia qa", "copa", "agieval",
    "boolq", "bool q", "swag", "race", "squad", "natural questions",
    "codexglue", "humaneval+", "bigcode", "swe-bench", "mmstar",
    "livecodebench", "livebench", "aime", "agi eval", "c-eval",
    "mt-bench", "mt bench", "mtbench", "chatbot arena",
    "alpacaeval", "alpaca eval", "arena-hard", "arena hard",
    "wildbench", "wild bench",
    # Non-English / multilingual benchmarks
    "cmmlu", "gaokao", "flores", "xnli", "xcopa", "xquad",
    "paws-x", "tydiqa", "mlqa", "msvamp", "cmb", "cmath",
    "arabicmmlu", "kmmlu", "jmmlu", "indommlu",
    "belebele", "mega", "xtreme",
    # Tool-use / agent benchmarks
    "bfcl", "nexus", "api-bank", "api-bench",
    # Multilingual aggregates
    "mgsm", "multilingual mmlu",
}

# Generic metric terms — too common alone (any table with an 'accuracy'
# column would match).  Require ≥2 distinct weak matches to count as
# sufficient evidence, reducing false positives from non-eval tables.
_BENCHMARK_KEYWORDS_WEAK: set[str] = {
    "accuracy", "bleu", "rouge", "f1", "em", "exact match",
}

# Combined set for any code that still needs the full list (e.g. when
# identifying which *columns* are benchmark columns in a parsed table).
_BENCHMARK_KEYWORDS: set[str] = _BENCHMARK_KEYWORDS_STRONG | _BENCHMARK_KEYWORDS_WEAK

# Pre-built alternation string used by ProseExtractor regex patterns.
# Sorted longest-first so that "mmlu-pro" / "human eval" etc. match before
# the shorter "mmlu" / "eval" prefixes that would shadow them.
_BENCH_ALT_RE: str = "|".join(
    re.escape(kw)
    for kw in sorted(_BENCHMARK_KEYWORDS_STRONG, key=len, reverse=True)
)

# ---------------------------------------------------------------------------
# benchmark / model name canonicalization
# ---------------------------------------------------------------------------

# Map lowercased variant → canonical benchmark name.  Applied early in
# _convert_model so duplicate variants collapse into one entry.
_BENCHMARK_CANONICAL: dict[str, str] = {
    # Hyphen variants
    "math500": "MATH-500",
    "math-500": "MATH-500",
    "mmlupro": "MMLU-Pro",
    "mmlu-pro": "MMLU-Pro",
    "mmlu pro": "MMLU-Pro",
    # Year variants
    "mt-aime24": "MT-AIME2024",
    "mt-aime2024": "MT-AIME2024",
    "mt-aime 2024": "MT-AIME2024",
    "aime 2024": "AIME 2024",
    "aime 2025": "AIME 2025",
    # IFEval scoring variant → parent benchmark
    "ifeval strict prompt": "IFEval",
    "ifeval": "IFEval",
    # GPQA variants
    "gpqa-diamond": "GPQA-Diamond",
    "gpqa diamond": "GPQA-Diamond",
    "gpqa": "GPQA",
    # Case normalization for common benchmarks
    "mmlu": "MMLU",
    "mmlu-redux": "MMLU-Redux",
    "mmlu redux": "MMLU-Redux",
    "mmmlu": "MMMLU",
    "bbh": "BBH",
    "bb hard": "BBH",
    "bb_hard": "BBH",
    "big-bench hard": "BBH",
    "big bench hard": "BBH",
    "bigbench hard": "BBH",
    "gsm8k": "GSM8K",
    "mbpp": "MBPP",
    "mbpp+": "MBPP+",
    "supergpqa": "SuperGPQA",
    "super gpqa": "SuperGPQA",
    "humaneval": "HumanEval",
    "humaneval+": "HumanEval+",
    "human eval": "HumanEval",
    "human-eval": "HumanEval",
    "winogrande": "WinoGrande",
    "hellaswag": "HellaSwag",
    "piqa": "PIQA",
    "siqa": "SIQA",
    # ARC variants — covers hyphen, space, and abbreviation forms
    "arc-c": "ARC-Challenge",
    "arc-e": "ARC-Easy",
    "arc-challenge": "ARC-Challenge",
    "arc challenge": "ARC-Challenge",
    "arc_challenge": "ARC-Challenge",
    "arc-easy": "ARC-Easy",
    "arc easy": "ARC-Easy",
    "arc_easy": "ARC-Easy",
    "arc c": "ARC-Challenge",
    "arc e": "ARC-Easy",
    "arc (challenge)": "ARC-Challenge",
    "arc (easy)": "ARC-Easy",
    "arc- challenge": "ARC-Challenge",
    "arc- easy": "ARC-Easy",
    "arc (c)": "ARC-Challenge",
    "arc (e)": "ARC-Easy",
    "commonsenseqa": "CommonsenseQA",
    "openbookqa": "OpenBookQA",
    "triviaqa": "TriviaQA",
    "trivia qa": "TriviaQA",
    "boolq": "BoolQ",
    "bool q": "BoolQ",
    "squad": "SQuAD",
    "quac": "QuAC",
    "race": "RACE",
    "drop": "DROP",
    "agieval": "AGIEval",
    "agi eval": "AGIEval",
    "arena-hard": "Arena-Hard",
    "arena hard": "Arena-Hard",
    "c-eval": "C-Eval",
    "c eval": "C-Eval",
    "ceval": "C-Eval",
    "livebench": "LiveBench",
    "livecodebench v5": "LiveCodeBench-v5",
    "livecodebench": "LiveCodeBench",
    "live code bench": "LiveCodeBench",
    "polymath": "PolyMath",
    "multi-if": "Multi-IF",
    "multi if": "Multi-IF",
    "multiif": "Multi-IF",
    "mlogiqa": "MLogiQA",
    "include": "INCLUDE",
    "iinclude": "INCLUDE",
    "mmmlu 14 languages": "MMMLU",
    "livecodebench-v5": "LiveCodeBench-v5",
    "bfcl v3": "BFCL-v3",
    "bfcl-v3": "BFCL-v3",
    "alignbench v1.1": "AlignBench-v1.1",
    "alignbench-v1.1": "AlignBench-v1.1",
    "creative writing v3": "Creative-Writing-v3",
    "creative writing": "Creative-Writing",
    "writingbench": "WritingBench",
    "writing bench": "WritingBench",
    "crux-o": "CRUX-O",
    "evalplus": "EvalPlus",
    "eval plus": "EvalPlus",
    "mbpp evalplus": "MBPP+",
    "mbpp evalplus (base)": "MBPP+ (base)",
    "humaneval evalplus": "HumanEval+",
    "multipl-e": "MultiPL-E",
    "mgsm": "MGSM",
    "worldsense": "WorldSense",
    "world sense": "WorldSense",
    "autologi": "AutoLogi",
    # Additional common benchmarks
    "truthfulqa": "TruthfulQA",
    "truthful qa": "TruthfulQA",
    "nq": "NQ",
    "natural questions": "NQ",
    "copa": "COPA",
    "swe-bench": "SWE-Bench",
    "swe bench": "SWE-Bench",
    "swebench": "SWE-Bench",
    "swe-bench verified": "SWE-Bench-Verified",
    "mt-bench": "MT-Bench",
    "mt bench": "MT-Bench",
    "mtbench": "MT-Bench",
    "alpacaeval": "AlpacaEval",
    "alpaca eval": "AlpacaEval",
    "alpacaeval 2.0": "AlpacaEval-2.0",
    "alpacaeval2": "AlpacaEval-2.0",
    "lambada": "LAMBADA",
    "lambada (openai)": "LAMBADA (OpenAI)",
    "anli r1": "ANLI-R1",
    "anli r2": "ANLI-R2",
    "anli r3": "ANLI-R3",
    "anli-r1": "ANLI-R1",
    "anli-r2": "ANLI-R2",
    "anli-r3": "ANLI-R3",
    "anli_r1": "ANLI-R1",
    "anli_r2": "ANLI-R2",
    "anli_r3": "ANLI-R3",
    "squadv1": "SQuAD v1",
    "squadv2": "SQuAD v2",
    "squad v1": "SQuAD v1",
    "squad v2": "SQuAD v2",
    "squad v2 (em)": "SQuAD v2 (EM)",
    "triviaqa (em)": "TriviaQA (EM)",
    "triviaqa-wiki": "TriviaQA",
    "naturalquestions (em)": "NaturalQuestions (EM)",
    "webquestions (em)": "WebQuestions (EM)",
    "multirc (f1)": "MultiRC (F1)",
    "xsum en": "XSum",
    "xsum english": "XSum",
    "xsum\u2014english": "XSum",
    "xsum\u2013english": "XSum",
    "xsum": "XSum",
    "mgsm.": "MGSM",
    "obqa": "OpenBookQA",
    "storycloze": "StoryCloze",
    "wic": "WiC",
    "wsc": "WSC",
    "record": "ReCoRD",
    "record (f1)": "ReCoRD",
    "cb": "CB",
    "rte": "RTE",
    "multirc": "MultiRC",
    # Truncated benchmark names (from Docling truncation)
    "hellas": "HellaSwag",
    "humane": "HumanEval",
    "triqa": "TriviaQA",
    "hella swag": "HellaSwag",
    "hella  swag": "HellaSwag",
    "hella__swag": "HellaSwag",
    "wino grande": "WinoGrande",
    "wino  grande": "WinoGrande",
    "wino__grande": "WinoGrande",
    # FLORES normalization
    "flores-101": "FLORES-101",
    "flores101": "FLORES-101",
    # BBH / BIG-Bench Hard subcategories → parent benchmark
    "bbh": "BBH",
    "bbh-alg": "BBH",
    "bbh-direct": "BBH",
    "bbh-nlp": "BBH",
    "bbh alg": "BBH",
    "bbh direct": "BBH",
    "bbh nlp": "BBH",
    "big-bench": "BBH",
    "big bench": "BBH",
    "big-bench hard": "BBH",
    "big bench hard": "BBH",
    "bigbench": "BBH",
    "bigbench hard": "BBH",
    # FRMT (variants with language/region)
    "frmt": "FRMT",
    "frmt bleurt": "FRMT",
    "frmt bleu": "FRMT",
    # MS MARCO
    "ms marco": "MS MARCO",
    "ms_marco": "MS MARCO",
    "msmarco": "MS MARCO",
    # GSM8K variants
    "gsm-8k": "GSM8K",
    "gsm 8k": "GSM8K",
    "musr": "MuSR",
    "codexglue": "CodeXGLUE",
    "bigcodebench": "BigCodeBench",
    "big code bench": "BigCodeBench",
    "mmstar": "MMStar",
    "wildbench": "WildBench",
    "wild bench": "WildBench",
    "judgebench": "JudgeBench",
    "zebra logic": "ZebraLogic",
    "zebralogic": "ZebraLogic",
    # Truncated/short variants
    "winog": "WinoGrande",
    # pass@k variants
    "pass@1": "HumanEval pass@1",
    "pass@10": "HumanEval pass@10",
    "pass@100": "HumanEval pass@100",
    "humaneval pass@1": "HumanEval pass@1",
    "humaneval pass@10": "HumanEval pass@10",
    "humaneval pass@100": "HumanEval pass@100",
    "mbpp pass@1": "MBPP pass@1",
    # Non-English benchmarks
    "cmmlu": "CMMLU",
    "gaokao": "GAOKAO",
    "gaokao-bench": "GAOKAO",
    "flores": "FLORES",
    "flores-200": "FLORES-200",
    "xnli": "XNLI",
    "xcopa": "XCOPA",
    "xquad": "XQuAD",
    "paws-x": "PAWS-X",
    "tydiqa": "TyDiQA",
    "mlqa": "MLQA",
    "belebele": "Belebele",
    "kmmlu": "KMMLU",
    "jmmlu": "JMMLU",
    "indommlu": "IndoMMLU",
    "arabicmmlu": "ArabicMMLU",
    # Parenthesized / descriptor variants that Docling converts to underscores
    "ifeval average": "IFEval",
    "ifeval (average)": "IFEval",
    "lambada openai": "LAMBADA (OpenAI)",
    "mbpp pass 1": "MBPP pass@1",
    "mbpp (pass 1)": "MBPP pass@1",
    "mbpp (pass@1)": "MBPP pass@1",
    "natural questions held-out": "NQ",
    "natural questions (held-out)": "NQ",
    "triviaqa held-out": "TriviaQA",
    "triviaqa (held-out)": "TriviaQA",
    "xnli zh": "XNLI",
    "xnli en": "XNLI",
    "squad v2 em": "SQuAD v2 (EM)",
    "squad v2 (em)": "SQuAD v2 (EM)",
    "triviaqa em": "TriviaQA (EM)",
    "triviaqa (em)": "TriviaQA (EM)",
    "tydiqa-goldp": "TyDiQA-GoldP",
    "tydiqa goldp": "TyDiQA-GoldP",
    "flores-101 bleu": "FLORES-101",
    "openrewrite-eval": "OpenRewrite-Eval",
}

# Non-benchmark names that should be filtered out — summary statistics,
# language-family subcategories, etc.
_NON_BENCHMARK_NAMES: frozenset[str] = frozenset({
    "average", "avg", "mean", "overall", "total",
    # Language family subcategories (from INCLUDE/MMMLU tables)
    "afro", "asiatic", "afro-asiatic", "afro asiatic",
    "germanic", "romance", "slavic", "uralic", "turkic",
    "sinitic", "japonic", "austronesian", "dravidian",
    "indo-iranian", "indo iranian", "italic", "celtic",
    "semitic", "bantu", "niger-congo", "sino-tibetan",
    # Modality labels
    "text", "& text",
    # Double-space / artifact forms of language families
    "afro  asiatic", "afro asiatic 85.9",
    # Verbatim/meta labels
    "verbatim memorization",
    # Contamination analysis labels (phi-1 Table 3 etc.)
    "similar", "non-similar",
    # Table header fragments from multi-level headers
    "by eval set", "by eval set,", "by language", "by language,",
    "disaggregated worst-case", "disaggregated",
    # Standalone metric names (require benchmark prefix)
    "rouge-2", "rouge-l", "rouge-1",
    # Standalone language names (too generic without parent benchmark)
    "arabic", "bengali", "chinese", "english", "french", "german",
    "hindi", "indonesian", "japanese", "korean", "portuguese",
    "russian", "spanish", "swahili", "telugu", "thai", "turkish",
    "urdu", "vietnamese", "dutch", "italian", "polish", "czech",
    "romanian", "hungarian", "finnish", "danish", "norwegian",
    "swedish", "greek", "hebrew", "persian", "malay",
    # Quoted pronouns from bias/fairness tables
    '"he"', '"she"', "\"he\"", "\"she\"",
    # Generic labels
    "score", "results", "task", "tasks", "benchmark", "dataset",
    # Category/aggregate labels from multi-task tables
    "avg.", "avg..", "code", "language",
})

# Generation-quality / NLG benchmarks scored with ROUGE/BLEU — not
# capability benchmarks.  No cross-paper collision pairs possible, so
# they inflate the unique-benchmark count without analytical value.
_GENERATION_TASK_BLOCKLIST: frozenset[str] = frozenset({
    "commongen", "e2enlg", "webnlg", "wikilingua", "xlsum",
    "commongen rouge", "e2enlg rouge", "webnlg rouge",
    "commongen_rouge-1", "e2enlg_rouge-1", "webnlg_rouge-1",
})


# Category labels that Docling prepends via multi-row header concatenation.
# These are NOT benchmark names and must be stripped.
_TABLE_CATEGORY_PREFIXES: frozenset[str] = frozenset({
    "commonsense understanding", "commonsense reasoning",
    "reading comprehension", "world knowledge",
    "math and reasoning", "math and text",
    "code", "coding", "general", "popular aggregated results",
    "language understanding", "knowledge", "reasoning",
    "exam", "tasks", "alignment tasks", "alignment",
    "multilingual", "multilingual tasks", "agent",
    "stem", "math", "text",
})


def _normalize_benchmark(name: str) -> str | None:
    """Canonicalize a benchmark name.

    Returns the canonical form, or None if the name should be filtered out.
    """
    # Fix mojibake arrow: the UTF-8 encoding of → (U+2192) is bytes E2 86 92,
    # which when re-interpreted as Latin-1 become â (E2) † (86 via CP1252) ' (92 via CP1252).
    # Python sees these as U+00E2 U+2020 U+2019 after Docling's double-decode.
    # Must run BEFORE quote normalization to catch U+2019 in context.
    name = name.replace("â†’", "→")
    name = name.replace("â†'", "→")
    # Normalize unicode apostrophes / primes to ASCII
    # Docling mojibake: Æ (U+00C6) appears where ‘ (left single quote) was expected
    name = name.replace("Æ", "'")
    name = name.replace("‘", "'").replace("’", "'").replace("′", "'")
    # Strip control characters (Docling artifacts)
    name = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', ' ', name)
    # Collapse multiple spaces into one (Docling artifact: "hella  swag")
    name = re.sub(r'\s{2,}', ' ', name)
    stripped = name.strip()

    # Strip leading '& ' — LaTeX table separator artifact from Docling
    stripped = re.sub(r'^[&]\s*', '', stripped).strip()

    # Early reject: names starting with a decimal (0.8, 0.85, 0.9, 0.95...)
    # These are contamination-analysis similarity thresholds, NOT benchmarks.
    if re.match(r'^0\.\d', stripped):
        return None

    # Early reject: average/aggregate rows — "Avg.", "Avg..MBPP", "Average"
    if re.match(r'^avg\.?(?:\.|$)', stripped, re.IGNORECASE):
        return None
    if stripped.lower() in ('average', 'mean', 'overall'):
        return None

    # Early reject: names that are just quoted pronouns or short tokens
    stripped_quotes = stripped.strip('"\'""''')
    if stripped_quotes.lower() in ('he', 'she', 'it', 'they'):
        return None

    # Strip parenthesized shot info: "(25-shot)", "(5-shot)", "(10-shot)"
    stripped = re.sub(r'\s*\(\d+-shot\)', '', stripped).strip()

    # Strip parenthesized descriptive qualifiers:
    # "(MCQ in 57 subjects)" → strip
    # "(for Instruct Models)" → strip
    # But preserve meaningful qualifiers like "(EM)", "(F1)", "(OpenAI)", "(pass@1)"
    stripped = re.sub(
        r'\s*\((?:MCQ\s+in\s+\d+\s+subjects|for\s+\w+\s+Models?)\)',
        '', stripped, flags=re.IGNORECASE,
    ).strip()

    # Strip non-parenthesized shot info: "HellaSwag 10-shot" → "HellaSwag"
    # Also handles: "ARC_Challenge_25-shot_", "GSM-8K_5-shot_"
    stripped = re.sub(r'[_\s]+\d+-shot[_,.]?\s*$', '', stripped, flags=re.IGNORECASE).strip()
    # Strip leading shot info: "5-shot ARC" → "ARC"
    stripped = re.sub(r'^\d+-shot[_\s]+', '', stripped, flags=re.IGNORECASE).strip()

    # Normalize "(pass 1)" → "(pass@1)" and "(pass 10)" → "(pass@10)"
    stripped = re.sub(
        r'\(pass\s+(\d+)\)',
        r'(pass@\1)', stripped, flags=re.IGNORECASE,
    )
    # Strip standalone " pass 1" / " pass@1" suffix without parens:
    # "MBPP pass 1" → "MBPP pass@1"
    stripped = re.sub(r'\s+pass\s+(\d+)$', r' pass@\1', stripped, flags=re.IGNORECASE)

    # Normalize WMT benchmark names with embedded scores:
    # "WMT '14 En→Fr BLEU 35.0 d" → "WMT14 En-Fr"
    # "WMT '16 De→En BLEU 41.2 e" → "WMT16 De-En"
    # Handle multiple arrow representations: → (U+2192), â†' (mojibake), ->, -->
    # Also handle Æ (U+00C6) — Docling mojibake of left single quote U+2018
    wmt_match = re.match(
        r"WMT\s*['''Æ]?(\d{2})\s+([\w]+)\s*(?:→|â†’|->|-->)\s*([\w]+)"
        r"(?:\s+(?:BLEU|BLEURT|chrF)\s+\d+\.?\d*\s*[a-z]?)?$",
        stripped, re.IGNORECASE,
    )
    if wmt_match:
        year = wmt_match.group(1)
        src = wmt_match.group(2)
        tgt = wmt_match.group(3)
        return f"WMT{year} {src}-{tgt}"

    # WMT fallback: arrow lost by Docling, just space between lang codes
    # "WMT '14 En Fr BLEU 35.0 d" → "WMT14 En-Fr"
    # "WMT  16 De En BLEU 41.2 e" → "WMT16 De-En"
    wmt_space = re.match(
        r"WMT[\s'''Æ]*?(\d{2})\s+([A-Z][a-z])\s+([A-Z][a-z])"
        r"(?:\s+(?:BLEU|BLEURT|chrF)\s+[\d\.]+\s*[a-z]?)?$",
        stripped, re.IGNORECASE,
    )
    if wmt_space:
        year = wmt_space.group(1)
        src = wmt_space.group(2).capitalize()
        tgt = wmt_space.group(3).capitalize()
        return f"WMT{year} {src}-{tgt}"

    # Strip trailing language-code lists from multilingual benchmarks:
    # "XLSum ar, bn, en, ja, in, sw, ko, ru, te, th, tr" → "XLSum"
    # "WikiLingua ar, ja, ko, ru, th, tr" → "WikiLingua"
    lang_list_match = re.match(
        r'^(.+?)\s+(?:[a-z]{2},\s*){2,}[a-z]{2}$',
        stripped, re.IGNORECASE,
    )
    if lang_list_match:
        candidate = lang_list_match.group(1).strip()
        if len(candidate) >= 2:
            stripped = candidate

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

    # Reject NLG generation-quality benchmarks (ROUGE-scored, no cross-paper value)
    if any(sig in lowered for sig in _GENERATION_TASK_BLOCKLIST):
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

    # FRMT with language/region/metric suffixes → just "FRMT"
    # "FRMT Chinese Mainland BLEURT" → "FRMT"
    # "FRMT Portuguese Brazil BLEU" → "FRMT"
    if re.match(r'^frmt\b', lowered):
        return "FRMT"

    # BBH with subcategory suffix → just "BBH"
    # "BBH-Boolean Expressions" → "BBH"
    # "BBH Causal Judgement" → "BBH"
    if re.match(r'^bbh[\s_-]', lowered):
        return "BBH"

    canonical = _BENCHMARK_CANONICAL.get(lowered)
    if canonical:
        return canonical

    # Double-underscore form cleanup (Docling artifact): replace __ with space and retry
    if '__' in lowered:
        cleaned = re.sub(r'_+', ' ', stripped).strip()
        # Re-run normalization on the cleaned form to catch WMT etc.
        return _normalize_benchmark(cleaned)

    return stripped


def _normalize_benchmark_with_shots(name: str) -> tuple[str | None, int | None]:
    """Like _normalize_benchmark but also returns the shot count if present.

    Call this instead of _normalize_benchmark anywhere the shot count
    needs to survive into the schema (i.e. in the table parser and converter).
    """
    # Extract shot count BEFORE stripping it
    shot_match = re.search(r'[\(\s](\d+)-shot[\)\s,.]?', name, re.IGNORECASE)
    shots = int(shot_match.group(1)) if shot_match else None

    normalized = _normalize_benchmark(name)  # strips the suffix as before
    return normalized, shots


def _normalize_model_name(name: str) -> str:
    """Normalize model name casing for consistency.

    - Uppercases parameter-size suffixes: "27b" → "27B"
    - Normalizes known casing: "gemma-3-27b-it" → "Gemma-3-27B-IT"
    - Strips score-like data that got concatenated from merged table cells
    """
    # Strip control characters
    name = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', ' ', name)
    # Strip trailing periods/commas that Docling leaves on model names
    name = name.rstrip('.').strip()
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
    # Remove duplicate size tokens: "Llama 1 33B 33B" → "Llama 1 33B"
    name = re.sub(r'\b(\d+\.?\d*B)\s+\1\b', r'\1', name)
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


# ---------------------------------------------------------------------------
# Resolve self-referential model names ("our model", "the proposed model")
# using the paper title to identify the model family.
# ---------------------------------------------------------------------------

# Model family names to search for in the paper title, ordered by specificity.
_TITLE_MODEL_FAMILIES: list[str] = [
    "Llama", "LLaMA", "GPT-NeoX", "GPT-Neo", "GPT", "Claude",
    "Gemini", "Gemma", "Mistral", "Mixtral", "Qwen", "DeepSeek",
    "Falcon", "Phi", "Command", "Yi", "Vicuna", "Alpaca",
    "StarCoder", "CodeGen", "Jamba", "DBRX", "InternLM", "Baichuan",
    "OLMo", "Grok", "Solar", "Nemotron", "Pythia", "FLAN",
    "Flan-T5", "Flan-PaLM", "PaLM", "T5", "Aya",
]

# Patterns that indicate a self-referential model name
_SELF_REF_PATTERN = re.compile(
    r'^(?:our|the\s+proposed|proposed)\s+'
    r'(?:(?:new|final|best|largest|smallest|base|full)\s+)?'
    r'(\d+[bBmM])?\s*'
    r'(?:model|system|architecture)$',
    re.IGNORECASE
)


def _resolve_self_referential_name(model_name: str, paper_title: str) -> str:
    """Resolve self-referential names like 'our 7B model' using the paper title.

    Extracts the model family from the title (e.g. 'Gemma' from
    'Gemma: Open Models Based on...') and combines it with any size
    suffix from the original name.

    Returns the original name unchanged if no resolution is possible.
    """
    m = _SELF_REF_PATTERN.match(model_name.strip())
    if not m:
        return model_name

    size_suffix = m.group(1) or ""

    # Find the model family in the paper title — prefer the one that
    # appears earliest (titles typically lead with the model name).
    title_lower = paper_title.lower()
    best_family: str | None = None
    best_pos = len(title_lower) + 1
    for family in _TITLE_MODEL_FAMILIES:
        pos = title_lower.find(family.lower())
        if pos >= 0 and pos < best_pos:
            best_pos = pos
            best_family = family

    if best_family:
        resolved = f"{best_family} {size_suffix}".strip() if size_suffix else best_family
        print(
            f"    [self-ref] resolved '{model_name}' → '{resolved}' "
            f"(from title: '{paper_title[:60]}')",
            file=sys.stderr,
        )
        return resolved

    return model_name


# ---------------------------------------------------------------------------
# Model alias normalization — canonical forms for cross-paper consistency.
# ---------------------------------------------------------------------------
_MODEL_ALIASES: dict[str, str] = {
    "gpt4o": "GPT-4o",
    "gpt-4o": "GPT-4o",
    "gpt4-o": "GPT-4o",
    "gpt-4o-mini": "GPT-4o-mini",
    "gpt-4o-mini-2024-07-18": "GPT-4o-mini",
    "gpt-4-turbo": "GPT-4-Turbo",
    "gpt4-turbo": "GPT-4-Turbo",
    "claude-3.5-sonnet": "Claude-3.5-Sonnet",
    "claude-3-5-sonnet": "Claude-3.5-Sonnet",
    "claude-3-opus": "Claude-3-Opus",
    "claude-3-sonnet": "Claude-3-Sonnet",
    "claude-3-haiku": "Claude-3-Haiku",
    "gemma-2-9b": "Gemma-2-9B",
    "gemma-2-27b": "Gemma-2-27B",
    "gemma-3-27b-it": "Gemma-3-27B-IT",
    "gemma-3-12b-it": "Gemma-3-12B-IT",
    "mistral-7b": "Mistral-7B",
    "mixtral-8x7b": "Mixtral-8x7B",
    "mixtral-8x22b": "Mixtral-8x22B",
}


# ---------------------------------------------------------------------------
# benchmark metadata — lower_is_better, score ranges, score types
# ---------------------------------------------------------------------------

# Benchmarks where a lower score is better.  Matched with substring search
# against the lowercased benchmark name extracted from the table.
_LOWER_IS_BETTER_PATTERNS: tuple[str, ...] = (
    "perplexity", "ppl", "wer", "cer",
    "word error", "char error",
    "bpw", "bpc", "bits per",
    "latency", "toxicity", "hallucination",
    "error rate", "false positive", "false negative",
)

# Benchmarks whose raw scores are genuinely unbounded (e.g. perplexity).
# For these we skip the ÷100 normalisation step and record the raw value.
_UNBOUNDED_SCORE_PATTERNS: tuple[str, ...] = (
    "perplexity", "ppl", "bpw", "bpc", "bits per",
    "wer", "cer", "word error", "char error",
    "latency",
)

# Benchmarks that are already reported on a 0–100 scale in most papers and
# should be kept on that scale rather than divided to 0–1.
# We record min_score=0, max_score=100 in the schema.
_SCALE_100_PATTERNS: tuple[str, ...] = (
    "bleu", "rouge",
)

# Benchmarks with native raw-score scales that should not be normalized.
_RAW_SCALE_BENCHMARKS: dict[str, tuple[float, float]] = {
    "mt-bench": (1.0, 10.0),
    "mt bench": (1.0, 10.0),
    "alignbench": (1.0, 10.0),
    "writingbench": (1.0, 10.0),
    "chatbot arena": (0.0, 3000.0),
    "elo": (0.0, 3000.0),
}

_RELATIVE_TABLE_PATTERNS: tuple[str, ...] = (
    "relative to",
    "compared to baseline",
    "improvement over",
    "delta",
    "vs. the final data mixture",
    "vs the final data mixture",
    "change from",
    "difference from",
    "gain over",
    "ablation",
    "\u0394",
)

# Signals that a table is NOT a standard benchmark results table —
# contamination analyses, ablation studies, etc.  Matched against the
# lowercased caption and header text.
_SKIP_TABLE_SIGNALS: tuple[str, ...] = (
    "contam",
    "contamination",
    "performance gain est",
    "training data overlap",
    "data contamination",
    "data leakage",
    "similarity threshold",
    "similar data",
    "memorization",
)

_SKIP_TABLE_CAPTIONS: tuple[str, ...] = (
    "percentage of evaluation sets considered to be contaminated",
    "estimated performance gain",
    "contaminated because similar data exists",
    "data decontamination",
    "contamination analysis",
    "similarity analysis",
)

_SECTION_HEADER_PATTERNS: tuple[str, ...] = (
    "tasks",
    "coding",
    "math",
    "math & text",
    "stem",
    "multilingual",
    "multilingual tasks",
    "general",
    "reasoning",
    "agent",
    "thinking mode",
    "non-thinking mode",
    "alignment tasks",
    "alignment",
)

_PROMPT_TEMPLATE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"chatml", re.IGNORECASE), "chatml"),
    (re.compile(r"alpaca format|alpaca template", re.IGNORECASE), "alpaca"),
    (re.compile(r"llama[ -]?3(?:\.[0-9]+)? chat template", re.IGNORECASE), "llama_chat"),
)

_MODEL_ALIGNMENT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bbase\b", re.IGNORECASE), "base"),
    (re.compile(r"\binstruct\b", re.IGNORECASE), "instruct"),
    (re.compile(r"\bchat\b|\bit\b|\balign", re.IGNORECASE), "chat"),
)

_QUANTIZATION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bint4\b|\bq4\b", re.IGNORECASE), "INT4"),
    (re.compile(r"\bint8\b|\bq8\b", re.IGNORECASE), "INT8"),
    (re.compile(r"\bbf16\b", re.IGNORECASE), "BF16"),
    (re.compile(r"\bfp16\b|\bf16\b", re.IGNORECASE), "FP16"),
)

_NON_MODEL_ROW_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(?:general|tasks|reasoning|coding|alignment|thinking mode|non-thinking mode|multilingual tasks)$", re.IGNORECASE),
    re.compile(r"^(?:afro|indo|austro|uralic|tai|turkic|dravidian|asiatic)", re.IGNORECASE),
    # Reject citation-style strings: "HumanEval (Chen et al., 2021), MBPP ..."
    re.compile(r"\(.*(?:et\s+al|\d{4}).*\)", re.IGNORECASE),
    # Reject strings that are mostly benchmark names with citations
    re.compile(r"(?:^|,\s*)(?:" + _BENCH_ALT_RE + r")\s*\(", re.IGNORECASE),
)

# Eval-framework text signatures found in paper body → canonical lib name.
# Longer / more-specific strings must appear before shorter prefix strings.
_EVAL_FRAMEWORK_SIGNATURES: tuple[tuple[str, str], ...] = (
    ("lm-evaluation-harness", "lm_eval"),
    ("lm_eval",               "lm_eval"),
    ("eleutherai/lm-eval",    "lm_eval"),
    ("inspect_ai",            "inspect_ai"),
    ("inspect ai",            "inspect_ai"),
    ("stanford-crfm/helm",    "helm"),
    ("openai evals",          "openai_evals"),
    ("big-bench",             "bigbench"),
    ("bigbench",              "bigbench"),
)

# regex matching common footnote / superscript markers attached to cell values
# e.g. "85.2†", "*73.1", "92‡", "81.0 a", "77.3^{1}"
_FOOTNOTE_RE = re.compile(r"[\*†‡§¶#^]+$|^[\*†‡§¶#]")

# developer lookup: lower-cased partial model name → developer org.
# Keys are matched via startswith / substring checks in _infer_developer.
# Longer / more-specific keys must appear before shorter prefix keys so
# that e.g. "codellama" is matched before the bare "llama" entry.
_DEVELOPER_MAP: dict[str, str] = {
    # ── OpenAI ──────────────────────────────────────────────────────────
    "gpt": "openai",
    "text-davinci": "openai",
    "text-curie": "openai",
    "davinci": "openai",
    "curie": "openai",
    "o1": "openai",
    "o3": "openai",
    "openai": "openai",
    "chatgpt": "openai",
    # ── Anthropic ───────────────────────────────────────────────────────
    "claude": "anthropic",
    # ── Google / DeepMind ────────────────────────────────────────────────
    "gemini": "google",
    "gemma": "google",
    "palm": "google",
    "t5": "google",
    "ul2": "google",
    "flan": "google",
    "chinchilla": "google-deepmind",
    "gopher": "google-deepmind",
    "sparrow": "google",
    # ── Meta ────────────────────────────────────────────────────────────
    "codellama": "meta-llama",    # must precede bare "llama"
    "llama": "meta-llama",
    "opt": "meta",
    # ── Mistral AI ──────────────────────────────────────────────────────
    "mistral": "mistralai",
    "mixtral": "mistralai",
    "open-mistral": "mistralai",
    "codestral": "mistralai",
    "mathstral": "mistralai",
    # ── Qwen (Alibaba) ──────────────────────────────────────────────────
    "qwen": "Qwen",
    "qwq": "Qwen",
    # ── Microsoft ───────────────────────────────────────────────────────
    "phi": "microsoft",
    "orca": "microsoft",
    "wizardlm": "microsoft",
    # ── TII ─────────────────────────────────────────────────────────────
    "falcon": "tiiuae",
    # ── BigScience / HuggingFace ─────────────────────────────────────────
    "bloom": "bigscience",
    # ── EleutherAI ──────────────────────────────────────────────────────
    "pythia": "EleutherAI",
    "gpt-j": "EleutherAI",
    "gpt-neox": "EleutherAI",
    # ── MosaicML / Databricks ────────────────────────────────────────────
    "mpt": "mosaicml",
    "dbrx": "databricks",
    # ── DeepSeek ────────────────────────────────────────────────────────
    "deepseek": "deepseek-ai",
    # ── 01.AI ───────────────────────────────────────────────────────────
    "yi": "01-ai",
    # ── Allen AI ────────────────────────────────────────────────────────
    "olmo": "allenai",
    "tulu": "allenai",
    # ── xAI ─────────────────────────────────────────────────────────────
    "grok": "xai",
    # ── Stanford ────────────────────────────────────────────────────────
    "alpaca": "stanford",
    # ── LMSYS ───────────────────────────────────────────────────────────
    "vicuna": "lmsys",
    # ── BigCode ─────────────────────────────────────────────────────────
    "starcoder": "bigcode",
    "santacoder": "bigcode",
    # ── Salesforce ──────────────────────────────────────────────────────
    "codegen": "salesforce",
    "xgen": "salesforce",
    # ── Cohere ──────────────────────────────────────────────────────────
    "command-r": "CohereForAI",   # must precede bare "command"
    "command": "CohereForAI",
    "aya": "CohereForAI",
    # ── AI21 Labs ────────────────────────────────────────────────────────
    "jamba": "ai21labs",
    "jurassic": "ai21labs",
    # ── NVIDIA ──────────────────────────────────────────────────────────
    "nemotron": "nvidia",
    "megatron": "nvidia",
    # ── Upstage ─────────────────────────────────────────────────────────
    "solar": "upstage",
    # ── LG AI Research ──────────────────────────────────────────────────
    "exaone": "LGAI-RESEARCH",
    # ── THUDM (Tsinghua) ────────────────────────────────────────────────
    "chatglm": "THUDM",
    "glm": "THUDM",
    "codegeex": "THUDM",
    # ── Shanghai AI Lab ──────────────────────────────────────────────────
    "internlm": "internlm",
    "internvl": "OpenGVLab",
    # ── Baichuan ────────────────────────────────────────────────────────
    "baichuan": "baichuan-inc",
    # ── Together AI ──────────────────────────────────────────────────────
    "redpajama": "togethercomputer",
    # ── Writer ──────────────────────────────────────────────────────────
    "palmyra": "Writer",
    # ── Teknium / NousResearch ───────────────────────────────────────────
    "openhermes": "teknium",
    "nous-hermes": "NousResearch",
    "hermes": "NousResearch",
}

# simple in-process cache so repeated unknown model names don't all hit the
# HF Hub API during a single batch run.
_HF_AUTHOR_CACHE: dict[str, str] = {}

# Pre-sorted by key length descending so longer / more-specific patterns
# (e.g. "gpt-j", "command-r") match before their shorter prefixes ("gpt",
# "command").  Dict insertion order alone is NOT sufficient because shorter
# keys like "gpt" appear first in _DEVELOPER_MAP and would shadow "gpt-j"
# in a plain iteration, mis-attributing EleutherAI models to openai.
_DEVELOPER_PATTERNS: tuple[tuple[str, str], ...] = tuple(
    sorted(_DEVELOPER_MAP.items(), key=lambda kv: len(kv[0]), reverse=True)
)

# Patterns whose developer attribution is frequently wrong for community
# fine-tunes that borrow a well-known model name as a prefix.  When
# _infer_developer resolves via one of these, a stderr warning is emitted
# so the caller knows to verify before submission.  Examples:
#   'alpaca'  — Stanford's original, but ~100 community fine-tunes share the prefix
#   'orca'    — Microsoft's Orca, but many non-MSFT 'orca-*' fine-tunes exist
#   'hermes'  — NousResearch, but widely cloned with the same name prefix
#   'vicuna'  — LMSYS, but many third-party vicuna-based derivatives exist
_AMBIGUOUS_DEVELOPER_PATTERNS: frozenset[str] = frozenset({
    "alpaca",
    "orca",
    "hermes",
    "vicuna",
})


# ---------------------------------------------------------------------------
# S: PDF downloader — fetch only
# ---------------------------------------------------------------------------


class PDFDownloader:
    """download a PDF from arXiv or a local path."""

    def fetch(self, source: str) -> Path:
        """return a local Path to the PDF for *source*.

        *source* is either a local path or an arXiv ID (e.g. '2407.21783').
        """
        local = Path(source)
        if local.exists():
            return local

        # treat source as arXiv ID
        arxiv_id = source.strip()
        dest = _PDF_DOWNLOAD_DIR / f"{arxiv_id}.pdf"
        if dest.exists():
            return dest

        url = _ARXIV_PDF_URL.format(arxiv_id=arxiv_id)
        print(f"  downloading {url} ...")
        resp = requests.get(url, timeout=_TIMEOUT, headers={"User-Agent": "EEE-pipeline/1.0"})
        resp.raise_for_status()

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        print(f"  saved to {dest}")
        return dest


# ---------------------------------------------------------------------------
# S: table extractor — Docling table detection only
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Docling imports and shared parser
# ---------------------------------------------------------------------------
# Auto-enable HF offline mode BEFORE importing Docling / huggingface_hub,
# because huggingface_hub caches HF_HUB_OFFLINE at import time into a
# module-level constant.  Setting os.environ after import has no effect.
if not os.environ.get("HF_HUB_OFFLINE"):
    _hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    _layout_cache = _hf_home / "hub" / "models--docling-project--docling-layout-heron"
    if _layout_cache.exists():
        os.environ["HF_HUB_OFFLINE"] = "1"
        print("  [init] Docling models cached — auto-enabled HF_HUB_OFFLINE=1")

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
import pandas as pd

# If we auto-set the env var above, huggingface_hub was imported by Docling
# with the correct value.  But if the user set it themselves after Python
# started (e.g. $env:HF_HUB_OFFLINE=1 in the same shell), the constant
# may still be False.  Patch it directly to be safe.
if os.environ.get("HF_HUB_OFFLINE") == "1":
    try:
        import huggingface_hub.constants as _hf_constants
        _hf_constants.HF_HUB_OFFLINE = True
    except ImportError:
        pass


class DoclingParser:
    """Shared Docling PDF parser with per-path result caching.

    Parse each PDF once; all extractors reuse the cached result to avoid
    redundant deep-learning inference across TableExtractor, ProseExtractor,
    LLMFallbackExtractor, and _detect_eval_library.
    """

    _PARSE_TIMEOUT = 300  # seconds — generous for large PDFs

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_table_structure = True
        pipeline_options.do_ocr = False
        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=PyPdfiumDocumentBackend,
                )
            }
        )

    def parse(self, pdf_path: Path) -> Any:
        """Return the cached Docling ConversionResult for *pdf_path*.

        Catches SSL / network errors that occur when Docling tries to
        download models and provides an actionable hint.
        """
        key = str(pdf_path.resolve())
        if key not in self._cache:
            try:
                self._cache[key] = self._converter.convert(key)
            except Exception as exc:
                msg = str(exc)
                if "SSL" in msg or "huggingface" in msg.lower() or "MaxRetryError" in msg:
                    print(
                        "\n  ERROR: Docling failed to download models due to an SSL/network error.\n"
                        "  Fix: set  $env:HF_HUB_OFFLINE=1  (PowerShell) or  export HF_HUB_OFFLINE=1  (bash)\n"
                        "  before running the pipeline. The models must be cached from a prior run.\n",
                        file=sys.stderr,
                    )
                raise
        return self._cache[key]

    def get_full_text(self, pdf_path: Path) -> str:
        """Return the full document as markdown text."""
        return self.parse(pdf_path).document.export_to_markdown()


# Module-level shared parser instance — constructed lazily on first use.
_docling_parser = DoclingParser()


# ---------------------------------------------------------------------------
# S: table extractor — Docling table detection only
# ---------------------------------------------------------------------------


def _split_merged_model_names(cell: str) -> list[str]:
    """Split a header cell containing multiple space-separated model names.

    Uses ``_DEVELOPER_PATTERNS`` to detect where one model name ends and the
    next begins.  Handles Docling's detached-hyphen artefacts (e.g.
    ``"DeepSeek-R1 -Distill-Qwen-32B"`` → single model).

    Returns a list of individual model name strings.  If only one model is
    detected, the original (stripped) text is returned as a single-element
    list.
    """
    cell = cell.strip()
    if not cell:
        return []
    # Re-join detached hyphens first: "GPT-4o -2024-11-20" → "GPT-4o-2024-11-20"
    cell = re.sub(r"\s+-", "-", cell)
    # Insert spaces where Docling concatenated model names without separators.
    # Common pattern: parameter-size suffix immediately followed by start of
    # next model name, e.g. "8BLlama" → "8B Llama", "405BGPT-3.5" → "405B GPT-3.5"
    cell = re.sub(r'(\d+[bBmM])([A-Z])', r'\1 \2', cell)
    # Also handle "InstructGPT" / "TurboNemotron" — insert space before known
    # model family names when preceded by lowercase (concatenation boundary).
    _FAMILY_PREFIXES = (
        "Llama", "GPT", "Claude", "Gemini", "Gemma", "Mistral", "Mixtral",
        "Qwen", "DeepSeek", "Nemotron", "Falcon", "Phi", "Command", "Yi",
        "Vicuna", "Alpaca", "StarCoder", "CodeGen", "Jamba", "DBRX",
        "InternLM", "Baichuan", "OLMo", "Grok", "Solar",
    )
    for prefix in _FAMILY_PREFIXES:
        cell = re.sub(rf'([a-z])({re.escape(prefix)})', r'\1 \2', cell)
    tokens = cell.split()
    if len(tokens) <= 1:
        return [cell]

    models: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        token_lower = token.lower()
        first_seg = token_lower.split("-")[0] if "-" in token_lower else token_lower
        is_developer_start = False
        for pat, _dev in _DEVELOPER_PATTERNS:
            if len(pat) < 3:
                continue
            if first_seg == pat or token_lower.startswith(pat):
                is_developer_start = True
                break
        if is_developer_start and current:
            models.append(current)
            current = [token]
        else:
            current.append(token)
    if current:
        models.append(current)

    if len(models) <= 1:
        return [cell]

    # Detect trailing variant suffixes ("Base", "Instruct", "Chat", etc.)
    # that Docling appends once per model but all land on the last model.
    # E.g. "Gemma-3-4B Qwen2.5-3B Qwen3-4B Base Base Base" → 3 models,
    # 3 trailing "Base" tokens should distribute one per model.
    _VARIANT_SUFFIXES = {"base", "instruct", "chat", "it", "hf", "pretrained", "pre-trained"}
    last_model_tokens = models[-1]
    if len(last_model_tokens) > 1:
        # Find where variant suffix tokens start in the last model
        suffix_start = len(last_model_tokens)
        for i in range(1, len(last_model_tokens)):
            if last_model_tokens[i].lower() in _VARIANT_SUFFIXES:
                suffix_start = i
                break
        trailing_suffixes = last_model_tokens[suffix_start:]
        if trailing_suffixes and all(t.lower() in _VARIANT_SUFFIXES for t in trailing_suffixes):
            num_models = len(models)
            core_last = last_model_tokens[:suffix_start]
            if len(trailing_suffixes) >= num_models:
                # Distribute one suffix per model
                for j, model_tokens in enumerate(models):
                    if j < len(models) - 1:
                        model_tokens.append(trailing_suffixes[j])
                models[-1] = core_last + [trailing_suffixes[num_models - 1]]
            elif len(trailing_suffixes) == 1:
                # Single trailing suffix — append to last model only
                models[-1] = core_last + trailing_suffixes
            else:
                # Mismatch — strip all trailing suffixes from last model
                models[-1] = core_last if core_last else last_model_tokens

    result = [" ".join(m).rstrip('.') for m in models]
    return result if len(result) > 1 else [cell.rstrip('.')]


def _extract_size_tokens_from_header_cell0(cell0: str) -> list[str]:
    """Extract model-size tokens (e.g. '7B', '13B', '2B') from the first
    header cell of a collapsed table.

    Docling often puts the sub-header row content into cell 0:
       "Benchmark metric 7B 13B 7B 2B 7B"
    This function extracts ['7B', '13B', '7B', '2B', '7B'].
    """
    tokens = cell0.split()
    sizes: list[str] = []
    for t in tokens:
        cleaned = t.rstrip("*†‡∗")
        if re.match(r'^\d+[bBmM]$', cleaned):
            sizes.append(cleaned.upper() if cleaned[-1].lower() == 'b' else cleaned)
    return sizes


def _try_expand_merged_model_columns(
    table_rows: list[list[str]],
) -> list[list[str]] | None:
    """Expand models-as-columns tables where Docling merged multiple model
    columns into single cells.

    Detection:
      - At least one header cell splits into ≥ 2 model names (via developer
        prefix boundaries).
      - ≥ 30 % of data rows produce exactly *N* trailing numeric / missing-
        value tokens (where *N* = total model count from the header).

    Reconstruction treats each row as a *flat* sequence of tokens (cell
    boundaries from Docling are unreliable when columns are collapsed) and
    extracts benchmark name + N trailing score-or-dash tokens.

    Enhanced to handle:
      - Duplicate model names across header cells (de-duplication)
      - Size tokens in cell 0 that pair with family names in other cells
      - Metric/description columns between benchmark name and scores
      - Footnote markers (*, †, ∗) on size tokens

    Returns the rebuilt table, or ``None`` if expansion was not applicable.
    """
    if len(table_rows) < 3:
        return None

    # --- Step 1: split header cells into individual model names ----------
    header = table_rows[0]
    expanded_models: list[str] = []
    any_merged = False
    # Collect model names from each non-empty header cell separately so we
    # can detect duplicates across cells.
    per_cell_models: list[list[str]] = []
    for cell in header:
        stripped = cell.strip()
        if not stripped:
            per_cell_models.append([])
            continue
        parts = _split_merged_model_names(stripped)
        if len(parts) > 1:
            any_merged = True
            per_cell_models.append(parts)
        else:
            per_cell_models.append([])
    if not any_merged:
        return None

    # Flatten, but de-duplicate: if the same sequence of model names appears
    # in multiple cells (Docling duplicates spanning headers), keep only one
    # copy.  E.g. header cells 2 and 3 both contain "LLaMA-2 Mistral Gemma."
    seen_sequences: list[tuple[str, ...]] = []
    for cell_models in per_cell_models:
        if not cell_models:
            continue
        seq = tuple(cell_models)
        if seq not in seen_sequences:
            seen_sequences.append(seq)
            expanded_models.extend(cell_models)

    if len(expanded_models) < 2:
        return None

    # --- Step 1b: pair family names with size tokens from cell 0 ---------
    # Header cell 0 often contains: "Benchmark metric 7B 13B 7B 2B 7B"
    # The size tokens tell us the actual number of model columns and can be
    # paired with the family names to build proper model identifiers.
    size_tokens = _extract_size_tokens_from_header_cell0(header[0])

    # If we found size tokens AND they outnumber the expanded model names,
    # the sizes are the true column count.  Pair family names with sizes
    # by trying all valid partitions and picking the one that minimizes
    # duplicate sizes within the same family.
    if size_tokens and len(size_tokens) > len(expanded_models):
        from itertools import combinations as _combinations

        num_families = len(expanded_models)
        num_sizes = len(size_tokens)

        # --- Infer column grouping from data row cell boundaries --------
        # Even though Docling merges columns, numeric values within each
        # cell hint at the number of original columns it spanned. Count
        # numeric tokens per cell in data rows and take the mode pattern.
        cell_count_patterns: list[tuple[int, ...]] = []
        for row in table_rows[1:]:
            row_text = " ".join(cell.strip() for cell in row if cell.strip())
            if not row_text:
                continue
            tokens = row_text.split()
            if all(_parse_numeric(t) is None and t not in ("-", "–", "—") for t in tokens):
                continue
            # Count numeric tokens per cell (skip cell 0 which has benchmark)
            counts: list[int] = []
            for ci, cell in enumerate(row):
                cell_stripped = cell.strip()
                if not cell_stripped:
                    counts.append(0)
                    continue
                cell_tokens = cell_stripped.split()
                n_numeric = sum(
                    1 for t in cell_tokens
                    if _parse_numeric(t) is not None or t in ("-", "–", "—")
                )
                counts.append(n_numeric)
            cell_count_patterns.append(tuple(counts))

        # Find the most common pattern of value counts per cell
        inferred_groups: tuple[int, ...] | None = None
        if cell_count_patterns:
            mode_pattern = Counter(cell_count_patterns).most_common(1)[0][0]
            # Extract non-zero counts from cells that contain values
            # (skip cells that only had benchmark text)
            val_groups = [c for c in mode_pattern if c > 0]
            if sum(val_groups) == num_sizes:
                inferred_groups = tuple(val_groups)
            else:
                # Try: sum all value cells (some patterns may vary)
                # Use the pattern that sums to num_sizes
                for pattern, _cnt in Counter(cell_count_patterns).most_common(5):
                    vg = [c for c in pattern if c > 0]
                    if sum(vg) == num_sizes:
                        inferred_groups = tuple(vg)
                        break

        if num_families <= num_sizes:
            # Enumerate all ways to split num_sizes items into num_families
            # contiguous groups (each group gets >= 1 item).
            best_paired: list[str] | None = None
            best_unique_score = -1
            best_balance_score = float('inf')
            best_cell_match = False
            best_mono_count = -1

            for split_points in _combinations(range(1, num_sizes), num_families - 1):
                groups: list[list[str]] = []
                prev = 0
                for sp in split_points:
                    groups.append(size_tokens[prev:sp])
                    prev = sp
                groups.append(size_tokens[prev:])

                # Score 1: maximize unique sizes per family
                unique_score = sum(len(set(g)) for g in groups)
                # Score 2: minimize size imbalance
                sizes_per_group = [len(g) for g in groups]
                balance_score = max(sizes_per_group) - min(sizes_per_group)
                # Score 3: does this partition match the inferred cell grouping?
                cell_match = (
                    inferred_groups is not None
                    and len(inferred_groups) == num_families
                    and tuple(len(g) for g in groups) == inferred_groups
                )
                # Score 4: count families where sizes are monotonically ordered
                # (ascending by numeric part). Papers typically list sizes in
                # order within each family group.
                mono_count = 0
                for g in groups:
                    if len(g) <= 1:
                        mono_count += 1
                    else:
                        nums = []
                        for sz in g:
                            m_sz = re.match(r'(\d+)', sz)
                            nums.append(int(m_sz.group(1)) if m_sz else 0)
                        if nums == sorted(nums):
                            mono_count += 1

                # Prefer: cell_match > unique_score > mono_count > balance_score
                is_better = False
                if cell_match and not best_cell_match:
                    is_better = True
                elif cell_match == best_cell_match:
                    if unique_score > best_unique_score:
                        is_better = True
                    elif unique_score == best_unique_score:
                        if mono_count > best_mono_count:
                            is_better = True
                        elif mono_count == best_mono_count and balance_score < best_balance_score:
                            is_better = True

                if is_better:
                    best_unique_score = unique_score
                    best_balance_score = balance_score
                    best_cell_match = cell_match
                    best_mono_count = mono_count
                    paired: list[str] = []
                    for fi, family in enumerate(expanded_models):
                        for sz in groups[fi]:
                            paired.append(f"{family} {sz}")
                    best_paired = paired

            if best_paired and len(best_paired) == num_sizes:
                expanded_models = best_paired
                if inferred_groups is not None:
                    print(
                        f"    [expand-pair] cell-inferred grouping: {inferred_groups}, "
                        f"cell_match={best_cell_match}",
                        file=sys.stderr,
                    )
            else:
                # Fallback: just use size tokens as model names
                expanded_models = size_tokens[:num_sizes]

    num_models = len(expanded_models)
    print(
        f"    [expand-check] found {num_models} merged model names: "
        f"{expanded_models[:6]}{'...' if num_models > 6 else ''}",
        file=sys.stderr,
    )

    # --- Step 2: verify data rows match the expected value count ---------
    # Try the primary count first; if that fails, also try count from
    # size_tokens (which may differ from expanded_models if pairing failed).
    candidate_counts = [num_models]
    if size_tokens and len(size_tokens) != num_models:
        candidate_counts.append(len(size_tokens))

    _SKIP_PREFIXES = frozenset({
        "architecture", "#", "total param", "activated param",
        "# total", "# activated",
    })

    best_count: int | None = None
    best_matching = 0
    best_total = 0

    for try_count in candidate_counts:
        matching_rows = 0
        total_data_rows = 0
        for row in table_rows[1:]:
            row_text = " ".join(cell.strip() for cell in row if cell.strip())
            if not row_text:
                continue
            row_lower = row_text.lower().strip()
            if any(row_lower.startswith(sp) for sp in _SKIP_PREFIXES):
                continue
            tokens = row_text.split()
            if all(_parse_numeric(t) is None and t not in ("-", "–", "—") for t in tokens):
                continue
            total_data_rows += 1

            # Strip footnote markers (∗, *, †) that Docling sometimes puts in a separate token
            cleaned = re.sub(r"\s*(?:\+/?-|±)\s*\d+\.?\d*", "", row_text)
            cleaned = re.sub(r"(\d)-(?=\s|$)", r"\1 -", cleaned)
            # Remove isolated footnote markers: "∗" or "*" as standalone tokens
            cleaned = re.sub(r'\s+[∗\*†‡]+(?=\s|$)', '', cleaned)
            ctokens = cleaned.split()
            values: list[str] = []
            remaining = list(ctokens)
            for _ in range(try_count):
                if not remaining:
                    break
                last = remaining[-1]
                if _parse_numeric(last) is not None or last in ("-", "–", "—"):
                    values.insert(0, remaining.pop())
                else:
                    break
            if try_count == candidate_counts[0] and total_data_rows <= 2:
                _last_tok = repr(ctokens[-1][:20]) if ctokens else "N/A"
                print(
                    f"    [expand-row-debug] row_text={row_text[:80]!r} "
                    f"values={len(values)} need={try_count} "
                    f"ctokens={len(ctokens)} last_tok={_last_tok}",
                    file=sys.stderr,
                )
            if len(values) == try_count and remaining:
                nums_in_bench = sum(
                    1 for t in remaining if _parse_numeric(t) is not None
                )
                if nums_in_bench < 2:
                    matching_rows += 1

        if total_data_rows >= 2 and matching_rows >= total_data_rows * 0.3:
            if matching_rows > best_matching:
                best_count = try_count
                best_matching = matching_rows
                best_total = total_data_rows

    if best_count is None:
        # --- Step 2b: fallback — count ALL numeric tokens per row ---------
        # When Docling scatters values across cells inconsistently, trailing-
        # from-right extraction may fail. Count total numeric tokens instead.
        all_num_counts: list[int] = []
        for row in table_rows[1:]:
            row_text = " ".join(cell.strip() for cell in row if cell.strip())
            if not row_text:
                continue
            row_lower = row_text.lower().strip()
            if any(row_lower.startswith(sp) for sp in _SKIP_PREFIXES):
                continue
            tokens = row_text.split()
            if all(_parse_numeric(t) is None and t not in ("-", "–", "—") for t in tokens):
                continue
            # Strip footnote markers
            cleaned = re.sub(r'\s+[∗\*†‡]+(?=\s|$)', '', row_text)
            cleaned = re.sub(r"\s*(?:\+/?-|±)\s*\d+\.?\d*", "", cleaned)
            ctokens = cleaned.split()
            n_vals = sum(1 for t in ctokens if _parse_numeric(t) is not None)
            all_num_counts.append(n_vals)

        if all_num_counts:
            mode_count = Counter(all_num_counts).most_common(1)[0][0]
            # Check if mode_count matches any of our candidates or size_tokens
            valid_targets = set(candidate_counts)
            if size_tokens:
                valid_targets.add(len(size_tokens))
            if mode_count in valid_targets and mode_count >= 2:
                best_count = mode_count
            elif mode_count >= 2:
                # The data has a consistent value count that doesn't match our
                # model names — use it as the column count anyway, building
                # model names from size tokens if available.
                best_count = mode_count

    if best_count is None:
        print(
            f"    [expand-check] validation failed: no candidate count matched "
            f"≥30% of data rows",
            file=sys.stderr,
        )
        return None

    # If best_count differs from num_models, rebuild model names
    if best_count != num_models:
        if size_tokens and len(size_tokens) == best_count:
            # Pair family names with sizes
            unique_families: list[str] = []
            for m in expanded_models:
                if m not in unique_families:
                    unique_families.append(m)
            if len(unique_families) <= best_count:
                # Re-do family-size pairing with known count
                paired_final: list[str] = []
                sizes_per_family = best_count // len(unique_families)
                remainder = best_count % len(unique_families)
                idx = 0
                for fi, fam in enumerate(unique_families):
                    n = sizes_per_family + (1 if fi < remainder else 0)
                    for _ in range(n):
                        if idx < len(size_tokens):
                            paired_final.append(f"{fam} {size_tokens[idx]}")
                            idx += 1
                if len(paired_final) == best_count:
                    expanded_models = paired_final
                else:
                    # Just use size tokens as model names
                    expanded_models = size_tokens[:best_count]
            else:
                expanded_models = size_tokens[:best_count]
        num_models = best_count

    # Trim expanded_models to match best_count
    if len(expanded_models) > num_models:
        expanded_models = expanded_models[:num_models]
    elif len(expanded_models) < num_models:
        # Pad with generic names
        for i in range(len(expanded_models), num_models):
            expanded_models.append(f"Model_{i+1}")

    # --- Step 3: rebuild the table with one column per model -------------
    new_table: list[list[str]] = [["Benchmark"] + expanded_models]

    for row in table_rows[1:]:
        row_text = " ".join(cell.strip() for cell in row if cell.strip())
        if not row_text:
            continue
        row_lower = row_text.lower().strip()
        if any(row_lower.startswith(sp) for sp in _SKIP_PREFIXES):
            continue
        tokens_raw = row_text.split()
        if all(_parse_numeric(t) is None and t not in ("-", "–", "—") for t in tokens_raw):
            if "thinking" in row_lower:
                new_table.append([row_text] + [""] * num_models)
            continue
        # Strip footnote markers and CIs
        cleaned = re.sub(r"\s*(?:\+/?-|±)\s*\d+\.?\d*", "", row_text)
        cleaned = re.sub(r"(\d)-(?=\s|$)", r"\1 -", cleaned)
        cleaned = re.sub(r'\s+[∗\*†‡]+(?=\s|$)', '', cleaned)
        ctokens = cleaned.split()
        values = []
        remaining = list(ctokens)
        for _ in range(num_models):
            if not remaining:
                break
            last = remaining[-1]
            if _parse_numeric(last) is not None or last in ("-", "–", "—"):
                values.insert(0, remaining.pop())
            else:
                break
        if len(values) == num_models and remaining:
            bench_name = " ".join(remaining)
            if bench_name.strip():
                nums_in_bench = sum(
                    1 for t in remaining if _parse_numeric(t) is not None
                )
                if nums_in_bench < 2:
                    new_table.append([bench_name] + values)
                    continue

        # Fallback: extract ALL numeric tokens from the row and take the
        # last num_models of them (handles rows where non-numeric tokens
        # like "5-shot," appear between benchmark name and scores).
        all_vals: list[tuple[int, str]] = []
        for ti, t in enumerate(ctokens):
            if _parse_numeric(t) is not None:
                all_vals.append((ti, t))
        if len(all_vals) >= num_models:
            score_vals = [v for _, v in all_vals[-num_models:]]
            first_score_idx = all_vals[-num_models][0]
            bench_tokens = ctokens[:first_score_idx]
            # Strip trailing metric descriptors like "5-shot," "top-1", "partial", etc.
            while bench_tokens and bench_tokens[-1].rstrip(",.;:") in (
                "top-1", "top-5", "0-shot", "5-shot", "3-shot", "4-shot",
                "7-shot", "8-shot", "10-shot", "25-shot",
                "pass@1", "pass@10", "pass@100",
                "maj@1", "partial", "scoring", "avg",
                "1-shot", "shot", "top-1,",
            ):
                bench_tokens.pop()
            # Also strip "N-shot" pattern and "metric" label
            while bench_tokens and re.match(r'^\d+-shot[,.]?$', bench_tokens[-1]):
                bench_tokens.pop()
            bench_name = " ".join(bench_tokens).strip().rstrip(",")
            if bench_name:
                new_table.append([bench_name] + score_vals)

    if len(new_table) >= 3:
        print(
            f"  [table-fix] Expanded merged model columns: "
            f"{num_models} models × {len(new_table) - 1} benchmarks"
        )
        return new_table

    return None


def _try_fix_collapsed_table(table_rows: list[list[str]]) -> list[list[str]]:
    """Re-split tables where Docling merged many columns into a few cells.

    When Docling's table structure detection fails, all values for a row may
    end up concatenated in a single cell (e.g. the whole row of benchmarks +
    scores becomes one long string).  This function detects that pattern and
    reconstructs proper columns by:

    1. Finding benchmark keywords in the concatenated header text.
    2. Counting numeric values from the right of each concatenated data row.
    3. Rebuilding with one model column + N benchmark columns.

    Only activates when the table is clearly collapsed (many benchmark
    keywords in header but very few non-empty actual columns).  Tables with
    multi-word benchmark names that cannot be reliably split are left alone
    for the LLM extractor to handle.
    """
    if len(table_rows) < 3:
        return table_rows

    # --- detect collapsed structure ---
    header_text = " ".join(cell.strip() for cell in table_rows[0] if cell.strip())
    header_lower = header_text.lower()

    # Need at least 3 strong benchmark keywords in the header
    bench_kw_count = sum(1 for kw in _BENCHMARK_KEYWORDS_STRONG if kw in header_lower)
    if bench_kw_count < 3:
        return table_rows

    # Check if actual non-empty column count is far smaller than benchmark
    # keyword count — the smoking gun for a collapsed table.
    non_empty_cols_per_row = [
        sum(1 for cell in row if cell.strip()) for row in table_rows
    ]
    avg_non_empty = (
        sum(non_empty_cols_per_row) / len(non_empty_cols_per_row)
        if non_empty_cols_per_row
        else 0
    )
    if avg_non_empty >= bench_kw_count:
        return table_rows  # columns already reasonably separated

    # --- count expected score columns from data rows ---
    score_counts: list[int] = []
    for row in table_rows[1:]:
        row_text = " ".join(cell.strip() for cell in row if cell.strip())
        if not row_text.strip():
            continue
        # Remove confidence intervals before tokenising
        row_text = re.sub(r"\s*(?:\+/?-|±)\s*\d+\.?\d*", "", row_text)
        tokens = row_text.split()
        count = 0
        for t in reversed(tokens):
            if _parse_numeric(t) is not None:
                count += 1
            else:
                break
        if count > 0:
            score_counts.append(count)

    if not score_counts:
        return table_rows

    num_scores = Counter(score_counts).most_common(1)[0][0]
    if num_scores < 2:
        return table_rows

    # --- extract benchmark headers ---
    header_tokens = header_text.split()
    if len(header_tokens) <= num_scores:
        return table_rows  # can't split header

    # Find first benchmark keyword token via startswith or two-word match
    first_bench_idx: int | None = None
    for i, token in enumerate(header_tokens):
        token_lower = token.lower()
        # Single-word keyword match (startswith to handle "Arc-e" matching "arc")
        for kw in _BENCHMARK_KEYWORDS_STRONG:
            if len(kw) >= 3 and (kw == token_lower or token_lower.startswith(kw)):
                first_bench_idx = i
                break
        if first_bench_idx is not None:
            break
        # Two-word keyword match (e.g. "chatbot arena", "mt bench")
        if i < len(header_tokens) - 1:
            two_word = f"{token_lower} {header_tokens[i + 1].lower()}"
            for kw in _BENCHMARK_KEYWORDS_STRONG:
                if " " in kw and kw in two_word:
                    first_bench_idx = i
                    break
        if first_bench_idx is not None:
            break

    if first_bench_idx is None:
        first_bench_idx = len(header_tokens) - num_scores

    bench_headers = header_tokens[first_bench_idx:]

    # Match benchmark header count to score count
    if len(bench_headers) == num_scores:
        pass  # perfect 1:1 match
    elif len(bench_headers) > num_scores:
        if len(bench_headers) / num_scores > 1.5:
            # Too many header tokens relative to scores — likely multi-word
            # benchmark names that we can't reliably split by whitespace.
            # Leave this for the LLM to handle.
            return table_rows
        bench_headers = bench_headers[:num_scores]
    else:
        return table_rows  # fewer headers than scores — can't label

    # --- rebuild table ---
    new_header = ["Model"] + bench_headers
    new_table: list[list[str]] = [new_header]

    for row in table_rows[1:]:
        row_text = " ".join(cell.strip() for cell in row if cell.strip())
        if not row_text.strip():
            continue
        row_text = re.sub(r"\s*(?:\+/?-|±)\s*\d+\.?\d*", "", row_text)
        tokens = row_text.split()
        scores: list[str] = []
        remaining = list(tokens)
        for _ in range(num_scores):
            if not remaining:
                break
            if _parse_numeric(remaining[-1]) is not None:
                scores.insert(0, remaining.pop())
            else:
                break
        model_name = " ".join(remaining)
        if len(scores) == num_scores and model_name.strip():
            new_table.append([model_name] + scores)

    if len(new_table) >= 3:  # header + at least 2 data rows
        # Validate: if most reconstructed "model names" are actually benchmark
        # keywords, this is a benchmarks-as-rows table, not models-as-rows.
        # Skip reconstruction — the table parser or LLM should handle it.
        bench_like = 0
        for rebuilt_row in new_table[1:]:
            name_lower = rebuilt_row[0].lower()
            if any(kw in name_lower for kw in _BENCHMARK_KEYWORDS_STRONG):
                bench_like += 1
        if bench_like > len(new_table[1:]) * 0.4:
            return table_rows

        print(
            f"  [table-fix] Reconstructed collapsed table: "
            f"{len(new_table) - 1} data rows × {num_scores} benchmarks"
        )
        return new_table

    return table_rows


class TableExtractor:
    """extract raw tables from a PDF using Docling."""

    def extract(self, pdf_path: Path) -> list[tuple[int, list[list[str]], str]]:
        """Return list of (page_number, table_rows, context_text) from *pdf_path*.

        *context_text* is the markdown text preceding the table (caption/section header).
        """
        result = _docling_parser.parse(pdf_path)
        tables_out: list[tuple[int, list[list[str]], str]] = []
        doc = result.document
        markdown = doc.export_to_markdown()
        for table in doc.tables:
            try:
                df = table.export_to_dataframe(doc)
            except Exception:
                d = table.export_to_dict()
                rows = d.get("rows") or d.get("data") or []
                df = pd.DataFrame(rows)
            # Prepend column headers as the first row so benchmark keywords
            # (e.g. "MMLU", "GSM8K") are visible to ResultsTableParser.
            header = [str(c) for c in df.columns.tolist()]
            data_rows = df.astype(str).fillna("").values.tolist()
            table_rows = [header] + data_rows
            table_rows = _try_fix_collapsed_table(table_rows)
            page_num = getattr(table, "page_num", 1)
            context_text = getattr(table, "caption", None)
            if not context_text:
                table_md = df.to_string(index=False)
                idx = markdown.find(table_md[:40]) if table_md else -1
                if idx > 0:
                    context_text = markdown[max(0, idx - 300):idx].strip()
                else:
                    context_text = ""
            tables_out.append((page_num, table_rows, context_text or ""))
        return tables_out


# ---------------------------------------------------------------------------
# S: prose extractor — regex-based extraction from running text and captions
# ---------------------------------------------------------------------------


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


class ResultsTableParser:
    """identify and parse results tables from raw extracted table data."""

    def __init__(self) -> None:
        self._last_relative_table = False
        self._last_relative_to: str | None = None
        self._last_section_header_rows_skipped = 0

    def is_results_table(self, table: list[list[str]], context: str = "", dynamic_keywords: set[str] | None = None) -> bool:
        """Return True if *table* looks like a model evaluation results table.

        Three checks must all pass:

        1. **Benchmark keyword** — a known benchmark name appears somewhere in
           the table *or in the context text above it* (checked header-first,
           then full table, then context).
        2. **Numeric density** — at least 25 % of non-empty body cells, or at
           least 5 cells, contain a parseable score (int, float, or %).
        3. **Minimum size** — at least header row + 2 data rows, ensuring we
           don't flag single-row summary lines.

        Parameters
        ----------
        table:
            Normalised table rows (all cells are str, None already replaced).
        context:
            Plain text extracted from the region above the table on the same
            page (e.g. caption, section header).  When the table cells
            themselves carry no benchmark keyword, this text is the last
            resort before the table is rejected — it catches cases like a
            multi-part results block where the benchmark name only appears in
            the preceding paragraph.
        """
        # check 0: reject non-results tables (contamination analyses, etc.)
        # These tables contain benchmark names and numeric data but report
        # contamination deltas or performance-gain estimates, not actual scores.
        caption_lower = context.lower()
        header_lower = " ".join(cell.lower() for cell in table[0] if cell) if table else ""
        for sig in _SKIP_TABLE_CAPTIONS:
            if sig in caption_lower:
                return False
        for sig in _SKIP_TABLE_SIGNALS:
            if sig in caption_lower or sig in header_lower:
                return False

        # check 3: minimum size
        if len(table) < 3:
            return False

        # check 1: benchmark keyword presence — two-tier strategy:
        #   • one *strong* match (specific benchmark name) is sufficient;
        #   • generic metric terms (_BENCHMARK_KEYWORDS_WEAK) require ≥2
        #     distinct matches to reduce false positives from non-eval tables
        #     that happen to mention "accuracy" or "f1" (e.g. a confusion
        #     matrix or a hyperparameter ablation table).
        # Scan order: header row → second row → full table → context text.
        def _has_eval_signal(text: str) -> bool:
            effective_strong = _BENCHMARK_KEYWORDS_STRONG | (dynamic_keywords or set())
            if any(kw in text for kw in effective_strong):
                return True
            weak_hits = sum(1 for kw in _BENCHMARK_KEYWORDS_WEAK if kw in text)
            return weak_hits >= 2

        header_text = " ".join(cell.lower() for cell in table[0] if cell)
        has_benchmark = _has_eval_signal(header_text)
        if not has_benchmark and len(table) > 1:
            second_text = " ".join(c.lower() for c in table[1] if c)
            has_benchmark = _has_eval_signal(second_text)
        # slow path: scan the entire table (handles multi-row headers and
        # tables where benchmark names appear in the first column)
        if not has_benchmark:
            all_text = " ".join(
                cell.lower() for row in table for cell in row if cell
            )
            has_benchmark = _has_eval_signal(all_text)
        # last resort: check the caption / section header above the table.
        # Accept only a strong keyword here to avoid false positives from
        # sections like "Experimental Setup" that mention metric names in
        # passing (weak matches require 2 hits, which is strict enough).
        if not has_benchmark and context:
            has_benchmark = _has_eval_signal(context.lower())
        if not has_benchmark:
            return False

        # check 2: numeric density — also match percentages and negatives
        # Use _parse_numeric to mirror the actual parsing logic, which handles
        # footnote markers (85.2†), bold residue (\textbf{85}), parenthesised
        # values, etc. that a simple regex would miss.
        numeric_cells = 0
        non_empty_cells = 0
        for row in table[1:]:
            for cell in row:
                stripped = cell.strip()
                if stripped:
                    non_empty_cells += 1
                    if _parse_numeric(stripped) is not None:
                        numeric_cells += 1

        if non_empty_cells == 0:
            return False
        # require either an absolute floor of 5 numeric cells (small tables)
        # or 25 % density (larger tables).  For large tables (8+ body rows)
        # with confirmed benchmark keywords, use a lower threshold (3 cells
        # or 15% density) to catch wide post-training results tables where
        # Docling sometimes fails to parse all numeric cells correctly.
        density = numeric_cells / non_empty_cells
        n_body_rows = len(table) - 1
        if n_body_rows >= 8:
            # Large table with benchmark keywords already confirmed (check 1)
            if not (numeric_cells >= 3 or density >= 0.15):
                return False
        else:
            if not (numeric_cells >= 5 or density >= 0.25):
                return False

        # check 4: reject ablation tables — first column dominated by ablation
        # marker language rather than distinct model names from different orgs.
        # Ablation tables pass checks 1–3 but generate EEE records where every
        # entry describes the same underlying model under ablation conditions,
        # making the schema data ambiguous to query.
        if self._is_ablation_table(table):
            return False

        # check 5: reject architecture/configuration tables — these list
        # layer counts, head counts, hidden dims etc. which pass density
        # checks but are not evaluation results.
        _ARCH_TABLE_KEYWORDS = (
            "model architecture", "architecture of", "model configuration",
            "hyperparameter", "training configuration", "model specification",
            "hidden size", "num layers", "num heads",
        )
        # caption_lower and header_lower already computed in check 0
        # Only apply caption-based architecture rejection when the table body
        # does NOT contain a strong benchmark keyword.  This prevents false
        # rejections of results tables whose context (preceding prose) happens
        # to mention "model architecture" or similar terms.
        table_body_text = " ".join(
            cell.lower() for row in table for cell in row if cell
        )
        _table_has_bench_kw = any(kw in table_body_text for kw in (_BENCHMARK_KEYWORDS_STRONG | (dynamic_keywords or set())))
        if not _table_has_bench_kw and any(kw in caption_lower for kw in _ARCH_TABLE_KEYWORDS):
            return False
        if any(kw in header_lower for kw in ("layers", "heads", "hidden dim", "hidden size", "vocab size", "params", "parameters")):
            # If header mentions architecture columns AND lacks benchmark names,
            # it's an architecture table
            arch_cols = sum(1 for kw in ("layers", "heads", "hidden", "vocab", "params", "embed") if kw in header_lower)
            if arch_cols >= 2:
                return False

        return True

    @staticmethod
    def _is_ablation_table(table: list[list[str]]) -> bool:
        """Return True when the first column looks like an ablation study.

        Ablation tables compare variants of a single model (e.g. "w/o layer
        norm", "base + FT", "+ DPO") rather than multiple independent models.
        Heuristic: if ≥ 40 % of non-empty first-column cells contain an
        ablation marker token, classify as ablation.  The 40 % threshold
        tolerates one legitimate baseline row while catching tables that are
        predominantly ablation rows.
        """
        _ABLATION_MARKERS: frozenset[str] = frozenset({
            "w/o", "without", "ablat", "no ", "+ ", " + ",
            "only", "ours w", "base model", "full model",
            "remove", "drop ", "variant",
        })
        first_col = [
            row[0].lower().strip()
            for row in table[1:]
            if row and row[0].strip()
        ]
        if not first_col:
            return False
        ablation_count = sum(
            1 for v in first_col
            if any(
                marker in v
                for marker in _ABLATION_MARKERS
                # skip the "no " marker when the cell mentions "prompt"
                if not (marker == "no " and "prompt" in v)
            )
        )
        return ablation_count / len(first_col) >= 0.40

    def extraction_confidence(self, table: list[list[str]], context: str = "", dynamic_keywords: set[str] | None = None) -> str:
        """Return a confidence tier for how reliably this table was extracted.

        Calibration
        -----------
        'high'   — ≥1 strong benchmark keyword in the table body AND numeric
                   density ≥ 50 %.  The table is unambiguously a results table.
        'medium' — ≥1 strong keyword with density 25–50 %, OR the table only
                   matched via the caption / context rather than its own cells.
        'low'    — Accepted on weak-keyword evidence (≥2 weak matches) or on
                   the numeric-count floor only (≥5 cells but < 25 % density).

        A 'low' result is not an error — the table still passed
        `is_results_table` and is included — but benchmark→score assignments
        should be treated with more scepticism and manually reviewed before
        submission.
        """
        all_text = " ".join(cell.lower() for row in table for cell in row if cell)
        context_lower = context.lower()

        effective_strong = _BENCHMARK_KEYWORDS_STRONG | (dynamic_keywords or set())
        strong_in_body = sum(
            1 for kw in effective_strong if kw in all_text
        )
        strong_in_context_only = (
            strong_in_body == 0
            and any(kw in context_lower for kw in effective_strong)
        )

        # recompute numeric density (same logic as is_results_table)
        numeric_cells = 0
        non_empty_cells = 0
        for row in table[1:]:
            for cell in row:
                stripped = cell.strip()
                if stripped:
                    non_empty_cells += 1
                    if _parse_numeric(stripped) is not None:
                        numeric_cells += 1
        density = (numeric_cells / non_empty_cells) if non_empty_cells > 0 else 0.0

        if strong_in_body >= 1 and density >= 0.50:
            return "high"
        if (strong_in_body >= 1 and density >= 0.25) or strong_in_context_only:
            return "medium"
        return "low"

    def is_relative_table(self, table: list[list[str]], context: str = "") -> tuple[bool, str | None]:
        """Detect tables that report relative/delta scores."""
        sign_prefixed = 0
        numeric_cells = 0
        for row in table[1:]:
            for cell in row:
                stripped = cell.strip()
                if not stripped:
                    continue
                if re.fullmatch(r"[+-]?\d+(?:\.\d+)?%?", stripped):
                    numeric_cells += 1
                    if stripped.startswith("+") or stripped.startswith("-"):
                        sign_prefixed += 1
        sign_ratio = (sign_prefixed / numeric_cells) if numeric_cells else 0.0
        context_lower = context.lower()
        has_relative_caption = any(phrase in context_lower for phrase in _RELATIVE_TABLE_PATTERNS)
        if sign_ratio > 0.50 or has_relative_caption:
            return True, self._extract_relative_to(context)
        return False, None

    def _extract_relative_to(self, context: str) -> str | None:
        patterns = (
            re.compile(r"relative to\s+([^\.;:,]+)", re.IGNORECASE),
            re.compile(r"compared to baseline\s+([^\.;:,]+)", re.IGNORECASE),
            re.compile(r"improvement over\s+([^\.;:,]+)", re.IGNORECASE),
            re.compile(r"vs\.?\s+the final data mixture", re.IGNORECASE),
        )
        for pattern in patterns:
            match = pattern.search(context)
            if match:
                if match.lastindex:
                    return match.group(1).strip()
                return match.group(0).strip()
        return None

    def _is_section_header_row(self, row: list[str]) -> bool:
        non_empty = [cell.strip() for cell in row if cell and cell.strip()]
        if len(non_empty) == 1:
            lowered = non_empty[0].lower()
            return any(pattern in lowered for pattern in _SECTION_HEADER_PATTERNS)
        return False

    # Patterns whose section header should be appended to model names to
    # disambiguate rows that would otherwise produce identical model entries.
    _THINKING_MODE_PATTERNS: tuple[str, ...] = (
        "thinking mode",
        "non-thinking mode",
    )

    def _get_thinking_mode_suffix(self, row: list[str]) -> str | None:
        """If *row* is a thinking-mode section header, return a suffix string."""
        non_empty = [cell.strip() for cell in row if cell and cell.strip()]
        if len(non_empty) != 1:
            return None
        lowered = non_empty[0].lower()
        if "non-thinking" in lowered or "non thinking" in lowered:
            return "(Non-Thinking)"
        if "thinking" in lowered and "non" not in lowered:
            return "(Thinking)"
        return None

    def _model_signal_count(self, value: str) -> int:
        lowered = value.lower()
        count = 0
        for pattern, _developer in _DEVELOPER_PATTERNS:
            if len(pattern) < 3:
                continue
            if lowered.startswith(pattern) or f" {pattern}" in lowered or f"-{pattern}" in lowered:
                count += 1
        return count

    def _looks_like_model_name(self, value: str) -> bool:
        lowered = value.lower().strip()
        if not lowered:
            return False
        if any(pattern.search(lowered) for pattern in _NON_MODEL_ROW_PATTERNS):
            return False
        # Check against both hardcoded and dynamically-detected benchmark names
        effective_bench = _BENCHMARK_KEYWORDS_STRONG | getattr(self, '_dynamic_keywords', set())
        if any(keyword in lowered for keyword in effective_bench):
            return False
        if self._model_signal_count(lowered) >= 2 and len(lowered.split()) > 3:
            return False
        if any(pattern in lowered for pattern in _SECTION_HEADER_PATTERNS):
            return False
        # Reject overly long strings — real model names are rarely > 60 chars
        if len(value.strip()) > 60:
            return False
        # Reject strings with citation markers (et al., year references)
        if re.search(r"et\s+al\.?", lowered):
            return False
        if "/" in value or any(ch.isdigit() for ch in value) or "(" in value or ")" in value:
            return True
        if self._model_signal_count(lowered) >= 1:
            return True
        return False

    def parse(
        self, table: list[list[str]], context: str = "",
        dynamic_keywords: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """parse *table* into a list of {model, benchmark, score} dicts.

        handles both orientations (models-as-rows and models-as-columns).
        Also attempts to collapse multi-row headers into a single header row
        when the first row is mostly empty (common in LaTeX multicolumn tables).
        """
        # Store dynamic keywords for use by _looks_like_model_name()
        self._dynamic_keywords = dynamic_keywords or set()
        results: list[dict[str, Any]] = []
        self._last_section_header_rows_skipped = 0
        self._last_relative_table, self._last_relative_to = self.is_relative_table(table, context)

        if not table or not table[0]:
            return results

        # Try expanding merged model columns FIRST — _collapse_multirow_header
        # would absorb metadata rows (Architecture, # Params) into the header
        # and destroy the model-name information we need for splitting.
        expanded = _try_expand_merged_model_columns(table)
        if expanded is not None:
            # Expansion already rebuilt the table into proper models-as-columns
            # form.  Skip the orientation heuristic — substring keyword checks
            # like "em" can false-positive on model names (e.g. "Gemini").
            table = expanded
            results = self._parse_models_as_columns(table)
        else:
            table = self._collapse_multirow_header(table)

            header = [_clean_cell(c) for c in table[0]]

            # determine orientation:
            # models-as-rows: first column contains model names, header has benchmarks
            # models-as-columns: first row is benchmarks, first column is metric names

            effective_bench_kw = _BENCHMARK_KEYWORDS | self._dynamic_keywords
            benchmarks_in_header = [
                h for h in header if any(kw in h.lower() for kw in effective_bench_kw)
            ]

            # Guard: if a header cell matches benchmark keywords but is
            # actually a model name containing a benchmark word (e.g.
            # "ARC-Instruct-7B"), verify it doesn't also look like a model.
            if benchmarks_in_header:
                confirmed = [
                    h for h in benchmarks_in_header
                    if not self._looks_like_model_name(h)
                ]
                if not confirmed:
                    benchmarks_in_header = []

            if benchmarks_in_header:
                results = self._parse_models_as_rows(header, table[1:])
            else:
                results = self._parse_models_as_columns(table)

        score_type = "relative" if self._last_relative_table else "absolute"
        for item in results:
            item.setdefault("score_type", score_type)
            item.setdefault("relative_to", self._last_relative_to)

        return results

    @staticmethod
    def _collapse_multirow_header(table: list[list[str]]) -> list[list[str]]:
        """merge row 0 and row 1 into a single header when row 0 is sparse.

        Many LaTeX-generated PDF tables have a top row with group labels and
        an immediately following subheader row with benchmark names.  When the
        first row has more than half its cells empty, we join the two rows
        cell-by-cell so downstream parsing sees a single enriched header.

        Smart joining: when row0 has a group label (e.g. "MMLU") spanning
        multiple columns and row1 has sub-benchmarks (e.g. "Pro", "Redux"),
        the merge produces "MMLU Pro", "MMLU Redux" — propagating the group
        label to each sub-column.
        """
        collapsed = table
        for _ in range(4):
            if len(collapsed) < 2:
                return collapsed
            row0 = collapsed[0]
            row1 = collapsed[1]
            header_text = " ".join(cell.lower() for cell in row0 if cell)

            # If row 0 already has benchmark keywords, it might be a real
            # header — but only if it's NOT a sparse group-label row.
            # A sparse row (>50 % cells empty) with benchmark keywords like
            # "MMLU" is a group-label row that still needs collapsing with
            # the sub-header row below it.
            non_empty_r0 = sum(1 for c in row0 if c and c.strip())
            is_sparse = non_empty_r0 < max(len(row0), 1) * 0.5
            if any(kw in header_text for kw in _BENCHMARK_KEYWORDS):
                if not is_sparse:
                    return collapsed
                # sparse group-label row — continue to collapse
            merged: list[str] = []

            # Propagate group labels: if row0 has a non-empty cell followed
            # by empty cells, those empty cells share the same group label.
            # e.g. row0 = ["Model", "MMLU", "", "", "Code", ""]
            #   →  groups = ["Model", "MMLU", "MMLU", "MMLU", "Code", "Code"]
            propagated_row0: list[str] = []
            last_label = ""
            for i in range(max(len(row0), len(row1))):
                c0 = row0[i].strip() if i < len(row0) else ""
                if c0:
                    last_label = c0
                    propagated_row0.append(c0)
                else:
                    propagated_row0.append(last_label)

            for i in range(max(len(propagated_row0), len(row1))):
                c0 = propagated_row0[i] if i < len(propagated_row0) else ""
                c1 = row1[i].strip() if i < len(row1) else ""
                if c0 and c1:
                    # Avoid "MMLU MMLU" duplication when both rows have
                    # the same text.
                    if c1.lower().startswith(c0.lower()):
                        merged.append(c1)
                    elif c0.lower().startswith(c1.lower()):
                        merged.append(c0)
                    else:
                        merged.append(f"{c0} {c1}")
                else:
                    merged.append(c0 or c1)
            collapsed = [merged] + collapsed[2:]
        return collapsed

    def _parse_models_as_rows(
        self, header: list[str], body: list[list[str]]
    ) -> list[dict]:
        """models are rows, benchmarks are columns."""
        results: list[dict] = []
        modality_suffixes = {
            "pretrained", "finetuned", "instruct", "chat", "rlhf",
            "sft", "base", "fine-tuned", "pre-trained",
        }
        # first column is model name; rest are benchmark columns
        bench_names = [_clean_cell(h) for h in header[1:]]
        expected_cols = len(bench_names)
        total_data_rows = 0
        misaligned_rows = 0
        current_mode_suffix: str | None = None
        for row in body:
            if not row or not row[0]:
                continue
            # Check for thinking mode section headers BEFORE the generic check
            mode_suffix = self._get_thinking_mode_suffix(row)
            if mode_suffix is not None:
                current_mode_suffix = mode_suffix
                self._last_section_header_rows_skipped += 1
                continue
            if self._is_section_header_row(row):
                # Non-thinking-mode section header (e.g. "Math", "Reasoning")
                # within a thinking mode section should NOT reset the mode.
                # Only a thinking/non-thinking header changes the current mode.
                # If we're not in any thinking context, it's just a category
                # header — leave mode as None.
                self._last_section_header_rows_skipped += 1
                continue
            model_name = _clean_cell(row[0])
            model_parts = model_name.split()
            if model_parts and model_parts[-1].lower() in modality_suffixes:
                model_name = " ".join(model_parts[:-1])
            # Handle "Thinking Mode <model>" prefix pattern (e.g. Qwen3 paper)
            _inline_thinking_mode: str | None = None
            if model_name.startswith("Thinking Mode "):
                _inline_thinking_mode = "(Thinking)"
                model_name = model_name[len("Thinking Mode "):]
            elif model_name.startswith("Non-Thinking Mode "):
                _inline_thinking_mode = "(Non-Thinking)"
                model_name = model_name[len("Non-Thinking Mode "):]
            if not model_name or _is_separator(model_name) or not self._looks_like_model_name(model_name):
                continue
            # Append thinking mode suffix to disambiguate models.
            # Priority: inline prefix > section header context.
            effective_mode = _inline_thinking_mode or current_mode_suffix
            if effective_mode:
                model_name = f"{model_name} {effective_mode}"
            # A row should have 1 model-name cell + expected_cols score cells.
            # Fewer columns means pdfplumber collapsed a \multicolumn span,
            # silently shifting all subsequent scores one or more positions left.
            actual_data_cols = len(row) - 1
            total_data_rows += 1
            if actual_data_cols != expected_cols:
                misaligned_rows += 1
            for i, bench in enumerate(bench_names):
                if not bench:
                    continue
                cell_idx = i + 1
                if cell_idx >= len(row):
                    continue
                raw_cell = row[cell_idx].strip()
                # Detect merged multi-column cells: if the header contains
                # multiple space-separated benchmark names and the data cell
                # contains the same number of space-separated numeric values,
                # split them into individual benchmark-score pairs.
                bench_parts = bench.split()
                cell_parts = raw_cell.split()
                # Extract shots from the bench name before any stripping
                shot_match = re.search(r'[\(\s](\d+)-shot[\)\s,.]?', bench, re.IGNORECASE)
                bench_shots = int(shot_match.group(1)) if shot_match else None
                if len(bench_parts) > 1 and len(bench_parts) == len(cell_parts):
                    parsed_parts = [_parse_numeric_with_pct(p) for p in cell_parts]
                    if all(v is not None for v, _ in parsed_parts):
                        for bp, (sp, had_pct) in zip(bench_parts, parsed_parts):
                            results.append(
                                {
                                    "model": model_name,
                                    "benchmark": bp,
                                    "score": sp,
                                    "score_type": "absolute",
                                    "relative_to": None,
                                    "_had_pct": had_pct,
                                    "_bench_shots": bench_shots,
                                }
                            )
                        continue
                score, had_pct = _parse_numeric_with_pct(raw_cell)
                if score is None:
                    continue
                results.append(
                    {
                        "model": model_name,
                        "benchmark": bench,
                        "score": score,
                        "score_type": "absolute",
                        "relative_to": None,
                        "_had_pct": had_pct,
                        "_bench_shots": bench_shots,
                    }
                )
        # Warn when misalignment is widespread — this means \multicolumn spanning
        # has corrupted the column→benchmark mapping for a significant fraction of
        # rows.  The threshold of 15 % is intentionally loose to avoid noise from
        # rare footnote rows that add an extra cell.
        if total_data_rows > 0 and misaligned_rows / total_data_rows > 0.15:
            print(
                f"  warning: {misaligned_rows}/{total_data_rows} data rows have a "
                f"column-count mismatch (expected {expected_cols} benchmark cols). "
                "Likely cause: \\multicolumn spanning — some scores may be "
                "assigned to the wrong benchmark. Review this table manually."
            )
        return results

    def _parse_models_as_columns(
        self, table: list[list[str]]
    ) -> list[dict]:
        """models are columns, benchmarks are rows."""
        results: list[dict] = []
        if len(table[0]) < 2:
            return results

        model_names = [
            _clean_cell(c) if self._looks_like_model_name(_clean_cell(c)) else ""
            for c in table[0][1:]
        ]

        current_mode_suffix: str | None = None
        for row in table[1:]:
            if not row or not row[0]:
                continue
            # Check for thinking mode section headers BEFORE the generic check
            mode_suffix = self._get_thinking_mode_suffix(row)
            if mode_suffix is not None:
                current_mode_suffix = mode_suffix
                self._last_section_header_rows_skipped += 1
                continue
            if self._is_section_header_row(row):
                # Preserve thinking mode context across sub-category headers
                self._last_section_header_rows_skipped += 1
                continue
            bench = _clean_cell(row[0])
            if not bench or _is_separator(bench):
                continue
            for i, model in enumerate(model_names):
                if not model:
                    continue
                effective_model = f"{model} {current_mode_suffix}" if current_mode_suffix else model
                cell_idx = i + 1
                if cell_idx >= len(row):
                    continue
                score, had_pct = _parse_numeric_with_pct(row[cell_idx])
                if score is None:
                    continue
                shot_match = re.search(r'[\(\s](\d+)-shot[\)\s,.]?', bench, re.IGNORECASE)
                bench_shots = int(shot_match.group(1)) if shot_match else None
                results.append(
                    {
                        "model": effective_model,
                        "benchmark": bench,
                        "score": score,
                        "score_type": "absolute",
                        "relative_to": None,
                        "_had_pct": had_pct,
                        "_bench_shots": bench_shots,
                    }
                )
        return results


# ---------------------------------------------------------------------------
# S: paper converter — schema conversion only
# ---------------------------------------------------------------------------


class PaperConverter:
    """convert extracted paper results to EEE schema dicts."""

    def __init__(self, gemini_extractor: "GeminiProtocolExtractor | None" = None) -> None:
        self.gemini_extractor = gemini_extractor or GeminiProtocolExtractor()

    def convert(
        self,
        extracted: list[dict[str, Any]],
        arxiv_id: str,
        retrieved_timestamp: str,
        paper_title: str = "",
        paper_authors: list[str] | None = None,
        eval_library: EvalLibrary | None = None,
        gemini_raw: dict[str, Any] | None = None,
        protocol_map: dict[str, dict[str, str | int]] | None = None,
    ) -> list[dict]:
        """group *extracted* by model and produce one EEE record per model."""
        # group results by model name
        by_model: dict[str, list[dict]] = {}
        for item in extracted:
            model = item["model"]
            by_model.setdefault(model, []).append(item)


        eval_name = _make_eval_name(arxiv_id)
        source_metadata = _make_source_metadata(arxiv_id, paper_title)
        lib = eval_library if eval_library is not None else EvalLibrary(name="unknown", version="unknown")

        records: list[dict] = []
        for model_name, items in by_model.items():
            try:
                record = self._convert_model(
                    model_name,
                    items,
                    eval_name,
                    arxiv_id,
                    retrieved_timestamp,
                    source_metadata,
                    lib,
                    paper_authors=paper_authors or [],
                    gemini_raw=gemini_raw,
                    protocol_map=protocol_map,
                )
                records.append(record)
            except Exception as exc:
                print(f"  skipping model {model_name!r}: {exc}")

        # Post-processing: infer "non-thinking" for untagged entries when a
        # thinking counterpart exists for the same base model.
        # Collect model IDs that have thinking-mode records.
        thinking_model_ids: set[str] = set()
        for rec in records:
            for er in rec.get("evaluation_results", []):
                ga = (er.get("generation_config") or {}).get("generation_args") or {}
                if ga.get("reasoning_mode") == "thinking":
                    thinking_model_ids.add(rec.get("model_info", {}).get("id", ""))
                    break
        # Tag untagged records for models that also have thinking entries.
        if thinking_model_ids:
            for rec in records:
                mid = rec.get("model_info", {}).get("id", "")
                if mid not in thinking_model_ids:
                    continue
                for er in rec.get("evaluation_results", []):
                    ga = (er.get("generation_config") or {}).get("generation_args")
                    if ga is None:
                        continue
                    if not ga.get("reasoning_mode"):
                        ga["reasoning_mode"] = "non-thinking"
                        ga["reasoning"] = False

        return records

    def _convert_model(
        self,
        model_name: str,
        items: list[dict],
        eval_name: str,
        arxiv_id: str,
        retrieved_timestamp: str,
        source_metadata: SourceMetadata,
        eval_library: EvalLibrary,
        paper_authors: list[str] | None = None,
        gemini_raw: dict[str, Any] | None = None,
        protocol_map: dict[str, dict[str, str | int]] | None = None,
    ) -> dict:
        """build one EvaluationLog dict for *model_name*."""
        # Extract thinking mode suffix before normalization
        thinking_mode: str | None = None
        if model_name.endswith("(Thinking)"):
            thinking_mode = "thinking"
            model_name = model_name.rsplit("(Thinking)", 1)[0].strip()
        elif model_name.endswith("(Non-Thinking)"):
            thinking_mode = "non-thinking"
            model_name = model_name.rsplit("(Non-Thinking)", 1)[0].strip()
        # Handle prefix-style thinking mode: "Thinking Mode Qwen3-32B"
        elif model_name.startswith("Thinking Mode "):
            thinking_mode = "thinking"
            model_name = model_name[len("Thinking Mode "):]
        elif model_name.startswith("Non-Thinking Mode "):
            thinking_mode = "non-thinking"
            model_name = model_name[len("Non-Thinking Mode "):]

        # Guard: only tag reasoning_mode for model families that actually
        # support thinking/non-thinking mode selection.  Baseline models
        # (e.g. GPT-4o) appearing in a "Non-Thinking" table section
        # should NOT be tagged.
        if thinking_mode and not _should_tag_thinking_mode(model_name):
            thinking_mode = None

        # Resolve self-referential names like "our 7B model" using paper title
        model_name = _resolve_self_referential_name(
            model_name, source_metadata.source_name or ""
        )
        # Normalise stray spaces before hyphens: "Gemma-3 -27B-IT" → "Gemma-3-27B-IT"
        model_name = re.sub(r'\s+-', '-', model_name)
        # Normalize parameter-size casing: "27b" → "27B"
        model_name = _normalize_model_name(model_name)

        if gemini_raw is not None and OPENROUTER_API_KEY:
            protocol_map = self.gemini_extractor.to_protocol_map(
                gemini_raw, model_name=model_name
            )
        elif protocol_map is None:
            protocol_map = {}

        developer = _infer_developer(model_name)
        if "/" in model_name:
            model_id = model_name
            developer = model_id.split("/")[0]
        else:
            model_id = f"{developer}/{model_name}"

        safe_id = model_id.replace("/", "_")
        evaluation_id = f"{eval_name}/{safe_id}/{retrieved_timestamp}"

        model_metadata = _infer_model_metadata(model_name)

        # Compute evaluator relationship for source_metadata.
        eval_relationship = _infer_evaluator_relationship(
            developer, paper_authors or [], source_metadata.source_name or ""
        )

        seen_benchmarks: dict[tuple[str, str], EvaluationResult] = {}
        for item in items:
            bench = item["benchmark"].strip()
            # Normalise stray spaces before hyphens: "GPQA -Diamond" → "GPQA-Diamond"
            bench = re.sub(r'\s+-', '-', bench)
            # Canonicalize benchmark name (MATH500→MATH-500, etc.)
            bench = _normalize_aime(bench)
            normalized = _normalize_benchmark(bench)
            if normalized is None:
                continue  # skip non-benchmarks like "Average"
            bench = normalized


            # Build per-benchmark source_data so sample_size is benchmark-specific
            source_data = SourceDataUrl(
                dataset_name="arXiv paper",
                source_type="url",
                url=[f"https://arxiv.org/abs/{arxiv_id}"],
                dataset_split=_infer_dataset_split(bench),
                sample_size=_infer_sample_size(bench),
            )
            bench_shots = item.get("_bench_shots")
            dedup_key = (bench, bench_shots, thinking_mode or "")
            if dedup_key in seen_benchmarks:
                # Keep higher-confidence entry; if same confidence, first wins
                existing = seen_benchmarks[dedup_key]
                existing_conf = existing.generation_config.additional_details.get(
                    "extraction_confidence", "low"
                ) if existing.generation_config and existing.generation_config.additional_details else "low"
                new_conf = item.get("_extraction_confidence", "low")
                _conf_order = {"high": 3, "medium": 2, "low": 1}
                if _conf_order.get(new_conf, 0) <= _conf_order.get(existing_conf, 0):
                    continue  # existing is same or higher confidence — keep it
                # else: new entry has higher confidence — replace below

            raw_score = float(item["score"])
            bench_lower = bench.lower()

            lower_is_better = any(p in bench_lower for p in _LOWER_IS_BETTER_PATTERNS)
            is_unbounded    = any(p in bench_lower for p in _UNBOUNDED_SCORE_PATTERNS)
            is_scale_100    = any(p in bench_lower for p in _SCALE_100_PATTERNS)
            raw_scale_range = next(
                (
                    score_range
                    for pattern, score_range in _RAW_SCALE_BENCHMARKS.items()
                    if pattern in bench_lower
                ),
                None,
            )

            if raw_scale_range is not None:
                score = round(raw_score, 4)
                min_score, max_score = raw_scale_range
            elif is_unbounded:
                # keep raw value; no meaningful upper bound
                score = round(raw_score, 4)
                min_score: float = 0.0
                max_score: float | None = None
            elif is_scale_100:
                # BLEU / ROUGE: keep on 0–100 scale
                score = round(raw_score, 4)
                min_score = 0.0
                max_score = 100.0
            elif raw_score > 1.0:
                # assume percentage — normalise to 0–1
                score = round(raw_score / 100.0, 4)
                min_score = 0.0
                max_score = 1.0
            elif _is_likely_fraction(bench, raw_score):
                # Score <= 1.0 on a benchmark typically reported as a
                # percentage (e.g. MMLU 0.623 = 62.3%).  Keep as-is on
                # the 0-1 scale — do NOT divide by 100.
                score = round(raw_score, 4)
                min_score = 0.0
                max_score = 1.0
            elif item.get("_had_pct") and raw_score <= 1.0:
                # explicit '%' sign present (e.g. "1.0%") — treat as
                # percentage even though the numeric value is <= 1
                score = round(raw_score / 100.0, 4)
                min_score = 0.0
                max_score = 1.0
            else:
                score = round(raw_score, 4)
                min_score = 0.0
                max_score = 1.0

            protocol = _lookup_protocol(protocol_map, bench, shots=bench_shots)
            # Shots priority: protocol map > benchmark name suffix > item-level (LLM-extracted)
            protocol_shots = protocol.get("shots")
            item_shots = item.get("shots")
            shots = protocol_shots if protocol_shots is not None else (
                bench_shots if bench_shots is not None else item_shots
            )
            scoring_method = protocol.get("method") or item.get("scoring_method")
            # Temperature, top_p, and chain_of_thought: protocol takes
            # precedence (paper-level methodology), item-level as fallback.
            temperature = _coerce_float(protocol.get("temperature") or item.get("temperature"))
            top_p = _coerce_float(protocol.get("top_p") or item.get("top_p"))
            top_k = _coerce_float(protocol.get("top_k") or item.get("top_k"))
            max_tokens = _coerce_int(protocol.get("max_tokens") or item.get("max_tokens"))
            chain_of_thought = _coerce_bool(protocol.get("chain_of_thought") if protocol.get("chain_of_thought") is not None else item.get("chain_of_thought"))
            prompt_template = protocol.get("prompt_template") or item.get("prompt_template")

            generation_details: dict[str, Any] = {
                "source": f"arXiv:{arxiv_id}",
                "extraction_confidence": item.get("_extraction_confidence", _aggregate_extraction_confidence(items)),
                "harness": eval_library.name,
            }
            if eval_library.version != "unknown":
                generation_details["eval_library_version"] = eval_library.version
            if scoring_method:
                generation_details["scoring_method"] = scoring_method
            if protocol.get("protocol_source"):
                generation_details["protocol_source"] = protocol["protocol_source"]
            if item.get("page_number") is not None:
                generation_details["page_number"] = item["page_number"]
            table_ref = item.get("table_id") or item.get("table")
            if table_ref:
                generation_details["table"] = table_ref
            if item.get("table_caption"):
                generation_details["table_caption"] = item["table_caption"]
            if thinking_mode:
                generation_details["thinking_mode"] = thinking_mode
            if len(generation_details) == 2:
                generation_details["note"] = "generation config not reported in source paper"

            score_details_extra: dict[str, Any] = {}
            score_type = item.get("score_type")
            if score_type and score_type != "unknown":
                score_details_extra["score_type"] = score_type
            relative_to = item.get("relative_to")
            if relative_to:
                score_details_extra["relative_to"] = relative_to
                score_details_extra["comparability_warning"] = (
                    "Relative score reported against a paper-specific baseline; avoid direct comparison to absolute leaderboard scores."
                )

            seen_benchmarks[dedup_key] = EvaluationResult(
                evaluation_name=bench,
                source_data=source_data,
                metric_config=MetricConfig(
                    evaluation_description=f"score on {bench} as reported in arXiv:{arxiv_id}",
                    metric_name=_infer_metric_name(bench),
                    lower_is_better=lower_is_better,
                    score_type=ScoreType.continuous,
                    min_score=min_score,
                    max_score=max_score,
                ),
                score_details=ScoreDetails(
                    score=score,
                    details=_stringify_detail_values(score_details_extra) or None,
                ),
                generation_config=GenerationConfig(
                    generation_args=(
                        GenerationArgs(
                            shots=_coerce_int(shots),
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            max_tokens=max_tokens,
                            reasoning=chain_of_thought or (thinking_mode == "thinking") or None,
                            reasoning_mode=thinking_mode,
                            chain_of_thought=chain_of_thought,
                            prompt_template=str(prompt_template) if prompt_template else None,
                        )
                        if any(
                            value is not None
                            for value in (
                                _coerce_int(shots),
                                temperature,
                                top_p,
                                top_k,
                                max_tokens,
                                chain_of_thought,
                                thinking_mode,
                                prompt_template,
                            )
                        )
                        else None
                    ),
                    additional_details=_stringify_detail_values(generation_details),
                ),
            )

        eval_results: list[EvaluationResult] = list(seen_benchmarks.values())

        model_source_metadata = SourceMetadata(
            source_name=source_metadata.source_name,
            source_type=source_metadata.source_type,
            source_organization_name=source_metadata.source_organization_name,
            source_organization_url=source_metadata.source_organization_url,
            evaluator_relationship=EvaluatorRelationship(eval_relationship),
        )

        log = EvaluationLog(
            schema_version=_SCHEMA_VERSION,
            evaluation_id=evaluation_id,
            retrieved_timestamp=retrieved_timestamp,
            source_metadata=model_source_metadata,
            eval_library=eval_library,
            model_info=ModelInfo(
                name=model_name,
                id=model_id,
                developer=developer,
                parameter_count=model_metadata["parameter_count"],
                model_alignment=model_metadata["model_alignment"],
                quantization=model_metadata["quantization"],
                # inference_platform left None — unknown from paper tables
            ),
            evaluation_results=eval_results,
        )

        record = log.model_dump(mode='json', exclude_none=True)
        return record


# ---------------------------------------------------------------------------
# S: paper writer — file I/O only
# ---------------------------------------------------------------------------


class PaperWriter:
    """write EEE schema dicts to disk under data/{benchmark}/{developer}/{model}/."""

    def write(self, records: list[dict], eval_name: str) -> list[Path]:
        """Write per-benchmark JSON files to data/{benchmark}/{developer}/{model}/.

        Each evaluation result in a record is written as a separate file
        following the EEE repository structure:
            data/{benchmark}/{developer_name}/{model_name}/{uuid}.json

        Deduplicates on (model_id, benchmark, score, source_url) to avoid
        writing the same result twice when both table and LLM extraction
        find it.
        """
        paths: list[Path] = []
        seen: set[tuple[str, str, str, str]] = set()

        def _sanitize_part(value: str) -> str:
            value = value.replace(" ", "_")
            return re.sub(r'[^A-Za-z0-9_.-]', "_", value)

        for rec in records:
            try:
                model_id: str = rec["model_info"]["id"]
                if "/" in model_id:
                    developer, model_name = model_id.split("/", 1)
                else:
                    developer = rec["model_info"].get("developer", "unknown")
                    model_name = model_id

                evaluation_results = rec.get("evaluation_results", [])
                for evaluation_result in evaluation_results:
                    benchmark_name = evaluation_result.get("evaluation_name", "unknown")
                    score = str(evaluation_result.get("score_details", {}).get("score", ""))
                    # reasoning_mode lives inside each evaluation_result's
                    # generation_config.generation_args, not at the record top level.
                    gen_args = (evaluation_result.get("generation_config") or {}).get("generation_args") or {}
                    reasoning_mode = gen_args.get("reasoning_mode", "") or ""
                    shots_val = str(gen_args.get("shots", ""))
                    dedup_key = (model_id, benchmark_name, score, reasoning_mode, shots_val)
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)

                    split_record = json.loads(json.dumps(rec))
                    split_record["evaluation_results"] = [evaluation_result]
                    split_dir = (
                        Path("data")
                        / _sanitize_part(benchmark_name)
                        / _sanitize_part(developer)
                        / _sanitize_part(model_name)
                    )
                    split_dir.mkdir(parents=True, exist_ok=True)
                    split_path = split_dir / f"{uuid.uuid4()}.json"
                    with open(split_path, "w", encoding="utf-8") as fh:
                        json.dump(split_record, fh, indent=2, ensure_ascii=False)
                    paths.append(split_path)
            except Exception as exc:
                mid = rec.get("model_info", {}).get("id", "?")
                print(f"  warning: could not write record for {mid}: {exc}")

        return paths


# ---------------------------------------------------------------------------
# S: LLM fallback extractor — prose and figure-caption extraction only
# ---------------------------------------------------------------------------


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

                    # Strip markdown code fences that some models wrap around JSON
                    _stripped = raw_json.strip()
                    if _stripped.startswith("```"):
                        # Remove opening fence line (```json or ```)
                        _stripped = _stripped.split("\n", 1)[-1]
                        # Remove closing fence
                        if _stripped.rstrip().endswith("```"):
                            _stripped = _stripped.rstrip()[:-3].rstrip()
                        raw_json = _stripped

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


class PaperExtractionPipeline:
    """orchestrate download → extract → parse → convert → write for a paper."""

    def __init__(
        self,
        downloader: PDFDownloader,
        table_extractor: TableExtractor,
        table_parser: ResultsTableParser,
        converter: PaperConverter,
        writer: PaperWriter,
        llm_fallback: LLMFallbackExtractor | None = None,
        prose_extractor: ProseExtractor | None = None,
        protocol_extractor: EvalProtocolExtractor | None = None,
    ) -> None:
        self._downloader = downloader
        self._extractor = table_extractor
        self._parser = table_parser
        self._converter = converter
        self._writer = writer
        self._llm_fallback = llm_fallback
        self._prose_extractor = prose_extractor
        self.gemini_extractor = GeminiProtocolExtractor()
        self._converter.gemini_extractor = self.gemini_extractor
        self.regex_extractor = protocol_extractor or EvalProtocolExtractor()

    def run(self, source: str, *, force_llm: bool = False) -> CoverageStats:
        """process *source* (arXiv ID or PDF path) end-to-end.

        Parameters
        ----------
        source:
            arXiv paper ID (e.g. ``2407.21783``) or path to a local PDF.
        force_llm:
            When True, run the LLM extractor unconditionally alongside table
            parsing.  When False (default) and ``OPENROUTER_API_KEY`` is set,
            the LLM extractor still runs as the **primary** extraction channel;
            table and prose parsing serve as validation and augmentation.  When
            no API key is available, falls back to table + prose only.

        Returns
        -------
        CoverageStats
            Per-paper extraction quality metrics suitable for batch aggregation.
        """
        stats = CoverageStats(arxiv_id=source)
        retrieved_timestamp = str(time.time())
        arxiv_id = _normalise_arxiv_id(source)
        stats.arxiv_id = arxiv_id
        eval_name = _make_eval_name(arxiv_id)

        print(f"\n[{arxiv_id}] fetching paper metadata from arXiv...")
        arxiv_meta = _fetch_arxiv_metadata(arxiv_id)
        paper_title = arxiv_meta.get("title", "")
        paper_authors: list[str] = arxiv_meta.get("authors", [])
        if paper_title:
            print(f"  title: {paper_title}")
        else:
            print("  title not found — using arXiv ID as fallback")
        if paper_authors:
            print(f"  authors: {len(paper_authors)} found")

        print(f"[{arxiv_id}] downloading PDF...")
        pdf_path = self._downloader.fetch(source)

        # ── Phase 1: Protocol + eval library detection (LLM-primary) ──
        protocol_map: dict[str, dict[str, str | int]] = {}
        shared_gemini_raw: dict[str, Any] | None = None

        print(f"[{arxiv_id}] detecting eval framework and protocol...")
        dynamic_keywords: set[str] = set()
        if OPENROUTER_API_KEY:
            # Run protocol extraction and benchmark identification sequentially
            # to avoid hitting rate limits from parallel large requests.
            gemini_raw = self.gemini_extractor.extract_raw(pdf_path, arxiv_id)
            bench_info = self.gemini_extractor.identify_benchmarks(pdf_path, arxiv_id)

            if self.gemini_extractor.last_call_failed:
                shared_gemini_raw = None
                eval_library = _detect_eval_library(pdf_path)
            else:
                eval_library_name = self.gemini_extractor.get_eval_library(gemini_raw)
                eval_library = EvalLibrary(name=eval_library_name, version="unknown")
                shared_gemini_raw = gemini_raw
                # Diagnostic: show protocol extraction summary
                dp = gemini_raw.get("default_protocol") or {}
                bc = dp.get("benchmark_configs") or {}
                spb = dp.get("shots_per_benchmark") or {}
                n_bench = len(set(bc.keys()) | set(spb.keys()))
                n_overrides = len(gemini_raw.get("model_overrides") or [])
                print(f"  protocol: {n_bench} benchmark configs, {n_overrides} model overrides")
                if dp.get("temperature") is not None:
                    print(f"  default temperature: {dp['temperature']}")
                if dp.get("scoring_method"):
                    print(f"  default scoring: {dp['scoring_method']}")

            dynamic_keywords = {kw.lower() for kw in bench_info.get("known_benchmarks", [])}
            if dynamic_keywords:
                print(f"  LLM-identified benchmarks: {', '.join(sorted(dynamic_keywords))}")
        else:
            shared_gemini_raw = None
            eval_library = _detect_eval_library(pdf_path)

        if shared_gemini_raw is not None:
            default_protocol = shared_gemini_raw.get("default_protocol") or {}
            override_entries = shared_gemini_raw.get("model_overrides") or []
            # Count benchmarks from both new (benchmark_configs) and old (shots_per_benchmark) formats
            bench_configs = default_protocol.get("benchmark_configs") or {}
            shots_per_bench = default_protocol.get("shots_per_benchmark") or {}
            default_count = len(set(bench_configs.keys()) | set(shots_per_bench.keys()))
            override_count = sum(
                len(
                    set((override.get("benchmark_configs") or {}).keys())
                    | set((override.get("shots_per_benchmark") or {}).keys())
                )
                for override in override_entries
            )
            stats.protocol_found = default_count + override_count
            if stats.protocol_found == 0:
                stats.protocol_missing = 1
        elif protocol_map:
            stats.protocol_found = sum(1 for key in protocol_map if key != "__default__")
        else:
            stats.protocol_missing = 1

        # ── Phase 2: LLM score extraction (primary when API key available) ──
        llm_extracted: list[dict] = []
        has_llm = self._llm_fallback is not None and OPENROUTER_API_KEY
        if has_llm:
            print(f"[{arxiv_id}] LLM score extraction (primary)...")
            llm_extracted = self._llm_fallback.extract(pdf_path, arxiv_id)
            stats.llm_data_points = len(llm_extracted)
            print(
                f"  LLM extracted {len(llm_extracted)} data point(s) "
                f"({len({it['model'] for it in llm_extracted})} models, "
                f"{len({it['benchmark'] for it in llm_extracted})} benchmarks)"
            )

        # ── Phase 3: Table parsing (validation + augmentation) ──
        print(f"[{arxiv_id}] extracting tables ({pdf_path.name})...")
        try:
            raw_tables = self._extractor.extract(pdf_path)
        except Exception as exc:
            msg = str(exc)
            if "SSL" in msg or "huggingface" in msg.lower() or "MaxRetryError" in msg:
                print(
                    f"  ERROR: Docling table extraction failed (SSL/network).\n"
                    f"  Set $env:HF_HUB_OFFLINE=1 before running the pipeline.\n"
                    f"  Continuing with LLM extraction only...",
                    file=sys.stderr,
                )
                raw_tables = []
            else:
                raise
        stats.tables_scanned = len(raw_tables)
        print(f"  found {len(raw_tables)} tables across all pages")

        _has_keyword_re = re.compile(
            "|".join(re.escape(kw) for kw in _BENCHMARK_KEYWORDS), re.IGNORECASE
        )

        table_extracted: list[dict] = []
        for page_num, table, context_text in raw_tables:
            table_text = " ".join(cell for row in table for cell in row if cell)
            has_kw = bool(_has_keyword_re.search(table_text)) or bool(
                _has_keyword_re.search(context_text)
            )

            # Log explicit skip reasons for contamination/non-results tables
            _ctx_lower = context_text.lower()
            _hdr_lower = " ".join(cell.lower() for cell in table[0] if cell) if table else ""
            _skip_reason = ""
            for _sig in _SKIP_TABLE_CAPTIONS:
                if _sig in _ctx_lower:
                    _skip_reason = f"caption matches: '{_sig}'"
                    break
            if not _skip_reason:
                for _sig in _SKIP_TABLE_SIGNALS:
                    if _sig in _ctx_lower or _sig in _hdr_lower:
                        _skip_reason = f"signal '{_sig}' in caption/header"
                        break
            if _skip_reason:
                print(
                    f"  page {page_num}: non-results table skipped "
                    f"({len(table)} rows) — {_skip_reason}"
                )
                continue

            if self._parser.is_results_table(table, context_text, dynamic_keywords=dynamic_keywords):
                confidence = self._parser.extraction_confidence(table, context_text, dynamic_keywords=dynamic_keywords)
                items = self._parser.parse(table, context_text, dynamic_keywords=dynamic_keywords)
                table_id = _extract_table_id(context_text)
                table_caption = _clean_caption(context_text)
                for item in items:
                    item["_extraction_confidence"] = confidence
                    item["page_number"] = page_num
                    item["table_id"] = table_id
                    item["table_caption"] = table_caption
                table_extracted.extend(items)
                stats.results_tables_accepted += 1
                if self._parser._last_relative_table:
                    stats.relative_tables += 1
                stats.section_header_rows_skipped += self._parser._last_section_header_rows_skipped
                print(
                    f"  page {page_num}: results table accepted "
                    f"({len(table)} rows, {len(items)} data points, "
                    f"confidence={confidence})"
                )
                if len(items) == 0 and len(table) >= 5:
                    # Log diagnostic for large accepted tables that yield nothing
                    _hdr = " | ".join(table[0][:6])
                    _r1 = " | ".join(table[1][:6]) if len(table) > 1 else ""
                    print(
                        f"    [diag] 0 data from {len(table)}-row table; "
                        f"header: {_hdr[:120]}",
                        file=sys.stderr,
                    )
                    print(
                        f"    [diag] row 1: {_r1[:120]}",
                        file=sys.stderr,
                    )
            elif has_kw:
                stats.density_rejected += 1
                print(
                    f"  page {page_num}: table has benchmark keywords "
                    f"but failed density check ({len(table)} rows) — skipped"
                )

        stats.table_data_points = len(table_extracted)

        # ── Phase 4: Prose extraction (always runs — lightweight regex) ──
        prose_new: list[dict] = []
        if self._prose_extractor is not None:
            print(f"[{arxiv_id}] scanning prose and figure captions...")
            prose_raw = self._prose_extractor.extract(pdf_path, arxiv_id)
            # Deduplicate against table
            table_key_set: set[tuple[str, str]] = {
                _merge_key(it["model"], it["benchmark"])
                for it in table_extracted
            }
            prose_new = [
                it for it in prose_raw
                if _merge_key(it["model"], it["benchmark"])
                not in table_key_set
            ]
            stats.prose_data_points = len(prose_new)

        # ── Phase 5: Merge — LLM-primary, table validates/augments ──
        if has_llm and llm_extracted:
            # LLM is the primary source.  Table-parsed results with HIGH
            # confidence override LLM results for the same (model, benchmark)
            # pair.  All other table results augment the LLM output (fill
            # gaps for benchmarks the LLM missed).
            llm_keys: dict[tuple[str, str], dict] = {
                _merge_key(it["model"], it["benchmark"]): it
                for it in llm_extracted
            }
            table_high_conf = [
                it for it in table_extracted
                if it.get("_extraction_confidence") in ("high", "medium")
            ]
            table_rest = [
                it for it in table_extracted
                if it.get("_extraction_confidence") not in ("high", "medium")
            ]

            # Pre-group table entries by merge key to detect score variants
            # (e.g. thinking vs non-thinking from different tables).
            from collections import defaultdict as _defaultdict
            _table_by_key: dict[tuple[str, str], list[dict]] = _defaultdict(list)
            for t_item in table_high_conf:
                key = _merge_key(t_item["model"], t_item["benchmark"])
                _table_by_key[key].append(t_item)

            # For each key, identify distinct score clusters and pick primary.
            overrides = 0
            variant_count = 0
            # primary_for_key: best table entry per key (overrides LLM)
            # variants_for_key: additional entries with significantly different scores
            primary_for_key: dict[tuple[str, str], dict] = {}
            variants_for_key: dict[tuple[str, str], list[dict]] = {}
            for key, t_items in _table_by_key.items():
                # Collect unique scores (rounded to avoid float noise)
                seen_scores: dict[float, dict] = {}
                for t in t_items:
                    try:
                        s = round(float(t.get("score", 0)), 6)
                        if s not in seen_scores:
                            seen_scores[s] = t
                    except (TypeError, ValueError):
                        pass
                if not seen_scores:
                    continue
                # Pick primary (highest score; for tie, first seen)
                sorted_items = sorted(
                    seen_scores.values(),
                    key=lambda x: float(x.get("score", 0)),
                    reverse=True,
                )
                primary = sorted_items[0]
                primary_for_key[key] = primary
                # All other distinct scores that differ by >10% from primary
                # are score variants (likely thinking/non-thinking).
                variants = []
                primary_score = float(primary.get("score", 0))
                for alt in sorted_items[1:]:
                    alt_score = float(alt.get("score", 0))
                    if primary_score and abs(primary_score - alt_score) / max(abs(primary_score), 0.01) > 0.10:
                        variants.append(alt)
                if variants:
                    variants_for_key[key] = variants

            # Override LLM entries with primary table entries
            for key, primary in primary_for_key.items():
                if key in llm_keys:
                    existing_score = llm_keys[key].get("score")
                    tbl_score = primary.get("score")
                    if existing_score is not None and tbl_score is not None:
                        try:
                            ex_f = float(existing_score)
                            tb_f = float(tbl_score)
                            diff = abs(ex_f - tb_f)
                            if diff > 0.5 or (ex_f and diff / max(abs(ex_f), 0.01) > 0.10):
                                print(
                                    f"  [cross-val] {primary['model']}\u00d7{primary['benchmark']}: "
                                    f"LLM={existing_score}, table={tbl_score} \u2014 "
                                    f"using table (high confidence)",
                                    file=sys.stderr,
                                )
                        except (TypeError, ValueError):
                            pass
                    llm_keys[key] = primary
                    overrides += 1

            # Start with (possibly overridden) LLM results
            merged: list[dict] = list(llm_keys.values())
            merged_key_set = set(llm_keys.keys())

            # Augment with high-confidence table results the LLM missed
            for key, primary in primary_for_key.items():
                if key not in merged_key_set:
                    merged.append(primary)
                    merged_key_set.add(key)

            # Add all score variants (thinking/non-thinking splits)
            for key, variants in variants_for_key.items():
                for v_item in variants:
                    merged.append(v_item)
                    variant_count += 1
                    print(
                        f"  [cross-val] {v_item['model']}\u00d7{v_item['benchmark']}: "
                        f"variant score={v_item.get('score')} "
                        f"(possible thinking/non-thinking split)",
                        file=sys.stderr,
                    )

            # Augment with remaining table results that the LLM missed
            for t_item in table_rest:
                key = _merge_key(t_item["model"], t_item["benchmark"])
                if key not in merged_key_set:
                    merged.append(t_item)
                    merged_key_set.add(key)

            # Augment with prose results that neither LLM nor table found
            for p_item in prose_new:
                key = _merge_key(p_item["model"], p_item["benchmark"])
                if key not in merged_key_set:
                    merged.append(p_item)
                    merged_key_set.add(key)

            extracted = merged
            if variant_count:
                print(
                    f"  [merge] {variant_count} variant score(s) kept "
                    f"(possible thinking/non-thinking splits)"
                )
            if overrides:
                print(
                    f"  [merge] {overrides} LLM result(s) overridden by "
                    f"high-confidence table data"
                )

        else:
            # No LLM available — fall back to table > prose merge (old behaviour)
            combined_so_far = table_extracted + prose_new
            # If table+prose yielded nothing and LLM extractor exists but
            # wasn't run yet (no API key), we can't help further.
            extracted = combined_so_far

        # Post-extraction sanity filter: drop citation-as-model, year-as-score, etc.
        extracted = _sanitize_extracted(extracted, known_benchmarks=dynamic_keywords)

        # Determine source tier(s) for the coverage report.
        sources_used: list[str] = []
        if has_llm and llm_extracted:
            sources_used.append("llm")
        if table_extracted:
            sources_used.append("table")
        if prose_new:
            sources_used.append("prose")
        stats.source = "+".join(sources_used) if sources_used else "none"

        stats.unique_models = len({item["model"] for item in extracted})
        stats.unique_benchmarks = len({item["benchmark"] for item in extracted})

        print(
            f"\n[{arxiv_id}] extraction summary:\n"
            f"  tables scanned:          {stats.tables_scanned}\n"
            f"  results tables accepted: {stats.results_tables_accepted}\n"
            f"  keyword-only rejects:    {stats.density_rejected}  "
            f"(potential false negatives — review manually)\n"
            f"  table data points:       {stats.table_data_points}\n"
            f"  prose data points (new): {stats.prose_data_points}\n"
            f"  llm data points:         {stats.llm_data_points}\n"
            f"  total data points:       {len(extracted)}\n"
            f"  unique models:           {stats.unique_models}\n"
            f"  unique benchmarks:       {stats.unique_benchmarks}\n"
            f"  relative tables:         {stats.relative_tables}\n"
            f"  protocol entries found:  {stats.protocol_found}\n"
            f"  section headers skipped: {stats.section_header_rows_skipped}\n"
            f"  source:                  {stats.source}"
        )

        if not extracted:
            print(f"[{arxiv_id}] no results found — skipping")
            return stats

        records = self._converter.convert(
            extracted, arxiv_id, retrieved_timestamp,
            paper_title=paper_title,
            paper_authors=paper_authors,
            eval_library=eval_library,
            gemini_raw=shared_gemini_raw,
            protocol_map=protocol_map,
        )
        print(f"[{arxiv_id}] converted {len(records)} model records")

        unknown_devs = [
            r["model_info"]["id"]
            for r in records
            if r.get("model_info", {}).get("developer") == "unknown"
        ]
        if unknown_devs:
            print(
                f"  [developer-inference] {len(unknown_devs)} model(s) have "
                f"developer='unknown' \u2014 verify these before submission:\n"
                + "\n".join(f"    {mid}" for mid in unknown_devs)
            )

        paths = self._writer.write(records, eval_name)
        stats.files_written = len(paths)
        print(f"[{arxiv_id}] written {len(paths)} files to data/{{benchmark}}/...")

        passed, failed = self._validate_written(paths, arxiv_id)
        stats.validation_passed = passed
        stats.validation_failed = failed

        return stats

    def _validate_written(self, paths: list[Path], arxiv_id: str) -> tuple[int, int]:
        """Validate each written JSON file against eval.schema.json.

        Returns
        -------
        tuple[int, int]
            ``(passed, failed)`` counts.  Failures are non-fatal so a batch
            run can continue; the caller should inspect warnings and fix
            source data before submitting a PR.
        """
        import json as _json
        from jsonschema.validators import validator_for

        schema_path = _ROOT / "eval.schema.json"
        if not schema_path.exists():
            print(f"  [validation] schema not found at {schema_path} — skipping")
            return 0, 0

        with schema_path.open() as fh:
            schema = _json.load(fh)
        validator_cls = validator_for(schema)
        validator = validator_cls(schema)

        passed = 0
        failed = 0
        for path in paths:
            try:
                with path.open() as fh:
                    instance = _json.load(fh)
                validator.validate(instance)
                passed += 1
            except Exception as exc:
                failed += 1
                print(f"  [validation] FAIL {path.name}: {exc}")

        status = "all passed" if failed == 0 else f"{failed} FAILED"
        print(
            f"[{arxiv_id}] schema validation: {passed} passed, {failed} failed — {status}"
        )
        return passed, failed


# ---------------------------------------------------------------------------
# pure helpers — no side effects
# ---------------------------------------------------------------------------


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

    # reject empty/None model names
    if not model or model.strip().lower() in ("none", "null", "n/a", ""):
        return False
    # reject model names that are prose fragments (start with articles/prepositions)
    model_stripped = model.strip()
    if re.match(r'^(?:as|the|a|an|of|in|on|for|with|from|to|is|are|was|were|this|that|our|their)\s',
                model_stripped, re.IGNORECASE):
        return False
    # reject model names that are descriptions rather than names
    _DESC_PATTERNS = (
        "parameter model", "trained on", "retrained on",
        "fine-tuned on", "finetuned on", "pruned data",
        "filtered subset", "unfiltered",
    )
    model_lower_tmp = model_stripped.lower()
    if any(p in model_lower_tmp for p in _DESC_PATTERNS):
        return False
    # reject Model_N placeholder names
    if re.fullmatch(r'Model_\d+', model_stripped):
        return False
    # reject model names with repeated shot/mode info:
    # "PaLM 2-L 1-shot 1-shot 1-shot 1-shot"
    if re.search(r'(\b\d+-shot\b).*\1.*\1', model_stripped):
        return False
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
    protocol_map: dict[str, dict[str, str | int]],
    benchmark: str,
    shots: int | None = None,
) -> dict[str, str | int]:
    if not protocol_map:
        return {}

    norm = _normalise_benchmark_key(benchmark)

    # Pass 1: exact match with shot count qualifier if we have one
    # e.g. key "MMLU (5-shot)" preferred over key "MMLU" when shots=5
    if shots is not None:
        shot_qualified = f"{norm} ({shots}-shot)"
        for key, value in protocol_map.items():
            if key == "__default__":
                continue
            if _normalise_benchmark_key(key) == shot_qualified:
                merged = dict(protocol_map.get("__default__", {}))
                merged.update(value)
                return merged

    # Pass 2: exact match on benchmark name alone
    for key, value in protocol_map.items():
        if key == "__default__":
            continue
        if _normalise_benchmark_key(key) == norm:
            merged = dict(protocol_map.get("__default__", {}))
            merged.update(value)
            return merged

    # Pass 3: best substring match (prefer longest matching key)
    best_key: str | None = None
    best_value: dict[str, str | int] | None = None
    best_len = 0
    for key, value in protocol_map.items():
        if key == "__default__":
            continue
        norm_key = _normalise_benchmark_key(key)
        if norm in norm_key or norm_key in norm:
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


def _detect_eval_library(pdf_path: Path) -> EvalLibrary:
    """scan the first few pages of *pdf_path* for eval-framework signatures.

    Looks for known framework strings in the paper body (abstract + intro)
    and returns the matching EvalLibrary.  Falls back to ``internal`` for
    papers that do not clearly name a public harness.
    """
    try:
        full_text = _docling_parser.get_full_text(pdf_path).lower()
    except Exception:  # noqa: BLE001
        return EvalLibrary(name="internal", version="unknown")
    cutoff_markers = (
        "references\n",
        "bibliography\n",
        "acknowledgements\n",
        "acknowledgments\n",
    )
    cutoff_positions = [full_text.find(marker) for marker in cutoff_markers if full_text.find(marker) != -1]
    search_text = full_text[:min(cutoff_positions)] if cutoff_positions else full_text
    for signature, lib_name in _EVAL_FRAMEWORK_SIGNATURES:
        if signature.lower() in search_text:
            print(f"  detected eval framework: {lib_name}")
            return EvalLibrary(name=lib_name, version="unknown")
    return EvalLibrary(name="internal", version="unknown")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def _build_pipeline(
    llm_fallback: LLMFallbackExtractor | None = None,
) -> PaperExtractionPipeline:
    """construct the default pipeline with all default implementations."""
    return PaperExtractionPipeline(
        downloader=PDFDownloader(),
        table_extractor=TableExtractor(),
        table_parser=ResultsTableParser(),
        converter=PaperConverter(),
        writer=PaperWriter(),
        llm_fallback=llm_fallback,
        prose_extractor=ProseExtractor(),
        protocol_extractor=EvalProtocolExtractor(),
    )


def main() -> None:
    """command-line entry point for the paper extraction pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="extract LLM evaluation results from arXiv PDFs to EEE schema"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--arxiv_id",
        metavar="ID",
        help="arXiv paper ID (e.g. 2407.21783)",
    )
    src.add_argument(
        "--pdf",
        metavar="PATH",
        help="path to a local PDF file",
    )
    src.add_argument(
        "--batch",
        metavar="FILE",
        help="text file with one arXiv ID per line",
    )
    src.add_argument(
        "--convert",
        metavar="PATH",
        help=(
            "convert evaluation framework logs (lm-eval, Inspect, HELM) "
            "to EEE schema. Pass a log file or directory."
        ),
    )
    parser.add_argument(
        "--framework",
        choices=["lm_eval", "inspect", "helm", "auto"],
        default="auto",
        help="evaluation framework (for --convert mode, default: auto-detect)",
    )
    parser.add_argument(
        "--llm-fallback",
        action="store_true",
        default=False,
        help=(
            "activate the LLM score extractor as the primary extraction "
            "channel. When set and OPENROUTER_API_KEY is available, the LLM "
            "extracts all scores first, then table parsing validates and "
            "augments the LLM output. Highly recommended for papers with "
            "non-standard table layouts or sparse prose-embedded results."
        ),
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        default=False,
        help=(
            "disable LLM extraction entirely; rely only on table parsing "
            "and prose regex. Useful when no API key is available or for "
            "testing the deterministic pipeline."
        ),
    )
    parser.add_argument(
        "--llm-model",
        metavar="MODEL",
        default="anthropic/claude-haiku-4.5",
        help="OpenRouter model to use for LLM extraction (default: anthropic/claude-haiku-4.5)",
    )
    args = parser.parse_args()

    # ── Converter mode: delegate to eval_converters ──
    if args.convert:
        from scripts.convert_eval_logs import main as _convert_main
        import sys as _sys
        convert_argv = ["--log_path", args.convert, "--framework", args.framework]
        _sys.argv = ["convert_eval_logs.py"] + convert_argv
        _convert_main()
        return

    # LLM extractor is the primary channel when API key is available,
    # unless explicitly disabled with --no-llm.  --llm-fallback is kept
    # for backwards compatibility and is equivalent to the new default.
    llm_fallback: LLMFallbackExtractor | None = None
    if not args.no_llm and OPENROUTER_API_KEY:
        llm_fallback = LLMFallbackExtractor(model=args.llm_model)
    elif args.llm_fallback:
        # explicit flag — construct even if API key check would fail
        # (LLMFallbackExtractor will raise at call time)
        llm_fallback = LLMFallbackExtractor(model=args.llm_model)

    pipeline = _build_pipeline(llm_fallback=llm_fallback)
    all_stats: list[CoverageStats] = []

    use_llm = llm_fallback is not None

    if args.arxiv_id:
        stats = pipeline.run(args.arxiv_id, force_llm=use_llm)
        all_stats.append(stats)
    elif args.pdf:
        stats = pipeline.run(args.pdf, force_llm=use_llm)
        all_stats.append(stats)
    elif args.batch:
        batch_file = Path(args.batch)
        if not batch_file.exists():
            print(f"batch file not found: {batch_file}", file=sys.stderr)
            sys.exit(1)
        ids = [
            line.strip()
            for line in batch_file.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        print(f"processing {len(ids)} arXiv papers from {args.batch}...")
        for i, arxiv_id in enumerate(ids):
            try:
                stats = pipeline.run(arxiv_id, force_llm=use_llm)
                all_stats.append(stats)
            except Exception as exc:
                print(f"  error processing {arxiv_id}: {exc}")
                all_stats.append(CoverageStats(arxiv_id=arxiv_id, source="none"))

            # Rate limit buffer between papers
            if i < len(ids) - 1:
                time.sleep(5)

        # Write machine-readable coverage report for batch runs.
        report_dir = Path("scripts/scrapers/raw")
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"coverage_report_{int(time.time())}.json"
        report = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "batch_file": str(args.batch),
            "total_papers": len(all_stats),
            "papers_with_results": sum(1 for s in all_stats if s.total_data_points > 0),
            "papers_with_no_results": sum(1 for s in all_stats if s.total_data_points == 0),
            "total_files_written": sum(s.files_written for s in all_stats),
            "total_validation_failed": sum(s.validation_failed for s in all_stats),
            # source is now a '+'-joined string of tiers used, e.g. "table+prose",
            # so we count substring membership rather than exact equality.
            "source_breakdown": {
                "table_only": sum(1 for s in all_stats if s.source == "table"),
                "prose_only": sum(1 for s in all_stats if s.source == "prose"),
                "llm_only": sum(1 for s in all_stats if s.source == "llm"),
                "has_table": sum(1 for s in all_stats if "table" in s.source),
                "has_prose": sum(1 for s in all_stats if "prose" in s.source),
                "has_llm": sum(1 for s in all_stats if "llm" in s.source),
                "none": sum(1 for s in all_stats if s.source == "none"),
            },
            "per_paper": [asdict(s) for s in all_stats],
        }
        with report_path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"\ncoverage report written to {report_path}")

    total = sum(s.files_written for s in all_stats)
    print(f"\ntotal files written: {total}")


if __name__ == "__main__":
    main()
