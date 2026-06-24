"""Constants for the extraction pipeline.

Benchmark keywords, canonical name maps, developer lookup tables, score
ranges, model aliases, and other shared data definitions.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '')
PROTOCOL_CACHE_DIR = Path('.cache/llm_protocol')
_PROTOCOL_CACHE_VERSION = 2

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
    # IFEval — preserve metric variant when specified
    "ifeval strict prompt": "IFEval (strict-prompt)",
    "ifeval strict-prompt": "IFEval (strict-prompt)",
    "ifeval strict instruction": "IFEval (strict-inst)",
    "ifeval strict-instruction": "IFEval (strict-inst)",
    "ifeval loose prompt": "IFEval (loose-prompt)",
    "ifeval loose-prompt": "IFEval (loose-prompt)",
    "ifeval loose instruction": "IFEval (loose-inst)",
    "ifeval loose-instruction": "IFEval (loose-inst)",
    "ifeval average": "IFEval",
    "ifeval": "IFEval",
    # GPQA variants
    "gpqa-diamond": "GPQA-Diamond",
    "gpqa diamond": "GPQA-Diamond",
    "gpqa-d": "GPQA-Diamond",
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
    "bigbench-hard": "BBH",
    "big-bench-hard": "BBH",
    # BIG-Bench (full 204-task suite) ≠ BBH (23-task hard subset)
    "big-bench": "BIG-Bench",
    "big bench": "BIG-Bench",
    "bigbench": "BIG-Bench",
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
    # ARC variants — all collapse to ARC-Challenge / ARC-Easy
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
    "truthfulqa mc1": "TruthfulQA-MC1",
    "truthfulqa-mc1": "TruthfulQA-MC1",
    "truthfulqa mc2": "TruthfulQA-MC2",
    "truthfulqa-mc2": "TruthfulQA-MC2",
    # MATH variants — distinct subsets, do not collapse
    "math lvl 5": "MATH Lvl 5",
    "math level 5": "MATH Lvl 5",
    "math-hard": "MATH Lvl 5",
    "math hard": "MATH Lvl 5",
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
    # pass@k variants — bare pass@k is ambiguous (could be HumanEval or MBPP)
    # Only map when benchmark context is explicit
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
)

_SKIP_TABLE_CAPTIONS: tuple[str, ...] = (
    "percentage of evaluation sets considered to be contaminated",
    "estimated performance gain",
    "contaminated because similar data exists",
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


# Organization name → canonical developer ID mapping
_ORG_MAP: dict[str, str] = {
    "openai": "openai", "google": "google", "meta": "meta-llama",
    "anthropic": "anthropic", "mistral": "mistralai", "alibaba": "Qwen",
    "microsoft": "microsoft", "deepseek": "deepseek-ai",
    "cohere": "CohereForAI", "01.ai": "01-ai", "nvidia": "nvidia",
    "ai21": "ai21labs", "tii": "tiiuae", "xai": "xai",
    "lmsys": "lmsys", "databricks": "databricks", "bigcode": "bigcode",
    "eleutherai": "EleutherAI", "mosaicml": "mosaicml",
    "allenai": "allenai", "together": "togethercomputer",
}


def infer_developer(model_name: str, organization: str = "") -> str:
    """Infer the developer/organization for a model name.

    Uses organization hint first, then falls back to _DEVELOPER_PATTERNS
    substring matching. Returns 'unknown' if no match found.
    """
    # Try organization hint first
    if organization:
        org_lower = organization.lower()
        for label, dev in _ORG_MAP.items():
            if label in org_lower:
                return dev

    # Fall back to model name pattern matching
    lower = model_name.lower()
    for prefix, dev in _DEVELOPER_PATTERNS:
        if lower.startswith(prefix) or f"-{prefix}" in lower or f"/{prefix}" in lower:
            return dev
    return "unknown"


# ---------------------------------------------------------------------------
# S: PDF downloader — fetch only
# ---------------------------------------------------------------------------

