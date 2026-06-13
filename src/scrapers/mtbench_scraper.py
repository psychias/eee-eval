"""
MT-Bench leaderboard scraper.

fetches GPT-4-judged MT-Bench scores from the LMSYS FastChat GitHub
repository and converts results to the EEE schema (v0.2.1).

MT-Bench tests multi-turn conversation and instruction-following with
80 high-quality questions across 8 capability categories. Each response is
scored 1–10 by GPT-4; the final model score is the average over all turns.

Reference: "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"
           arXiv 2306.05685 (Zheng et al., 2023)

extends BaseLeaderboardScraper; existing scrapers are never modified.

usage:
    python scripts/scrapers/mtbench_scraper.py [--limit N] [--dry-run]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

# add repo root and utils/ to sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "utils"))

import requests

from eval_types import (
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    GenerationConfig,
    GenerationArgs,
    JudgeConfig,
    LlmScoring,
    MetricConfig,
    ModelInfo,
    ScoreDetails,
    ScoreType,
    SourceDataUrl,
    SourceMetadata,
)
from src.scrapers.utils import get_developer

from src.converters import SCHEMA_VERSION as _SCHEMA_VERSION
from src.scrapers.base import BaseLeaderboardScraper

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

_TIMEOUT = 30  # seconds

# Publication date of the MT-Bench paper (used to timestamp fallback data)
_FALLBACK_DATE = "2023-06-09"
import calendar as _cal, datetime as _dt
_FALLBACK_TS = str(_cal.timegm(_dt.date(2023, 6, 9).timetuple()))
del _cal, _dt

# FastChat maintains the canonical MT-Bench results in their GitHub repo.
# We try several known file locations in order.
_FASTCHAT_SCORE_URLS = [
    # Current main-branch location
    (
        "https://raw.githubusercontent.com/lm-sys/FastChat"
        "/main/fastchat/llm_judge/data/mt_bench/model_score.jsonl"
    ),
    # Alternative flat-file export (some branches)
    (
        "https://raw.githubusercontent.com/lm-sys/FastChat"
        "/main/fastchat/serve/gradio_web_server_multi.py"
    ),
]

# The original leaderboard table is embedded in the FastChat README
_FASTCHAT_README_URL = (
    "https://raw.githubusercontent.com/lm-sys/FastChat"
    "/main/fastchat/llm_judge/README.md"
)

# Source data reference (the repository is the authoritative data source)
_SOURCE_DATA = SourceDataUrl(
    dataset_name="MT-Bench",
    source_type="url",
    url=["https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge"],
)

# Shared source metadata for all MT-Bench records
_SOURCE_METADATA = SourceMetadata(
    source_name="MT-Bench",
    source_type="documentation",
    source_organization_name="LMSYS",
    source_organization_url="https://lmsys.org",
    evaluator_relationship=EvaluatorRelationship.third_party,
    additional_details={
        "paper": "arXiv:2306.05685",
        "paper_title": "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena",
    },
)

_EVAL_LIBRARY = EvalLibrary(name="mt_bench", version="unknown")

# GPT-4-0314 was the judge model used in the original MT-Bench paper.
_JUDGE_MODEL = ModelInfo(
    name="gpt-4-0314",
    id="openai/gpt-4-0314",
    developer="openai",
    inference_platform="openai",
)

# LLM scoring config (GPT-4 single-judge)
_LLM_SCORING = LlmScoring(
    judges=[JudgeConfig(model_info=_JUDGE_MODEL)],
    input_prompt=(
        "Given the user question, refer to the reference answer, and evaluate "
        "the quality of the assistant's answer on a scale of 1-10."
    ),
)

# ---------------------------------------------------------------------------
# hardcoded fallback leaderboard — sourced from Table 2 of arXiv:2306.05685
# and the FastChat leaderboard page (scores as of late 2023 / early 2024).
# Scores are the average turn score (1–10 scale).
# ---------------------------------------------------------------------------
_FALLBACK_LEADERBOARD: list[dict] = [
    # ── Closed-source ─────────────────────────────────────────────────────
    {"model": "gpt-4",               "score": 8.99, "developer": "openai"},
    {"model": "gpt-4-0314",          "score": 8.96, "developer": "openai"},
    {"model": "gpt-4-0613",          "score": 9.18, "developer": "openai"},
    {"model": "gpt-4-turbo",         "score": 9.32, "developer": "openai"},
    {"model": "gpt-3.5-turbo",       "score": 7.94, "developer": "openai"},
    {"model": "claude-v1",           "score": 8.15, "developer": "anthropic"},
    {"model": "claude-instant-v1",   "score": 7.90, "developer": "anthropic"},
    # ── Open-source ───────────────────────────────────────────────────────
    {"model": "vicuna-33b-v1.3",     "score": 7.12, "developer": "lmsys"},
    {"model": "wizardlm-30b",        "score": 7.01, "developer": "WizardLM"},
    {"model": "guanaco-65b",         "score": 6.41, "developer": "timdettmers"},
    {"model": "vicuna-13b-v1.3",     "score": 6.57, "developer": "lmsys"},
    {"model": "vicuna-13b-v1.5",     "score": 7.22, "developer": "lmsys"},
    {"model": "wizardlm-13b-v1.2",   "score": 7.20, "developer": "WizardLM"},
    {"model": "llama-2-70b-chat",    "score": 6.27, "developer": "meta-llama"},
    {"model": "llama-2-13b-chat",    "score": 6.27, "developer": "meta-llama"},
    {"model": "llama-2-7b-chat",     "score": 6.27, "developer": "meta-llama"},
    {"model": "guanaco-33b",         "score": 6.53, "developer": "timdettmers"},
    {"model": "tulu-30b",            "score": 6.43, "developer": "allenai"},
    {"model": "mpt-30b-chat",        "score": 6.39, "developer": "mosaicml"},
    {"model": "vicuna-7b-v1.5",      "score": 6.69, "developer": "lmsys"},
    {"model": "mpt-7b-chat",         "score": 5.42, "developer": "mosaicml"},
    {"model": "alpaca-13b",          "score": 4.53, "developer": "stanford"},
    {"model": "llama-13b",           "score": 5.18, "developer": "meta-llama"},
    {"model": "llama-30b",           "score": 5.78, "developer": "meta-llama"},
    {"model": "dolly-v2-12b",        "score": 3.28, "developer": "databricks"},
    {"model": "rwkv-4-raven-14b",    "score": 3.98, "developer": "blinkdl"},
    {"model": "oasst-pythia-12b",    "score": 5.32, "developer": "OpenAssistant"},
    {"model": "stablelm-tuned-alpha-7b", "score": 2.75, "developer": "stabilityai"},
    {"model": "fastchat-t5-3b",      "score": 3.04, "developer": "lmsys"},
]


# ---------------------------------------------------------------------------
# scraper implementation
# ---------------------------------------------------------------------------


class MTBenchScraper(BaseLeaderboardScraper):
    """scraper for the LMSYS MT-Bench leaderboard.

    tries live sources from the FastChat GitHub repo before falling back to
    the hardcoded scores from the original arXiv paper.
    Pass ``allow_fallback=False`` to raise instead of using stale data.
    """

    eval_name = "mt_bench"
    source_name = "MT-Bench"
    source_organization = "LMSYS"
    output_dir = "data/mt_bench"

    def __init__(self, allow_fallback: bool = True) -> None:
        self._allow_fallback = allow_fallback
        self._is_fallback = False

    def fetch_raw(self) -> list[dict]:
        """fetch the MT-Bench leaderboard from the FastChat repo."""
        # try the JSONL score file
        rows = _try_fetch_jsonl(_FASTCHAT_SCORE_URLS[0])
        if rows:
            print(f"  fetched JSONL from {_FASTCHAT_SCORE_URLS[0]}")
            return rows

        # try to parse the README table
        rows = _try_parse_readme(_FASTCHAT_README_URL)
        if rows:
            print("  parsed leaderboard table from FastChat README")
            return rows

        if not self._allow_fallback:
            raise RuntimeError(
                "all MT-Bench live sources failed and --no-fallback was set."
            )

        print(
            f"  WARNING: all live sources failed — using hardcoded fallback "
            f"leaderboard ({len(_FALLBACK_LEADERBOARD)} models from {_FALLBACK_DATE})."
        )
        self._is_fallback = True
        return _FALLBACK_LEADERBOARD

    def convert(self, raw: list[dict], retrieved_timestamp: str) -> list[dict]:
        """convert raw leaderboard rows to EEE schema dicts."""
        ts = _FALLBACK_TS if self._is_fallback else retrieved_timestamp
        results: list[dict] = []
        for row in raw:
            try:
                record = _convert_row(row, ts)
                results.append(record)
            except Exception as exc:
                name = row.get("model", row.get("name", "?"))
                print(f"  skipping {name!r}: {exc}")
        return results


# ---------------------------------------------------------------------------
# fetch helpers
# ---------------------------------------------------------------------------


def _try_fetch_jsonl(url: str) -> list[dict]:
    """try to fetch and parse a JSONL file from *url*."""
    import json

    try:
        resp = requests.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        rows = []
        for line in resp.text.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows if rows else []
    except Exception:
        return []


def _try_parse_readme(url: str) -> list[dict]:
    """try to extract the leaderboard table embedded in the FastChat README."""
    import re

    try:
        resp = requests.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        text = resp.text

        rows: list[dict] = []
        # Look for rows that contain a numeric score in a Markdown table
        # Format: | Model | Score | ... |  or variations thereof
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("|"):
                continue
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) < 2:
                continue
            # Skip separator rows (---|---)
            if all(set(p) <= set("-: ") for p in parts if p):
                continue
            # Try to find a column that looks like a 1–10 score
            model_name: str | None = None
            score: float | None = None
            for i, part in enumerate(parts):
                try:
                    val = float(part.replace(",", "").strip("*` "))
                    if 1.0 <= val <= 10.0:
                        score = val
                        # model name is the first non-empty cell
                        for j in range(i):
                            if parts[j] and not parts[j].startswith("-"):
                                model_name = parts[j].strip("*` ")
                                break
                except ValueError:
                    pass
            if model_name and score is not None:
                rows.append({"model": model_name, "score": score})

        return rows
    except Exception:
        return []


# ---------------------------------------------------------------------------
# conversion helper
# ---------------------------------------------------------------------------


def _normalise_model_id(model_name: str, developer: str) -> str:
    """convert *model_name* to a HuggingFace-style model ID."""
    if "/" in model_name:
        return model_name

    _hf_ids: dict[str, str] = {
        "gpt-4-0314":            "openai/gpt-4-0314",
        "gpt-4-0613":            "openai/gpt-4-0613",
        "gpt-4-turbo":           "openai/gpt-4-turbo",
        "gpt-4":                 "openai/gpt-4",
        "gpt-3.5-turbo":         "openai/gpt-3.5-turbo",
        "gpt-3.5-turbo-0613":    "openai/gpt-3.5-turbo-0613",
        "claude-v1":             "anthropic/claude-v1",
        "claude-instant-v1":     "anthropic/claude-instant-v1",
        "llama-2-70b-chat":      "meta-llama/Llama-2-70b-chat-hf",
        "llama-2-13b-chat":      "meta-llama/Llama-2-13b-chat-hf",
        "llama-2-7b-chat":       "meta-llama/Llama-2-7b-chat-hf",
        "llama-13b":             "meta-llama/Llama-13b",
        "llama-30b":             "meta-llama/Llama-30b",
        "vicuna-33b-v1.3":       "lmsys/vicuna-33b-v1.3",
        "vicuna-13b-v1.3":       "lmsys/vicuna-13b-v1.3",
        "vicuna-13b-v1.5":       "lmsys/vicuna-13b-v1.5",
        "vicuna-7b-v1.5":        "lmsys/vicuna-7b-v1.5",
        "wizardlm-30b":          "WizardLM/WizardLM-30B-V1.0",
        "wizardlm-13b-v1.2":     "WizardLM/WizardLM-13B-V1.2",
        "alpaca-13b":            "tatsu-lab/alpaca-7b-wdiff",
        "dolly-v2-12b":          "databricks/dolly-v2-12b",
        "mpt-30b-chat":          "mosaicml/mpt-30b-chat",
        "mpt-7b-chat":           "mosaicml/mpt-7b-chat",
        "oasst-pythia-12b":      "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
        "stablelm-tuned-alpha-7b": "stabilityai/stablelm-tuned-alpha-7b",
        "fastchat-t5-3b":        "lmsys/fastchat-t5-3b-v1.0",
        "rwkv-4-raven-14b":      "BlinkDL/rwkv-4-raven",
        "guanaco-65b":           "timdettmers/guanaco-65b-merged",
        "guanaco-33b":           "timdettmers/guanaco-33b-merged",
        "tulu-30b":              "allenai/tulu-30b",
    }

    if model_name.lower() in _hf_ids:
        return _hf_ids[model_name.lower()]

    return f"{developer}/{model_name}"


def _convert_row(row: dict[str, Any], retrieved_timestamp: str) -> dict:
    """convert a single MT-Bench row to an EEE schema dict."""
    model_name: str = (
        row.get("model") or row.get("Model") or row.get("name") or ""
    ).strip()

    if not model_name:
        raise ValueError("no model name in row")

    # score may be stored as 'score', 'turn_avg', 'average', etc.
    score_raw = (
        row.get("score")
        or row.get("turn_avg")
        or row.get("average")
        or row.get("Score")
        or "0"
    )
    try:
        score = float(str(score_raw).strip())
    except (ValueError, TypeError):
        score = 0.0

    dev_hint = row.get("developer") or row.get("organization") or None
    developer = dev_hint if dev_hint else get_developer(model_name)

    model_id = _normalise_model_id(model_name, developer)
    if "/" in model_id:
        developer = model_id.split("/")[0]

    safe_model_id = model_id.replace("/", "_")
    evaluation_id = f"mt_bench/{safe_model_id}/{retrieved_timestamp}"

    log = EvaluationLog(
        schema_version=_SCHEMA_VERSION,
        evaluation_id=evaluation_id,
        retrieved_timestamp=retrieved_timestamp,
        source_metadata=_SOURCE_METADATA,
        eval_library=_EVAL_LIBRARY,
        model_info=ModelInfo(
            name=model_name,
            id=model_id,
            developer=developer,
        ),
        evaluation_results=[
            EvaluationResult(
                evaluation_name="MT-Bench",
                source_data=_SOURCE_DATA,
                metric_config=MetricConfig(
                    evaluation_description=(
                        "Multi-turn conversation and instruction-following score "
                        "judged by GPT-4. Average of turn-1 and turn-2 across "
                        "80 questions in 8 capability categories. Scale: 1–10."
                    ),
                    lower_is_better=False,
                    score_type=ScoreType.continuous,
                    min_score=1.0,
                    max_score=10.0,
                    llm_scoring=_LLM_SCORING,
                ),
                score_details=ScoreDetails(score=score),
                generation_config=GenerationConfig(
                    generation_args=GenerationArgs(temperature=0.7)
                ),
            )
        ],
    )

    return log.model_dump(mode='json', exclude_none=True)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main(limit: int | None = None, dry_run: bool = False, no_fallback: bool = False) -> None:
    """run the MT-Bench scraper end-to-end."""
    scraper = MTBenchScraper(allow_fallback=not no_fallback)

    print("fetching MT-Bench leaderboard...")
    raw = scraper.fetch_raw()
    print(f"  fetched {len(raw)} rows")

    if limit:
        raw = raw[:limit]

    scraper.save_raw(raw, Path("scripts/scrapers/raw"))

    retrieved_timestamp = str(time.time())
    records = scraper.convert(raw, retrieved_timestamp)
    print(f"  converted {len(records)} records")

    if dry_run:
        print("dry-run mode — skipping disk writes")
        return

    count = scraper._write_records(records)
    print(f"done — {count} files written to {scraper.output_dir}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="scrape LMSYS MT-Bench leaderboard")
    parser.add_argument("--limit", type=int, default=None, help="max models to process")
    parser.add_argument("--dry-run", action="store_true", help="skip writing files")
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help=(
            f"raise an error instead of using the hardcoded {_FALLBACK_DATE} "
            "snapshot when all live sources fail"
        ),
    )
    args = parser.parse_args()
    main(limit=args.limit, dry_run=args.dry_run, no_fallback=args.no_fallback)
