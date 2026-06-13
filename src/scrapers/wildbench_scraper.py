"""
WildBench leaderboard scraper.

fetches WB-Score results from the WildBench-v2.0 leaderboard
(Allen Institute for AI) and converts them to the EEE schema (v0.2.1).

WildBench evaluates instruction-following quality by collecting 1,024
challenging real-world user queries and using GPT-4 as a judge to compare
model responses against a reference. The primary metric is WB-Score, a
length-controlled win-margin over weaker baseline models (higher is better;
0.0 is the performance of the reference/baseline model).

Reference: "WildBench: Benchmarking LLMs with Challenging Tasks from
            Real Users in the Wild", arXiv 2406.04770 (Lin et al., 2024)

extends BaseLeaderboardScraper; existing scrapers are never modified.

usage:
    python scripts/scrapers/wildbench_scraper.py [--limit N] [--dry-run]
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

# Publication date of the WildBench v2.0 paper
_FALLBACK_DATE = "2024-06-06"
import calendar as _cal, datetime as _dt
_FALLBACK_TS = str(_cal.timegm(_dt.date(2024, 6, 6).timetuple()))
del _cal, _dt

# WildBench publishes its leaderboard data directly in the GitHub repo
_GITHUB_LEADERBOARD_URLS = [
    # Structured leaderboard JSON (preferred)
    (
        "https://raw.githubusercontent.com/allenai/WildBench"
        "/main/leaderboard/data/leaderboard.json"
    ),
    # Alternative path (older layout)
    (
        "https://raw.githubusercontent.com/allenai/WildBench"
        "/main/leaderboard/data_v2/leaderboard.json"
    ),
]

# HuggingFace space API (Gradio)
_HF_SPACE_API_URL = (
    "https://allenai-wildguard.hf.space/api/predict"
)

# Source data reference (the leaderboard dataset)
_SOURCE_DATA = SourceDataUrl(
    dataset_name="WildBench-v2",
    source_type="url",
    url=["https://huggingface.co/spaces/allenai/WildBench"],
)

# Shared source metadata for all WildBench records
_SOURCE_METADATA = SourceMetadata(
    source_name="WildBench-v2.0",
    source_type="documentation",
    source_organization_name="Allen Institute for AI (AI2)",
    source_organization_url="https://allenai.org",
    evaluator_relationship=EvaluatorRelationship.third_party,
    additional_details={
        "paper": "arXiv:2406.04770",
        "paper_title": "WildBench: Benchmarking LLMs with Challenging Tasks from Real Users",
        "benchmark_version": "v2.0",
        "num_queries": "1024",
        "judge_model": "gpt-4-turbo-2024-04-09",
    },
)

_EVAL_LIBRARY = EvalLibrary(
    name="wildbench",
    version="2.0",
    additional_details={"metric": "WB-Score (length-controlled win margin)"},
)

# GPT-4-Turbo was the judge in WildBench v2.0
_JUDGE_MODEL = ModelInfo(
    name="gpt-4-turbo-2024-04-09",
    id="openai/gpt-4-turbo-2024-04-09",
    developer="openai",
    inference_platform="openai",
)

_LLM_SCORING = LlmScoring(
    judges=[JudgeConfig(model_info=_JUDGE_MODEL)],
    input_prompt=(
        "Given a challenging task from real users, compare the two model responses "
        "and determine which one is better. Score each response on helpfulness, "
        "accuracy, and user satisfaction."
    ),
    additional_details={
        "scoring_type": "pairwise comparison with length control",
        "baseline": "mixture of Llama-2-70B-chat and Haiku",
    },
)

# ---------------------------------------------------------------------------
# hardcoded fallback leaderboard
# Sourced from Table 2 of arXiv:2406.04770 + WildBench leaderboard (2024-06-06)
# WB-Score: higher is better; 0.0 ≈ performance of the reference baseline.
# The range is approximately [-50, 100] in practice; we use -100/100 as bounds.
# ---------------------------------------------------------------------------
_FALLBACK_LEADERBOARD: list[dict] = [
    # ── Closed-source ─────────────────────────────────────────────────────
    {
        "model": "gpt-4-turbo-2024-04-09",    "developer": "openai",
        "wb_score": 57.3,  "inference_platform": "openai",
    },
    {
        "model": "gpt-4-0125-preview",         "developer": "openai",
        "wb_score": 56.4,  "inference_platform": "openai",
    },
    {
        "model": "gpt-4-0613",                 "developer": "openai",
        "wb_score": 38.5,  "inference_platform": "openai",
    },
    {
        "model": "gpt-3.5-turbo-0125",         "developer": "openai",
        "wb_score": 27.9,  "inference_platform": "openai",
    },
    {
        "model": "claude-3-opus-20240229",      "developer": "anthropic",
        "wb_score": 56.9,  "inference_platform": "anthropic",
    },
    {
        "model": "claude-3-sonnet-20240229",    "developer": "anthropic",
        "wb_score": 46.0,  "inference_platform": "anthropic",
    },
    {
        "model": "claude-3-haiku-20240307",     "developer": "anthropic",
        "wb_score": 35.3,  "inference_platform": "anthropic",
    },
    {
        "model": "gemini-1.5-pro",              "developer": "google",
        "wb_score": 50.7,  "inference_platform": "google",
    },
    {
        "model": "command-r-plus",              "developer": "cohere",
        "wb_score": 35.2,  "inference_platform": "cohere",
    },
    {
        "model": "mistral-large-2402",          "developer": "mistralai",
        "wb_score": 34.2,  "inference_platform": "mistralai",
    },
    # ── Open-source ───────────────────────────────────────────────────────
    {
        "model": "meta-llama/Meta-Llama-3-70B-Instruct", "developer": "meta-llama",
        "wb_score": 38.8,
    },
    {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",  "developer": "meta-llama",
        "wb_score": 21.5,
    },
    {
        "model": "meta-llama/Llama-2-70b-chat-hf",       "developer": "meta-llama",
        "wb_score": 12.1,
    },
    {
        "model": "meta-llama/Llama-2-13b-chat-hf",       "developer": "meta-llama",
        "wb_score": 5.2,
    },
    {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1", "developer": "mistralai",
        "wb_score": 27.5,
    },
    {
        "model": "mistralai/Mistral-7B-Instruct-v0.2",   "developer": "mistralai",
        "wb_score": 5.9,
    },
    {
        "model": "Qwen/Qwen1.5-72B-Chat",                "developer": "Qwen",
        "wb_score": 23.7,
    },
    {
        "model": "microsoft/Phi-3-mini-128k-instruct",   "developer": "microsoft",
        "wb_score": 17.6,
    },
    {
        "model": "deepseek-ai/DeepSeek-V2-Chat",         "developer": "deepseek-ai",
        "wb_score": 43.8,
    },
    {
        "model": "01-ai/Yi-1.5-34B-Chat",                "developer": "01-ai",
        "wb_score": 32.3,
    },
    {
        "model": "cohere/command-r",                     "developer": "cohere",
        "wb_score": 18.3,
    },
    {
        "model": "databricks/dbrx-instruct",             "developer": "databricks",
        "wb_score": 28.3,
    },
]


# ---------------------------------------------------------------------------
# scraper implementation
# ---------------------------------------------------------------------------


class WildBenchScraper(BaseLeaderboardScraper):
    """scraper for the WildBench-v2.0 leaderboard.

    tries GitHub JSON sources before falling back to the hardcoded snapshot
    from the WildBench paper and leaderboard.
    Pass ``allow_fallback=False`` to raise instead of using stale data.
    """

    eval_name = "wildbench"
    source_name = "WildBench-v2.0"
    source_organization = "Allen Institute for AI"
    output_dir = "data/wildbench"

    def __init__(self, allow_fallback: bool = True) -> None:
        self._allow_fallback = allow_fallback
        self._is_fallback = False

    def fetch_raw(self) -> list[dict]:
        """fetch the leaderboard from GitHub JSON → fallback."""
        for url in _GITHUB_LEADERBOARD_URLS:
            rows = _try_fetch_json(url)
            if rows:
                print(f"  fetched leaderboard JSON from {url}")
                return rows

        if not self._allow_fallback:
            raise RuntimeError(
                "all WildBench live sources failed and --no-fallback was set."
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


def _try_fetch_json(url: str) -> list[dict]:
    """try to fetch and parse a JSON leaderboard from *url*."""
    try:
        resp = requests.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        # may be a list of dicts or a dict keyed by model name
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            rows = []
            for model_name, values in data.items():
                row = {"model": model_name}
                if isinstance(values, dict):
                    row.update(values)
                rows.append(row)
            return rows
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# conversion helper
# ---------------------------------------------------------------------------


def _normalise_model_id(model_name: str, developer: str) -> str:
    """convert *model_name* to a HuggingFace-style model ID."""
    if "/" in model_name:
        return model_name
    return f"{developer}/{model_name}"


def _extract_wb_score(row: dict[str, Any]) -> float | None:
    """extract the WB-Score from a raw leaderboard row."""
    for key in (
        "wb_score", "WB-Score", "score", "wb_score_adjusted",
        "wb_avg_score", "reward", "win_rate",
    ):
        val = row.get(key)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
    return None


def _convert_row(row: dict[str, Any], retrieved_timestamp: str) -> dict:
    """convert a single WildBench row to an EEE schema dict."""
    model_name: str = (
        row.get("model") or row.get("Model") or row.get("model_name") or ""
    ).strip()

    if not model_name:
        raise ValueError("no model name in row")

    wb_score = _extract_wb_score(row)
    if wb_score is None:
        raise ValueError(f"no WB-Score found for {model_name!r}")

    dev_hint = row.get("developer") or row.get("organization") or None
    developer = dev_hint if dev_hint else get_developer(model_name)

    model_id = _normalise_model_id(model_name, developer)
    if "/" in model_id:
        developer = model_id.split("/")[0]

    inference_platform: str | None = (
        row.get("inference_platform") or row.get("api") or None
    )
    if isinstance(inference_platform, str):
        inference_platform = inference_platform.strip().lower() or None

    safe_model_id = model_id.replace("/", "_")
    evaluation_id = f"wildbench/{safe_model_id}/{retrieved_timestamp}"

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
            inference_platform=inference_platform,
        ),
        evaluation_results=[
            EvaluationResult(
                evaluation_name="WildBench-v2 WB-Score",
                source_data=_SOURCE_DATA,
                metric_config=MetricConfig(
                    evaluation_description=(
                        "WB-Score from WildBench-v2.0: length-controlled pairwise "
                        "win-margin over a weaker reference model (mixture of "
                        "Llama-2-70B-chat and Claude-3-Haiku), judged by GPT-4-Turbo. "
                        "Score 0.0 ≈ reference model performance. Higher is better."
                    ),
                    lower_is_better=False,
                    score_type=ScoreType.continuous,
                    min_score=-100.0,
                    max_score=100.0,
                    llm_scoring=_LLM_SCORING,
                ),
                score_details=ScoreDetails(score=wb_score),
            )
        ],
    )

    return log.model_dump(mode='json', exclude_none=True)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main(limit: int | None = None, dry_run: bool = False, no_fallback: bool = False) -> None:
    """run the WildBench scraper end-to-end."""
    scraper = WildBenchScraper(allow_fallback=not no_fallback)

    print("fetching WildBench leaderboard...")
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

    parser = argparse.ArgumentParser(description="scrape WildBench leaderboard")
    parser.add_argument("--limit", type=int, default=None, help="max models to process")
    parser.add_argument("--dry-run", action="store_true", help="skip writing files")
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help=(
            f"raise instead of using the hardcoded {_FALLBACK_DATE} "
            "snapshot when all live sources fail"
        ),
    )
    args = parser.parse_args()
    main(limit=args.limit, dry_run=args.dry_run, no_fallback=args.no_fallback)
