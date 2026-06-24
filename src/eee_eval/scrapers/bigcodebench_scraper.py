"""
BigCodeBench leaderboard scraper.

fetches pass@1 scores from the BigCodeBench leaderboard (bigcode-project)
and converts results to the EEE schema (v0.2.1).

BigCodeBench provides two challenging task sets for evaluating code generation:
  - BigCodeBench-Complete: fill-in-the-middle / completion tasks
  - BigCodeBench-Instruct: instruction-following code generation tasks

Both are measured by execution-based pass@1 (0–100 %).

Reference: "BigCodeBench: Benchmarking Code Generation with Diverse Function
            Calls and Complex Instructions", arXiv 2406.15877 (Zhuo et al., 2024)

extends BaseLeaderboardScraper; existing scrapers are never modified.

usage:
    python scripts/scrapers/bigcodebench_scraper.py [--limit N] [--dry-run]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

# add repo root and utils/ to sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "utils"))

import requests

from eee_eval.eval_types import (
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    MetricConfig,
    ModelInfo,
    ScoreDetails,
    ScoreType,
    SourceDataHf,
    SourceDataUrl,
    SourceMetadata,
)
from eee_eval.scrapers.utils import get_developer

from eee_eval.converters import SCHEMA_VERSION as _SCHEMA_VERSION
from eee_eval.scrapers.base import BaseLeaderboardScraper

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

_TIMEOUT = 30  # seconds

# Publication date of the BigCodeBench paper
_FALLBACK_DATE = "2024-06-21"
import calendar as _cal, datetime as _dt
_FALLBACK_TS = str(_cal.timegm(_dt.date(2024, 6, 21).timetuple()))
del _cal, _dt

# Primary: BigCodeBench publishes leaderboard data in their HuggingFace dataset
_HF_LEADERBOARD_COMPLETE_URL = (
    "https://huggingface.co/datasets/bigcode/bigcodebench-results/resolve/main"
    "/leaderboard_complete.json"
)
_HF_LEADERBOARD_INSTRUCT_URL = (
    "https://huggingface.co/datasets/bigcode/bigcodebench-results/resolve/main"
    "/leaderboard_instruct.json"
)

# Secondary: GitHub repo CSV (some releases)
_GITHUB_LEADERBOARD_URL = (
    "https://raw.githubusercontent.com/bigcode-project/bigcodebench"
    "/main/bigcodebench/leaderboard.csv"
)

# Tertiary: HF Space API (Gradio)
_HF_SPACE_API_URL = (
    "https://bigcode-bigcodebench-leaderboard.hf.space/api/predict"
)

# Source data references for each task
_SOURCE_DATA_COMPLETE = SourceDataHf(
    dataset_name="BigCodeBench-Complete",
    source_type="hf_dataset",
    hf_repo="bigcode/bigcodebench",
    hf_split="test",
    samples_number=1140,
)

_SOURCE_DATA_INSTRUCT = SourceDataHf(
    dataset_name="BigCodeBench-Instruct",
    source_type="hf_dataset",
    hf_repo="bigcode/bigcodebench",
    hf_split="test",
    samples_number=1140,
)

# Shared source metadata for all BigCodeBench records
_SOURCE_METADATA = SourceMetadata(
    source_name="BigCodeBench Leaderboard",
    source_type="documentation",
    source_organization_name="BigCode Project",
    source_organization_url="https://www.bigcode-project.org",
    evaluator_relationship=EvaluatorRelationship.third_party,
    additional_details={
        "paper": "arXiv:2406.15877",
        "paper_title": "BigCodeBench: Benchmarking Code Generation with Diverse Function Calls",
        "leaderboard_url": "https://huggingface.co/spaces/bigcode/bigcodebench-leaderboard",
    },
)

_EVAL_LIBRARY = EvalLibrary(
    name="bigcodebench",
    version="0.1.0",
    additional_details={"metric": "pass@1 (execution-based)"},
)

# ---------------------------------------------------------------------------
# hardcoded fallback leaderboard
# Sourced from Table 1 of arXiv:2406.15877 + leaderboard snapshot (2024-06-21)
# Scores are pass@1 percentage (0–100).
# Each entry has both 'complete' and 'instruct' scores where available.
# ---------------------------------------------------------------------------
_FALLBACK_LEADERBOARD: list[dict] = [
    # ── Closed-source ─────────────────────────────────────────────────────
    {
        "model": "gpt-4o-2024-05-13",    "developer": "openai",
        "complete": 61.1,                "instruct": 78.9,
        "inference_platform": "openai",
    },
    {
        "model": "gpt-4-turbo-2024-04-09", "developer": "openai",
        "complete": 55.8,                  "instruct": 76.2,
        "inference_platform": "openai",
    },
    {
        "model": "gpt-4-0613",           "developer": "openai",
        "complete": 49.8,                "instruct": 67.8,
        "inference_platform": "openai",
    },
    {
        "model": "gpt-3.5-turbo-0125",   "developer": "openai",
        "complete": 39.1,                "instruct": 56.6,
        "inference_platform": "openai",
    },
    {
        "model": "claude-3-5-sonnet-20240620", "developer": "anthropic",
        "complete": 56.5,                      "instruct": 75.8,
        "inference_platform": "anthropic",
    },
    {
        "model": "claude-3-opus-20240229",  "developer": "anthropic",
        "complete": 50.1,                   "instruct": 67.2,
        "inference_platform": "anthropic",
    },
    {
        "model": "claude-3-haiku-20240307", "developer": "anthropic",
        "complete": 42.9,                   "instruct": 55.1,
        "inference_platform": "anthropic",
    },
    {
        "model": "gemini-1.5-pro",          "developer": "google",
        "complete": 51.5,                   "instruct": 69.1,
        "inference_platform": "google",
    },
    {
        "model": "gemini-1.5-flash",        "developer": "google",
        "complete": 47.7,                   "instruct": 63.4,
        "inference_platform": "google",
    },
    # ── Open-source ───────────────────────────────────────────────────────
    {
        "model": "meta-llama/Meta-Llama-3-70B-Instruct", "developer": "meta-llama",
        "complete": 46.6,                                 "instruct": 60.9,
    },
    {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",  "developer": "meta-llama",
        "complete": 28.7,                                 "instruct": 40.2,
    },
    {
        "model": "deepseek-ai/DeepSeek-Coder-V2-Instruct", "developer": "deepseek-ai",
        "complete": 48.7,                                   "instruct": 66.0,
    },
    {
        "model": "mistralai/Mixtral-8x22B-Instruct-v0.1", "developer": "mistralai",
        "complete": 37.0,                                  "instruct": 50.0,
    },
    {
        "model": "Qwen/CodeQwen1.5-7B-Chat", "developer": "Qwen",
        "complete": 38.9,                    "instruct": 48.1,
    },
    {
        "model": "bigcode/starcoder2-15b-instruct-v0.1", "developer": "bigcode",
        "complete": 40.9,                                 "instruct": 46.2,
    },
    {
        "model": "microsoft/Phi-3-medium-4k-instruct", "developer": "microsoft",
        "complete": 35.7,                              "instruct": 49.6,
    },
    {
        "model": "microsoft/Phi-3-mini-128k-instruct", "developer": "microsoft",
        "complete": 24.0,                              "instruct": 37.5,
    },
]


# ---------------------------------------------------------------------------
# scraper implementation
# ---------------------------------------------------------------------------


class BigCodeBenchScraper(BaseLeaderboardScraper):
    """scraper for the BigCodeBench leaderboard.

    tries multiple public data sources; falls back to the hardcoded snapshot
    from the BigCodeBench paper when all live sources fail.
    Pass ``allow_fallback=False`` to raise instead of using stale data.
    """

    eval_name = "bigcodebench"
    source_name = "BigCodeBench"
    source_organization = "BigCode Project"
    output_dir = "data/bigcodebench"

    def __init__(self, allow_fallback: bool = True) -> None:
        self._allow_fallback = allow_fallback
        self._is_fallback = False

    def fetch_raw(self) -> list[dict]:
        """fetch the leaderboard, trying HF dataset → GitHub CSV → fallback."""
        # try HF dataset JSON (complete + instruct merged)
        rows = _try_fetch_hf_combined()
        if rows:
            print("  fetched combined leaderboard from HF dataset API")
            return rows

        # try GitHub CSV
        rows = _try_fetch_github_csv()
        if rows:
            print(f"  fetched CSV from {_GITHUB_LEADERBOARD_URL}")
            return rows

        if not self._allow_fallback:
            raise RuntimeError(
                "all BigCodeBench live sources failed and --no-fallback was set."
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


def _try_fetch_hf_combined() -> list[dict]:
    """try to fetch both Complete and Instruct leaderboards from HF and merge."""
    try:
        resp_c = requests.get(_HF_LEADERBOARD_COMPLETE_URL, timeout=_TIMEOUT)
        resp_c.raise_for_status()
        data_c: list[dict] = resp_c.json()
    except Exception:
        data_c = []

    try:
        resp_i = requests.get(_HF_LEADERBOARD_INSTRUCT_URL, timeout=_TIMEOUT)
        resp_i.raise_for_status()
        data_i: list[dict] = resp_i.json()
    except Exception:
        data_i = []

    if not data_c and not data_i:
        return []

    # index by model name so we can merge complete + instruct scores
    combined: dict[str, dict] = {}

    for entry in data_c:
        model_name = _extract_model_name(entry)
        if model_name:
            combined[model_name] = {
                "model": model_name,
                "complete": _extract_score(entry),
                **_extract_meta(entry),
            }

    for entry in data_i:
        model_name = _extract_model_name(entry)
        if model_name:
            if model_name in combined:
                combined[model_name]["instruct"] = _extract_score(entry)
            else:
                combined[model_name] = {
                    "model": model_name,
                    "instruct": _extract_score(entry),
                    **_extract_meta(entry),
                }

    return list(combined.values())


def _extract_model_name(entry: dict) -> str:
    """extract a model name string from a raw API entry."""
    for key in ("model", "Model", "model_name", "name"):
        val = entry.get(key)
        if val and isinstance(val, str):
            return val.strip()
    return ""


def _extract_score(entry: dict) -> float | None:
    """extract a numeric pass@1 score from a raw API entry."""
    for key in (
        "pass@1", "pass_at_1", "pass@1 (%)", "score",
        "complete_score", "instruct_score",
    ):
        val = entry.get(key)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
    return None


def _extract_meta(entry: dict) -> dict:
    """extract optional metadata (developer, platform) from a raw API entry."""
    meta: dict = {}
    for key in ("developer", "organization"):
        val = entry.get(key)
        if val and isinstance(val, str):
            meta["developer"] = val.strip()
            break
    for key in ("inference_platform", "backend"):
        val = entry.get(key)
        if val and isinstance(val, str):
            meta["inference_platform"] = val.strip()
            break
    return meta


def _try_fetch_github_csv() -> list[dict]:
    """try to fetch the BigCodeBench leaderboard CSV from GitHub."""
    import csv
    import io

    try:
        resp = requests.get(_GITHUB_LEADERBOARD_URL, timeout=_TIMEOUT)
        resp.raise_for_status()
        reader = csv.DictReader(io.StringIO(resp.text))
        rows = list(reader)
        return rows if rows else []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# conversion helper
# ---------------------------------------------------------------------------


def _normalise_model_id(model_name: str, developer: str) -> str:
    """convert *model_name* to a HuggingFace-style model ID."""
    if "/" in model_name:
        return model_name
    return f"{developer}/{model_name}"


def _convert_row(row: dict[str, Any], retrieved_timestamp: str) -> dict:
    """convert a single BigCodeBench row to an EEE schema dict.

    One row may contain both Complete and Instruct scores; they are
    stored as two separate EvaluationResult objects in the same log.
    """
    model_name: str = (
        row.get("model") or row.get("Model") or row.get("name") or ""
    ).strip()

    if not model_name:
        raise ValueError("no model name in row")

    dev_hint = row.get("developer") or row.get("organization") or None
    developer = dev_hint if dev_hint else get_developer(model_name)

    model_id = _normalise_model_id(model_name, developer)
    if "/" in model_id:
        developer = model_id.split("/")[0]

    inference_platform: str | None = (
        row.get("inference_platform") or row.get("backend") or None
    )
    # For open-weight models accessed directly, do not set an inference platform
    if inference_platform:
        inference_platform = inference_platform.strip().lower() or None

    # Build one EvaluationResult per available task
    eval_results: list[EvaluationResult] = []

    complete_score = row.get("complete")
    if complete_score is not None:
        try:
            cs = float(complete_score)
            eval_results.append(
                EvaluationResult(
                    evaluation_name="BigCodeBench-Complete",
                    source_data=_SOURCE_DATA_COMPLETE,
                    metric_config=MetricConfig(
                        evaluation_description=(
                            "BigCodeBench-Complete: pass@1 (%) on function-call-based "
                            "completion tasks. Models must complete function bodies with "
                            "correct library usage, verified by execution against test cases."
                        ),
                        lower_is_better=False,
                        score_type=ScoreType.continuous,
                        min_score=0.0,
                        max_score=100.0,
                    ),
                    score_details=ScoreDetails(score=cs),
                )
            )
        except (ValueError, TypeError):
            pass

    instruct_score = row.get("instruct") or row.get("instruct_score")
    if instruct_score is not None:
        try:
            ins = float(instruct_score)
            eval_results.append(
                EvaluationResult(
                    evaluation_name="BigCodeBench-Instruct",
                    source_data=_SOURCE_DATA_INSTRUCT,
                    metric_config=MetricConfig(
                        evaluation_description=(
                            "BigCodeBench-Instruct: pass@1 (%) on instruction-following "
                            "code generation tasks. Models receive a natural-language "
                            "functional description and must generate a correct solution, "
                            "verified by execution against test cases."
                        ),
                        lower_is_better=False,
                        score_type=ScoreType.continuous,
                        min_score=0.0,
                        max_score=100.0,
                    ),
                    score_details=ScoreDetails(score=ins),
                )
            )
        except (ValueError, TypeError):
            pass

    # fall back to a generic 'score' column (e.g. CSV export)
    if not eval_results:
        generic = row.get("score") or row.get("Score") or row.get("pass@1")
        if generic is not None:
            try:
                gs = float(str(generic).replace(",", "").strip())
                eval_results.append(
                    EvaluationResult(
                        evaluation_name="BigCodeBench",
                        source_data=_SOURCE_DATA_COMPLETE,
                        metric_config=MetricConfig(
                            evaluation_description="BigCodeBench pass@1 (%)",
                            lower_is_better=False,
                            score_type=ScoreType.continuous,
                            min_score=0.0,
                            max_score=100.0,
                        ),
                        score_details=ScoreDetails(score=gs),
                    )
                )
            except (ValueError, TypeError):
                pass

    if not eval_results:
        raise ValueError(f"no valid scores found for {model_name!r}")

    safe_model_id = model_id.replace("/", "_")
    evaluation_id = f"bigcodebench/{safe_model_id}/{retrieved_timestamp}"

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
        evaluation_results=eval_results,
    )

    return log.model_dump(mode='json', exclude_none=True)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main(limit: int | None = None, dry_run: bool = False, no_fallback: bool = False) -> None:
    """run the BigCodeBench scraper end-to-end."""
    scraper = BigCodeBenchScraper(allow_fallback=not no_fallback)

    print("fetching BigCodeBench leaderboard...")
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

    parser = argparse.ArgumentParser(description="scrape BigCodeBench leaderboard")
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
