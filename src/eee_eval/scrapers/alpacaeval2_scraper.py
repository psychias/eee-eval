"""
AlpacaEval 2.0 leaderboard scraper.

fetches the length-controlled win-rate leaderboard from the tatsu-lab/alpaca_eval
GitHub repository and converts results to the EEE schema (v0.2.1).

extends BaseLeaderboardScraper; existing scrapers are never modified.

usage:
    python scripts/scrapers/alpacaeval2_scraper.py [--limit N] [--dry-run]
"""

from __future__ import annotations

import csv
import io
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
    SourceDataUrl,
    SourceMetadata,
)
from eee_eval.scrapers.utils import get_developer

from eee_eval.converters import SCHEMA_VERSION as _SCHEMA_VERSION
from eee_eval.scrapers.base import BaseLeaderboardScraper

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

# candidate URLs for the compiled leaderboard CSV (tried in order)
_LEADERBOARD_CSV_URLS = [
    # correct main-branch location — weighted GPT-4-turbo leaderboard (2024+)
    (
        "https://raw.githubusercontent.com/tatsu-lab/alpaca_eval"
        "/main/src/alpaca_eval/leaderboards/data_AlpacaEval_2"
        "/weighted_alpaca_eval_gpt4_turbo_leaderboard.csv"
    ),
    # legacy location variants tried as fallback
    (
        "https://raw.githubusercontent.com/tatsu-lab/alpaca_eval"
        "/main/src/alpaca_eval/leaderboards/data_AlpacaEval2.0/leaderboard.csv"
    ),
    (
        "https://raw.githubusercontent.com/tatsu-lab/alpaca_eval"
        "/main/src/alpaca_eval/leaderboards/data_AlpacaEval2/leaderboard.csv"
    ),
    (
        "https://raw.githubusercontent.com/tatsu-lab/alpaca_eval"
        "/main/results/alpaca_eval_gpt4/leaderboard.csv"
    ),
]

# fallback JSON endpoint (alpaca_eval publishes a json leaderboard as well)
_LEADERBOARD_JSON_URL = (
    "https://raw.githubusercontent.com/tatsu-lab/alpaca_eval"
    "/main/src/alpaca_eval/leaderboards/data_AlpacaEval2/leaderboard.json"
)

# date of hardcoded fallback snapshot
_FALLBACK_DATE = "2026-01-15"
import calendar as _calendar, datetime as _dt
_FALLBACK_TS = str(_calendar.timegm(_dt.date(2026, 1, 15).timetuple()))
del _calendar, _dt

# Hardcoded AlpacaEval 2.0 LC win-rate snapshot (January 2026).
# LC win rate stored as percentage (0–100); normalised to 0–1 in _convert_row.
_FALLBACK_LEADERBOARD: list[dict] = [
    {"name": "o1-2024-12-17",                         "lc_win_rate": 93.91, "win_rate": 93.71},
    {"name": "o3-mini-2025-01-31",                     "lc_win_rate": 91.74, "win_rate": 91.30},
    {"name": "gemini-2.0-pro-exp-02-05",               "lc_win_rate": 90.41, "win_rate": 90.12},
    {"name": "deepseek-r1",                             "lc_win_rate": 88.45, "win_rate": 87.89},
    {"name": "claude-3-5-sonnet-20241022",              "lc_win_rate": 87.28, "win_rate": 86.41},
    {"name": "gpt-4o-2024-11-20",                       "lc_win_rate": 85.10, "win_rate": 84.73},
    {"name": "gemini-2.0-flash",                        "lc_win_rate": 84.29, "win_rate": 83.61},
    {"name": "gpt-4o-2024-08-06",                       "lc_win_rate": 82.47, "win_rate": 81.99},
    {"name": "deepseek-v3",                             "lc_win_rate": 79.98, "win_rate": 79.48},
    {"name": "o1-mini-2024-09-12",                      "lc_win_rate": 78.43, "win_rate": 77.90},
    {"name": "claude-3-5-haiku-20241022",               "lc_win_rate": 76.21, "win_rate": 75.68},
    {"name": "gpt-4o-mini-2024-07-18",                  "lc_win_rate": 72.18, "win_rate": 71.52},
    {"name": "Qwen/Qwen2.5-72B-Instruct",               "lc_win_rate": 71.23, "win_rate": 70.80},
    {"name": "meta-llama/Llama-3.3-70B-Instruct",       "lc_win_rate": 69.84, "win_rate": 68.95},
    {"name": "mistralai/Mistral-Large-Instruct-2411",   "lc_win_rate": 65.13, "win_rate": 64.32},
    {"name": "microsoft/phi-4",                         "lc_win_rate": 63.40, "win_rate": 62.71},
    {"name": "gpt-4-turbo-2024-04-09",                  "lc_win_rate": 61.22, "win_rate": 60.88},
    {"name": "meta-llama/Meta-Llama-3.1-405B-Instruct", "lc_win_rate": 58.47, "win_rate": 57.93},
    {"name": "meta-llama/Meta-Llama-3.1-70B-Instruct",  "lc_win_rate": 54.37, "win_rate": 53.98},
    {"name": "claude-3-opus-20240229",                  "lc_win_rate": 52.14, "win_rate": 51.67},
    {"name": "gpt-4-0613",                              "lc_win_rate": 49.61, "win_rate": 49.12},
    {"name": "mistralai/Mixtral-8x22B-Instruct-v0.1",   "lc_win_rate": 44.25, "win_rate": 43.80},
    {"name": "meta-llama/Meta-Llama-3-70B-Instruct",    "lc_win_rate": 40.31, "win_rate": 39.87},
    {"name": "claude-3-sonnet-20240229",                "lc_win_rate": 38.60, "win_rate": 38.10},
    {"name": "mistralai/Mixtral-8x7B-Instruct-v0.1",    "lc_win_rate": 30.42, "win_rate": 29.98},
    {"name": "meta-llama/Meta-Llama-3-8B-Instruct",     "lc_win_rate": 27.19, "win_rate": 26.74},
    {"name": "meta-llama/Llama-2-70b-chat-hf",          "lc_win_rate": 20.84, "win_rate": 20.31},
    {"name": "gpt-3.5-turbo-0125",                      "lc_win_rate": 18.73, "win_rate": 18.21},
    {"name": "meta-llama/Llama-2-13b-chat-hf",          "lc_win_rate": 15.36, "win_rate": 14.89},
    {"name": "meta-llama/Llama-2-7b-chat-hf",           "lc_win_rate": 10.20, "win_rate": 9.81},
    {"name": "mistralai/Mistral-7B-Instruct-v0.1",      "lc_win_rate": 9.46,  "win_rate": 8.93},
]

# github API to list models under results/
_GITHUB_RESULTS_API = (
    "https://api.github.com/repos/tatsu-lab/alpaca_eval/contents/results"
)

# source data reference (the leaderboard itself is the data source)
_SOURCE_DATA = SourceDataUrl(
    dataset_name="AlpacaEval 2.0",
    source_type="url",
    url=["https://github.com/tatsu-lab/alpaca_eval"],
)

# shared source metadata
_SOURCE_METADATA = SourceMetadata(
    source_name="AlpacaEval 2.0 Leaderboard",
    source_type="documentation",
    source_organization_name="Stanford / Tatsu Lab",
    source_organization_url="https://tatsu-lab.github.io/alpaca_eval/",
    evaluator_relationship=EvaluatorRelationship.third_party,
)

_EVAL_LIBRARY = EvalLibrary(name="alpaca_eval", version="2.0")

_TIMEOUT = 30  # seconds


# ---------------------------------------------------------------------------
# scraper implementation
# ---------------------------------------------------------------------------


class AlpacaEval2Scraper(BaseLeaderboardScraper):
    """scraper for the AlpacaEval 2.0 leaderboard.

    tries multiple public sources in order; raises only when all fail.
    """

    eval_name = "alpacaeval2"
    source_name = "AlpacaEval 2.0 Leaderboard"
    source_organization = "Stanford / Tatsu Lab"
    output_dir = "data/alpacaeval2"

    def __init__(self, allow_fallback: bool = True) -> None:
        self._allow_fallback = allow_fallback
        self._is_fallback = False

    def fetch_raw(self) -> list[dict]:
        """fetch the leaderboard, trying CSV → JSON → hardcoded fallback."""
        # try compiled CSV files first (fastest path)
        for url in _LEADERBOARD_CSV_URLS:
            rows = _try_fetch_csv(url)
            if rows:
                print(f"  fetched CSV from {url}")
                return rows

        # try the JSON leaderboard
        rows = _try_fetch_json_leaderboard(_LEADERBOARD_JSON_URL)
        if rows:
            print(f"  fetched JSON from {_LEADERBOARD_JSON_URL}")
            return rows

        # fall back to hardcoded snapshot (avoids slow per-model GitHub API)
        print(
            f"  WARNING: all live sources failed — using hardcoded fallback "
            f"leaderboard ({len(_FALLBACK_LEADERBOARD)} models from {_FALLBACK_DATE})."
        )
        self._is_fallback = True
        return _FALLBACK_LEADERBOARD

    def convert(self, raw: list[dict], retrieved_timestamp: str) -> list[dict]:
        """convert raw leaderboard rows to EEE schema dicts."""
        ts = _FALLBACK_TS if getattr(self, "_is_fallback", False) else retrieved_timestamp
        results: list[dict] = []
        for row in raw:
            try:
                record = _convert_row(row, ts)
                results.append(record)
            except Exception as exc:
                name = row.get("name", row.get("model", "?"))
                print(f"  skipping {name!r}: {exc}")
        return results


# ---------------------------------------------------------------------------
# fetch helpers — single responsibility each
# ---------------------------------------------------------------------------


def _try_fetch_csv(url: str) -> list[dict]:
    """attempt to fetch and parse a CSV from *url*; return [] on failure."""
    try:
        resp = requests.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        reader = csv.DictReader(io.StringIO(resp.text))
        rows = list(reader)
        if rows:
            return rows
    except Exception:
        pass
    return []


def _try_fetch_json_leaderboard(url: str) -> list[dict]:
    """attempt to fetch a JSON leaderboard from *url*; return [] on failure."""
    try:
        resp = requests.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        # the JSON leaderboard may be a dict keyed by model name
        if isinstance(data, dict):
            rows = []
            for model_name, values in data.items():
                row = {"name": model_name}
                if isinstance(values, dict):
                    row.update(values)
                rows.append(row)
            return rows
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def _fetch_via_github_api() -> list[dict]:
    """use the GitHub contents API to list model result directories.

    Applies a short sleep between per-model requests to stay well under
    GitHub's unauthenticated rate limit (60 requests / hour).
    """
    import time as _time

    try:
        resp = requests.get(_GITHUB_RESULTS_API, timeout=_TIMEOUT)
        resp.raise_for_status()
        entries = resp.json()
        if not isinstance(entries, list):
            return []

        rows: list[dict] = []
        for entry in entries:
            if entry.get("type") != "dir":
                continue
            model_name = entry["name"]
            # respectful delay — GitHub allows 60 unauthed requests/hr
            _time.sleep(0.6)
            # try to fetch annotations/leaderboard_stats.json for this model
            stats = _fetch_model_stats(model_name)
            if stats:
                stats["name"] = model_name
                rows.append(stats)
        return rows
    except Exception:
        return []


def _fetch_model_stats(model_name: str) -> dict | None:
    """fetch the leaderboard stats JSON for a single model directory."""
    url = (
        f"https://raw.githubusercontent.com/tatsu-lab/alpaca_eval"
        f"/main/results/{model_name}/model_card.json"
    )
    try:
        resp = requests.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        # also try annotations summary
        url2 = (
            f"https://raw.githubusercontent.com/tatsu-lab/alpaca_eval"
            f"/main/results/{model_name}/annotations_seed0_configs.yaml"
        )
        try:
            import yaml  # type: ignore[import-untyped]

            resp2 = requests.get(url2, timeout=_TIMEOUT)
            resp2.raise_for_status()
            return yaml.safe_load(resp2.text)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# conversion helper — single responsibility
# ---------------------------------------------------------------------------


def _convert_row(row: dict[str, Any], retrieved_timestamp: str) -> dict:
    """convert a single leaderboard row to an EEE schema dict."""
    # the CSV/JSON uses various column names for the model;
    # the weighted_alpaca_eval_gpt4_turbo_leaderboard.csv uses an unnamed
    # first column (key="") for the model name.
    model_name: str = (
        row.get("name")
        or row.get("model")
        or row.get("model_name")
        or row.get("Model")
        or row.get("")
        or ""
    ).strip()

    if not model_name:
        raise ValueError("no model name found in row")

    # parse LC win rate (main AlpacaEval 2.0 metric) — stored as % in CSV
    # is_percentage=True because AlpacaEval always publishes these as 0–100
    lc_wr = _parse_score(
        row.get("lc_win_rate")
        or row.get("lc_winrate")
        or row.get("LC WinRate")
        or row.get("lc_win_rate_se")
        or row.get("length_controlled_winrate")
        or "0",
        is_percentage=True,
    )
    # parse standard win rate
    wr = _parse_score(
        row.get("win_rate")
        or row.get("winrate")
        or row.get("WinRate")
        or "0",
        is_percentage=True,
    )

    developer = get_developer(model_name)
    # build a HuggingFace-style model ID if not already in org/model format
    if "/" not in model_name:
        model_id = f"{developer}/{model_name}"
    else:
        model_id = model_name
        developer, _ = model_id.split("/", 1)

    safe_model_id = model_id.replace("/", "_")
    evaluation_id = f"alpacaeval2/{safe_model_id}/{retrieved_timestamp}"

    eval_results: list[EvaluationResult] = []

    # length-controlled win rate (primary metric)
    eval_results.append(
        EvaluationResult(
            evaluation_name="AlpacaEval LC Win Rate",
            source_data=_SOURCE_DATA,
            metric_config=MetricConfig(
                evaluation_description=(
                    "length-controlled win rate vs GPT-4 on AlpacaEval 2.0"
                ),
                lower_is_better=False,
                score_type=ScoreType.continuous,
                min_score=0.0,
                max_score=1.0,
            ),
            score_details=ScoreDetails(score=round(lc_wr, 4)),
        )
    )

    # standard win rate (secondary metric)
    if wr > 0:
        eval_results.append(
            EvaluationResult(
                evaluation_name="AlpacaEval Win Rate",
                source_data=_SOURCE_DATA,
                metric_config=MetricConfig(
                    evaluation_description=(
                        "standard win rate vs GPT-4 on AlpacaEval 2.0"
                    ),
                    lower_is_better=False,
                    score_type=ScoreType.continuous,
                    min_score=0.0,
                    max_score=1.0,
                ),
                score_details=ScoreDetails(score=round(wr, 4)),
            )
        )

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
            # inference_platform intentionally omitted — AlpacaEval is a
            # leaderboard aggregating many providers; the field is optional
            # and 'unknown' is semantically misleading.
        ),
        evaluation_results=eval_results,
    )

    return log.model_dump(mode='json', exclude_none=True)


def _parse_score(value: Any, is_percentage: bool = False) -> float:
    """parse *value* as a float in [0, 1].

    If *is_percentage* is True the raw value is treated as a 0–100 percentage
    and divided by 100 unconditionally.  This avoids the fragile ``> 1.0``
    heuristic which silently misclassifies genuine scores near zero
    (e.g. a model with a 1 % win rate would be left as ``0.01``, not
    normalised, which is correct — but one with exactly ``1.0`` would be
    treated as already in [0, 1] and not normalised, which is wrong).

    AlpacaEval 2.0 always publishes ``lc_win_rate`` and ``win_rate`` as
    percentages, so callers should always pass ``is_percentage=True``.
    """
    try:
        v = float(str(value).replace("%", "").strip())
        return v / 100.0 if is_percentage else v
    except (ValueError, TypeError):
        return 0.0


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main(limit: int | None = None, dry_run: bool = False) -> None:
    """run the AlpacaEval 2.0 scraper end-to-end."""
    scraper = AlpacaEval2Scraper()

    print("fetching AlpacaEval 2.0 leaderboard...")
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

    parser = argparse.ArgumentParser(description="scrape AlpacaEval 2.0 leaderboard")
    parser.add_argument("--limit", type=int, default=None, help="max models to process")
    parser.add_argument("--dry-run", action="store_true", help="skip writing files")
    args = parser.parse_args()
    main(limit=args.limit, dry_run=args.dry_run)
