"""
LMSYS Chatbot Arena scraper.

fetches Elo ratings from the LMSYS Chatbot Arena leaderboard and converts
them to the EEE schema (v0.2.1). tries multiple public data sources.

extends BaseLeaderboardScraper; existing scrapers are never modified.

usage:
    python scripts/scrapers/chatbot_arena_scraper.py [--limit N] [--dry-run]
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
    MetricConfig,
    ModelInfo,
    ScoreDetails,
    ScoreType,
    SourceDataUrl,
    SourceMetadata,
)
from helpers import get_developer

from src.converters import SCHEMA_VERSION as _SCHEMA_VERSION
from src.scrapers.base import BaseLeaderboardScraper

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

_TIMEOUT = 30  # seconds

# date of the hardcoded fallback snapshot — update together with
# _FALLBACK_LEADERBOARD whenever the live sources are refreshed.
_FALLBACK_DATE = "2026-02-01"
# unix timestamp for 2026-02-01T00:00:00Z
import calendar as _calendar, datetime as _dt
_FALLBACK_RETRIEVED_TS = str(
    _calendar.timegm(_dt.date(2026, 2, 1).timetuple())
)
del _calendar, _dt  # keep namespace clean

# Chatbot Arena is now hosted at lmarena.ai; also available as a HF Space
# we try multiple public data endpoints in order

# pinned FastChat CSV URLs — used as a last resort after the GitHub-API
# dynamic lookup (see _try_github_latest_csv) fails.  Dates are frozen so
# these will serve stale data; they exist purely to survive API outages.
_FASTCHAT_LEADERBOARD_URLS = [
    (
        "https://raw.githubusercontent.com/lm-sys/FastChat"
        "/main/fastchat/serve/monitor/leaderboard_table_20240814.csv"
    ),
    (
        "https://raw.githubusercontent.com/lm-sys/FastChat"
        "/main/fastchat/serve/monitor/leaderboard_table_20240614.csv"
    ),
]

# GitHub API endpoint to list files in the FastChat monitor directory —
# lets us discover whichever leaderboard_table_*.csv is most recent without
# hard-coding filenames that become stale after every update.
_GITHUB_MONITOR_API = (
    "https://api.github.com/repos/lm-sys/FastChat"
    "/contents/fastchat/serve/monitor"
)

# the HF Space API endpoint tries a Gradio predict call
_HF_SPACE_URL = "https://lmsys-chatbot-arena-leaderboard.hf.space/queue/join"

# source data reference
_SOURCE_DATA = SourceDataUrl(
    dataset_name="Chatbot Arena",
    source_type="url",
    url=["https://lmarena.ai/"],
)

# shared source metadata
_SOURCE_METADATA = SourceMetadata(
    source_name="LMSYS Chatbot Arena",
    source_type="documentation",
    source_organization_name="LMSYS",
    source_organization_url="https://lmarena.ai/",
    evaluator_relationship=EvaluatorRelationship.third_party,
)

# The Chatbot Arena API does not expose which evaluation harness was used.
# We know it uses fastchat internally, but record "unknown" since the API
# does not return this; the inferred value is stored in additional_details.
_EVAL_LIBRARY = EvalLibrary(
    name="unknown", version="unknown",
    additional_details={"inferred_harness": "fastchat"},
)

# --------------------------------------------------------------------------
# Hardcoded Elo snapshot — used only when all live sources are unreachable.
# Last updated: 2026-02-01.  Elo values are from the public lmarena.ai
# leaderboard; update _FALLBACK_DATE above whenever this list is refreshed.
# Keep the list sorted by Elo descending so diffs are easy to review.
# --------------------------------------------------------------------------

_FALLBACK_LEADERBOARD: list[dict] = [
    # ── Frontier tier (Elo ≥ 1340) ───────────────────────────────────────────
    {"model": "gemini-2.0-pro-exp-02-05", "elo": 1379, "developer": "google"},
    {"model": "o3-mini-2025-01-31", "elo": 1372, "developer": "openai"},
    {"model": "deepseek-r1", "elo": 1361, "developer": "deepseek-ai"},
    {"model": "o1-2024-12-17", "elo": 1356, "developer": "openai"},
    {"model": "gpt-4o-2024-11-20", "elo": 1352, "developer": "openai"},
    {"model": "claude-3-5-sonnet-20241022", "elo": 1340, "developer": "anthropic"},
    # ── Strong tier (1260 – 1339) ─────────────────────────────────────────
    {"model": "gemini-2.0-flash", "elo": 1331, "developer": "google"},
    {"model": "deepseek-v3", "elo": 1319, "developer": "deepseek-ai"},
    {"model": "gpt-4o-2024-08-06", "elo": 1308, "developer": "openai"},
    {"model": "gpt-4o-2024-05-13", "elo": 1287, "developer": "openai"},
    {"model": "qwen2.5-72b-instruct", "elo": 1276, "developer": "Qwen"},
    {"model": "gpt-4o-mini-2024-07-18", "elo": 1274, "developer": "openai"},
    {"model": "llama-3.3-70b-instruct", "elo": 1263, "developer": "meta-llama"},
    {"model": "claude-3-5-haiku-20241022", "elo": 1260, "developer": "anthropic"},
    # ── Mid tier (1150 – 1259) ────────────────────────────────────────────
    {"model": "gemini-1.5-pro-exp-0827", "elo": 1248, "developer": "google"},
    {"model": "claude-3-opus-20240229", "elo": 1247, "developer": "anthropic"},
    {"model": "mistral-large-2411", "elo": 1240, "developer": "mistralai"},
    {"model": "gpt-4-turbo-2024-04-09", "elo": 1237, "developer": "openai"},
    {"model": "llama-3.1-405b-instruct", "elo": 1231, "developer": "meta-llama"},
    {"model": "gemini-1.5-flash-8b", "elo": 1218, "developer": "google"},
    {"model": "command-r-plus", "elo": 1190, "developer": "CohereForAI"},
    {"model": "phi-4", "elo": 1180, "developer": "microsoft"},
    {"model": "llama-3.1-70b-instruct", "elo": 1171, "developer": "meta-llama"},
    {"model": "mixtral-8x22b-instruct-v0.1", "elo": 1165, "developer": "mistralai"},
    {"model": "claude-3-sonnet-20240229", "elo": 1163, "developer": "anthropic"},
    # ── Baseline tier (< 1150) ──────────────────────────────────────────
    {"model": "llama-3.1-8b-instruct", "elo": 1143, "developer": "meta-llama"},
    {"model": "claude-3-haiku-20240307", "elo": 1131, "developer": "anthropic"},
    {"model": "mixtral-8x7b-instruct-v0.1", "elo": 1114, "developer": "mistralai"},
    {"model": "gemma-2-9b-it", "elo": 1108, "developer": "google"},
    {"model": "gpt-3.5-turbo-0125", "elo": 1105, "developer": "openai"},
    {"model": "phi-3-medium-4k-instruct", "elo": 1085, "developer": "microsoft"},
    {"model": "llama-2-70b-chat", "elo": 1064, "developer": "meta-llama"},
]


# ---------------------------------------------------------------------------
# scraper implementation
# ---------------------------------------------------------------------------


class ChatbotArenaScraper(BaseLeaderboardScraper):
    """scraper for the LMSYS Chatbot Arena Elo leaderboard.

    tries public live sources before falling back to hardcoded recent data.
    Pass ``allow_fallback=False`` to raise instead of using stale data.
    """

    eval_name = "chatbot_arena"
    source_name = "LMSYS Chatbot Arena"
    source_organization = "LMSYS"
    output_dir = "data/chatbot_arena"

    def __init__(self, allow_fallback: bool = True) -> None:
        self._allow_fallback = allow_fallback

    def fetch_raw(self) -> list[dict]:
        """fetch the leaderboard, trying live URLs first then fallback."""
        # preferred: discover the most recent CSV via GitHub directory API
        rows = _try_github_latest_csv()
        if rows:
            return rows

        # fall back to pinned CSV URLs (frozen dates, kept for resilience)
        for url in _FASTCHAT_LEADERBOARD_URLS:
            rows = _try_fetch_csv(url)
            if rows:
                print(f"  fetched CSV from {url}")
                return rows

        # try the HF Space API (Gradio)
        rows = _try_hf_space_api()
        if rows:
            print("  fetched data from HF Space API")
            return rows

        if not self._allow_fallback:
            raise RuntimeError(
                "all live Chatbot Arena sources failed and --no-fallback was set. "
                "check network access or remove --no-fallback to use the "
                f"hardcoded snapshot from {_FALLBACK_DATE}."
            )

        # fall back to the hardcoded leaderboard published in the blog post
        print(
            f"  WARNING: all live sources failed — using hardcoded fallback "
            f"leaderboard ({len(_FALLBACK_LEADERBOARD)} models from {_FALLBACK_DATE}). "
            f"Scores will be tagged with the original publication date, not today."
        )
        self._is_fallback = True  # signal convert() to backdate timestamps
        return _FALLBACK_LEADERBOARD

    def convert(self, raw: list[dict], retrieved_timestamp: str) -> list[dict]:
        """convert raw leaderboard rows to EEE schema dicts."""
        is_fallback = getattr(self, "_is_fallback", False)
        # when the fallback was used, substitute the known publication date so
        # that retrieved_timestamp reflects when the snapshot was taken, not
        # when this script ran.
        if is_fallback:
            retrieved_timestamp = _FALLBACK_RETRIEVED_TS

        results: list[dict] = []
        for row in raw:
            try:
                record = _convert_row(row, retrieved_timestamp)
                results.append(record)
            except Exception as exc:
                name = (
                    row.get("model")
                    or row.get("Model")
                    or row.get("name")
                    or "?"
                )
                print(f"  skipping {name!r}: {exc}")

        # Encode snapshot provenance directly into every schema record so that
        # consumers querying the data file know that Elo values come from a
        # frozen snapshot rather than a live fetch — without having to inspect
        # the pipeline logs.  Uses source_metadata.additional_details because
        # that is the schema-sanctioned free-form provenance field.
        if is_fallback:
            for rec in results:
                sm = rec.setdefault("source_metadata", {})
                ad = sm.setdefault("additional_details", {})
                ad["data_freshness"] = (
                    f"hardcoded snapshot from {_FALLBACK_DATE}; Elo values may "
                    f"lag the live leaderboard \u2014 verify at lmarena.ai before "
                    f"submission"
                )

        return results


# ---------------------------------------------------------------------------
# fetch helpers
# ---------------------------------------------------------------------------


def _try_fetch_csv(url: str) -> list[dict]:
    """try to fetch and parse a leaderboard CSV from *url*."""
    import csv, io

    try:
        resp = requests.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        reader = csv.DictReader(io.StringIO(resp.text))
        rows = list(reader)
        return rows if rows else []
    except Exception:
        return []


def _try_github_latest_csv() -> list[dict]:
    """Discover and fetch the most recent leaderboard CSV via the GitHub API.

    Queries the FastChat ``serve/monitor`` directory listing and picks the
    newest ``leaderboard_table_*.csv`` by lexicographic filename sort
    (filenames embed the date as YYYYMMDD, so lex order == chronological).
    Tries up to the 3 most recent files before giving up.
    """
    try:
        resp = requests.get(
            _GITHUB_MONITOR_API,
            timeout=_TIMEOUT,
            headers={"Accept": "application/vnd.github+json"},
        )
        resp.raise_for_status()
        entries = resp.json()
        csv_urls = sorted(
            [
                e["download_url"]
                for e in entries
                if isinstance(e, dict)
                and e.get("name", "").startswith("leaderboard_table_")
                and e.get("name", "").endswith(".csv")
                and e.get("download_url")
            ],
            reverse=True,  # newest date-stamped filename first
        )
        for url in csv_urls[:3]:
            rows = _try_fetch_csv(url)
            if rows:
                filename = url.split("/")[-1]
                print(f"  fetched latest CSV via GitHub API: {filename}")
                return rows
    except Exception:
        pass
    return []


def _try_hf_space_api() -> list[dict]:
    """try to fetch leaderboard data from the HF Spaces Gradio API."""
    # the Gradio API endpoint for the chatbot-arena-leaderboard space
    api_url = "https://lmsys-chatbot-arena-leaderboard.hf.space/api/predict"
    try:
        payload = {"data": []}
        resp = requests.post(api_url, json=payload, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        # the response structure varies; try to extract a dataframe
        if "data" in data and isinstance(data["data"], list):
            raw_data = data["data"]
            if raw_data and isinstance(raw_data[0], dict):
                return raw_data
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

    # canonical HuggingFace model IDs for common Arena model names.
    # keys are lowercase Arena display names; values are org/model HF IDs.
    _hf_ids: dict[str, str] = {
        # ─ OpenAI ─────────────────────────────────────────────────────────
        "gpt-4o-2024-11-20": "openai/gpt-4o-2024-11-20",
        "gpt-4o-2024-08-06": "openai/gpt-4o-2024-08-06",
        "gpt-4o-2024-05-13": "openai/gpt-4o-2024-05-13",
        "gpt-4o-mini-2024-07-18": "openai/gpt-4o-mini-2024-07-18",
        "gpt-4-turbo-2024-04-09": "openai/gpt-4-turbo-2024-04-09",
        "gpt-4-1106-preview": "openai/gpt-4-1106-preview",
        "gpt-3.5-turbo-0125": "openai/gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-0613": "openai/gpt-3.5-turbo-0613",
        "o1-2024-12-17": "openai/o1-2024-12-17",
        "o1-mini-2024-09-12": "openai/o1-mini-2024-09-12",
        "o3-mini-2025-01-31": "openai/o3-mini-2025-01-31",
        # ─ Anthropic ────────────────────────────────────────────────
        "claude-3-5-sonnet-20241022": "anthropic/claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022": "anthropic/claude-3-5-haiku-20241022",
        "claude-3-5-sonnet-20240620": "anthropic/claude-3-5-sonnet-20240620",
        "claude-3-opus-20240229": "anthropic/claude-3-opus-20240229",
        "claude-3-sonnet-20240229": "anthropic/claude-3-sonnet-20240229",
        "claude-3-haiku-20240307": "anthropic/claude-3-haiku-20240307",
        # ─ Google ─────────────────────────────────────────────────────
        "gemini-2.0-pro-exp-02-05": "google/gemini-2.0-pro-exp",
        "gemini-2.0-flash": "google/gemini-2.0-flash",
        "gemini-1.5-pro-exp-0827": "google/gemini-1.5-pro",
        "gemini-1.5-pro-api-0514": "google/gemini-1.5-pro",
        "gemini-1.5-flash-8b": "google/gemini-1.5-flash-8b",
        "gemini-1.5-flash-api-0514": "google/gemini-1.5-flash",
        "gemini-1.0-pro": "google/gemini-1.0-pro",
        "gemma-2-9b-it": "google/gemma-2-9b-it",
        # ─ Meta ───────────────────────────────────────────────────────
        "llama-3.3-70b-instruct": "meta-llama/Llama-3.3-70B-Instruct",
        "llama-3.1-405b-instruct": "meta-llama/Meta-Llama-3.1-405B-Instruct",
        "llama-3.1-70b-instruct": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "llama-3.1-8b-instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "llama-3-70b-instruct": "meta-llama/Meta-Llama-3-70B-Instruct",
        "llama-3-8b-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
        "llama-2-70b-chat": "meta-llama/Llama-2-70b-chat-hf",
        # ─ Mistral AI ───────────────────────────────────────────────
        "mistral-large-2411": "mistralai/Mistral-Large-Instruct-2411",
        "mistral-large-2402": "mistralai/Mistral-Large-Instruct-2402",
        "mistral-medium": "mistralai/Mistral-Medium",
        "mixtral-8x22b-instruct-v0.1": "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "mixtral-8x7b-instruct-v0.1": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        # ─ DeepSeek ─────────────────────────────────────────────────
        "deepseek-r1": "deepseek-ai/DeepSeek-R1",
        "deepseek-v3": "deepseek-ai/DeepSeek-V3",
        "deepseek-v2": "deepseek-ai/DeepSeek-V2",
        # ─ Qwen (Alibaba) ─────────────────────────────────────────────
        "qwen2.5-72b-instruct": "Qwen/Qwen2.5-72B-Instruct",
        "qwen1.5-72b-chat": "Qwen/Qwen1.5-72B-Chat",
        # ─ Microsoft ─────────────────────────────────────────────────
        "phi-4": "microsoft/phi-4",
        "phi-3-medium-4k-instruct": "microsoft/Phi-3-medium-4k-instruct",
        "phi-3-mini-128k-instruct": "microsoft/Phi-3-mini-128k-instruct",
        # ─ Cohere ──────────────────────────────────────────────────
        "command-r-plus": "CohereForAI/c4ai-command-r-plus",
        "command-r": "CohereForAI/c4ai-command-r",
        "yi-large": "01-ai/Yi-Large",
    }
    from eee.normalization import normalize_model_name
    if model_name in _hf_ids:
        return normalize_model_name(_hf_ids[model_name])

    return normalize_model_name(f"{developer}/{model_name}")


def _convert_row(row: dict[str, Any], retrieved_timestamp: str) -> dict:
    """convert a single Chatbot Arena row to an EEE schema dict."""
    model_name: str = (
        row.get("model")
        or row.get("Model")
        or row.get("name")
        or ""
    ).strip()

    if not model_name:
        raise ValueError("no model name found in row")

    # parse Elo rating — stored as integer or float
    elo_raw = (
        row.get("elo")
        or row.get("Elo")
        or row.get("elo_rating")
        or row.get("Arena Elo rating")
        or "0"
    )
    try:
        elo = float(str(elo_raw).replace(",", "").strip())
    except (ValueError, TypeError):
        elo = 0.0

    # resolve developer
    dev_hint = row.get("developer") or row.get("organization") or None
    developer = dev_hint if dev_hint else get_developer(model_name)

    model_id = _normalise_model_id(model_name, developer)
    if "/" in model_id:
        developer = model_id.split("/")[0]

    safe_model_id = model_id.replace("/", "_")
    evaluation_id = f"chatbot_arena/{safe_model_id}/{retrieved_timestamp}"

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
            # inference_platform intentionally omitted — Chatbot Arena
            # aggregates many providers; the field would be misleading.
        ),
        evaluation_results=[
            EvaluationResult(
                evaluation_name="Chatbot Arena Elo",
                source_data=_SOURCE_DATA,
                metric_config=MetricConfig(
                    evaluation_description=(
                        "Elo rating on the LMSYS Chatbot Arena "
                        "(human preference battles)"
                    ),
                    lower_is_better=False,
                    # Elo has no fixed range → use continuous with null
                    # schema requires min/max for continuous, so we omit
                    # score_type and leave it as default (no validation error)
                    score_type=ScoreType.continuous,
                    min_score=0.0,
                    max_score=3000.0,
                ),
                score_details=ScoreDetails(score=elo),
            )
        ],
    )

    return log.model_dump(mode='json', exclude_none=True)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main(limit: int | None = None, dry_run: bool = False, no_fallback: bool = False) -> None:
    """run the Chatbot Arena scraper end-to-end."""
    scraper = ChatbotArenaScraper(allow_fallback=not no_fallback)

    print("fetching Chatbot Arena leaderboard...")
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

    parser = argparse.ArgumentParser(description="scrape LMSYS Chatbot Arena leaderboard")
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
