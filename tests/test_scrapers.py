"""Unit tests for leaderboard scrapers.

All HTTP calls are mocked — no network access is required.  Each test
validates that:
  1. The scraper's fetch_raw() + convert() produce the expected number of records.
  2. Every converted record passes validation against eval.schema.json.
  3. Key schema fields (evaluation_name, score range, timestamp backdating on
     fallback) have the correct values.

Coverage:
  MT-Bench        — JSONL primary path, fallback path, no-fallback-raises
  AlpacaEval 2.0  — CSV primary path, score normalisation (0-100 → 0-1)
  Chatbot Arena   — fallback path (all live sources fail), no-fallback-raises
  WildBench v2    — JSON list layout, JSON dict layout
  HF Open LLM v2  — API primary path, score normalisation, malformed entry skip
"""

from __future__ import annotations

import json
import sys
import textwrap
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# ensure repo root and utils/ are importable
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "utils") not in sys.path:
    sys.path.insert(0, str(_ROOT / "utils"))

# ---------------------------------------------------------------------------
# JSON-schema validator — skip gracefully when jsonschema is unavailable
# ---------------------------------------------------------------------------

try:
    from jsonschema.validators import validator_for as _validator_for

    _SCHEMA_PATH = _ROOT / "eval.schema.json"
    with _SCHEMA_PATH.open() as _fh:
        _SCHEMA = json.load(_fh)
    _VALIDATOR = _validator_for(_SCHEMA)(_SCHEMA)
    _HAS_JSONSCHEMA = True
except Exception:
    _HAS_JSONSCHEMA = False


def _assert_schema_valid(record: dict) -> None:
    """Validate *record* against eval.schema.json; skip if validator unavailable."""
    if not _HAS_JSONSCHEMA:
        return
    _VALIDATOR.validate(record)


# ---------------------------------------------------------------------------
# shared mock helpers
# ---------------------------------------------------------------------------


def _mock_ok(data: Any, *, text: str | None = None) -> MagicMock:
    """Return a mock requests.Response that succeeds with *data*."""
    m = MagicMock()
    m.status_code = 200
    m.json.return_value = data
    m.text = text if text is not None else (
        json.dumps(data) if isinstance(data, (dict, list)) else str(data)
    )
    m.raise_for_status = MagicMock()  # no-op — does not raise
    return m


def _mock_fail() -> MagicMock:
    """Return a mock that raises ConnectionError on every call."""
    import requests as _req

    return MagicMock(side_effect=_req.exceptions.ConnectionError("mocked network failure"))


# ===========================================================================
# MT-Bench
# ===========================================================================

_MTBENCH_JSONL = textwrap.dedent("""\
    {"model": "gpt-4-0613", "score": 9.18}
    {"model": "llama-2-70b-chat", "score": 6.27}
    {"model": "vicuna-13b-v1.3", "score": 6.57}
""")


def test_mtbench_jsonl_primary_path():
    """Converts three JSONL rows into three schema-valid MT-Bench records."""
    from scripts.scrapers.mtbench_scraper import MTBenchScraper

    with patch(
        "scripts.scrapers.mtbench_scraper.requests.get",
        return_value=_mock_ok(None, text=_MTBENCH_JSONL),
    ):
        scraper = MTBenchScraper(allow_fallback=False)
        raw = scraper.fetch_raw()

    assert len(raw) == 3
    assert scraper._is_fallback is False

    records = scraper.convert(raw, str(time.time()))
    assert len(records) == 3

    for rec in records:
        _assert_schema_valid(rec)
        result = rec["evaluation_results"][0]
        assert result["evaluation_name"] == "MT-Bench"
        score = result["score_details"]["score"]
        assert 1.0 <= score <= 10.0, f"MT-Bench score out of [1,10]: {score}"


def test_mtbench_fallback_backdates_timestamp():
    """When all live sources fail, the fallback timestamp must match the hardcoded date."""
    from scripts.scrapers.mtbench_scraper import (
        MTBenchScraper,
        _FALLBACK_LEADERBOARD,
        _FALLBACK_TS,
    )

    with patch("scripts.scrapers.mtbench_scraper.requests.get", _mock_fail()):
        scraper = MTBenchScraper(allow_fallback=True)
        raw = scraper.fetch_raw()

    assert raw is _FALLBACK_LEADERBOARD
    assert scraper._is_fallback is True

    records = scraper.convert(raw, str(time.time()))
    assert len(records) == len(_FALLBACK_LEADERBOARD)

    # Every record must carry the hardcoded fallback timestamp, not wall-clock now.
    for rec in records:
        assert rec["retrieved_timestamp"] == _FALLBACK_TS, (
            "fallback records must be backdated to the original publication date"
        )
        _assert_schema_valid(rec)


def test_mtbench_no_fallback_raises():
    """MTBenchScraper(allow_fallback=False) must raise RuntimeError when offline."""
    from scripts.scrapers.mtbench_scraper import MTBenchScraper

    with patch("scripts.scrapers.mtbench_scraper.requests.get", _mock_fail()):
        scraper = MTBenchScraper(allow_fallback=False)
        with pytest.raises(RuntimeError, match="no-fallback"):
            scraper.fetch_raw()


# ===========================================================================
# AlpacaEval 2.0
# ===========================================================================

_ALPACA_CSV = textwrap.dedent("""\
    name,lc_win_rate,win_rate
    gpt-4-turbo,50.12,55.34
    meta-llama/Meta-Llama-3-70B-Instruct,15.20,18.90
    mistralai/Mixtral-8x7B-Instruct-v0.1,8.40,9.10
""")


def test_alpacaeval2_csv_primary_path():
    """Parses three CSV rows; LC win rates are correctly normalised from 0-100 to 0-1."""
    from scripts.scrapers.alpacaeval2_scraper import AlpacaEval2Scraper

    with patch(
        "scripts.scrapers.alpacaeval2_scraper.requests.get",
        return_value=_mock_ok(None, text=_ALPACA_CSV),
    ):
        scraper = AlpacaEval2Scraper()
        raw = scraper.fetch_raw()

    assert len(raw) == 3

    records = scraper.convert(raw, str(time.time()))
    assert len(records) == 3

    for rec in records:
        _assert_schema_valid(rec)
        eval_names = {r["evaluation_name"] for r in rec["evaluation_results"]}
        assert "AlpacaEval LC Win Rate" in eval_names

    # Verify score normalisation: 50.12 / 100 = 0.5012
    gpt4_rec = next(
        r for r in records if "gpt-4" in r["model_info"]["id"].lower()
    )
    lc_result = next(
        r for r in gpt4_rec["evaluation_results"]
        if r["evaluation_name"] == "AlpacaEval LC Win Rate"
    )
    assert abs(lc_result["score_details"]["score"] - 0.5012) < 1e-4, (
        "LC win rate must be divided by 100 to normalise from % to proportion"
    )


def test_alpacaeval2_zero_standard_winrate_omits_secondary_result():
    """When win_rate is 0 or absent, only the LC win rate result is emitted."""
    from scripts.scrapers.alpacaeval2_scraper import AlpacaEval2Scraper

    csv_no_wr = "name,lc_win_rate\nmeta-llama/Llama-2-7b-chat-hf,3.50\n"

    with patch(
        "scripts.scrapers.alpacaeval2_scraper.requests.get",
        return_value=_mock_ok(None, text=csv_no_wr),
    ):
        scraper = AlpacaEval2Scraper()
        raw = scraper.fetch_raw()

    records = scraper.convert(raw, str(time.time()))
    assert len(records) == 1

    result_names = [r["evaluation_name"] for r in records[0]["evaluation_results"]]
    assert result_names == ["AlpacaEval LC Win Rate"], (
        "secondary Win Rate result must be omitted when win_rate is absent or zero"
    )


# ===========================================================================
# Chatbot Arena
# ===========================================================================


def test_chatbot_arena_fallback_backdates_timestamp():
    """All live sources fail → uses hardcoded snapshot timestamped to 2024-06-27."""
    from scripts.scrapers.chatbot_arena_scraper import (
        ChatbotArenaScraper,
        _FALLBACK_LEADERBOARD,
        _FALLBACK_RETRIEVED_TS,
    )

    with (
        patch("scripts.scrapers.chatbot_arena_scraper.requests.get", _mock_fail()),
        patch("scripts.scrapers.chatbot_arena_scraper.requests.post", _mock_fail()),
    ):
        scraper = ChatbotArenaScraper(allow_fallback=True)
        raw = scraper.fetch_raw()

    assert raw is _FALLBACK_LEADERBOARD

    records = scraper.convert(raw, str(time.time()))
    assert len(records) == len(_FALLBACK_LEADERBOARD)

    for rec in records:
        assert rec["retrieved_timestamp"] == _FALLBACK_RETRIEVED_TS, (
            "fallback Arena records must be backdated to the original publication date"
        )
        _assert_schema_valid(rec)
        result = rec["evaluation_results"][0]
        assert result["evaluation_name"] == "Chatbot Arena Elo"
        # All fallback models have Elo >= 1000
        assert result["score_details"]["score"] >= 1000.0


def test_chatbot_arena_no_fallback_raises():
    """ChatbotArenaScraper(allow_fallback=False) must raise when offline."""
    from scripts.scrapers.chatbot_arena_scraper import ChatbotArenaScraper

    with (
        patch("scripts.scrapers.chatbot_arena_scraper.requests.get", _mock_fail()),
        patch("scripts.scrapers.chatbot_arena_scraper.requests.post", _mock_fail()),
    ):
        scraper = ChatbotArenaScraper(allow_fallback=False)
        with pytest.raises(RuntimeError):
            scraper.fetch_raw()


# ===========================================================================
# WildBench v2
# ===========================================================================

_WILDBENCH_LIST = [
    {
        "model": "gpt-4-turbo-2024-04-09",
        "developer": "openai",
        "wb_score": 57.3,
        "inference_platform": "openai",
    },
    {
        "model": "meta-llama/Meta-Llama-3-70B-Instruct",
        "developer": "meta-llama",
        "wb_score": 38.8,
    },
    {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "developer": "mistralai",
        "wb_score": 27.5,
    },
]

_WILDBENCH_DICT = {
    "gpt-4-turbo-2024-04-09": {"developer": "openai", "wb_score": 57.3},
    "meta-llama/Meta-Llama-3-70B-Instruct": {"developer": "meta-llama", "wb_score": 38.8},
}


def test_wildbench_list_layout():
    """WildBenchScraper handles the list-of-dicts JSON layout."""
    from scripts.scrapers.wildbench_scraper import WildBenchScraper

    with patch(
        "scripts.scrapers.wildbench_scraper.requests.get",
        return_value=_mock_ok(_WILDBENCH_LIST),
    ):
        scraper = WildBenchScraper(allow_fallback=False)
        raw = scraper.fetch_raw()

    assert len(raw) == 3

    records = scraper.convert(raw, str(time.time()))
    assert len(records) == 3

    for rec in records:
        _assert_schema_valid(rec)
        result = rec["evaluation_results"][0]
        assert result["evaluation_name"] == "WildBench-v2 WB-Score"
        score = result["score_details"]["score"]
        assert -100.0 <= score <= 100.0, f"WB-Score out of bounds: {score}"


def test_wildbench_dict_layout():
    """WildBenchScraper handles the dict-keyed JSON layout (model → metrics)."""
    from scripts.scrapers.wildbench_scraper import WildBenchScraper

    with patch(
        "scripts.scrapers.wildbench_scraper.requests.get",
        return_value=_mock_ok(_WILDBENCH_DICT),
    ):
        scraper = WildBenchScraper(allow_fallback=False)
        raw = scraper.fetch_raw()

    # dict with 2 keys → 2 rows after normalisation
    assert len(raw) == 2

    records = scraper.convert(raw, str(time.time()))
    assert len(records) == 2
    for rec in records:
        _assert_schema_valid(rec)


def test_wildbench_no_fallback_raises():
    """WildBenchScraper(allow_fallback=False) must raise when offline."""
    from scripts.scrapers.wildbench_scraper import WildBenchScraper

    with patch("scripts.scrapers.wildbench_scraper.requests.get", _mock_fail()):
        scraper = WildBenchScraper(allow_fallback=False)
        with pytest.raises(RuntimeError):
            scraper.fetch_raw()


# ===========================================================================
# HF Open LLM Leaderboard v2
# ===========================================================================

_HFOPENLLM_DATA = [
    {
        "model": {
            "name": "meta-llama/Meta-Llama-3-8B-Instruct",
            "precision": "float16",
            "architecture": "LlamaForCausalLM",
        },
        "evaluations": {
            "ifeval": {"name": "IFEval", "value": 78.42},
            "bbh": {"name": "BBH", "value": 62.30},
            "math": {"name": "MATH Level 5", "value": 20.50},
        },
        "metadata": {"params_billions": 8.0},
    },
    {
        "model": {"name": "mistralai/Mistral-7B-Instruct-v0.3"},
        "evaluations": {
            "ifeval": {"name": "IFEval", "value": 55.21},
            "bbh": {"name": "BBH", "value": 47.88},
        },
        "metadata": {},
    },
]


def test_hfopenllm_v2_primary_path():
    """Converts two API entries into schema-valid records with normalised scores."""
    from scripts.scrapers.hfopenllm_v2_scraper import HFOpenLLMv2Scraper

    with patch(
        "helpers.fetch.requests.get",
        return_value=_mock_ok(_HFOPENLLM_DATA),
    ):
        scraper = HFOpenLLMv2Scraper()
        raw = scraper.fetch_raw()

    assert len(raw) == 2

    records = scraper.convert(raw, str(time.time()))
    assert len(records) == 2

    for rec in records:
        _assert_schema_valid(rec)
        assert rec["model_info"]["inference_platform"] == "huggingface"
        for result in rec["evaluation_results"]:
            score = result["score_details"]["score"]
            assert 0.0 <= score <= 1.0, (
                f"Score {score!r} for {result['evaluation_name']!r} must be "
                f"normalised to [0, 1] — check division by 100 logic"
            )

    # Verify scores for the first model: 78.42 / 100 = 0.7842
    llama_rec = next(r for r in records if "llama" in r["model_info"]["id"].lower())
    ifeval_result = next(
        r for r in llama_rec["evaluation_results"] if "IFEval" in r["evaluation_name"]
    )
    assert abs(ifeval_result["score_details"]["score"] - 0.7842) < 1e-4, (
        "IFEval score must be divided by 100 (78.42 % → 0.7842)"
    )


def test_hfopenllm_v2_skips_non_hf_model_id():
    """Entries without 'org/model' format must be skipped with a warning, not crash."""
    from scripts.scrapers.hfopenllm_v2_scraper import HFOpenLLMv2Scraper

    bad_data = [
        {
            "model": {"name": "just-a-plain-model"},
            "evaluations": {"ifeval": {"name": "IFEval", "value": 70.0}},
            "metadata": {},
        }
    ]

    with patch("helpers.fetch.requests.get", return_value=_mock_ok(bad_data)):
        scraper = HFOpenLLMv2Scraper()
        raw = scraper.fetch_raw()

    records = scraper.convert(raw, str(time.time()))
    # The bad entry must be skipped — not crash and not produce a record
    assert len(records) == 0, (
        "model IDs not in 'org/model' format should be skipped during conversion"
    )
