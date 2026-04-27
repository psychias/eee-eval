"""
add_leaderboard_records.py — fetch live leaderboard scores and write EEE records.

NO hardcoded scores.  Every number is fetched from the authoritative source
at runtime and cached to disk for 24 hours.

HuggingFace leaderboard fetchers use huggingface_hub + pandas to download
the actual parquet files — NOT the datasets-server rows API, which has
unreliable column names, rate limits, and no authentication support.

Requirements
------------
  pip install huggingface_hub pandas pyarrow requests

Authentication
--------------
  HF datasets that require a token (v1, v2 leaderboards).
  The token is loaded in this priority order:

    1. HF_TOKEN environment variable          (export HF_TOKEN=hf_...)
    2. .env file in the repo root             (HF_TOKEN=hf_...)
    3. .env file next to this script

  python-dotenv is used if installed; otherwise the .env file is parsed
  manually so there is no hard dependency.  Without a valid token the two
  HF leaderboard fetchers raise a clear error; all other sources still run.

Cache
-----
  .cache/leaderboards/{source_key}.json   (24-hour TTL)
  Bypass with --refresh.

Usage
-----
  python add_leaderboard_records.py                        # all sources
  python add_leaderboard_records.py --source bigcodebench  # one source
  python add_leaderboard_records.py --refresh              # bypass cache
  python add_leaderboard_records.py --list                 # list sources
  python add_leaderboard_records.py --probe open_llm_v1    # print columns
"""
from __future__ import annotations

import csv
import io
import json
import os
import pathlib
import re
import sys
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import requests

_ROOT     = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

DATA_DIR  = _ROOT / "data"
CACHE_DIR = pathlib.Path(".cache/leaderboards")
CACHE_TTL = 24 * 3600
_TIMEOUT  = 30

# ---------------------------------------------------------------------------
# .env loader — reads HF_TOKEN from a .env file when not already set in the
# environment.  Tries python-dotenv first (cleaner), then falls back to a
# minimal manual parser so there is no hard dependency.
#
# Search order for .env:
#   1. repo root  (_ROOT / ".env")
#   2. directory containing this script
# ---------------------------------------------------------------------------

def _load_dotenv() -> None:
    """Load .env into os.environ if HF_TOKEN is not already set."""
    if os.environ.get("HF_TOKEN"):
        return  # already present — nothing to do

    candidates = [
        _ROOT / ".env",
        pathlib.Path(__file__).resolve().parent / ".env",
    ]

    for env_path in candidates:
        if not env_path.exists():
            continue

        # Try python-dotenv first
        try:
            from dotenv import load_dotenv
            load_dotenv(dotenv_path=str(env_path), override=False)
            if os.environ.get("HF_TOKEN"):
                print(f"  [env] HF_TOKEN loaded via python-dotenv from {env_path}")
                return
        except ImportError:
            pass  # python-dotenv not installed — fall through to manual parse

        # Manual .env parser — handles KEY=value and KEY="value" lines,
        # skips blank lines and # comments, does not expand variables.
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, raw_val = line.partition("=")
                key     = key.strip()
                raw_val = raw_val.strip()
                # Strip optional surrounding quotes  "value" or 'value'
                if len(raw_val) >= 2 and raw_val[0] in ('"', "'") and raw_val[0] == raw_val[-1]:
                    raw_val = raw_val[1:-1]
                if key and key not in os.environ:
                    os.environ[key] = raw_val
            if os.environ.get("HF_TOKEN"):
                print(f"  [env] HF_TOKEN loaded from {env_path}")
                return
        except Exception as exc:
            print(f"  [env] could not parse {env_path}: {exc}", file=sys.stderr)


_load_dotenv()

# HF token — required for gated datasets (both leaderboards are gated).
# Resolved after _load_dotenv() so the .env file is already in os.environ.
_HF_TOKEN: str = os.environ.get("HF_TOKEN", "")


# ---------------------------------------------------------------------------
# Shared schema helpers
# ---------------------------------------------------------------------------

def _already_exists(source_dir: str, model_id: str) -> bool:
    d = DATA_DIR / source_dir
    if not d.exists():
        return False
    # Build a cached set of model IDs per source_dir to avoid O(n*m) file reads
    cache = _already_exists._cache  # type: ignore[attr-defined]
    if source_dir not in cache:
        ids: set[str] = set()
        for f in d.rglob("*.json"):
            try:
                mid = json.loads(f.read_text(encoding="utf-8")).get("model_info", {}).get("id")
                if mid:
                    ids.add(mid)
            except Exception:
                pass
        cache[source_dir] = ids
    return model_id in cache[source_dir]

_already_exists._cache: dict[str, set[str]] = {}  # type: ignore[attr-defined]


def _save(rec: dict, source_dir: str, developer: str, model_slug: str) -> pathlib.Path:
    # Sanitise path components for Windows (remove :, ?, *, <, >, |, ")
    safe_dev  = re.sub(r'[<>:"/\\|?*]', '_', developer).strip(". ")
    safe_slug = re.sub(r'[<>:"/\\|?*]', '_', model_slug).strip(". ")
    out = DATA_DIR / source_dir / safe_dev / safe_slug
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{uuid.uuid4()}.json"
    p.write_text(json.dumps(rec, indent=2))
    return p


def _record(
    source_dir: str, source_name: str, source_org: str,
    source_org_url: str, eval_library: str,
    model_id: str, model_name: str, developer: str,
    results: list[dict],
    evaluator_relationship: str = "third_party",
    eval_date: str = "",
) -> dict:
    ts = str(int(time.time()))
    return {
        "schema_version": "0.2.1",
        "evaluation_id": f"{source_dir}/{model_id.replace('/', '_')}/{ts}",
        **({
            "evaluation_timestamp": eval_date,
        } if eval_date else {}),
        "retrieved_timestamp": ts,
        "source_metadata": {
            "source_name": source_name,
            "source_type": "documentation",
            "source_organization_name": source_org,
            "source_organization_url": source_org_url,
            "evaluator_relationship": evaluator_relationship,
        },
        "eval_library": {"name": eval_library, "version": "unknown"},
        "model_info": {"name": model_name, "id": model_id, "developer": developer},
        "evaluation_results": [
            {
                "evaluation_name": r["bench"],
                **({"evaluation_timestamp": eval_date} if eval_date else {}),
                "source_data": {
                    "dataset_name": r["bench"],
                    "source_type": "url",
                    "url": [source_org_url],
                },
                "metric_config": {
                    "metric_name":    r.get("metric_name", "accuracy"),
                    "lower_is_better": r.get("lower_is_better", False),
                    "score_type":     "continuous",
                    "min_score":      r["min_score"],
                    "max_score":      r["max_score"],
                },
                "score_details": {
                    "score": r["score"],
                    **({"details": r["details"]} if r.get("details") else {}),
                },
                "generation_config": {
                    "generation_args": {
                        **({"shots": r["shots"]} if r.get("shots") is not None else {}),
                        **({"temperature": r["temperature"]} if r.get("temperature") is not None else {}),
                        **({"chain_of_thought": r["chain_of_thought"]} if r.get("chain_of_thought") is not None else {}),
                    },
                    "additional_details": {
                        "source": source_dir,
                        **({"note": r["note"]} if r.get("note") else {}),
                    },
                },
            }
            for r in results
        ],
    }


def _add(
    source_dir: str, source_name: str, source_org: str,
    source_org_url: str, eval_library: str,
    model_id: str, model_name: str, developer: str,
    results: list[dict],
    evaluator_relationship: str = "third_party",
    eval_date: str = "",
) -> int:
    if _already_exists(source_dir, model_id):
        return 0
    rec = _record(
        source_dir, source_name, source_org, source_org_url,
        eval_library, model_id, model_name, developer, results,
        evaluator_relationship, eval_date,
    )
    slug = model_id.split("/", 1)[-1]
    _save(rec, source_dir, developer, slug)
    # Update the cache so subsequent checks within this run are instant
    _already_exists._cache.setdefault(source_dir, set()).add(model_id)  # type: ignore[attr-defined]
    return 1


# ---------------------------------------------------------------------------
# Developer inference
# ---------------------------------------------------------------------------

_DEV_PATTERNS: list[tuple[str, str]] = sorted([
    ("gpt",       "openai"),    ("o1",        "openai"),
    ("o3",        "openai"),    ("openai",    "openai"),
    ("claude",    "anthropic"), ("gemini",    "google"),
    ("gemma",     "google"),    ("palm",      "google"),
    ("codellama", "meta-llama"),("llama",     "meta-llama"),
    ("mistral",   "mistralai"), ("mixtral",   "mistralai"),
    ("codestral", "mistralai"), ("qwen",      "Qwen"),
    ("qwq",       "Qwen"),      ("phi",       "microsoft"),
    ("falcon",    "tiiuae"),    ("bloom",     "bigscience"),
    ("pythia",    "EleutherAI"),("gpt-j",     "EleutherAI"),
    ("mpt",       "mosaicml"), ("dbrx",       "databricks"),
    ("deepseek",  "deepseek-ai"),("yi",        "01-ai"),
    ("olmo",      "allenai"),  ("command",    "CohereForAI"),
    ("aya",       "CohereForAI"),("jamba",    "ai21labs"),
    ("nemotron",  "nvidia"),   ("starcoder",  "bigcode"),
    ("vicuna",    "lmsys"),    ("alpaca",     "stanford"),
    ("grok",      "xai"),      ("glm",        "THUDM"),
    ("internlm",  "internlm"), ("baichuan",   "baichuan-inc"),
], key=lambda kv: len(kv[0]), reverse=True)

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


def _infer_developer(model_name: str, organization: str = "") -> str:
    if organization:
        for label, dev in _ORG_MAP.items():
            if label in organization.lower():
                return dev
    lower = model_name.lower()
    for prefix, dev in _DEV_PATTERNS:
        if lower.startswith(prefix) or f"-{prefix}" in lower or f"/{prefix}" in lower:
            return dev
    return "unknown"


def _extract_meta_from_text(text: str) -> dict:
    """Extract shots / temperature / CoT from a benchmark or model name string.

    Returns a dict with keys 'shots', 'temperature', 'chain_of_thought'
    that are set only when detected (otherwise absent).
    """
    meta: dict = {}
    lower = text.lower()
    # --- shots ---
    m = re.search(r'(\d+)[- ]shot', lower)
    if m:
        meta['shots'] = int(m.group(1))
    elif re.search(r'zero[- ]?shot', lower):
        meta['shots'] = 0
    elif re.search(r'one[- ]?shot', lower):
        meta['shots'] = 1
    elif re.search(r'few[- ]?shot', lower):
        meta['shots'] = -1  # sentinel for "few"
    # --- chain-of-thought ---
    if re.search(r'\bcot\b|chain[- ]of[- ]thought', lower):
        meta['chain_of_thought'] = True
    # --- temperature ---
    tm = re.search(r'temp(?:erature)?\s*=?\s*([0-9.]+)', lower)
    if tm:
        try:
            meta['temperature'] = float(tm.group(1))
        except ValueError:
            pass
    return meta


def _normalize_model_name(name: str) -> str:
    return re.sub(r'(\d+)([bB])\b', lambda m: m.group(1) + 'B', name).strip()


def _model_id(name: str, org: str = "") -> str:
    if "/" in name:
        return name
    return f"{_infer_developer(name, org)}/{name}"


# ---------------------------------------------------------------------------
# FetchedRow
# ---------------------------------------------------------------------------

@dataclass
class FetchedRow:
    model_name:       str
    organization:     str                            = ""
    scores:           dict[str, float]               = field(default_factory=dict)
    shots:            dict[str, int | None]          = field(default_factory=dict)
    metric_names:     dict[str, str]                 = field(default_factory=dict)
    scales:           dict[str, tuple[float, float]] = field(default_factory=dict)
    details:          dict[str, dict]                = field(default_factory=dict)
    notes:            dict[str, str]                 = field(default_factory=dict)
    chain_of_thought: dict[str, bool]                = field(default_factory=dict)
    temperature:      dict[str, float | None]        = field(default_factory=dict)
    eval_date:        str                            = ""


# ---------------------------------------------------------------------------
# Base fetcher
# ---------------------------------------------------------------------------

class LiveLeaderboardFetcher(ABC):
    SOURCE_DIR:  str = ""
    SOURCE_NAME: str = ""
    SOURCE_ORG:  str = ""
    SOURCE_URL:  str = ""
    EVAL_LIB:    str = "unknown"
    CACHE_KEY:   str = ""

    def fetch(self, force_refresh: bool = False) -> list[FetchedRow]:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = CACHE_DIR / f"{self.CACHE_KEY}.json"
        if not force_refresh and cache_file.exists():
            age = time.time() - cache_file.stat().st_mtime
            if age < CACHE_TTL:
                try:
                    rows = [FetchedRow(**r) for r in json.loads(cache_file.read_text())]
                    print(f"  [{self.CACHE_KEY}] {len(rows)} rows from cache "
                          f"({age / 3600:.1f}h old)")
                    return rows
                except Exception:
                    pass
        print(f"  [{self.CACHE_KEY}] fetching ...")
        try:
            rows = self._fetch_rows()
        except Exception as exc:
            print(f"  [{self.CACHE_KEY}] FAILED: {exc}", file=sys.stderr)
            return []
        try:
            cache_file.write_text(json.dumps([r.__dict__ for r in rows], indent=2))
        except Exception:
            pass
        print(f"  [{self.CACHE_KEY}] {len(rows)} rows fetched")
        return rows

    def write_rows(self, rows: list[FetchedRow]) -> int:
        n = 0
        for row in rows:
            mn  = _normalize_model_name(row.model_name)
            mid = _model_id(mn, row.organization)
            dev = _infer_developer(mn, row.organization)
            results = []
            for bench, score in row.scores.items():
                r_dict: dict = {
                    "bench":       bench,
                    "score":       score,
                    "min_score":   row.scales.get(bench, (0.0, 100.0))[0],
                    "max_score":   row.scales.get(bench, (0.0, 100.0))[1],
                    "shots":       row.shots.get(bench),
                    "metric_name": row.metric_names.get(bench, "accuracy"),
                }
                if bench in row.details:
                    r_dict["details"] = row.details[bench]
                if bench in row.notes:
                    r_dict["note"] = row.notes[bench]
                if bench in row.chain_of_thought:
                    r_dict["chain_of_thought"] = row.chain_of_thought[bench]
                if bench in row.temperature:
                    r_dict["temperature"] = row.temperature[bench]
                # Also extract metadata from bench name & model name
                bench_meta = _extract_meta_from_text(bench)
                name_meta = _extract_meta_from_text(row.model_name)
                # bench-level meta takes precedence, then model-level, then explicit
                if r_dict.get("shots") is None and "shots" in bench_meta:
                    r_dict["shots"] = bench_meta["shots"]
                if r_dict.get("shots") is None and "shots" in name_meta:
                    r_dict["shots"] = name_meta["shots"]
                if r_dict.get("chain_of_thought") is None and "chain_of_thought" in bench_meta:
                    r_dict["chain_of_thought"] = bench_meta["chain_of_thought"]
                if r_dict.get("chain_of_thought") is None and "chain_of_thought" in name_meta:
                    r_dict["chain_of_thought"] = name_meta["chain_of_thought"]
                if r_dict.get("temperature") is None and "temperature" in bench_meta:
                    r_dict["temperature"] = bench_meta["temperature"]
                if r_dict.get("temperature") is None and "temperature" in name_meta:
                    r_dict["temperature"] = name_meta["temperature"]
                results.append(r_dict)
            if results:
                n += _add(self.SOURCE_DIR, self.SOURCE_NAME, self.SOURCE_ORG,
                          self.SOURCE_URL, self.EVAL_LIB, mid, mn, dev, results,
                          eval_date=row.eval_date)
        return n

    def probe_columns(self) -> None:
        """Print actual column names from the live source — use before running
        for real to verify the column map matches the current schema."""
        print(f"\n[{self.CACHE_KEY}] probing columns ...")
        try:
            rows = self._fetch_rows()
            if rows:
                print(f"  sample row keys: {list(rows[0].scores.keys())[:10]}")
                print(f"  first model: {rows[0].model_name}")
                print(f"  total rows: {len(rows)}")
            else:
                print("  no rows returned")
        except Exception as exc:
            print(f"  probe failed: {exc}")

    @abstractmethod
    def _fetch_rows(self) -> list[FetchedRow]: ...

    # HTTP helpers
    @staticmethod
    def _get(url: str, token: str = "") -> requests.Response:
        headers = {"User-Agent": "EEE-pipeline/1.0"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        r = requests.get(url, timeout=_TIMEOUT, headers=headers)
        r.raise_for_status()
        return r

    @classmethod
    def _get_json(cls, url: str, token: str = "") -> Any:
        return cls._get(url, token).json()

    @classmethod
    def _get_csv_rows(cls, url: str, token: str = "") -> list[dict]:
        return list(csv.DictReader(io.StringIO(cls._get(url, token).text)))

    @staticmethod
    def _pct(val: Any) -> float:
        """Return score in % range regardless of whether source used 0-1 or 0-100."""
        v = float(val)
        return round(v * 100 if v <= 1.0 else v, 2)


# ---------------------------------------------------------------------------
# HuggingFace parquet helper
# Uses huggingface_hub to download the actual parquet files — more reliable
# than the datasets-server rows API, which has rate limits and column-name
# drift as the leaderboard schema evolves.
# ---------------------------------------------------------------------------

def _load_hf_dataset_as_df(
    dataset_id: str,
    config: str = "default",
    split: str = "train",
    token: str = "",
) -> "pd.DataFrame":
    """Download a HuggingFace dataset as a pandas DataFrame.

    Uses huggingface_hub.snapshot_download to fetch the parquet shard(s)
    for the requested split.  Falls back to the datasets-server rows API
    if huggingface_hub is not installed or the download fails.

    Parameters
    ----------
    dataset_id:  e.g. "open-llm-leaderboard/contents"
    config:      dataset configuration name (usually "default")
    split:       "train", "test", etc.
    token:       HuggingFace API token (required for gated datasets)
    """
    # Temporarily ensure HF_HUB_OFFLINE is not blocking downloads.
    # Must set env var BEFORE importing huggingface_hub because the library
    # caches HF_HUB_OFFLINE in huggingface_hub.constants at import time.
    prev_offline = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "0"

    try:
        import pandas as pd
        import huggingface_hub.constants
        from huggingface_hub import snapshot_download

        # Also patch the cached constant in case the module was already imported
        huggingface_hub.constants.HF_HUB_OFFLINE = False
    except ImportError:
        raise RuntimeError(
            "huggingface_hub and pandas are required for HF leaderboard fetching.\n"
            "Install with: pip install huggingface_hub pandas pyarrow"
        )

    # Download only the parquet files for this split
    local_dir = pathlib.Path(
        f".cache/hf_datasets/{dataset_id.replace('/', '__')}/{config}/{split}"
    )
    local_dir.mkdir(parents=True, exist_ok=True)

    kwargs: dict[str, Any] = {
        "repo_id":   dataset_id,
        "repo_type": "dataset",
        "local_dir": str(local_dir),
        # Download only parquet files for this split; ignore everything else
        "allow_patterns": [f"data/{split}*.parquet", f"{split}*.parquet", "*.parquet"],
        "ignore_patterns": ["*.json", "*.md", "*.py", "*.txt", "*.yaml"],
    }
    if token:
        kwargs["token"] = token

    try:
        snapshot_download(**kwargs)
    finally:
        if prev_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prev_offline
        # Restore the cached constant too
        if prev_offline and prev_offline.strip() == "1":
            huggingface_hub.constants.HF_HUB_OFFLINE = True

    # Find all downloaded parquet files and concatenate
    parquet_files = list(local_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found after download of {dataset_id}. "
            f"Check that the dataset exists, the split name is correct, "
            f"and that your HF_TOKEN has access."
        )

    frames = [pd.read_parquet(str(f)) for f in sorted(parquet_files)]
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    return df


# ===========================================================================
# TIER 1 — Direct benchmark overlap with paper sources
# ===========================================================================

class OpenLLMLeaderboardV1Fetcher(LiveLeaderboardFetcher):
    """
    Open LLM Leaderboard v1 (archived 2024-06).
    Dataset: open-llm-leaderboard/results
    Benchmarks: ARC-Challenge (25-shot), HellaSwag (10-shot), MMLU (5-shot),
                TruthfulQA (0-shot MC1), WinoGrande (5-shot), GSM8K (5-shot)
    Harness: EleutherAI lm-evaluation-harness 0.3.0

    Column schema (verified from the archived dataset parquet):
    ┌─────────────────────────────────────────────────────┬──────────┐
    │ column                                              │ type     │
    ├─────────────────────────────────────────────────────┼──────────┤
    │ model_name_or_path                                  │ string   │
    │ results.harness|arc:challenge|25.acc_norm            │ float    │
    │ results.harness|hellaswag|10.acc_norm                │ float    │
    │ results.harness|truthfulqa:mc1|0.mc1                 │ float    │
    │ results.harness|winogrande|5.acc                     │ float    │
    │ results.harness|gsm8k|5.acc                          │ float    │
    │ results.harness|hendrycksTest-{subject}|5.acc_norm   │ float×57 │
    └─────────────────────────────────────────────────────┴──────────┘
    All score columns store values in 0–1; we multiply to %.

    Requires HF_TOKEN — the dataset is gated (agree to terms on HF Hub first).
    """
    SOURCE_DIR  = "open_llm_leaderboard_v1"
    SOURCE_NAME = "Open LLM Leaderboard v1"
    SOURCE_ORG  = "HuggingFace"
    SOURCE_URL  = "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard"
    EVAL_LIB    = "lm_eval"
    CACHE_KEY   = "open_llm_v1"

    # Archived dataset ID — confirmed from HF Hub as of 2024-12
    _DATASET_ID = "open-llm-leaderboard/results"

    # Exact column names in the parquet (dot-separated nested structs)
    _DIRECT_COLS: dict[str, tuple[str, int, str]] = {
        "results.harness|arc:challenge|25.acc_norm":  ("ARC-Challenge", 25, "accuracy"),
        "results.harness|hellaswag|10.acc_norm":       ("HellaSwag",     10, "accuracy"),
        "results.harness|truthfulqa:mc1|0.mc1":        ("TruthfulQA",     0, "mc1"),
        "results.harness|winogrande|5.acc":            ("WinoGrande",     5, "accuracy"),
        "results.harness|gsm8k|5.acc":                 ("GSM8K",          5, "accuracy"),
    }
    _MODEL_COL   = "model_name_or_path"
    _MMLU_PREFIX = "results.harness|hendrycksTest-"
    _MMLU_SUFFIX = "|5.acc_norm"

    def _fetch_rows(self) -> list[FetchedRow]:
        # The open-llm-leaderboard/results dataset no longer contains
        # v1-format parquet data. It consists of individual JSON files per model
        # using v2-era benchmark keys (leaderboard_bbh, etc.)  and cannot be
        # loaded via _load_hf_dataset_as_df (no parquet files).
        # Use OpenLLMLeaderboardV2Fetcher instead for current data.
        raise RuntimeError(
            "Open LLM Leaderboard v1 dataset (open-llm-leaderboard/results) "
            "no longer provides v1-format parquet data.  The dataset now uses "
            "v2-era benchmark keys and individual JSON files per model.  "
            "Use open_llm_leaderboard_v2 instead."
        )

        df = _load_hf_dataset_as_df(
            self._DATASET_ID, config="default", split="train",
            token=_HF_TOKEN,
        )

        # Diagnostic: show actual columns so the user can verify the map
        print(f"  [{self.CACHE_KEY}] columns in dataset ({len(df.columns)} total):")
        sample_cols = [c for c in df.columns if "arc" in c.lower() or
                       "hellaswag" in c.lower() or "mmlu" in c.lower() or
                       c == self._MODEL_COL][:12]
        for c in sample_cols:
            print(f"    {c}")

        rows: list[FetchedRow] = []
        for _, row in df.iterrows():
            model_name = str(row.get(self._MODEL_COL, "")).strip()
            if not model_name:
                continue
            org     = model_name.split("/")[0] if "/" in model_name else ""
            display = model_name.split("/")[-1]
            fr      = FetchedRow(model_name=display, organization=org)

            for col, (bench, shots, metric) in self._DIRECT_COLS.items():
                val = row.get(col)
                if val is not None and str(val) not in ("", "nan", "None"):
                    try:
                        fr.scores[bench]       = self._pct(val)
                        fr.shots[bench]        = shots
                        fr.metric_names[bench] = metric
                        fr.scales[bench]       = (0.0, 100.0)
                    except (TypeError, ValueError):
                        pass

            # MMLU: average across all 57 hendrycksTest subjects
            mmlu_cols = [c for c in df.columns
                         if c.startswith(self._MMLU_PREFIX) and c.endswith(self._MMLU_SUFFIX)]
            mmlu_vals = []
            for c in mmlu_cols:
                v = row.get(c)
                if v is not None and str(v) not in ("", "nan", "None"):
                    try:
                        mmlu_vals.append(float(v))
                    except (TypeError, ValueError):
                        pass
            if mmlu_vals:
                fr.scores["MMLU"]       = round(sum(mmlu_vals) / len(mmlu_vals) * 100, 2)
                fr.shots["MMLU"]        = 5
                fr.metric_names["MMLU"] = "accuracy"
                fr.scales["MMLU"]       = (0.0, 100.0)

            if fr.scores:
                rows.append(fr)

        return rows


class OpenLLMLeaderboardV2Fetcher(LiveLeaderboardFetcher):
    """
    Open LLM Leaderboard v2 (active as of 2025).
    Dataset: open-llm-leaderboard/contents
    Benchmarks: IFEval (0-shot), BBH (3-shot CoT), MATH Lvl 5 (4-shot CoT),
                GPQA (0-shot CoT), MuSR (0-shot CoT), MMLU-PRO (5-shot CoT)
    Harness: EleutherAI lm-evaluation-harness 0.4.x

    Column schema (verified from the live dataset parquet):
    ┌───────────────────┬────────┬───────────────────────────────┐
    │ column            │ type   │ notes                         │
    ├───────────────────┼────────┼───────────────────────────────┤
    │ model             │ string │ full HF model ID              │
    │ model_type        │ string │ "pretrained", "instruct", etc │
    │ architecture      │ string │                               │
    │ average           │ float  │ macro-average of 6 tasks      │
    │ IFEval            │ float  │ 0–100 already                 │
    │ BBH               │ float  │ 0–100 already                 │
    │ MATH Lvl 5        │ float  │ 0–100 already                 │
    │ GPQA              │ float  │ 0–100 already                 │
    │ MuSR              │ float  │ 0–100 already                 │
    │ MMLU-PRO          │ float  │ 0–100 already                 │
    │ hub_hearts        │ int    │ HF model card likes           │
    │ hub_license       │ string │                               │
    │ params_billions   │ float  │ parameter count in billions   │
    └───────────────────┴────────┴───────────────────────────────┘
    Scores are already in 0–100 (unlike v1 which used 0–1).

    Requires HF_TOKEN for access.
    """
    SOURCE_DIR  = "open_llm_leaderboard_v2"
    SOURCE_NAME = "Open LLM Leaderboard v2"
    SOURCE_ORG  = "HuggingFace"
    SOURCE_URL  = "https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard"
    EVAL_LIB    = "lm_eval"
    CACHE_KEY   = "open_llm_v2"

    _DATASET_ID = "open-llm-leaderboard/contents"
    _MODEL_COL  = "fullname"  # "fullname" has org/model; "Model" is display-only

    # Exact column names → (bench_name, shots, metric, note, cot)
    _BENCH_COLS: dict[str, tuple[str, int, str, str, bool]] = {
        "IFEval":     ("IFEval",   0, "prompt_level_strict_acc", "",                         False),
        "BBH":        ("BBH",      3, "accuracy",                "3-shot CoT",               True),
        "MATH Lvl 5": ("MATH-500", 4, "accuracy",                "4-shot CoT, level 5 only", True),
        "GPQA":       ("GPQA",     0, "accuracy",                "0-shot CoT",               True),
        "MUSR":       ("MuSR",     0, "accuracy",                "0-shot CoT",               True),
        "MMLU-PRO":   ("MMLU-Pro", 5, "accuracy",                "5-shot CoT",               True),
    }

    def _fetch_rows(self) -> list[FetchedRow]:
        if not _HF_TOKEN:
            raise RuntimeError(
                "HF_TOKEN is not set.  The Open LLM Leaderboard v2 dataset is "
                "gated.  Accept terms at "
                "https://huggingface.co/datasets/open-llm-leaderboard/contents "
                "and set:  export HF_TOKEN=hf_..."
            )

        df = _load_hf_dataset_as_df(
            self._DATASET_ID, config="default", split="train",
            token=_HF_TOKEN,
        )

        # Diagnostic
        missing = [c for c in self._BENCH_COLS if c not in df.columns]
        if missing:
            print(
                f"  [{self.CACHE_KEY}] WARNING — expected columns not found: {missing}\n"
                f"  Actual columns: {list(df.columns)}\n"
                f"  The leaderboard schema may have changed.  Update _BENCH_COLS.",
                file=sys.stderr,
            )

        rows: list[FetchedRow] = []
        for _, row in df.iterrows():
            model_name = str(row.get(self._MODEL_COL, "")).strip()
            if not model_name:
                continue
            org     = model_name.split("/")[0] if "/" in model_name else ""
            display = model_name.split("/")[-1]
            fr      = FetchedRow(model_name=display, organization=org)

            for col, (bench, shots, metric, note, cot) in self._BENCH_COLS.items():
                val = row.get(col)
                if val is not None and str(val) not in ("", "nan", "None"):
                    try:
                        # v2 scores are already 0–100
                        score = float(val)
                        fr.scores[bench]       = round(score, 2)
                        fr.shots[bench]        = shots
                        fr.metric_names[bench] = metric
                        fr.scales[bench]       = (0.0, 100.0)
                        if note:
                            fr.notes[bench]    = note
                        if cot:
                            fr.chain_of_thought[bench] = True
                    except (TypeError, ValueError):
                        pass

            if fr.scores:
                rows.append(fr)

        return rows


# ===========================================================================
# TIER 2 — Coding / agentic (no paper overlap, distinct signal)
# ===========================================================================

class EvalPlusFetcher(LiveLeaderboardFetcher):
    """
    EvalPlus — HumanEval+ and MBPP+ pass@1 (0–100 %).
    Source: evalplus/evalplus GitHub JSON.

    Column schema:
      [ { "model": "...", "humaneval+": 88.4, "mbpp+": 84.5, ... }, ... ]
    or
      { "models": [ { "name": "...", "pass@1": { "humaneval+": 88.4, ... } } ] }
    The parser handles both layouts.
    """
    SOURCE_DIR  = "evalplus"
    SOURCE_NAME = "EvalPlus Leaderboard"
    SOURCE_ORG  = "EvalPlus"
    SOURCE_URL  = "https://evalplus.github.io/leaderboard.html"
    EVAL_LIB    = "evalplus"
    CACHE_KEY   = "evalplus"

    _URLS = [
        "https://raw.githubusercontent.com/evalplus/evalplus.github.io/main/results.json",
        "https://raw.githubusercontent.com/evalplus/evalplus/master/evalplus/leaderboard.json",
        "https://evalplus.github.io/leaderboard.json",
    ]

    def _fetch_rows(self) -> list[FetchedRow]:
        last: Exception | None = None
        for url in self._URLS:
            try:
                return self._parse(self._get_json(url))
            except Exception as exc:
                last = exc
        raise RuntimeError(f"EvalPlus: all URLs failed. Last: {last}")

    def _parse(self, data: Any) -> list[FetchedRow]:
        rows: list[FetchedRow] = []

        # New format from results.json: {model_name: {pass@1: {humaneval+: x, mbpp+: y}}}
        if isinstance(data, dict) and all(
            isinstance(v, dict) and "pass@1" in v for v in list(data.values())[:3]
        ):
            for model, info in data.items():
                nested = info.get("pass@1", {})
                he   = nested.get("humaneval+")
                mbpp = nested.get("mbpp+")
                if he is None and mbpp is None:
                    continue
                fr = FetchedRow(model_name=str(model).split("/")[-1],
                                organization=str(model).split("/")[0] if "/" in str(model) else "")
                for bench, val, shots in [("HumanEval+", he, 0), ("MBPP+", mbpp, 3)]:
                    if val is not None:
                        try:
                            fr.scores[bench]       = self._pct(val)
                            fr.shots[bench]        = shots
                            fr.metric_names[bench] = "pass@1"
                            fr.scales[bench]       = (0.0, 100.0)
                        except (TypeError, ValueError):
                            pass
                if fr.scores:
                    rows.append(fr)
            return rows

        # Old format: list or {models: [...]}
        entries = data if isinstance(data, list) else data.get("models", data.get("results", []))
        for entry in entries:
            # Flat format: {"model": "...", "humaneval+": 88.4, "mbpp+": 84.5}
            model = entry.get("model") or entry.get("name", "")
            if not model:
                continue
            # Nested format: {"pass@1": {"humaneval+": ..., "mbpp+": ...}}
            nested = entry.get("pass@1", {})
            he   = nested.get("humaneval+") or entry.get("humaneval+") or entry.get("HumanEval+")
            mbpp = nested.get("mbpp+")      or entry.get("mbpp+")      or entry.get("MBPP+")
            if he is None and mbpp is None:
                continue
            fr = FetchedRow(model_name=str(model).split("/")[-1],
                            organization=str(model).split("/")[0] if "/" in str(model) else "")
            for bench, val, shots in [("HumanEval+", he, 0), ("MBPP+", mbpp, 3)]:
                if val is not None:
                    try:
                        fr.scores[bench]       = self._pct(val)
                        fr.shots[bench]        = shots
                        fr.metric_names[bench] = "pass@1"
                        fr.scales[bench]       = (0.0, 100.0)
                    except (TypeError, ValueError):
                        pass
            if fr.scores:
                rows.append(fr)
        return rows


class BFCLFetcher(LiveLeaderboardFetcher):
    """
    Berkeley Function Calling Leaderboard v3.
    Source: gorilla.cs.berkeley.edu/data_overall.csv (live leaderboard CSV).

    Column schema (data_overall.csv):
      Rank, Overall Acc, Model, Model Link, Total Cost ($), Latency Mean (s),
      Latency Standard Deviation (s), Latency 95th Percentile (s),
      Non-Live AST Acc, ... (36 columns total)
    """
    SOURCE_DIR  = "bfcl"
    SOURCE_NAME = "BFCL v3"
    SOURCE_ORG  = "UC Berkeley"
    SOURCE_URL  = "https://gorilla.cs.berkeley.edu/leaderboard.html"
    EVAL_LIB    = "gorilla_eval"
    CACHE_KEY   = "bfcl"

    _OVERALL_CSV = "https://gorilla.cs.berkeley.edu/data_overall.csv"
    _LIVE_CSV    = "https://gorilla.cs.berkeley.edu/data_live.csv"

    def _fetch_rows(self) -> list[FetchedRow]:
        try:
            records = self._get_csv_rows(self._OVERALL_CSV)
        except Exception as exc:
            raise RuntimeError(f"BFCL: failed to fetch data_overall.csv: {exc}")

        # Also try to get live scores
        live_by_model: dict[str, dict] = {}
        try:
            live_records = self._get_csv_rows(self._LIVE_CSV)
            for rec in live_records:
                model = (rec.get("Model") or "").strip()
                if model:
                    live_by_model[model] = rec
        except Exception:
            pass

        rows: list[FetchedRow] = []
        for rec in records:
            model = (rec.get("Model") or "").strip()
            if not model:
                continue
            overall_raw = (rec.get("Overall Acc") or "").strip().rstrip("%")
            try:
                overall = float(overall_raw)
            except (TypeError, ValueError):
                continue

            fr = FetchedRow(model_name=str(model))

            # Overall accuracy
            fr.scores["BFCL-v3"]       = round(overall, 2)
            fr.shots["BFCL-v3"]        = 0
            fr.metric_names["BFCL-v3"] = "accuracy"
            fr.scales["BFCL-v3"]       = (0.0, 100.0)

            # Non-Live AST accuracy
            ast_raw = (rec.get("Non-Live AST Acc") or "").strip().rstrip("%")
            try:
                ast_val = float(ast_raw)
                fr.scores["BFCL-v3 AST"]       = round(ast_val, 2)
                fr.shots["BFCL-v3 AST"]        = 0
                fr.metric_names["BFCL-v3 AST"] = "accuracy"
                fr.scales["BFCL-v3 AST"]       = (0.0, 100.0)
            except (TypeError, ValueError):
                pass

            # Live summary (from live CSV if available)
            live_rec = live_by_model.get(model)
            if live_rec:
                live_raw = (live_rec.get("Live Overall Acc") or "").strip().rstrip("%")
                try:
                    live_val = float(live_raw)
                    fr.scores["BFCL-v3 Live"]       = round(live_val, 2)
                    fr.shots["BFCL-v3 Live"]        = 0
                    fr.metric_names["BFCL-v3 Live"] = "accuracy"
                    fr.scales["BFCL-v3 Live"]       = (0.0, 100.0)
                except (TypeError, ValueError):
                    pass

            # Add cost/latency details
            details: dict[str, str] = {}
            for key in ["Total Cost ($)", "Latency Mean (s)"]:
                val = (rec.get(key) or "").strip()
                if val:
                    details[key] = val
            if details:
                fr.details["BFCL-v3"] = details

            if fr.scores:
                rows.append(fr)
        return rows


class SWEBenchFetcher(LiveLeaderboardFetcher):
    """
    SWE-bench Verified — resolved % on real GitHub issues (agentic).
    Source: swe-bench/experiments GitHub JSON.

    Column schema:
      { "model_name": { "resolved": N, "total": N, "unresolved": N }, ... }
    """
    SOURCE_DIR  = "swe_bench"
    SOURCE_NAME = "SWE-bench Verified"
    SOURCE_ORG  = "Princeton NLP"
    SOURCE_URL  = "https://www.swebench.com/"
    EVAL_LIB    = "swebench"
    CACHE_KEY   = "swe_bench"

    _TOTAL_VERIFIED = 500  # SWE-bench Verified has exactly 500 instances
    _GITHUB_API = "https://api.github.com/repos/SWE-bench/experiments/contents/evaluation/verified"
    _RAW_BASE = "https://raw.githubusercontent.com/SWE-bench/experiments/main/evaluation/verified"
    _NOTE = "agentic scaffolding; not directly comparable to non-agentic benchmarks"

    def _fetch_rows(self) -> list[FetchedRow]:
        # List submission directories via GitHub API
        try:
            headers = {"User-Agent": "EEE-pipeline/1.0", "Accept": "application/vnd.github.v3+json"}
            r = requests.get(self._GITHUB_API, timeout=_TIMEOUT, headers=headers)
            r.raise_for_status()
            dirs = [item["name"] for item in r.json() if item.get("type") == "dir"]
        except Exception as exc:
            raise RuntimeError(f"SWE-bench: could not list submissions: {exc}")

        rows: list[FetchedRow] = []
        errors = 0
        for dirname in dirs:
            url = f"{self._RAW_BASE}/{dirname}/results/results.json"
            try:
                data = self._get_json(url)
                resolved = data.get("resolved", [])
                total = self._TOTAL_VERIFIED
                pct = round(len(resolved) / total * 100, 2)
                # Extract model/agent info from directory name
                # Format: YYYYMMDD_agent_name  or  YYYYMMDD_agent_model
                parts = dirname.split("_", 1)
                agent_name = parts[1] if len(parts) > 1 else dirname
                # Extract evaluation date from YYYYMMDD prefix
                eval_date = ""
                if len(parts[0]) == 8 and parts[0].isdigit():
                    d = parts[0]
                    eval_date = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
                fr = FetchedRow(model_name=agent_name, eval_date=eval_date)
                fr.scores["SWE-Bench-Verified"]       = pct
                fr.shots["SWE-Bench-Verified"]        = 0
                fr.metric_names["SWE-Bench-Verified"] = "resolved_pct"
                fr.scales["SWE-Bench-Verified"]       = (0.0, 100.0)
                fr.notes["SWE-Bench-Verified"]        = self._NOTE
                fr.details["SWE-Bench-Verified"]      = {
                    "resolved": str(len(resolved)),
                    "total": str(total),
                    "submission": dirname,
                }
                rows.append(fr)
            except Exception:
                errors += 1
                continue

        if not rows:
            raise RuntimeError(f"SWE-bench: no results fetched ({errors} errors)")
        print(f"  [swe_bench] parsed {len(rows)} submissions ({errors} skipped)")
        return rows


# ===========================================================================
# EXISTING SOURCES — normalisation fixed
# ===========================================================================

class AlpacaEval2Fetcher(LiveLeaderboardFetcher):
    """
    AlpacaEval 2.0 — LC Win Rate and Win Rate (raw %, 0–100).
    Source: tatsu-lab/alpaca_eval GitHub leaderboard.csv

    Column schema (leaderboard.csv):
      model, win_rate, std_error, avg_length, mode
    where mode ∈ {weighted_alpaca_eval_gpt4_turbo → LC, alpaca_eval_gpt4 → WR}
    win_rate is already 0–100.

    FIXED: original code divided by 100 — removed.
    """
    SOURCE_DIR  = "alpacaeval2"
    SOURCE_NAME = "AlpacaEval 2.0"
    SOURCE_ORG  = "Stanford"
    SOURCE_URL  = "https://tatsu-lab.github.io/alpaca_eval/"
    EVAL_LIB    = "alpaca_eval"
    CACHE_KEY   = "alpacaeval2"

    _CSV_URLS = [
        # Primary: the weighted LC win-rate leaderboard CSV (has LC + raw win rate)
        "https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/"
        "main/src/alpaca_eval/leaderboards/data_AlpacaEval_2/"
        "weighted_alpaca_eval_gpt4_turbo_leaderboard.csv",
        # Fallback: the old top-level CSV (may be removed)
        "https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/"
        "main/leaderboard.csv",
    ]

    def _fetch_rows(self) -> list[FetchedRow]:
        last: Exception | None = None
        for url in self._CSV_URLS:
            try:
                rows = self._parse_csv(self._get_csv_rows(url))
                if rows:
                    return rows
            except Exception as exc:
                last = exc
        raise RuntimeError(f"AlpacaEval2: all URLs failed. Last: {last}")

    def _parse_csv(self, records: list[dict]) -> list[FetchedRow]:
        by_model: dict[str, FetchedRow] = {}
        for rec in records:
            # New format: unnamed first column is model name (key "")
            model = (rec.get("model") or rec.get("") or
                     rec.get("Unnamed: 0", "")).strip()
            if not model:
                continue
            fr = by_model.setdefault(model, FetchedRow(model_name=model))

            # Length-controlled win rate
            lc_wr = rec.get("length_controlled_winrate") or rec.get("lc_win_rate")
            if lc_wr is not None:
                try:
                    fr.scores["AlpacaEval 2.0 LC"] = round(float(lc_wr), 2)
                    fr.shots["AlpacaEval 2.0 LC"] = 0
                    fr.metric_names["AlpacaEval 2.0 LC"] = "lc_win_rate"
                    fr.scales["AlpacaEval 2.0 LC"] = (0.0, 100.0)
                except (TypeError, ValueError):
                    pass

            # Raw win rate
            wr = rec.get("win_rate")
            if wr is not None:
                try:
                    score = float(wr)
                    fr.scores["AlpacaEval 2.0"] = round(score, 2)
                    fr.shots["AlpacaEval 2.0"] = 0
                    fr.metric_names["AlpacaEval 2.0"] = "win_rate"
                    fr.scales["AlpacaEval 2.0"] = (0.0, 100.0)
                except (TypeError, ValueError):
                    pass

            # Old format: mode-based disambiguation
            if "mode" in rec and "AlpacaEval 2.0 LC" not in fr.scores:
                mode = rec.get("mode", "").strip().lower()
                if wr is not None:
                    try:
                        score = float(wr)
                        bench = "AlpacaEval 2.0 LC" if ("weighted" in mode or "lc" in mode) else "AlpacaEval 2.0"
                        fr.scores[bench] = round(score, 2)
                        fr.shots[bench] = 0
                        fr.metric_names[bench] = "lc_win_rate" if "LC" in bench else "win_rate"
                        fr.scales[bench] = (0.0, 100.0)
                    except (TypeError, ValueError):
                        pass

        return list(by_model.values())


class ChatbotArenaFetcher(LiveLeaderboardFetcher):
    """
    LMSYS Chatbot Arena — Elo rating (raw, scale 0–3000, practical ~800–1400).
    Source: mathewhe/chatbot-arena-elo HuggingFace dataset (elo.csv).

    Column schema (elo.csv):
      Rank* (UB), Rank (StyleCtrl), Model Markup, Model, Arena Score,
      95% CI, Votes, Organization, License, Knowledge Cutoff
    """
    SOURCE_DIR  = "chatbot_arena"
    SOURCE_NAME = "LMSYS Chatbot Arena"
    SOURCE_ORG  = "LMSYS"
    SOURCE_URL  = "https://lmarena.ai/"
    EVAL_LIB    = "fastchat"
    CACHE_KEY   = "chatbot_arena"

    _CSV_URL = (
        "https://huggingface.co/datasets/mathewhe/chatbot-arena-elo/"
        "resolve/main/elo.csv"
    )

    def _fetch_rows(self) -> list[FetchedRow]:
        records = self._get_csv_rows(self._CSV_URL)
        rows: list[FetchedRow] = []
        for rec in records:
            model = (rec.get("Model") or "").strip()
            if not model:
                continue
            elo_raw = (rec.get("Arena Score") or "").strip()
            try:
                elo = float(elo_raw)
            except (TypeError, ValueError):
                continue
            org = (rec.get("Organization") or "").strip()
            fr = FetchedRow(model_name=model, organization=org)

            details: dict[str, Any] = {}
            ci = (rec.get("95% CI") or "").strip()
            m = re.match(r'^\+?(\d+\.?\d*)[/\-](\d+\.?\d*)$', ci)
            if m:
                details["ci_upper"] = str(round(elo + float(m.group(1)), 1))
                details["ci_lower"] = str(round(elo - float(m.group(2)), 1))
            else:
                pm = re.match(r'^[±]?(\d+\.?\d*)$', ci)
                if pm:
                    d = float(pm.group(1))
                    details["ci_upper"] = str(round(elo + d, 1))
                    details["ci_lower"] = str(round(elo - d, 1))
            votes = (rec.get("Votes") or "0").replace(",", "")
            try:
                details["num_battles"] = str(int(float(votes)))
            except (TypeError, ValueError):
                pass

            bench = "Arena Elo"
            fr.scores[bench]       = elo
            fr.shots[bench]        = None
            fr.metric_names[bench] = "elo"
            fr.scales[bench]       = (0.0, 3000.0)
            fr.notes[bench]        = "raw Elo; practical range ~800–1400"
            if details:
                fr.details[bench]  = details
            rows.append(fr)
        return rows


class MTBenchFetcher(LiveLeaderboardFetcher):
    """
    MT-Bench — GPT-4 judge score (raw 0–10 scale).
    Source: OfirArviv/mt_bench_single_score_gpt4_judgement HF dataset CSV.

    The CSV contains per-question per-turn GPT-4 judgments. We compute
    per-model averages across all questions and turns.

    FIXED: original FastChat CSV no longer available.  Using HF mirror.
    """
    SOURCE_DIR  = "mt_bench"
    SOURCE_NAME = "MT-Bench"
    SOURCE_ORG  = "LMSYS"
    SOURCE_URL  = "https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge"
    EVAL_LIB    = "fastchat"
    CACHE_KEY   = "mt_bench"

    _CSV_URL = (
        "https://huggingface.co/datasets/OfirArviv/"
        "mt_bench_single_score_gpt4_judgement/resolve/main/"
        "mt_bench_single_score_gpt4_judgement.csv"
    )

    _CATEGORIES = [
        "writing", "roleplay", "reasoning", "math",
        "coding", "extraction", "stem", "humanities",
    ]

    def _fetch_rows(self) -> list[FetchedRow]:
        import pandas as pd
        try:
            records = self._get_csv_rows(self._CSV_URL)
        except Exception as exc:
            raise RuntimeError(f"MT-Bench: failed to fetch CSV: {exc}")

        df = pd.DataFrame(records)
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
        df = df.dropna(subset=["score", "model_id"])

        # Compute per-model overall average
        model_avgs = df.groupby("model_id")["score"].mean()
        # Compute per-model per-category averages
        cat_avgs = df.groupby(["model_id", "category"])["score"].mean().unstack(fill_value=float("nan"))

        rows: list[FetchedRow] = []
        for model_id, avg_score in model_avgs.items():
            fr = FetchedRow(model_name=str(model_id))
            fr.scores["MT-Bench"]       = round(float(avg_score), 2)
            fr.shots["MT-Bench"]        = None
            fr.metric_names["MT-Bench"] = "judge_score"
            fr.scales["MT-Bench"]       = (0.0, 10.0)
            fr.notes["MT-Bench"]        = "GPT-4 judge score averaged over 8 categories"
            # Add per-category detail
            if model_id in cat_avgs.index:
                cat_details: dict[str, str] = {}
                for cat in self._CATEGORIES:
                    if cat in cat_avgs.columns:
                        v = cat_avgs.loc[model_id, cat]
                        if pd.notna(v):
                            cat_details[cat] = str(round(float(v), 2))
                if cat_details:
                    fr.details["MT-Bench"] = cat_details
            rows.append(fr)
        return rows


class WildBenchFetcher(LiveLeaderboardFetcher):
    """
    WildBench v2 — WB Score (raw -100 to 100).
    Source: allenai/WildBench HF dataset JSON or Space JSON.

    Column schema:
      [ { "model_id": "...", "wb_score": 56.4 }, ... ]
    or
      { "rows": [ { "row": { "model_id": "...", "wb_score": 56.4 } } ] }

    FIXED: original code applied (score+100)/200 — removed.  Raw score stored.
    """
    SOURCE_DIR  = "wildbench"
    SOURCE_NAME = "WildBench v2"
    SOURCE_ORG  = "AI2"
    SOURCE_URL  = "https://huggingface.co/spaces/allenai/WildBench"
    EVAL_LIB    = "wildeval"
    CACHE_KEY   = "wildbench"

    _URLS = [
        "https://raw.githubusercontent.com/allenai/WildBench/main/leaderboard/data_dir/score.json",
        "https://huggingface.co/datasets/allenai/WildBench-v2-dev/resolve/main/leaderboard.json",
        "https://datasets-server.huggingface.co/rows"
        "?dataset=allenai%2Fwildbench&config=default&split=train&offset=0&length=200",
    ]
    _NOTE = "GPT-4-based pairwise judge vs GPT-4-Turbo baseline; scale -100 to 100"

    def _fetch_rows(self) -> list[FetchedRow]:
        last: Exception | None = None
        for url in self._URLS:
            try:
                rows = self._parse(self._get_json(url))
                if rows:
                    return rows
            except Exception as exc:
                last = exc
        raise RuntimeError(f"WildBench: all URLs failed. Last: {last}")

    def _parse(self, data: Any) -> list[FetchedRow]:
        rows: list[FetchedRow] = []

        # New format from score.json: {model_name: {score: ..., task_macro_score: ...}}
        if isinstance(data, dict) and all(
            isinstance(v, dict) and "score" in v for v in list(data.values())[:3]
        ):
            for model, info in data.items():
                raw_score = info.get("score")      # on ~1-10 scale
                adj_score = info.get("adjusted_score")  # centered around 0
                task_score = info.get("task_macro_score")
                if raw_score is None:
                    continue
                fr = FetchedRow(model_name=str(model).split("/")[-1],
                                organization=str(model).split("/")[0] if "/" in str(model) else "")
                # WB Score (raw 1-10 scale)
                fr.scores["WildBench v2"]       = round(float(raw_score), 2)
                fr.shots["WildBench v2"]        = None
                fr.metric_names["WildBench v2"] = "wb_score"
                fr.scales["WildBench v2"]       = (1.0, 10.0)
                fr.notes["WildBench v2"]        = "GPT-4 judge score; scale 1-10"
                # WB Score (adjusted, centered)
                if adj_score is not None:
                    fr.scores["WildBench v2 (adj)"]       = round(float(adj_score), 2)
                    fr.shots["WildBench v2 (adj)"]        = None
                    fr.metric_names["WildBench v2 (adj)"] = "wb_score_adjusted"
                    fr.scales["WildBench v2 (adj)"]       = (-100.0, 100.0)
                rows.append(fr)
            return rows

        # Old format: list or {rows: [...]}
        entries = data if isinstance(data, list) else data.get("models", data.get("rows", []))
        for entry in entries:
            if isinstance(entry, dict) and "row" in entry:
                entry = entry["row"]
            model = entry.get("model_id") or entry.get("model", "")
            score = entry.get("wb_score") or entry.get("score")
            if not model or score is None:
                continue
            fr = FetchedRow(model_name=str(model).split("/")[-1],
                            organization=str(model).split("/")[0] if "/" in str(model) else "")
            fr.scores["WildBench v2"]       = float(score)   # raw -100..100
            fr.shots["WildBench v2"]        = None
            fr.metric_names["WildBench v2"] = "wb_score"
            fr.scales["WildBench v2"]       = (-100.0, 100.0)
            fr.notes["WildBench v2"]        = self._NOTE
            rows.append(fr)
        return rows


class BigCodeBenchFetcher(LiveLeaderboardFetcher):
    """
    BigCodeBench — Complete and Instruct pass@1 (raw %, 0–100).
    Source: bigcode/bigcodebench-results HuggingFace dataset (parquet).

    Column schema:
      model, link, moe, size, act_param, type, complete, instruct, date, prefill
    """
    SOURCE_DIR  = "bigcodebench"
    SOURCE_NAME = "BigCodeBench"
    SOURCE_ORG  = "BigCode"
    SOURCE_URL  = "https://bigcode-bench.github.io/"
    EVAL_LIB    = "bigcodebench"
    CACHE_KEY   = "bigcodebench"

    _HF_DATASET = "bigcode/bigcodebench-results"

    def _fetch_rows(self) -> list[FetchedRow]:
        import pandas as pd
        try:
            df = _load_hf_dataset_as_df(self._HF_DATASET)
        except Exception as exc:
            raise RuntimeError(f"BigCodeBench: failed to load HF dataset: {exc}")

        rows: list[FetchedRow] = []
        for _, entry in df.iterrows():
            model = str(entry.get("model", "")).strip()
            if not model:
                continue
            complete = entry.get("complete")
            instruct = entry.get("instruct")
            if pd.isna(complete) and pd.isna(instruct):
                continue

            fr = FetchedRow(
                model_name=model.split("/")[-1] if "/" in model else model,
                organization=model.split("/")[0] if "/" in model else "",
            )
            # Extract evaluation date if available
            date_val = entry.get("date")
            if date_val is not None and not pd.isna(date_val):
                date_str = str(date_val).strip()
                if re.match(r"^\d{4}-\d{2}-\d{2}", date_str):
                    fr.eval_date = date_str[:10]
                elif re.match(r"^\d{4}/\d{2}/\d{2}", date_str):
                    fr.eval_date = date_str[:10].replace("/", "-")
                elif re.match(r"^\d{8}$", date_str):
                    fr.eval_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            for bench, val in [("BigCodeBench-Complete", complete),
                                ("BigCodeBench-Instruct", instruct)]:
                if val is not None and not pd.isna(val):
                    try:
                        fr.scores[bench]       = self._pct(val)
                        fr.shots[bench]        = 0
                        fr.metric_names[bench] = "pass@1"
                        fr.scales[bench]       = (0.0, 100.0)
                    except (TypeError, ValueError):
                        pass
            if fr.scores:
                rows.append(fr)
        return rows


# ===========================================================================
# Registry + orchestration
# ===========================================================================

ALL_FETCHERS: list[type[LiveLeaderboardFetcher]] = [
    # Tier 1 — benchmark overlap with paper sources
    OpenLLMLeaderboardV1Fetcher,
    OpenLLMLeaderboardV2Fetcher,
    # Tier 2 — coding / agentic
    EvalPlusFetcher,
    BFCLFetcher,
    SWEBenchFetcher,
    # Existing (normalisation fixed)
    AlpacaEval2Fetcher,
    ChatbotArenaFetcher,
    MTBenchFetcher,
    WildBenchFetcher,
    BigCodeBenchFetcher,
]
_SOURCE_MAP = {F.SOURCE_DIR: F for F in ALL_FETCHERS}


def run_all(force_refresh: bool = False) -> dict[str, int]:
    counts: dict[str, int] = {}
    for FetcherClass in ALL_FETCHERS:
        fetcher = FetcherClass()
        rows    = fetcher.fetch(force_refresh=force_refresh)
        counts[fetcher.SOURCE_DIR] = fetcher.write_rows(rows)
    return counts


def run_source(source_key: str, force_refresh: bool = False) -> int:
    if source_key not in _SOURCE_MAP:
        raise ValueError(f"Unknown source: {source_key!r}. Valid: {list(_SOURCE_MAP)}")
    fetcher = _SOURCE_MAP[source_key]()
    return fetcher.write_rows(fetcher.fetch(force_refresh=force_refresh))


def probe_source(source_key: str) -> None:
    """Print actual columns from the live source to verify the column map."""
    if source_key not in _SOURCE_MAP:
        raise ValueError(f"Unknown source: {source_key!r}.")
    _SOURCE_MAP[source_key]().probe_columns()


# ===========================================================================
# CLI
# ===========================================================================

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(
        description="Fetch live leaderboard data and write EEE records."
    )
    p.add_argument("--source",  metavar="KEY", help="Run only this source.")
    p.add_argument("--refresh", action="store_true", help="Bypass disk cache.")
    p.add_argument("--list",    action="store_true", help="List sources and exit.")
    p.add_argument(
        "--probe", metavar="KEY",
        help=(
            "Print actual column names from the live source without writing "
            "any records.  Use this to verify the column map after a schema "
            "change on the leaderboard side."
        ),
    )
    args = p.parse_args()

    if args.list:
        print("Available leaderboard sources:")
        for F in ALL_FETCHERS:
            print(f"  {F.SOURCE_DIR:<42}  {F.SOURCE_URL}")
        return

    if args.probe:
        probe_source(args.probe)
        return

    if not _HF_TOKEN:
        print(
            "WARNING: HF_TOKEN not found.  The Open LLM Leaderboard v1 and v2 "
            "datasets are gated.\n"
            "  Options:\n"
            "    1. Add  HF_TOKEN=hf_...  to your .env file in the repo root\n"
            "    2. Set  export HF_TOKEN=hf_...  in your shell\n"
            "  All other sources (Arena, MT-Bench, AlpacaEval, etc.) will still run.\n",
            file=sys.stderr,
        )

    if args.source:
        n = run_source(args.source, force_refresh=args.refresh)
        print(f"\n{args.source}: +{n} records written")
    else:
        print("Fetching all leaderboard sources ...\n")
        counts = run_all(force_refresh=args.refresh)
        groups = {
            "Tier 1 — benchmark overlap with paper sources": [
                "open_llm_leaderboard_v1", "open_llm_leaderboard_v2",
            ],
            "Tier 2 — coding / agentic": ["evalplus", "bfcl", "swe_bench"],
            "Existing sources (normalisation fixed)": [
                "alpacaeval2", "chatbot_arena", "mt_bench",
                "wildbench", "bigcodebench",
            ],
        }
        for heading, keys in groups.items():
            print(f"\n{heading}:")
            for k in keys:
                print(f"  +{counts.get(k, 0):4d}  {k}")
        print(f"\nTotal new records: {sum(counts.values())}")

    data  = pathlib.Path(__file__).resolve().parent.parent.parent / "data"
    total = sum(1 for _ in data.rglob("*.json"))
    print(f"Grand total records: {total}")
    print("\nPer-source record counts:")
    for d in sorted(data.iterdir()):
        if d.is_dir():
            cnt = sum(1 for _ in d.rglob("*.json"))
            print(f"  {cnt:6d}  {d.name}")


if __name__ == "__main__":
    main()