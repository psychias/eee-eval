"""Shared scraper utilities — eliminates duplication across leaderboard scrapers.

Functions:
  - normalize_model_id: Convert model names to HuggingFace org/model format
  - fetch_csv: Fetch and parse a CSV URL into list of dicts
  - fetch_json: Fetch a JSON URL (handles list and dict responses)
  - fetch_jsonl: Fetch and parse JSONL format
  - extract_score: Try multiple dict keys to extract a numeric score
  - extract_model_name: Try multiple dict keys to extract model name
  - get_developer: Heuristic developer extraction from model name
"""
from __future__ import annotations

import csv
import io
import json
import logging
from typing import Any

import requests

log = logging.getLogger(__name__)

_TIMEOUT = 30  # seconds


# ─── Model identity ──────────────────────────────────────────────────────────


def normalize_model_id(
    model_name: str,
    developer: str,
    hf_mapping: dict[str, str] | None = None,
) -> str:
    """Normalize model_name to HuggingFace-style 'org/model' format.

    Priority: model_name already has '/' → return as-is
              model_name in hf_mapping   → return mapped value
              fallback                    → developer/model_name
    """
    if "/" in model_name:
        return model_name
    if hf_mapping and model_name in hf_mapping:
        return hf_mapping[model_name]
    return f"{developer}/{model_name}"


def extract_model_name(row: dict[str, Any], keys: list[str] | None = None) -> str:
    """Try several common key names to extract model name from a row.

    Returns stripped model name or empty string.
    """
    if keys is None:
        keys = ["model", "Model", "model_name", "name", "model_id", "model_id_or_alias"]
    for key in keys:
        val = row.get(key)
        if val and str(val).strip():
            return str(val).strip()
    return ""


def get_developer(model_name: str) -> str:
    """Extract developer/organisation from a model name.

    Delegates to the single canonical implementation
    (``src.extraction.constants.infer_developer``) so every source agrees on
    org IDs (e.g. ``meta-llama``, ``Qwen``) and the same model never lands
    under two different developer folders — which previously manufactured
    phantom cross-source score conflicts.
    """
    from src.extraction.constants import infer_developer
    return infer_developer(model_name)


# ─── Score extraction ─────────────────────────────────────────────────────────


def extract_score(
    data: dict[str, Any],
    possible_keys: list[str],
    is_percentage: bool = False,
    default: float | None = None,
) -> float | None:
    """Try multiple keys in order to extract a numeric score.

    If is_percentage, divides by 100. Returns default if no key matches.
    """
    for key in possible_keys:
        val = data.get(key)
        if val is not None:
            try:
                score = float(str(val).replace("%", "").strip())
                return score / 100.0 if is_percentage else score
            except (ValueError, TypeError):
                continue
    return default


# ─── Network fetching ─────────────────────────────────────────────────────────


def fetch_csv(url: str, timeout: int = _TIMEOUT) -> list[dict]:
    """Fetch a CSV URL and return parsed rows as list of dicts.

    Returns empty list on any error.
    """
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        reader = csv.DictReader(io.StringIO(resp.text))
        rows = list(reader)
        return rows if rows else []
    except Exception as exc:
        log.debug("fetch_csv(%s) failed: %s", url, exc)
        return []


def fetch_json(url: str, timeout: int = _TIMEOUT) -> list[dict]:
    """Fetch a JSON URL. Handles both list and dict responses.

    For dict responses, converts to list of {model: key, **values} dicts.
    Returns empty list on error.
    """
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            rows = []
            for model_name, values in data.items():
                row = {"model": model_name}
                if isinstance(values, dict):
                    row.update(values)
                elif isinstance(values, (int, float, str)):
                    row["score"] = values
                rows.append(row)
            return rows
    except Exception as exc:
        log.debug("fetch_json(%s) failed: %s", url, exc)
    return []


def fetch_jsonl(url: str, timeout: int = _TIMEOUT) -> list[dict]:
    """Fetch a JSONL (newline-delimited JSON) URL.

    Returns empty list on error.
    """
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        rows = []
        for line in resp.text.strip().split("\n"):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows
    except Exception as exc:
        log.debug("fetch_jsonl(%s) failed: %s", url, exc)
        return []
