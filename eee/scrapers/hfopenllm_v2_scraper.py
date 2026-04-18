"""
HuggingFace Open LLM Leaderboard v2 scraper.

fetches results from the official HF Space API and converts them to the
EEE schema (v0.2.1). extends BaseLeaderboardScraper; never modifies the
base class.

usage:
    python scripts/scrapers/hfopenllm_v2_scraper.py [--limit N] [--dry-run]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

# add repo root to sys.path so eval_types and utils.helpers are importable
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "utils"))

from eval_types import (
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    MetricConfig,
    ScoreDetails,
    ScoreType,
    SourceDataHf,
    SourceMetadata,
)
from helpers import fetch_json, get_developer, sanitize_filename, save_evaluation_log

from src.converters import SCHEMA_VERSION as _SCHEMA_VERSION
from src.scrapers.base import BaseLeaderboardScraper

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

# primary API endpoint for the HF Open LLM leaderboard v2
_API_URL = (
    "https://open-llm-leaderboard-open-llm-leaderboard.hf.space"
    "/api/leaderboard/formatted"
)

# mapping from API eval keys → human-readable display names
_EVAL_DISPLAY = {
    "ifeval": "IFEval",
    "bbh": "BBH",
    "math": "MATH Level 5",
    "gpqa": "GPQA",
    "musr": "MUSR",
    "mmlu_pro": "MMLU-PRO",
}

# per-benchmark HuggingFace dataset references
_SOURCE_DATA: dict[str, SourceDataHf] = {
    "ifeval": SourceDataHf(
        dataset_name="IFEval",
        source_type="hf_dataset",
        hf_repo="google/IFEval",
    ),
    "bbh": SourceDataHf(
        dataset_name="BBH",
        source_type="hf_dataset",
        hf_repo="SaylorTwift/bbh",
    ),
    "math": SourceDataHf(
        dataset_name="MATH Level 5",
        source_type="hf_dataset",
        hf_repo="DigitalLearningGmbH/MATH-lighteval",
    ),
    "gpqa": SourceDataHf(
        dataset_name="GPQA",
        source_type="hf_dataset",
        hf_repo="Idavidrein/gpqa",
    ),
    "musr": SourceDataHf(
        dataset_name="MUSR",
        source_type="hf_dataset",
        hf_repo="TAUR-Lab/MuSR",
    ),
    "mmlu_pro": SourceDataHf(
        dataset_name="MMLU-PRO",
        source_type="hf_dataset",
        hf_repo="TIGER-Lab/MMLU-Pro",
    ),
}

# shared source metadata for all HF Open LLM v2 records
_SOURCE_METADATA = SourceMetadata(
    source_name="HuggingFace Open LLM Leaderboard v2",
    source_type="documentation",
    source_organization_name="Hugging Face",
    source_organization_url="https://huggingface.co/spaces/open-llm-leaderboard",
    evaluator_relationship=EvaluatorRelationship.third_party,
)

# shared eval_library for all HF Open LLM v2 records
# the leaderboard uses lighteval internally but the version is not exposed by API
_EVAL_LIBRARY = EvalLibrary(name="lighteval", version="unknown")


# ---------------------------------------------------------------------------
# scraper implementation
# ---------------------------------------------------------------------------


class HFOpenLLMv2Scraper(BaseLeaderboardScraper):
    """scraper for the HuggingFace Open LLM Leaderboard v2.

    extends BaseLeaderboardScraper; adding new scrapers never modifies this.
    """

    eval_name = "hfopenllm_v2"
    source_name = "HuggingFace Open LLM Leaderboard v2"
    source_organization = "Hugging Face"
    output_dir = "data/hfopenllm_v2"

    def fetch_raw(self) -> list[dict]:
        """fetch the full leaderboard JSON from the HF Space API."""
        return fetch_json(_API_URL)

    def convert(self, raw: list[dict], retrieved_timestamp: str) -> list[dict]:
        """convert raw leaderboard entries to EEE schema dicts."""
        results: list[dict] = []
        for entry in raw:
            try:
                record = _convert_entry(entry, retrieved_timestamp)
                results.append(record)
            except Exception as exc:
                model_id = entry.get("model", {}).get("name", "?")
                print(f"  skipping {model_id}: {exc}")
        return results


# ---------------------------------------------------------------------------
# conversion helpers — single responsibility, not mixed with I/O
# ---------------------------------------------------------------------------


def _make_eval_results(evaluations: dict[str, Any]) -> list[EvaluationResult]:
    """build EvaluationResult objects from the 'evaluations' sub-dict."""
    results: list[EvaluationResult] = []
    for key, data in evaluations.items():
        display = data.get("name", _EVAL_DISPLAY.get(key, key))
        source = _SOURCE_DATA.get(key)
        if source is None:
            # fallback for unknown benchmark keys
            source = SourceDataHf(
                dataset_name=display,
                source_type="hf_dataset",
                hf_repo=None,
            )

        raw_score = data.get("value")
        score = round(float(raw_score), 4) if raw_score is not None else 0.0
        # The HF Open LLM v2 API returns scores on a 0–100 scale for some
        # benchmarks (e.g. IFEval, BBH, MMLU-PRO).  Normalise to 0–1 so that
        # all records in the database are on a consistent scale.  A score
        # exactly equal to 1.0 is rare but valid (perfect), so we only
        # normalise when the value is strictly greater than 1.
        if score > 1.0:
            score = round(score / 100.0, 4)

        results.append(
            EvaluationResult(
                evaluation_name=display,
                source_data=source,
                metric_config=MetricConfig(
                    evaluation_description=f"accuracy on {display} (normalised to 0–1)",
                    lower_is_better=False,
                    score_type=ScoreType.continuous,
                    min_score=0.0,
                    max_score=1.0,
                ),
                score_details=ScoreDetails(score=score),
            )
        )
    return results


def _build_additional_details(entry: dict[str, Any]) -> dict[str, str] | None:
    """extract optional model-level details as a string-valued dict."""
    model = entry.get("model", {})
    meta = entry.get("metadata", {})
    details: dict[str, str] = {}

    for field in ("precision", "architecture", "model_type"):
        if val := model.get(field):
            details[field] = str(val)

    if params := meta.get("params_billions"):
        details["params_billions"] = str(params)

    return details if details else None


def _convert_entry(entry: dict[str, Any], retrieved_timestamp: str) -> dict:
    """convert a single leaderboard entry to an EEE schema dict."""
    from eval_types import ModelInfo

    model_id: str = entry["model"]["name"]
    if "/" not in model_id:
        raise ValueError(f"expected 'org/model' format, got: {model_id!r}")

    developer, model_name = model_id.split("/", 1)

    eval_results = _make_eval_results(entry.get("evaluations", {}))
    additional_details = _build_additional_details(entry)

    # build evaluation_id following the convention: eval_name/developer_model/ts
    safe_model_id = model_id.replace("/", "_")
    evaluation_id = f"hfopenllm_v2/{safe_model_id}/{retrieved_timestamp}"

    log = EvaluationLog(
        schema_version=_SCHEMA_VERSION,
        evaluation_id=evaluation_id,
        retrieved_timestamp=retrieved_timestamp,
        source_metadata=_SOURCE_METADATA,
        eval_library=_EVAL_LIBRARY,
        model_info=ModelInfo(
            name=model_id,
            id=model_id,
            developer=developer,
            inference_platform="huggingface",
            additional_details=additional_details,
        ),
        evaluation_results=eval_results,
    )

    return log.model_dump(mode='json', exclude_none=True)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main(limit: int | None = None, dry_run: bool = False) -> None:
    """run the HF Open LLM v2 scraper end-to-end."""
    scraper = HFOpenLLMv2Scraper()

    print(f"fetching leaderboard from {_API_URL} ...")
    raw: list[dict] = scraper.fetch_raw()
    print(f"  fetched {len(raw)} entries")

    if limit:
        raw = raw[:limit]
        print(f"  limited to {limit} entries")

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

    parser = argparse.ArgumentParser(description="scrape HF Open LLM Leaderboard v2")
    parser.add_argument("--limit", type=int, default=None, help="max models to process")
    parser.add_argument("--dry-run", action="store_true", help="skip writing files")
    args = parser.parse_args()
    main(limit=args.limit, dry_run=args.dry_run)
