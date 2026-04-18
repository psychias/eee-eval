"""
Track 2 reference implementation: proprietary evaluation data converter.

This module is a *concrete, runnable* example of how to subclass
``BaseProprietaryConverter`` to convert an internal evaluation format into
the Every Eval Ever schema.  Copy and adapt it for your own benchmark.

Expected input format (CSV)
---------------------------
Required columns:

    model_id,benchmark,score

Optional generation-config columns (filled into ``generation_config``):

    num_shots,temperature,max_tokens,eval_date

Optional inference columns:

    inference_platform   — remote API provider (e.g. ``openai``, ``anthropic``,
                           ``huggingface``).  Set this when the model was called
                           through an API; leave blank for local runs (the
                           adapter auto-detects local engines via
                           ``_detect_inference_engine()``).

Optional instance-level columns (triggers companion JSONL output):

    sample_id,model_input,model_output,reference,is_correct

When instance-level columns are present the adapter writes a companion
``{uuid}.jsonl`` file alongside ``{uuid}.json`` and links them via the
``detailed_evaluation_results`` field.  Omit these columns to produce
aggregate-only output.

Example rows::

    anthropic/claude-3-5-sonnet-20241022,internal_safety_bench,0.87,0,0.0,1024,2025-11-01
    openai/gpt-4o-2024-11-20,internal_safety_bench,0.91,0,0.0,1024,2025-11-02

Fields that cannot be shared publicly are omitted from the output JSON; only
the aggregate score is emitted.

Usage
-----
    # dry run (no files written):
    python utils/proprietary_template/adapter.py --source path/to/results.csv --dry-run

    # real run:
    python utils/proprietary_template/adapter.py --source path/to/results.csv

The adapter writes to ``data/proprietary_template/{developer}/{model}/``.

Track 2 documentation
---------------------
See ``utils/proprietary_template/methodology.txt`` (generated alongside this
file) for the methodology note required by the shared task deliverables.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "utils"))

from eval_types import (
    DetailedEvaluationResults,
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    Format,
    GenerationConfig,
    GenerationArgs,
    HashAlgorithm,
    InferenceEngine,
    MetricConfig,
    ModelInfo,
    ScoreDetails,
    ScoreType,
    SourceDataPrivate,
    SourceMetadata,
)
from instance_level_types import (
    AnswerAttributionItem,
    Evaluation,
    Input,
    InstanceLevelEvaluationLog,
    InteractionType,
    Output,
)
from src.converters import SCHEMA_VERSION as _SCHEMA_VERSION
from src.scrapers.base import BaseProprietaryConverter, _sanitize


# ---------------------------------------------------------------------------
# module-level helpers
# ---------------------------------------------------------------------------


def _detect_inference_engine() -> InferenceEngine | None:
    """Probe the runtime environment for a known local inference engine.

    Checks common local inference packages in priority order and returns the
    first one found with its installed version.  Returns ``None`` when none
    are present, which is the correct signal that inference ran via a remote
    API (use ``ModelInfo.inference_platform`` in that case instead).

    This is intentionally lightweight — it uses ``importlib.metadata`` rather
    than importing the packages, so it has no side-effects on the current
    process.
    """
    for pkg, canonical_name in (
        ("vllm",      "vllm"),
        ("ollama",    "ollama"),
        ("llama_cpp", "llama.cpp"),
        ("mlx_lm",   "mlx-lm"),
    ):
        try:
            version = importlib.metadata.version(pkg)
            return InferenceEngine(name=canonical_name, version=version)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _sample_hash(raw_input: str, references: list[str]) -> str:
    """Compute the SHA-256 sample hash used by the instance-level schema.

    The hash is over ``input.raw + input.reference[0]`` (or just input.raw
    when no reference exists), matching the convention in the existing
    instance-level JSONL files in the repository.
    """
    reference = references[0] if references else ""
    payload = (raw_input + reference).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _detect_eval_library() -> EvalLibrary:
    """Probe the runtime environment for a known evaluation harness.

    Resolution order:
    1. Check ``importlib.metadata`` for known eval harness packages in
       priority order.  Returns name + installed version on first match.
    2. Fall back to ``name='internal'``.  The version is sourced from the
       ``EVAL_HARNESS_VERSION`` environment variable when set, so CI pipelines
       can pin it without modifying source code.  When unavailable, records
       ``'unknown'`` and adds a note in ``additional_details`` so consumers
       know the field is incomplete rather than genuinely absent.
    """
    import os

    for pkg, canonical_name in (
        ("lm_eval",    "lm_eval"),
        ("inspect_ai", "inspect_ai"),
        ("crfm_helm",  "helm"),
    ):
        try:
            version = importlib.metadata.version(pkg)
            return EvalLibrary(
                name=canonical_name,
                version=version,
                additional_details={"detected_via": "importlib.metadata"},
            )
        except importlib.metadata.PackageNotFoundError:
            continue

    # No known harness installed — this is a proprietary / internal harness.
    env_version = os.environ.get("EVAL_HARNESS_VERSION", "").strip()
    return EvalLibrary(
        name="internal",
        version=env_version if env_version else "unknown",
        additional_details={
            "note": (
                "no standard eval harness detected in environment; "
                "set EVAL_HARNESS_VERSION env var to record the internal harness version"
            )
        },
    )


# ---------------------------------------------------------------------------
# Tune these to match your benchmark
# ---------------------------------------------------------------------------

class ProprietaryTemplateConverter(BaseProprietaryConverter):
    """Convert an internal benchmark CSV into EEE schema records.

    Replace the class-level attributes with your benchmark's details.
    """

    eval_name: str = "proprietary_template"
    source_name: str = "Internal Safety Benchmark v1.0"
    source_organization: str = "MyOrg"
    output_dir: str = "data/proprietary_template"

    # "first_party"  → your org ran the evaluation on your own models
    # "third_party"  → your org evaluated other orgs' models
    evaluator_relationship: str = "first_party"

    # Lower score = worse safety; set True only if lower is genuinely better
    # for your metric (e.g. toxicity rate, error rate).
    LOWER_IS_BETTER: bool = False
    MIN_SCORE: float = 0.0
    MAX_SCORE: float = 1.0

    # Name shown inside evaluation_results[].evaluation_name
    METRIC_DISPLAY_NAME: str = "internal_safety_score"
    METRIC_DESCRIPTION: str = (
        "Aggregate safety score on the internal safety benchmark. "
        "Higher is safer. Details withheld under NDA."
    )

    # ---------------------------------------------------------------------------
    # Initialisation
    # ---------------------------------------------------------------------------

    def __init__(self) -> None:
        # Instance-level logs accumulated during convert(); keyed by evaluation_id.
        # Must live on the instance — not the class — to prevent state leaking
        # across back-to-back calls on different instances.
        self._instance_logs: dict[str, list[dict]] = {}

    # ---------------------------------------------------------------------------
    # Required: load raw data
    # ---------------------------------------------------------------------------

    def load_raw(self, source_path: Path) -> list[dict[str, str]]:
        """Read a CSV file and return a list of row dicts."""
        with open(source_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        missing = {"model_id", "score"} - set(rows[0].keys() if rows else [])
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {missing}. "
                f"Found: {set(rows[0].keys()) if rows else 'empty file'}"
            )
        return rows

    # ---------------------------------------------------------------------------
    # Required: convert to EEE schema
    # ---------------------------------------------------------------------------

    def convert(self, raw: list[dict[str, str]], retrieved_timestamp: str) -> list[dict]:
        """Convert raw CSV rows to EEE schema dicts, one dict per model."""
        # Reset instance-log accumulator for this conversion run so that
        # back-to-back calls to run() on the same converter don't cross-contaminate.
        self._instance_logs = {}

        # Group rows by model_id so one EvaluationLog covers all benchmarks
        # for a given model (matching the pattern in the existing converters).
        by_model: dict[str, list[dict[str, str]]] = {}
        for row in raw:
            model_id = row["model_id"].strip()
            by_model.setdefault(model_id, []).append(row)

        records: list[dict] = []
        for model_id, rows in by_model.items():
            try:
                records.append(
                    self._convert_model(model_id, rows, retrieved_timestamp)
                )
            except Exception as exc:
                print(f"  skipping model {model_id!r}: {exc}")

        return records

    def _convert_model(
        self,
        model_id: str,
        rows: list[dict[str, str]],
        retrieved_timestamp: str,
    ) -> dict:
        """Build one EvaluationLog dict for *model_id*."""

        # --- model info ---
        if "/" in model_id:
            developer, model_name = model_id.split("/", 1)
        else:
            developer = "unknown"
            model_name = model_id

        evaluation_id = (
            f"{self.eval_name}/{model_id.replace('/', '_')}/{retrieved_timestamp}"
        )

        # --- source metadata ---
        # source_type="documentation" → results were published/documented
        # source_type="evaluation_run" → results were produced locally by us
        source_metadata = SourceMetadata(
            source_name=self.source_name,
            source_type="evaluation_run",
            source_organization_name=self.source_organization,
            evaluator_relationship=EvaluatorRelationship(self.evaluator_relationship),
        )

        # --- source data (private — not publicly available) ---
        # Use SourceDataPrivate (source_type="other") when the underlying
        # dataset cannot be released.  Record what you *can* say publicly
        # in additional_details.
        source_data = SourceDataPrivate(
            dataset_name=self.source_name,
            source_type="other",
            additional_details={
                "availability": "internal only — data not publicly available",
                "contact": "reach out to the contributing org for access",
            },
        )

        # --- generation config ---
        # Populate from the first row; explicitly document *which* fields are
        # absent rather than emitting a single catch-all note, so consumers
        # know exactly what context is missing.
        first_row = rows[0]
        gen_args: dict[str, float] = {}
        missing_gen_fields: list[str] = []
        for field, key in [
            ("temperature", "temperature"),
            ("max_tokens",  "max_tokens"),
        ]:
            raw_val = first_row.get(key, "").strip()
            if raw_val:
                try:
                    gen_args[field] = float(raw_val)
                except ValueError:
                    missing_gen_fields.append(f"{key} (unparseable: {raw_val!r})")
            else:
                missing_gen_fields.append(key)

        # num_shots goes into additional_details (not a GenerationArgs field)
        extra_details: dict[str, str] = {}
        raw_shots = first_row.get("num_shots", "").strip()
        if raw_shots:
            extra_details["num_shots"] = raw_shots
        else:
            missing_gen_fields.append("num_shots")

        if missing_gen_fields:
            extra_details["missing_fields"] = ", ".join(missing_gen_fields)

        generation_config = GenerationConfig(
            generation_args=GenerationArgs(**gen_args) if gen_args else None,
            additional_details=extra_details or None,
        )

        # --- evaluation results (deduplicated by benchmark name) ---
        seen: dict[str, EvaluationResult] = {}
        for row in rows:
            bench = row.get("benchmark", self.METRIC_DISPLAY_NAME).strip()
            if bench in seen:
                continue  # first occurrence wins

            try:
                raw_score = float(row["score"])
            except (ValueError, KeyError) as exc:
                print(f"  skipping row for {model_id}/{bench}: {exc}")
                continue

            # Normalise: if score was submitted as a percentage, convert to 0–1
            if self.MAX_SCORE == 1.0 and raw_score > 1.0:
                raw_score = raw_score / 100.0

            seen[bench] = EvaluationResult(
                evaluation_name=bench,
                source_data=source_data,
                metric_config=MetricConfig(
                    evaluation_description=self.METRIC_DESCRIPTION,
                    lower_is_better=self.LOWER_IS_BETTER,
                    score_type=ScoreType.continuous,
                    min_score=self.MIN_SCORE,
                    max_score=self.MAX_SCORE,
                ),
                score_details=ScoreDetails(score=round(raw_score, 4)),
                # Capture the per-result evaluation timestamp if provided
                evaluation_timestamp=row.get("eval_date") or None,
            )

        if not seen:
            raise ValueError(f"no valid scores found for {model_id}")

        # --- instance-level logs (built when CSV has output columns) ---
        # Columns: sample_id, model_input, model_output, reference, is_correct
        # All five must be present in *at least one row* for JSONL to be written.
        _INSTANCE_COLS = {"model_input", "model_output", "reference", "is_correct"}
        has_instance_data = _INSTANCE_COLS.issubset(rows[0].keys())

        instance_entries: list[dict] = []
        if has_instance_data:
            for idx, row in enumerate(rows):
                raw_input  = row.get("model_input", "").strip()
                raw_output = row.get("model_output", "").strip()
                reference  = row.get("reference", "").strip()
                is_correct_raw = row.get("is_correct", "false").strip().lower()
                is_correct = is_correct_raw in ("1", "true", "yes")
                bench = row.get("benchmark", self.METRIC_DISPLAY_NAME).strip()

                # Derive a score consistent with is_correct for single-label evals.
                # If the CSV carries a per-row score, honour that instead.
                try:
                    inst_score = float(row["score"])
                except (KeyError, ValueError):
                    inst_score = 1.0 if is_correct else 0.0

                sample_id = row.get("sample_id", "").strip() or f"{bench}_{idx:05d}"
                references = [reference] if reference else []

                instance_log = InstanceLevelEvaluationLog(
                    schema_version=_SCHEMA_VERSION,
                    evaluation_id=evaluation_id,
                    model_id=model_id,
                    evaluation_name=bench,
                    sample_id=sample_id,
                    sample_hash=_sample_hash(raw_input, references) if raw_input else None,
                    interaction_type=InteractionType.single_turn,
                    input=Input(raw=raw_input, reference=references),
                    output=Output(raw=[raw_output]),
                    answer_attribution=[
                        AnswerAttributionItem(
                            turn_idx=0,
                            source="output.raw",
                            extracted_value=raw_output[:512],  # guard against huge outputs
                            extraction_method="exact_match",
                            is_terminal=True,
                        )
                    ],
                    evaluation=Evaluation(score=inst_score, is_correct=is_correct),
                )
                instance_entries.append(instance_log.model_dump(exclude_none=True))

        if instance_entries:
            self._instance_logs[evaluation_id] = instance_entries

        # --- inference engine / platform ---
        # The schema distinguishes *local* inference (inference_engine, e.g. vLLM)
        # from *remote* API access (inference_platform, e.g. "openai").
        # Priority: if the CSV carries an explicit inference_platform value, use
        # it and skip engine auto-detection (which only makes sense for local runs).
        raw_platform = first_row.get("inference_platform", "").strip()
        inferred_engine = None if raw_platform else _detect_inference_engine()

        log = EvaluationLog(
            schema_version=_SCHEMA_VERSION,
            evaluation_id=evaluation_id,
            retrieved_timestamp=retrieved_timestamp,
            source_metadata=source_metadata,
            eval_library=_detect_eval_library(),
            model_info=ModelInfo(
                name=model_name,
                id=model_id,
                developer=developer,
                # Use inference_platform for remote-API evals; use
                # inference_engine (auto-detected) for local runs.
                inference_platform=raw_platform or None,
                inference_engine=inferred_engine,
            ),
            generation_config=generation_config,
            evaluation_results=list(seen.values()),
            # detailed_evaluation_results is wired in _write_records() once
            # the JSONL path and checksum are known.
        )

        return log.model_dump(exclude_none=True)


    # ---------------------------------------------------------------------------
    # Instance-level JSONL writing
    # ---------------------------------------------------------------------------

    def _write_records(self, records: list[dict]) -> int:
        """Write aggregate JSON files and companion JSONL when instance data exists.

        For each model record:
        1. If ``self._instance_logs[evaluation_id]`` is populated, write a
           ``{uuid}.jsonl`` file alongside the aggregate JSON.
        2. Compute the SHA-256 checksum of the JSONL file and inject a
           ``detailed_evaluation_results`` block into the aggregate dict before
           serialising it, so consumers can validate the link.
        3. Write the (possibly updated) aggregate ``{uuid}.json``.
        """
        count = 0
        for rec in records:
            try:
                model_id: str = rec["model_info"]["id"]
                if "/" in model_id:
                    developer, model_name = model_id.split("/", 1)
                else:
                    developer = rec["model_info"].get("developer", "unknown")
                    model_name = model_id

                out_dir = (
                    Path(self.output_dir)
                    / _sanitize(developer)
                    / _sanitize(model_name)
                )
                out_dir.mkdir(parents=True, exist_ok=True)

                file_uuid = str(uuid.uuid4())
                evaluation_id: str = rec.get("evaluation_id", "")
                instance_rows = self._instance_logs.get(evaluation_id, [])

                if instance_rows:
                    # --- write JSONL ---
                    jsonl_path = out_dir / f"{file_uuid}.jsonl"
                    with open(jsonl_path, "w", encoding="utf-8") as fh:
                        for row in instance_rows:
                            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

                    # --- checksum ---
                    sha256 = hashlib.sha256(
                        jsonl_path.read_bytes()
                    ).hexdigest()

                    # --- wire into aggregate ---
                    # Use the Pydantic model so field names and enum values
                    # are guaranteed to match the schema (e.g. Format.jsonl
                    # serialises to "jsonl", HashAlgorithm.sha256 to "sha256").
                    rec["detailed_evaluation_results"] = DetailedEvaluationResults(
                        format=Format.jsonl,
                        file_path=str(jsonl_path),
                        hash_algorithm=HashAlgorithm.sha256,
                        checksum=sha256,
                        total_rows=len(instance_rows),
                    ).model_dump(exclude_none=True)

                # --- write JSON ---
                json_path = out_dir / f"{file_uuid}.json"
                with open(json_path, "w", encoding="utf-8") as fh:
                    json.dump(rec, fh, indent=2, ensure_ascii=False)

                count += 1
            except Exception as exc:
                mid = rec.get("model_info", {}).get("id", "?")
                print(f"  warning: could not write record for {mid}: {exc}")

        return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Track 2 converter: proprietary evaluation CSV → EEE schema JSON. "
            "Adapt ProprietaryTemplateConverter for your benchmark."
        )
    )
    parser.add_argument(
        "--source",
        required=True,
        metavar="CSV",
        help="path to the internal results CSV file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="convert without writing any files to disk",
    )
    args = parser.parse_args()

    converter = ProprietaryTemplateConverter()
    n = converter.run(Path(args.source), dry_run=args.dry_run)
    print(f"done — {n} records {'(dry-run, nothing written)' if args.dry_run else 'written'}.")


if __name__ == "__main__":
    main()
