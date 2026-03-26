"""Paper converter — maps extracted data to EEE schema records."""
from __future__ import annotations

import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval_types import (
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    GenerationArgs,
    GenerationConfig,
    MetricConfig,
    ModelInfo,
    ScoreDetails,
    ScoreType,
    SourceDataUrl,
    SourceMetadata,
)
from eval_converters import SCHEMA_VERSION as _SCHEMA_VERSION

from .constants import (
    _LOWER_IS_BETTER_PATTERNS,
    _UNBOUNDED_SCORE_PATTERNS,
    _SCALE_100_PATTERNS,
    _RAW_SCALE_BENCHMARKS,
    OPENROUTER_API_KEY,
)
from .normalize import (
    _normalize_benchmark,
    _normalize_model_name,
    _normalize_aime,
    _should_tag_thinking_mode,
    _is_likely_fraction,
)
from .helpers import (
    _coerce_int,
    _coerce_float,
    _coerce_bool,
    _infer_model_metadata,
    _infer_metric_name,
    _infer_dataset_split,
    _infer_sample_size,
    _infer_developer,
    _infer_evaluator_relationship,
    _make_eval_name,
    _make_source_metadata,
    _lookup_protocol,
    _stringify_detail_values,
    _aggregate_extraction_confidence,
)

class PaperConverter:
    """convert extracted paper results to EEE schema dicts."""

    def __init__(self, gemini_extractor: "GeminiProtocolExtractor | None" = None) -> None:
        self.gemini_extractor = gemini_extractor or GeminiProtocolExtractor()

    def convert(
        self,
        extracted: list[dict[str, Any]],
        arxiv_id: str,
        retrieved_timestamp: str,
        paper_title: str = "",
        paper_authors: list[str] | None = None,
        eval_library: EvalLibrary | None = None,
        gemini_raw: dict[str, Any] | None = None,
        protocol_map: dict[str, dict[str, str | int]] | None = None,
    ) -> list[dict]:
        """group *extracted* by model and produce one EEE record per model."""
        # group results by model name
        by_model: dict[str, list[dict]] = {}
        for item in extracted:
            model = item["model"]
            by_model.setdefault(model, []).append(item)


        eval_name = _make_eval_name(arxiv_id)
        source_metadata = _make_source_metadata(arxiv_id, paper_title)
        lib = eval_library if eval_library is not None else EvalLibrary(name="unknown", version="unknown")

        records: list[dict] = []
        for model_name, items in by_model.items():
            try:
                record = self._convert_model(
                    model_name,
                    items,
                    eval_name,
                    arxiv_id,
                    retrieved_timestamp,
                    source_metadata,
                    lib,
                    paper_authors=paper_authors or [],
                    gemini_raw=gemini_raw,
                    protocol_map=protocol_map,
                )
                records.append(record)
            except Exception as exc:
                print(f"  skipping model {model_name!r}: {exc}")

        # Post-processing: infer "non-thinking" for untagged entries when a
        # thinking counterpart exists for the same base model.
        # Collect model IDs that have thinking-mode records.
        thinking_model_ids: set[str] = set()
        for rec in records:
            for er in rec.get("evaluation_results", []):
                ga = (er.get("generation_config") or {}).get("generation_args") or {}
                if ga.get("reasoning_mode") == "thinking":
                    thinking_model_ids.add(rec.get("model_info", {}).get("id", ""))
                    break
        # Tag untagged records for models that also have thinking entries.
        if thinking_model_ids:
            for rec in records:
                mid = rec.get("model_info", {}).get("id", "")
                if mid not in thinking_model_ids:
                    continue
                for er in rec.get("evaluation_results", []):
                    ga = (er.get("generation_config") or {}).get("generation_args")
                    if ga is None:
                        continue
                    if not ga.get("reasoning_mode"):
                        ga["reasoning_mode"] = "non-thinking"
                        ga["reasoning"] = False

        return records

    def _convert_model(
        self,
        model_name: str,
        items: list[dict],
        eval_name: str,
        arxiv_id: str,
        retrieved_timestamp: str,
        source_metadata: SourceMetadata,
        eval_library: EvalLibrary,
        paper_authors: list[str] | None = None,
        gemini_raw: dict[str, Any] | None = None,
        protocol_map: dict[str, dict[str, str | int]] | None = None,
    ) -> dict:
        """build one EvaluationLog dict for *model_name*."""
        # Extract thinking mode suffix before normalization
        thinking_mode: str | None = None
        if model_name.endswith("(Thinking)"):
            thinking_mode = "thinking"
            model_name = model_name.rsplit("(Thinking)", 1)[0].strip()
        elif model_name.endswith("(Non-Thinking)"):
            thinking_mode = "non-thinking"
            model_name = model_name.rsplit("(Non-Thinking)", 1)[0].strip()
        # Handle prefix-style thinking mode: "Thinking Mode Qwen3-32B"
        elif model_name.startswith("Thinking Mode "):
            thinking_mode = "thinking"
            model_name = model_name[len("Thinking Mode "):]
        elif model_name.startswith("Non-Thinking Mode "):
            thinking_mode = "non-thinking"
            model_name = model_name[len("Non-Thinking Mode "):]

        # Guard: only tag reasoning_mode for model families that actually
        # support thinking/non-thinking mode selection.  Baseline models
        # (e.g. GPT-4o) appearing in a "Non-Thinking" table section
        # should NOT be tagged.
        if thinking_mode and not _should_tag_thinking_mode(model_name):
            thinking_mode = None

        # Normalise stray spaces before hyphens: "Gemma-3 -27B-IT" → "Gemma-3-27B-IT"
        model_name = re.sub(r'\s+-', '-', model_name)
        # Normalize parameter-size casing: "27b" → "27B"
        model_name = _normalize_model_name(model_name)

        if gemini_raw is not None and OPENROUTER_API_KEY:
            protocol_map = self.gemini_extractor.to_protocol_map(
                gemini_raw, model_name=model_name
            )
        elif protocol_map is None:
            protocol_map = {}

        developer = _infer_developer(model_name)
        if "/" in model_name:
            model_id = model_name
            developer = model_id.split("/")[0]
        else:
            model_id = f"{developer}/{model_name}"

        safe_id = model_id.replace("/", "_")
        evaluation_id = f"{eval_name}/{safe_id}/{retrieved_timestamp}"

        model_metadata = _infer_model_metadata(model_name)

        # Compute evaluator relationship for source_metadata.
        eval_relationship = _infer_evaluator_relationship(
            developer, paper_authors or [], source_metadata.source_name or ""
        )

        seen_benchmarks: dict[tuple[str, str], EvaluationResult] = {}
        for item in items:
            bench = item["benchmark"].strip()
            # Normalise stray spaces before hyphens: "GPQA -Diamond" → "GPQA-Diamond"
            bench = re.sub(r'\s+-', '-', bench)
            # Canonicalize benchmark name (MATH500→MATH-500, etc.)
            bench = _normalize_aime(bench)
            normalized = _normalize_benchmark(bench)
            if normalized is None:
                continue  # skip non-benchmarks like "Average"
            bench = normalized


            # Build per-benchmark source_data so sample_size is benchmark-specific
            source_data = SourceDataUrl(
                dataset_name="arXiv paper",
                source_type="url",
                url=[f"https://arxiv.org/abs/{arxiv_id}"],
                dataset_split=_infer_dataset_split(bench),
                sample_size=_infer_sample_size(bench),
            )
            dedup_key = (bench, thinking_mode or "")
            if dedup_key in seen_benchmarks:
                # Keep higher-confidence entry; if same confidence, first wins
                existing = seen_benchmarks[dedup_key]
                existing_conf = existing.generation_config.additional_details.get(
                    "extraction_confidence", "low"
                ) if existing.generation_config and existing.generation_config.additional_details else "low"
                new_conf = item.get("_extraction_confidence", "low")
                _conf_order = {"high": 3, "medium": 2, "low": 1}
                if _conf_order.get(new_conf, 0) <= _conf_order.get(existing_conf, 0):
                    continue  # existing is same or higher confidence — keep it
                # else: new entry has higher confidence — replace below

            raw_score = float(item["score"])
            bench_lower = bench.lower()

            lower_is_better = any(p in bench_lower for p in _LOWER_IS_BETTER_PATTERNS)
            is_unbounded    = any(p in bench_lower for p in _UNBOUNDED_SCORE_PATTERNS)
            is_scale_100    = any(p in bench_lower for p in _SCALE_100_PATTERNS)
            raw_scale_range = next(
                (
                    score_range
                    for pattern, score_range in _RAW_SCALE_BENCHMARKS.items()
                    if pattern in bench_lower
                ),
                None,
            )

            if raw_scale_range is not None:
                score = round(raw_score, 4)
                min_score, max_score = raw_scale_range
            elif is_unbounded:
                # keep raw value; no meaningful upper bound
                score = round(raw_score, 4)
                min_score: float = 0.0
                max_score: float | None = None
            elif is_scale_100:
                # BLEU / ROUGE: keep on 0–100 scale
                score = round(raw_score, 4)
                min_score = 0.0
                max_score = 100.0
            elif raw_score > 1.0:
                # assume percentage — normalise to 0–1
                score = round(raw_score / 100.0, 4)
                min_score = 0.0
                max_score = 1.0
            elif _is_likely_fraction(bench, raw_score):
                # Score <= 1.0 on a benchmark typically reported as a
                # percentage (e.g. MMLU 0.623 = 62.3%).  Keep as-is on
                # the 0-1 scale — do NOT divide by 100.
                score = round(raw_score, 4)
                min_score = 0.0
                max_score = 1.0
            elif item.get("_had_pct") and raw_score <= 1.0:
                # explicit '%' sign present (e.g. "1.0%") — treat as
                # percentage even though the numeric value is <= 1
                score = round(raw_score / 100.0, 4)
                min_score = 0.0
                max_score = 1.0
            else:
                score = round(raw_score, 4)
                min_score = 0.0
                max_score = 1.0

            protocol = _lookup_protocol(protocol_map, bench)
            # Shots priority: the paper's evaluation protocol applies to
            # ALL models in the comparison tables — it describes how each
            # reported score was obtained.  Protocol always wins when
            # available; item-level shots (from chunk-based LLM fallback
            # with limited context) are only used as a fallback.
            protocol_shots = protocol.get("shots")
            item_shots = item.get("shots")
            shots = protocol_shots if protocol_shots is not None else item_shots
            scoring_method = protocol.get("method") or item.get("scoring_method")
            # Temperature, top_p, and chain_of_thought: protocol takes
            # precedence (paper-level methodology), item-level as fallback.
            temperature = _coerce_float(protocol.get("temperature") or item.get("temperature"))
            top_p = _coerce_float(protocol.get("top_p") or item.get("top_p"))
            top_k = _coerce_float(protocol.get("top_k") or item.get("top_k"))
            max_tokens = _coerce_int(protocol.get("max_tokens") or item.get("max_tokens"))
            chain_of_thought = _coerce_bool(protocol.get("chain_of_thought") if protocol.get("chain_of_thought") is not None else item.get("chain_of_thought"))
            prompt_template = protocol.get("prompt_template") or item.get("prompt_template")

            generation_details: dict[str, Any] = {
                "source": f"arXiv:{arxiv_id}",
                "extraction_confidence": item.get("_extraction_confidence", _aggregate_extraction_confidence(items)),
                "harness": eval_library.name,
            }
            if eval_library.version != "unknown":
                generation_details["eval_library_version"] = eval_library.version
            if scoring_method:
                generation_details["scoring_method"] = scoring_method
            if protocol.get("protocol_source"):
                generation_details["protocol_source"] = protocol["protocol_source"]
            if item.get("page_number") is not None:
                generation_details["page_number"] = item["page_number"]
            table_ref = item.get("table_id") or item.get("table")
            if table_ref:
                generation_details["table"] = table_ref
            if item.get("table_caption"):
                generation_details["table_caption"] = item["table_caption"]
            if thinking_mode:
                generation_details["thinking_mode"] = thinking_mode
            if len(generation_details) == 2:
                generation_details["note"] = "generation config not reported in source paper"

            score_details_extra: dict[str, Any] = {}
            score_type = item.get("score_type")
            if score_type and score_type != "unknown":
                score_details_extra["score_type"] = score_type
            relative_to = item.get("relative_to")
            if relative_to:
                score_details_extra["relative_to"] = relative_to
                score_details_extra["comparability_warning"] = (
                    "Relative score reported against a paper-specific baseline; avoid direct comparison to absolute leaderboard scores."
                )

            seen_benchmarks[dedup_key] = EvaluationResult(
                evaluation_name=bench,
                source_data=source_data,
                metric_config=MetricConfig(
                    evaluation_description=f"score on {bench} as reported in arXiv:{arxiv_id}",
                    metric_name=_infer_metric_name(bench),
                    lower_is_better=lower_is_better,
                    score_type=ScoreType.continuous,
                    min_score=min_score,
                    max_score=max_score,
                ),
                score_details=ScoreDetails(
                    score=score,
                    details=_stringify_detail_values(score_details_extra) or None,
                ),
                generation_config=GenerationConfig(
                    generation_args=(
                        GenerationArgs(
                            shots=_coerce_int(shots),
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            max_tokens=max_tokens,
                            reasoning=chain_of_thought or (thinking_mode == "thinking") or None,
                            reasoning_mode=thinking_mode,
                            chain_of_thought=chain_of_thought,
                            prompt_template=str(prompt_template) if prompt_template else None,
                        )
                        if any(
                            value is not None
                            for value in (
                                _coerce_int(shots),
                                temperature,
                                top_p,
                                top_k,
                                max_tokens,
                                chain_of_thought,
                                thinking_mode,
                                prompt_template,
                            )
                        )
                        else None
                    ),
                    additional_details=_stringify_detail_values(generation_details),
                ),
            )

        eval_results: list[EvaluationResult] = list(seen_benchmarks.values())

        model_source_metadata = SourceMetadata(
            source_name=source_metadata.source_name,
            source_type=source_metadata.source_type,
            source_organization_name=source_metadata.source_organization_name,
            source_organization_url=source_metadata.source_organization_url,
            evaluator_relationship=EvaluatorRelationship(eval_relationship),
        )

        log = EvaluationLog(
            schema_version=_SCHEMA_VERSION,
            evaluation_id=evaluation_id,
            retrieved_timestamp=retrieved_timestamp,
            source_metadata=model_source_metadata,
            eval_library=eval_library,
            model_info=ModelInfo(
                name=model_name,
                id=model_id,
                developer=developer,
                parameter_count=model_metadata["parameter_count"],
                model_alignment=model_metadata["model_alignment"],
                quantization=model_metadata["quantization"],
                # inference_platform left None — unknown from paper tables
            ),
            evaluation_results=eval_results,
        )

        record = log.model_dump(mode='json', exclude_none=True)
        return record


# ---------------------------------------------------------------------------
# S: paper writer — file I/O only
# ---------------------------------------------------------------------------


class PaperWriter:
    """write EEE schema dicts to disk under data/{benchmark}/{developer}/{model}/."""

    def write(self, records: list[dict], eval_name: str) -> list[Path]:
        """Write per-benchmark JSON files to data/{benchmark}/{developer}/{model}/.

        Each evaluation result in a record is written as a separate file
        following the EEE repository structure:
            data/{benchmark}/{developer_name}/{model_name}/{uuid}.json

        Deduplicates on (model_id, benchmark, score, source_url) to avoid
        writing the same result twice when both table and LLM extraction
        find it.
        """
        paths: list[Path] = []
        seen: set[tuple[str, str, str, str]] = set()

        def _sanitize_part(value: str) -> str:
            value = value.replace(" ", "_")
            return re.sub(r'[^A-Za-z0-9_.-]', "_", value)

        for rec in records:
            try:
                model_id: str = rec["model_info"]["id"]
                if "/" in model_id:
                    developer, model_name = model_id.split("/", 1)
                else:
                    developer = rec["model_info"].get("developer", "unknown")
                    model_name = model_id

                evaluation_results = rec.get("evaluation_results", [])
                for evaluation_result in evaluation_results:
                    benchmark_name = evaluation_result.get("evaluation_name", "unknown")
                    score = str(evaluation_result.get("score_details", {}).get("score", ""))
                    # reasoning_mode lives inside each evaluation_result's
                    # generation_config.generation_args, not at the record top level.
                    gen_args = (evaluation_result.get("generation_config") or {}).get("generation_args") or {}
                    reasoning_mode = gen_args.get("reasoning_mode", "") or ""
                    dedup_key = (model_id, benchmark_name, score, reasoning_mode)
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)

                    split_record = json.loads(json.dumps(rec))
                    split_record["evaluation_results"] = [evaluation_result]
                    split_dir = (
                        Path("data")
                        / _sanitize_part(benchmark_name)
                        / _sanitize_part(developer)
                        / _sanitize_part(model_name)
                    )
                    split_dir.mkdir(parents=True, exist_ok=True)
                    split_path = split_dir / f"{uuid.uuid4()}.json"
                    with open(split_path, "w", encoding="utf-8") as fh:
                        json.dump(split_record, fh, indent=2, ensure_ascii=False)
                    paths.append(split_path)
            except Exception as exc:
                mid = rec.get("model_info", {}).get("id", "?")
                print(f"  warning: could not write record for {mid}: {exc}")

        return paths


# ---------------------------------------------------------------------------
# S: LLM fallback extractor — prose and figure-caption extraction only
# ---------------------------------------------------------------------------


