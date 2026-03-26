"""Main extraction pipeline — orchestrates all stages."""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / 'utils') not in sys.path:
    sys.path.insert(0, str(_ROOT / 'utils'))

from eval_types import EvalLibrary

from .constants import (
    _BENCHMARK_KEYWORDS,
    _BENCHMARK_KEYWORDS_STRONG,
    _EVAL_FRAMEWORK_SIGNATURES,
    OPENROUTER_API_KEY,
)
from .normalize import _merge_key
from .helpers import (
    _extract_table_id,
    _clean_caption,
    _fetch_arxiv_metadata,
    _normalise_arxiv_id,
    _make_eval_name,
    _sanitize_extracted,
)
from .download import PDFDownloader
from .docling_parser import TableExtractor, _docling_parser
from .table_parser import ResultsTableParser
from .converter import PaperConverter, PaperWriter
from .llm_fallback import LLMFallbackExtractor, CoverageStats
from .prose import ProseExtractor
from .protocol import EvalProtocolExtractor, GeminiProtocolExtractor

# _detect_eval_library lives here (not helpers) to avoid circular
# import with docling_parser.
def _detect_eval_library(pdf_path: Path) -> EvalLibrary:
    """scan the first few pages of *pdf_path* for eval-framework signatures.

    Looks for known framework strings in the paper body (abstract + intro)
    and returns the matching EvalLibrary.  Falls back to ``internal`` for
    papers that do not clearly name a public harness.
    """
    try:
        full_text = _docling_parser.get_full_text(pdf_path).lower()
    except Exception:  # noqa: BLE001
        return EvalLibrary(name="internal", version="unknown")
    cutoff_markers = (
        "references\n",
        "bibliography\n",
        "acknowledgements\n",
        "acknowledgments\n",
    )
    cutoff_positions = [full_text.find(marker) for marker in cutoff_markers if full_text.find(marker) != -1]
    search_text = full_text[:min(cutoff_positions)] if cutoff_positions else full_text
    for signature, lib_name in _EVAL_FRAMEWORK_SIGNATURES:
        if signature.lower() in search_text:
            print(f"  detected eval framework: {lib_name}")
            return EvalLibrary(name=lib_name, version="unknown")
    return EvalLibrary(name="internal", version="unknown")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------




class PaperExtractionPipeline:
    """orchestrate download → extract → parse → convert → write for a paper."""

    def __init__(
        self,
        downloader: PDFDownloader,
        table_extractor: TableExtractor,
        table_parser: ResultsTableParser,
        converter: PaperConverter,
        writer: PaperWriter,
        llm_fallback: LLMFallbackExtractor | None = None,
        prose_extractor: ProseExtractor | None = None,
        protocol_extractor: EvalProtocolExtractor | None = None,
    ) -> None:
        self._downloader = downloader
        self._extractor = table_extractor
        self._parser = table_parser
        self._converter = converter
        self._writer = writer
        self._llm_fallback = llm_fallback
        self._prose_extractor = prose_extractor
        self.gemini_extractor = GeminiProtocolExtractor()
        self._converter.gemini_extractor = self.gemini_extractor
        self.regex_extractor = protocol_extractor or EvalProtocolExtractor()

    def run(self, source: str, *, force_llm: bool = False) -> CoverageStats:
        """process *source* (arXiv ID or PDF path) end-to-end.

        Parameters
        ----------
        source:
            arXiv paper ID (e.g. ``2407.21783``) or path to a local PDF.
        force_llm:
            When True, run the LLM extractor unconditionally alongside table
            parsing.  When False (default) and ``OPENROUTER_API_KEY`` is set,
            the LLM extractor still runs as the **primary** extraction channel;
            table and prose parsing serve as validation and augmentation.  When
            no API key is available, falls back to table + prose only.

        Returns
        -------
        CoverageStats
            Per-paper extraction quality metrics suitable for batch aggregation.
        """
        stats = CoverageStats(arxiv_id=source)
        retrieved_timestamp = str(time.time())
        arxiv_id = _normalise_arxiv_id(source)
        stats.arxiv_id = arxiv_id
        eval_name = _make_eval_name(arxiv_id)

        print(f"\n[{arxiv_id}] fetching paper metadata from arXiv...")
        arxiv_meta = _fetch_arxiv_metadata(arxiv_id)
        paper_title = arxiv_meta.get("title", "")
        paper_authors: list[str] = arxiv_meta.get("authors", [])
        if paper_title:
            print(f"  title: {paper_title}")
        else:
            print("  title not found — using arXiv ID as fallback")
        if paper_authors:
            print(f"  authors: {len(paper_authors)} found")

        print(f"[{arxiv_id}] downloading PDF...")
        pdf_path = self._downloader.fetch(source)

        # ── Phase 1: Protocol + eval library detection (LLM-primary) ──
        protocol_map: dict[str, dict[str, str | int]] = {}
        shared_gemini_raw: dict[str, Any] | None = None

        print(f"[{arxiv_id}] detecting eval framework and protocol...")
        dynamic_keywords: set[str] = set()
        if OPENROUTER_API_KEY:
            # Run protocol extraction and benchmark identification sequentially
            # to avoid hitting rate limits from parallel large requests.
            gemini_raw = self.gemini_extractor.extract_raw(pdf_path, arxiv_id)
            bench_info = self.gemini_extractor.identify_benchmarks(pdf_path, arxiv_id)

            if self.gemini_extractor.last_call_failed:
                shared_gemini_raw = None
                eval_library = _detect_eval_library(pdf_path)
            else:
                eval_library_name = self.gemini_extractor.get_eval_library(gemini_raw)
                eval_library = EvalLibrary(name=eval_library_name, version="unknown")
                shared_gemini_raw = gemini_raw
                # Diagnostic: show protocol extraction summary
                dp = gemini_raw.get("default_protocol") or {}
                bc = dp.get("benchmark_configs") or {}
                spb = dp.get("shots_per_benchmark") or {}
                n_bench = len(set(bc.keys()) | set(spb.keys()))
                n_overrides = len(gemini_raw.get("model_overrides") or [])
                print(f"  protocol: {n_bench} benchmark configs, {n_overrides} model overrides")
                if dp.get("temperature") is not None:
                    print(f"  default temperature: {dp['temperature']}")
                if dp.get("scoring_method"):
                    print(f"  default scoring: {dp['scoring_method']}")

            dynamic_keywords = {kw.lower() for kw in bench_info.get("known_benchmarks", [])}
            if dynamic_keywords:
                print(f"  LLM-identified benchmarks: {', '.join(sorted(dynamic_keywords))}")
        else:
            shared_gemini_raw = None
            eval_library = _detect_eval_library(pdf_path)

        if shared_gemini_raw is not None:
            default_protocol = shared_gemini_raw.get("default_protocol") or {}
            override_entries = shared_gemini_raw.get("model_overrides") or []
            # Count benchmarks from both new (benchmark_configs) and old (shots_per_benchmark) formats
            bench_configs = default_protocol.get("benchmark_configs") or {}
            shots_per_bench = default_protocol.get("shots_per_benchmark") or {}
            default_count = len(set(bench_configs.keys()) | set(shots_per_bench.keys()))
            override_count = sum(
                len(
                    set((override.get("benchmark_configs") or {}).keys())
                    | set((override.get("shots_per_benchmark") or {}).keys())
                )
                for override in override_entries
            )
            stats.protocol_found = default_count + override_count
            if stats.protocol_found == 0:
                stats.protocol_missing = 1
        elif protocol_map:
            stats.protocol_found = sum(1 for key in protocol_map if key != "__default__")
        else:
            stats.protocol_missing = 1

        # ── Phase 2: LLM score extraction (primary when API key available) ──
        llm_extracted: list[dict] = []
        has_llm = self._llm_fallback is not None and OPENROUTER_API_KEY
        if has_llm:
            print(f"[{arxiv_id}] LLM score extraction (primary)...")
            llm_extracted = self._llm_fallback.extract(pdf_path, arxiv_id)
            stats.llm_data_points = len(llm_extracted)
            print(
                f"  LLM extracted {len(llm_extracted)} data point(s) "
                f"({len({it['model'] for it in llm_extracted})} models, "
                f"{len({it['benchmark'] for it in llm_extracted})} benchmarks)"
            )

        # ── Phase 3: Table parsing (validation + augmentation) ──
        print(f"[{arxiv_id}] extracting tables ({pdf_path.name})...")
        try:
            raw_tables = self._extractor.extract(pdf_path)
        except Exception as exc:
            msg = str(exc)
            if "SSL" in msg or "huggingface" in msg.lower() or "MaxRetryError" in msg:
                print(
                    f"  ERROR: Docling table extraction failed (SSL/network).\n"
                    f"  Set $env:HF_HUB_OFFLINE=1 before running the pipeline.\n"
                    f"  Continuing with LLM extraction only...",
                    file=sys.stderr,
                )
                raw_tables = []
            else:
                raise
        stats.tables_scanned = len(raw_tables)
        print(f"  found {len(raw_tables)} tables across all pages")

        _has_keyword_re = re.compile(
            "|".join(re.escape(kw) for kw in _BENCHMARK_KEYWORDS), re.IGNORECASE
        )

        table_extracted: list[dict] = []
        for page_num, table, context_text in raw_tables:
            table_text = " ".join(cell for row in table for cell in row if cell)
            has_kw = bool(_has_keyword_re.search(table_text)) or bool(
                _has_keyword_re.search(context_text)
            )

            # Log explicit skip reasons for contamination/non-results tables
            _ctx_lower = context_text.lower()
            _hdr_lower = " ".join(cell.lower() for cell in table[0] if cell) if table else ""
            _skip_reason = ""
            for _sig in _SKIP_TABLE_CAPTIONS:
                if _sig in _ctx_lower:
                    _skip_reason = f"caption matches: '{_sig}'"
                    break
            if not _skip_reason:
                for _sig in _SKIP_TABLE_SIGNALS:
                    if _sig in _ctx_lower or _sig in _hdr_lower:
                        _skip_reason = f"signal '{_sig}' in caption/header"
                        break
            if _skip_reason:
                print(
                    f"  page {page_num}: non-results table skipped "
                    f"({len(table)} rows) — {_skip_reason}"
                )
                continue

            if self._parser.is_results_table(table, context_text, dynamic_keywords=dynamic_keywords):
                confidence = self._parser.extraction_confidence(table, context_text, dynamic_keywords=dynamic_keywords)
                items = self._parser.parse(table, context_text, dynamic_keywords=dynamic_keywords)
                table_id = _extract_table_id(context_text)
                table_caption = _clean_caption(context_text)
                for item in items:
                    item["_extraction_confidence"] = confidence
                    item["page_number"] = page_num
                    item["table_id"] = table_id
                    item["table_caption"] = table_caption
                table_extracted.extend(items)
                stats.results_tables_accepted += 1
                if self._parser._last_relative_table:
                    stats.relative_tables += 1
                stats.section_header_rows_skipped += self._parser._last_section_header_rows_skipped
                print(
                    f"  page {page_num}: results table accepted "
                    f"({len(table)} rows, {len(items)} data points, "
                    f"confidence={confidence})"
                )
                if len(items) == 0 and len(table) >= 5:
                    # Log diagnostic for large accepted tables that yield nothing
                    _hdr = " | ".join(table[0][:6])
                    _r1 = " | ".join(table[1][:6]) if len(table) > 1 else ""
                    print(
                        f"    [diag] 0 data from {len(table)}-row table; "
                        f"header: {_hdr[:120]}",
                        file=sys.stderr,
                    )
                    print(
                        f"    [diag] row 1: {_r1[:120]}",
                        file=sys.stderr,
                    )
            elif has_kw:
                stats.density_rejected += 1
                print(
                    f"  page {page_num}: table has benchmark keywords "
                    f"but failed density check ({len(table)} rows) — skipped"
                )

        stats.table_data_points = len(table_extracted)

        # ── Phase 4: Prose extraction (always runs — lightweight regex) ──
        prose_new: list[dict] = []
        if self._prose_extractor is not None:
            print(f"[{arxiv_id}] scanning prose and figure captions...")
            prose_raw = self._prose_extractor.extract(pdf_path, arxiv_id)
            # Deduplicate against table
            table_key_set: set[tuple[str, str]] = {
                _merge_key(it["model"], it["benchmark"])
                for it in table_extracted
            }
            prose_new = [
                it for it in prose_raw
                if _merge_key(it["model"], it["benchmark"])
                not in table_key_set
            ]
            stats.prose_data_points = len(prose_new)

        # ── Phase 5: Merge — LLM-primary, table validates/augments ──
        if has_llm and llm_extracted:
            # LLM is the primary source.  Table-parsed results with HIGH
            # confidence override LLM results for the same (model, benchmark)
            # pair.  All other table results augment the LLM output (fill
            # gaps for benchmarks the LLM missed).
            llm_keys: dict[tuple[str, str], dict] = {
                _merge_key(it["model"], it["benchmark"]): it
                for it in llm_extracted
            }
            table_high_conf = [
                it for it in table_extracted
                if it.get("_extraction_confidence") in ("high", "medium")
            ]
            table_rest = [
                it for it in table_extracted
                if it.get("_extraction_confidence") not in ("high", "medium")
            ]

            # Pre-group table entries by merge key to detect score variants
            # (e.g. thinking vs non-thinking from different tables).
            from collections import defaultdict as _defaultdict
            _table_by_key: dict[tuple[str, str], list[dict]] = _defaultdict(list)
            for t_item in table_high_conf:
                key = _merge_key(t_item["model"], t_item["benchmark"])
                _table_by_key[key].append(t_item)

            # For each key, identify distinct score clusters and pick primary.
            overrides = 0
            variant_count = 0
            # primary_for_key: best table entry per key (overrides LLM)
            # variants_for_key: additional entries with significantly different scores
            primary_for_key: dict[tuple[str, str], dict] = {}
            variants_for_key: dict[tuple[str, str], list[dict]] = {}
            for key, t_items in _table_by_key.items():
                # Collect unique scores (rounded to avoid float noise)
                seen_scores: dict[float, dict] = {}
                for t in t_items:
                    try:
                        s = round(float(t.get("score", 0)), 6)
                        if s not in seen_scores:
                            seen_scores[s] = t
                    except (TypeError, ValueError):
                        pass
                if not seen_scores:
                    continue
                # Pick primary (highest score; for tie, first seen)
                sorted_items = sorted(
                    seen_scores.values(),
                    key=lambda x: float(x.get("score", 0)),
                    reverse=True,
                )
                primary = sorted_items[0]
                primary_for_key[key] = primary
                # All other distinct scores that differ by >10% from primary
                # are score variants (likely thinking/non-thinking).
                variants = []
                primary_score = float(primary.get("score", 0))
                for alt in sorted_items[1:]:
                    alt_score = float(alt.get("score", 0))
                    if primary_score and abs(primary_score - alt_score) / max(abs(primary_score), 0.01) > 0.10:
                        variants.append(alt)
                if variants:
                    variants_for_key[key] = variants

            # Override LLM entries with primary table entries
            for key, primary in primary_for_key.items():
                if key in llm_keys:
                    existing_score = llm_keys[key].get("score")
                    tbl_score = primary.get("score")
                    if existing_score is not None and tbl_score is not None:
                        try:
                            ex_f = float(existing_score)
                            tb_f = float(tbl_score)
                            diff = abs(ex_f - tb_f)
                            if diff > 0.5 or (ex_f and diff / max(abs(ex_f), 0.01) > 0.10):
                                print(
                                    f"  [cross-val] {primary['model']}\u00d7{primary['benchmark']}: "
                                    f"LLM={existing_score}, table={tbl_score} \u2014 "
                                    f"using table (high confidence)",
                                    file=sys.stderr,
                                )
                        except (TypeError, ValueError):
                            pass
                    llm_keys[key] = primary
                    overrides += 1

            # Start with (possibly overridden) LLM results
            merged: list[dict] = list(llm_keys.values())
            merged_key_set = set(llm_keys.keys())

            # Augment with high-confidence table results the LLM missed
            for key, primary in primary_for_key.items():
                if key not in merged_key_set:
                    merged.append(primary)
                    merged_key_set.add(key)

            # Add all score variants (thinking/non-thinking splits)
            for key, variants in variants_for_key.items():
                for v_item in variants:
                    merged.append(v_item)
                    variant_count += 1
                    print(
                        f"  [cross-val] {v_item['model']}\u00d7{v_item['benchmark']}: "
                        f"variant score={v_item.get('score')} "
                        f"(possible thinking/non-thinking split)",
                        file=sys.stderr,
                    )

            # Augment with remaining table results that the LLM missed
            for t_item in table_rest:
                key = _merge_key(t_item["model"], t_item["benchmark"])
                if key not in merged_key_set:
                    merged.append(t_item)
                    merged_key_set.add(key)

            # Augment with prose results that neither LLM nor table found
            for p_item in prose_new:
                key = _merge_key(p_item["model"], p_item["benchmark"])
                if key not in merged_key_set:
                    merged.append(p_item)
                    merged_key_set.add(key)

            extracted = merged
            if variant_count:
                print(
                    f"  [merge] {variant_count} variant score(s) kept "
                    f"(possible thinking/non-thinking splits)"
                )
            if overrides:
                print(
                    f"  [merge] {overrides} LLM result(s) overridden by "
                    f"high-confidence table data"
                )

        else:
            # No LLM available — fall back to table > prose merge (old behaviour)
            combined_so_far = table_extracted + prose_new
            # If table+prose yielded nothing and LLM extractor exists but
            # wasn't run yet (no API key), we can't help further.
            extracted = combined_so_far

        # Post-extraction sanity filter: drop citation-as-model, year-as-score, etc.
        extracted = _sanitize_extracted(extracted, known_benchmarks=dynamic_keywords)

        # Determine source tier(s) for the coverage report.
        sources_used: list[str] = []
        if has_llm and llm_extracted:
            sources_used.append("llm")
        if table_extracted:
            sources_used.append("table")
        if prose_new:
            sources_used.append("prose")
        stats.source = "+".join(sources_used) if sources_used else "none"

        stats.unique_models = len({item["model"] for item in extracted})
        stats.unique_benchmarks = len({item["benchmark"] for item in extracted})

        print(
            f"\n[{arxiv_id}] extraction summary:\n"
            f"  tables scanned:          {stats.tables_scanned}\n"
            f"  results tables accepted: {stats.results_tables_accepted}\n"
            f"  keyword-only rejects:    {stats.density_rejected}  "
            f"(potential false negatives — review manually)\n"
            f"  table data points:       {stats.table_data_points}\n"
            f"  prose data points (new): {stats.prose_data_points}\n"
            f"  llm data points:         {stats.llm_data_points}\n"
            f"  total data points:       {len(extracted)}\n"
            f"  unique models:           {stats.unique_models}\n"
            f"  unique benchmarks:       {stats.unique_benchmarks}\n"
            f"  relative tables:         {stats.relative_tables}\n"
            f"  protocol entries found:  {stats.protocol_found}\n"
            f"  section headers skipped: {stats.section_header_rows_skipped}\n"
            f"  source:                  {stats.source}"
        )

        if not extracted:
            print(f"[{arxiv_id}] no results found — skipping")
            return stats

        records = self._converter.convert(
            extracted, arxiv_id, retrieved_timestamp,
            paper_title=paper_title,
            paper_authors=paper_authors,
            eval_library=eval_library,
            gemini_raw=shared_gemini_raw,
            protocol_map=protocol_map,
        )
        print(f"[{arxiv_id}] converted {len(records)} model records")

        unknown_devs = [
            r["model_info"]["id"]
            for r in records
            if r.get("model_info", {}).get("developer") == "unknown"
        ]
        if unknown_devs:
            print(
                f"  [developer-inference] {len(unknown_devs)} model(s) have "
                f"developer='unknown' \u2014 verify these before submission:\n"
                + "\n".join(f"    {mid}" for mid in unknown_devs)
            )

        paths = self._writer.write(records, eval_name)
        stats.files_written = len(paths)
        print(f"[{arxiv_id}] written {len(paths)} files to data/{{benchmark}}/...")

        passed, failed = self._validate_written(paths, arxiv_id)
        stats.validation_passed = passed
        stats.validation_failed = failed

        return stats

    def _validate_written(self, paths: list[Path], arxiv_id: str) -> tuple[int, int]:
        """Validate each written JSON file against eval.schema.json.

        Returns
        -------
        tuple[int, int]
            ``(passed, failed)`` counts.  Failures are non-fatal so a batch
            run can continue; the caller should inspect warnings and fix
            source data before submitting a PR.
        """
        import json as _json
        from jsonschema.validators import validator_for

        schema_path = _ROOT / "eval.schema.json"
        if not schema_path.exists():
            print(f"  [validation] schema not found at {schema_path} — skipping")
            return 0, 0

        with schema_path.open() as fh:
            schema = _json.load(fh)
        validator_cls = validator_for(schema)
        validator = validator_cls(schema)

        passed = 0
        failed = 0
        for path in paths:
            try:
                with path.open() as fh:
                    instance = _json.load(fh)
                validator.validate(instance)
                passed += 1
            except Exception as exc:
                failed += 1
                print(f"  [validation] FAIL {path.name}: {exc}")

        status = "all passed" if failed == 0 else f"{failed} FAILED"
        print(
            f"[{arxiv_id}] schema validation: {passed} passed, {failed} failed — {status}"
        )
        return passed, failed


# ---------------------------------------------------------------------------
# pure helpers — no side effects
# ---------------------------------------------------------------------------




def _build_pipeline(
    llm_fallback: LLMFallbackExtractor | None = None,
) -> PaperExtractionPipeline:
    """construct the default pipeline with all default implementations."""
    return PaperExtractionPipeline(
        downloader=PDFDownloader(),
        table_extractor=TableExtractor(),
        table_parser=ResultsTableParser(),
        converter=PaperConverter(),
        writer=PaperWriter(),
        llm_fallback=llm_fallback,
        prose_extractor=ProseExtractor(),
        protocol_extractor=EvalProtocolExtractor(),
    )


def main() -> None:
    """command-line entry point for the paper extraction pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="extract LLM evaluation results from arXiv PDFs to EEE schema"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--arxiv_id",
        metavar="ID",
        help="arXiv paper ID (e.g. 2407.21783)",
    )
    src.add_argument(
        "--pdf",
        metavar="PATH",
        help="path to a local PDF file",
    )
    src.add_argument(
        "--batch",
        metavar="FILE",
        help="text file with one arXiv ID per line",
    )
    src.add_argument(
        "--convert",
        metavar="PATH",
        help=(
            "convert evaluation framework logs (lm-eval, Inspect, HELM) "
            "to EEE schema. Pass a log file or directory."
        ),
    )
    parser.add_argument(
        "--framework",
        choices=["lm_eval", "inspect", "helm", "auto"],
        default="auto",
        help="evaluation framework (for --convert mode, default: auto-detect)",
    )
    parser.add_argument(
        "--llm-fallback",
        action="store_true",
        default=False,
        help=(
            "activate the LLM score extractor as the primary extraction "
            "channel. When set and OPENROUTER_API_KEY is available, the LLM "
            "extracts all scores first, then table parsing validates and "
            "augments the LLM output. Highly recommended for papers with "
            "non-standard table layouts or sparse prose-embedded results."
        ),
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        default=False,
        help=(
            "disable LLM extraction entirely; rely only on table parsing "
            "and prose regex. Useful when no API key is available or for "
            "testing the deterministic pipeline."
        ),
    )
    parser.add_argument(
        "--llm-model",
        metavar="MODEL",
        default="anthropic/claude-haiku-4.5",
        help="OpenRouter model to use for LLM extraction (default: anthropic/claude-haiku-4.5)",
    )
    args = parser.parse_args()

    # ── Converter mode: delegate to eval_converters ──
    if args.convert:
        from scripts.convert_eval_logs import main as _convert_main
        import sys as _sys
        convert_argv = ["--log_path", args.convert, "--framework", args.framework]
        _sys.argv = ["convert_eval_logs.py"] + convert_argv
        _convert_main()
        return

    # LLM extractor is the primary channel when API key is available,
    # unless explicitly disabled with --no-llm.  --llm-fallback is kept
    # for backwards compatibility and is equivalent to the new default.
    llm_fallback: LLMFallbackExtractor | None = None
    if not args.no_llm and OPENROUTER_API_KEY:
        llm_fallback = LLMFallbackExtractor(model=args.llm_model)
    elif args.llm_fallback:
        # explicit flag — construct even if API key check would fail
        # (LLMFallbackExtractor will raise at call time)
        llm_fallback = LLMFallbackExtractor(model=args.llm_model)

    pipeline = _build_pipeline(llm_fallback=llm_fallback)
    all_stats: list[CoverageStats] = []

    use_llm = llm_fallback is not None

    if args.arxiv_id:
        stats = pipeline.run(args.arxiv_id, force_llm=use_llm)
        all_stats.append(stats)
    elif args.pdf:
        stats = pipeline.run(args.pdf, force_llm=use_llm)
        all_stats.append(stats)
    elif args.batch:
        batch_file = Path(args.batch)
        if not batch_file.exists():
            print(f"batch file not found: {batch_file}", file=sys.stderr)
            sys.exit(1)
        ids = [
            line.strip()
            for line in batch_file.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        print(f"processing {len(ids)} arXiv papers from {args.batch}...")
        for i, arxiv_id in enumerate(ids):
            try:
                stats = pipeline.run(arxiv_id, force_llm=use_llm)
                all_stats.append(stats)
            except Exception as exc:
                print(f"  error processing {arxiv_id}: {exc}")
                all_stats.append(CoverageStats(arxiv_id=arxiv_id, source="none"))

            # Rate limit buffer between papers
            if i < len(ids) - 1:
                time.sleep(5)

        # Write machine-readable coverage report for batch runs.
        report_dir = Path("scripts/scrapers/raw")
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"coverage_report_{int(time.time())}.json"
        report = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "batch_file": str(args.batch),
            "total_papers": len(all_stats),
            "papers_with_results": sum(1 for s in all_stats if s.total_data_points > 0),
            "papers_with_no_results": sum(1 for s in all_stats if s.total_data_points == 0),
            "total_files_written": sum(s.files_written for s in all_stats),
            "total_validation_failed": sum(s.validation_failed for s in all_stats),
            # source is now a '+'-joined string of tiers used, e.g. "table+prose",
            # so we count substring membership rather than exact equality.
            "source_breakdown": {
                "table_only": sum(1 for s in all_stats if s.source == "table"),
                "prose_only": sum(1 for s in all_stats if s.source == "prose"),
                "llm_only": sum(1 for s in all_stats if s.source == "llm"),
                "has_table": sum(1 for s in all_stats if "table" in s.source),
                "has_prose": sum(1 for s in all_stats if "prose" in s.source),
                "has_llm": sum(1 for s in all_stats if "llm" in s.source),
                "none": sum(1 for s in all_stats if s.source == "none"),
            },
            "per_paper": [asdict(s) for s in all_stats],
        }
        with report_path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"\ncoverage report written to {report_path}")

    total = sum(s.files_written for s in all_stats)
    print(f"\ntotal files written: {total}")


if __name__ == "__main__":
    main()
