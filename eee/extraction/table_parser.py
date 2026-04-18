"""Results table parser — heuristic identification and score parsing."""
from __future__ import annotations

import re
from typing import Any

from .constants import (
    _BENCHMARK_KEYWORDS,
    _BENCHMARK_KEYWORDS_STRONG,
    _BENCHMARK_KEYWORDS_WEAK,
    _DEVELOPER_PATTERNS,
    _NON_MODEL_ROW_PATTERNS,
    _RELATIVE_TABLE_PATTERNS,
    _SECTION_HEADER_PATTERNS,
    _SKIP_TABLE_CAPTIONS,
    _SKIP_TABLE_SIGNALS,
)
from .helpers import _clean_cell, _parse_numeric, _parse_numeric_with_pct, _is_separator
from .docling_parser import _try_expand_merged_model_columns

class ResultsTableParser:
    """identify and parse results tables from raw extracted table data."""

    def __init__(self) -> None:
        self._last_relative_table = False
        self._last_relative_to: str | None = None
        self._last_section_header_rows_skipped = 0

    def is_results_table(self, table: list[list[str]], context: str = "", dynamic_keywords: set[str] | None = None) -> bool:
        """Return True if *table* looks like a model evaluation results table.

        Three checks must all pass:

        1. **Benchmark keyword** — a known benchmark name appears somewhere in
           the table *or in the context text above it* (checked header-first,
           then full table, then context).
        2. **Numeric density** — at least 25 % of non-empty body cells, or at
           least 5 cells, contain a parseable score (int, float, or %).
        3. **Minimum size** — at least header row + 2 data rows, ensuring we
           don't flag single-row summary lines.

        Parameters
        ----------
        table:
            Normalised table rows (all cells are str, None already replaced).
        context:
            Plain text extracted from the region above the table on the same
            page (e.g. caption, section header).  When the table cells
            themselves carry no benchmark keyword, this text is the last
            resort before the table is rejected — it catches cases like a
            multi-part results block where the benchmark name only appears in
            the preceding paragraph.
        """
        # check 0: reject non-results tables (contamination analyses, etc.)
        # These tables contain benchmark names and numeric data but report
        # contamination deltas or performance-gain estimates, not actual scores.
        caption_lower = context.lower()
        header_lower = " ".join(cell.lower() for cell in table[0] if cell) if table else ""
        for sig in _SKIP_TABLE_CAPTIONS:
            if sig in caption_lower:
                return False
        for sig in _SKIP_TABLE_SIGNALS:
            if sig in caption_lower or sig in header_lower:
                return False

        # check 3: minimum size
        if len(table) < 3:
            return False

        # check 1: benchmark keyword presence — two-tier strategy:
        #   • one *strong* match (specific benchmark name) is sufficient;
        #   • generic metric terms (_BENCHMARK_KEYWORDS_WEAK) require ≥2
        #     distinct matches to reduce false positives from non-eval tables
        #     that happen to mention "accuracy" or "f1" (e.g. a confusion
        #     matrix or a hyperparameter ablation table).
        # Scan order: header row → second row → full table → context text.
        def _has_eval_signal(text: str) -> bool:
            effective_strong = _BENCHMARK_KEYWORDS_STRONG | (dynamic_keywords or set())
            if any(kw in text for kw in effective_strong):
                return True
            weak_hits = sum(1 for kw in _BENCHMARK_KEYWORDS_WEAK if kw in text)
            return weak_hits >= 2

        header_text = " ".join(cell.lower() for cell in table[0] if cell)
        has_benchmark = _has_eval_signal(header_text)
        if not has_benchmark and len(table) > 1:
            second_text = " ".join(c.lower() for c in table[1] if c)
            has_benchmark = _has_eval_signal(second_text)
        # slow path: scan the entire table (handles multi-row headers and
        # tables where benchmark names appear in the first column)
        if not has_benchmark:
            all_text = " ".join(
                cell.lower() for row in table for cell in row if cell
            )
            has_benchmark = _has_eval_signal(all_text)
        # last resort: check the caption / section header above the table.
        # Accept only a strong keyword here to avoid false positives from
        # sections like "Experimental Setup" that mention metric names in
        # passing (weak matches require 2 hits, which is strict enough).
        if not has_benchmark and context:
            has_benchmark = _has_eval_signal(context.lower())
        if not has_benchmark:
            return False

        # check 2: numeric density — also match percentages and negatives
        # Use _parse_numeric to mirror the actual parsing logic, which handles
        # footnote markers (85.2†), bold residue (\textbf{85}), parenthesised
        # values, etc. that a simple regex would miss.
        numeric_cells = 0
        non_empty_cells = 0
        for row in table[1:]:
            for cell in row:
                stripped = cell.strip()
                if stripped:
                    non_empty_cells += 1
                    if _parse_numeric(stripped) is not None:
                        numeric_cells += 1

        if non_empty_cells == 0:
            return False
        # require either an absolute floor of 5 numeric cells (small tables)
        # or 25 % density (larger tables).  For large tables (8+ body rows)
        # with confirmed benchmark keywords, use a lower threshold (3 cells
        # or 15% density) to catch wide post-training results tables where
        # Docling sometimes fails to parse all numeric cells correctly.
        density = numeric_cells / non_empty_cells
        n_body_rows = len(table) - 1
        if n_body_rows >= 8:
            # Large table with benchmark keywords already confirmed (check 1)
            if not (numeric_cells >= 3 or density >= 0.15):
                return False
        else:
            if not (numeric_cells >= 5 or density >= 0.25):
                return False

        # check 4: reject ablation tables — first column dominated by ablation
        # marker language rather than distinct model names from different orgs.
        # Ablation tables pass checks 1–3 but generate EEE records where every
        # entry describes the same underlying model under ablation conditions,
        # making the schema data ambiguous to query.
        if self._is_ablation_table(table):
            return False

        # check 5: reject architecture/configuration tables — these list
        # layer counts, head counts, hidden dims etc. which pass density
        # checks but are not evaluation results.
        _ARCH_TABLE_KEYWORDS = (
            "model architecture", "architecture of", "model configuration",
            "hyperparameter", "training configuration", "model specification",
            "hidden size", "num layers", "num heads",
        )
        # caption_lower and header_lower already computed in check 0
        # Only apply caption-based architecture rejection when the table body
        # does NOT contain a strong benchmark keyword.  This prevents false
        # rejections of results tables whose context (preceding prose) happens
        # to mention "model architecture" or similar terms.
        table_body_text = " ".join(
            cell.lower() for row in table for cell in row if cell
        )
        _table_has_bench_kw = any(kw in table_body_text for kw in (_BENCHMARK_KEYWORDS_STRONG | (dynamic_keywords or set())))
        if not _table_has_bench_kw and any(kw in caption_lower for kw in _ARCH_TABLE_KEYWORDS):
            return False
        if any(kw in header_lower for kw in ("layers", "heads", "hidden dim", "hidden size", "vocab size", "params", "parameters")):
            # If header mentions architecture columns AND lacks benchmark names,
            # it's an architecture table
            arch_cols = sum(1 for kw in ("layers", "heads", "hidden", "vocab", "params", "embed") if kw in header_lower)
            if arch_cols >= 2:
                return False

        return True

    @staticmethod
    def _is_ablation_table(table: list[list[str]]) -> bool:
        """Return True when the first column looks like an ablation study.

        Ablation tables compare variants of a single model (e.g. "w/o layer
        norm", "base + FT", "+ DPO") rather than multiple independent models.
        Heuristic: if ≥ 40 % of non-empty first-column cells contain an
        ablation marker token, classify as ablation.  The 40 % threshold
        tolerates one legitimate baseline row while catching tables that are
        predominantly ablation rows.
        """
        _ABLATION_MARKERS: frozenset[str] = frozenset({
            "w/o", "without", "ablat", "no ", "+ ", " + ",
            "only", "ours w", "base model", "full model",
            "remove", "drop ", "variant",
        })
        first_col = [
            row[0].lower().strip()
            for row in table[1:]
            if row and row[0].strip()
        ]
        if not first_col:
            return False
        ablation_count = sum(
            1 for v in first_col
            if any(
                marker in v
                for marker in _ABLATION_MARKERS
                # skip the "no " marker when the cell mentions "prompt"
                if not (marker == "no " and "prompt" in v)
            )
        )
        return ablation_count / len(first_col) >= 0.40

    def extraction_confidence(self, table: list[list[str]], context: str = "", dynamic_keywords: set[str] | None = None) -> str:
        """Return a confidence tier for how reliably this table was extracted.

        Calibration
        -----------
        'high'   — ≥1 strong benchmark keyword in the table body AND numeric
                   density ≥ 50 %.  The table is unambiguously a results table.
        'medium' — ≥1 strong keyword with density 25–50 %, OR the table only
                   matched via the caption / context rather than its own cells.
        'low'    — Accepted on weak-keyword evidence (≥2 weak matches) or on
                   the numeric-count floor only (≥5 cells but < 25 % density).

        A 'low' result is not an error — the table still passed
        `is_results_table` and is included — but benchmark→score assignments
        should be treated with more scepticism and manually reviewed before
        submission.
        """
        all_text = " ".join(cell.lower() for row in table for cell in row if cell)
        context_lower = context.lower()

        effective_strong = _BENCHMARK_KEYWORDS_STRONG | (dynamic_keywords or set())
        strong_in_body = sum(
            1 for kw in effective_strong if kw in all_text
        )
        strong_in_context_only = (
            strong_in_body == 0
            and any(kw in context_lower for kw in effective_strong)
        )

        # recompute numeric density (same logic as is_results_table)
        numeric_cells = 0
        non_empty_cells = 0
        for row in table[1:]:
            for cell in row:
                stripped = cell.strip()
                if stripped:
                    non_empty_cells += 1
                    if _parse_numeric(stripped) is not None:
                        numeric_cells += 1
        density = (numeric_cells / non_empty_cells) if non_empty_cells > 0 else 0.0

        if strong_in_body >= 1 and density >= 0.50:
            return "high"
        if (strong_in_body >= 1 and density >= 0.25) or strong_in_context_only:
            return "medium"
        return "low"

    def is_relative_table(self, table: list[list[str]], context: str = "") -> tuple[bool, str | None]:
        """Detect tables that report relative/delta scores."""
        sign_prefixed = 0
        numeric_cells = 0
        for row in table[1:]:
            for cell in row:
                stripped = cell.strip()
                if not stripped:
                    continue
                if re.fullmatch(r"[+-]?\d+(?:\.\d+)?%?", stripped):
                    numeric_cells += 1
                    if stripped.startswith("+") or stripped.startswith("-"):
                        sign_prefixed += 1
        sign_ratio = (sign_prefixed / numeric_cells) if numeric_cells else 0.0
        context_lower = context.lower()
        has_relative_caption = any(phrase in context_lower for phrase in _RELATIVE_TABLE_PATTERNS)
        if sign_ratio > 0.50 or has_relative_caption:
            return True, self._extract_relative_to(context)
        return False, None

    def _extract_relative_to(self, context: str) -> str | None:
        patterns = (
            re.compile(r"relative to\s+([^\.;:,]+)", re.IGNORECASE),
            re.compile(r"compared to baseline\s+([^\.;:,]+)", re.IGNORECASE),
            re.compile(r"improvement over\s+([^\.;:,]+)", re.IGNORECASE),
            re.compile(r"vs\.?\s+the final data mixture", re.IGNORECASE),
        )
        for pattern in patterns:
            match = pattern.search(context)
            if match:
                if match.lastindex:
                    return match.group(1).strip()
                return match.group(0).strip()
        return None

    def _is_section_header_row(self, row: list[str]) -> bool:
        non_empty = [cell.strip() for cell in row if cell and cell.strip()]
        if len(non_empty) == 1:
            lowered = non_empty[0].lower()
            return any(pattern in lowered for pattern in _SECTION_HEADER_PATTERNS)
        return False

    # Patterns whose section header should be appended to model names to
    # disambiguate rows that would otherwise produce identical model entries.
    _THINKING_MODE_PATTERNS: tuple[str, ...] = (
        "thinking mode",
        "non-thinking mode",
    )

    def _get_thinking_mode_suffix(self, row: list[str]) -> str | None:
        """If *row* is a thinking-mode section header, return a suffix string."""
        non_empty = [cell.strip() for cell in row if cell and cell.strip()]
        if len(non_empty) != 1:
            return None
        lowered = non_empty[0].lower()
        if "non-thinking" in lowered or "non thinking" in lowered:
            return "(Non-Thinking)"
        if "thinking" in lowered and "non" not in lowered:
            return "(Thinking)"
        return None

    def _model_signal_count(self, value: str) -> int:
        lowered = value.lower()
        count = 0
        for pattern, _developer in _DEVELOPER_PATTERNS:
            if len(pattern) < 3:
                continue
            if lowered.startswith(pattern) or f" {pattern}" in lowered or f"-{pattern}" in lowered:
                count += 1
        return count

    def _looks_like_model_name(self, value: str) -> bool:
        lowered = value.lower().strip()
        if not lowered:
            return False
        if any(pattern.search(lowered) for pattern in _NON_MODEL_ROW_PATTERNS):
            return False
        # Check against both hardcoded and dynamically-detected benchmark names
        effective_bench = _BENCHMARK_KEYWORDS_STRONG | getattr(self, '_dynamic_keywords', set())
        if any(keyword in lowered for keyword in effective_bench):
            return False
        if self._model_signal_count(lowered) >= 2 and len(lowered.split()) > 3:
            return False
        if any(pattern in lowered for pattern in _SECTION_HEADER_PATTERNS):
            return False
        # Reject overly long strings — real model names are rarely > 60 chars
        if len(value.strip()) > 60:
            return False
        # Reject strings with citation markers (et al., year references)
        if re.search(r"et\s+al\.?", lowered):
            return False
        if "/" in value or any(ch.isdigit() for ch in value) or "(" in value or ")" in value:
            return True
        if self._model_signal_count(lowered) >= 1:
            return True
        return False

    def parse(
        self, table: list[list[str]], context: str = "",
        dynamic_keywords: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """parse *table* into a list of {model, benchmark, score} dicts.

        handles both orientations (models-as-rows and models-as-columns).
        Also attempts to collapse multi-row headers into a single header row
        when the first row is mostly empty (common in LaTeX multicolumn tables).
        """
        # Store dynamic keywords for use by _looks_like_model_name()
        self._dynamic_keywords = dynamic_keywords or set()
        results: list[dict[str, Any]] = []
        self._last_section_header_rows_skipped = 0
        self._last_relative_table, self._last_relative_to = self.is_relative_table(table, context)

        if not table or not table[0]:
            return results

        # Try expanding merged model columns FIRST — _collapse_multirow_header
        # would absorb metadata rows (Architecture, # Params) into the header
        # and destroy the model-name information we need for splitting.
        expanded = _try_expand_merged_model_columns(table)
        if expanded is not None:
            # Expansion already rebuilt the table into proper models-as-columns
            # form.  Skip the orientation heuristic — substring keyword checks
            # like "em" can false-positive on model names (e.g. "Gemini").
            table = expanded
            results = self._parse_models_as_columns(table)
        else:
            table = self._collapse_multirow_header(table)

            header = [_clean_cell(c) for c in table[0]]

            # determine orientation:
            # models-as-rows: first column contains model names, header has benchmarks
            # models-as-columns: first row is benchmarks, first column is metric names

            effective_bench_kw = _BENCHMARK_KEYWORDS | self._dynamic_keywords
            benchmarks_in_header = [
                h for h in header if any(kw in h.lower() for kw in effective_bench_kw)
            ]

            # Guard: if a header cell matches benchmark keywords but is
            # actually a model name containing a benchmark word (e.g.
            # "ARC-Instruct-7B"), verify it doesn't also look like a model.
            if benchmarks_in_header:
                confirmed = [
                    h for h in benchmarks_in_header
                    if not self._looks_like_model_name(h)
                ]
                if not confirmed:
                    benchmarks_in_header = []

            if benchmarks_in_header:
                results = self._parse_models_as_rows(header, table[1:])
            else:
                results = self._parse_models_as_columns(table)

        score_type = "relative" if self._last_relative_table else "absolute"
        for item in results:
            item.setdefault("score_type", score_type)
            item.setdefault("relative_to", self._last_relative_to)

        return results

    @staticmethod
    def _collapse_multirow_header(table: list[list[str]]) -> list[list[str]]:
        """merge row 0 and row 1 into a single header when row 0 is sparse.

        Many LaTeX-generated PDF tables have a top row with group labels and
        an immediately following subheader row with benchmark names.  When the
        first row has more than half its cells empty, we join the two rows
        cell-by-cell so downstream parsing sees a single enriched header.

        Smart joining: when row0 has a group label (e.g. "MMLU") spanning
        multiple columns and row1 has sub-benchmarks (e.g. "Pro", "Redux"),
        the merge produces "MMLU Pro", "MMLU Redux" — propagating the group
        label to each sub-column.
        """
        collapsed = table
        for _ in range(4):
            if len(collapsed) < 2:
                return collapsed
            row0 = collapsed[0]
            row1 = collapsed[1]
            header_text = " ".join(cell.lower() for cell in row0 if cell)

            # If row 0 already has benchmark keywords, it might be a real
            # header — but only if it's NOT a sparse group-label row.
            # A sparse row (>50 % cells empty) with benchmark keywords like
            # "MMLU" is a group-label row that still needs collapsing with
            # the sub-header row below it.
            non_empty_r0 = sum(1 for c in row0 if c and c.strip())
            is_sparse = non_empty_r0 < max(len(row0), 1) * 0.5
            if any(kw in header_text for kw in _BENCHMARK_KEYWORDS):
                if not is_sparse:
                    return collapsed
                # sparse group-label row — continue to collapse
            merged: list[str] = []

            # Propagate group labels: if row0 has a non-empty cell followed
            # by empty cells, those empty cells share the same group label.
            # e.g. row0 = ["Model", "MMLU", "", "", "Code", ""]
            #   →  groups = ["Model", "MMLU", "MMLU", "MMLU", "Code", "Code"]
            propagated_row0: list[str] = []
            last_label = ""
            for i in range(max(len(row0), len(row1))):
                c0 = row0[i].strip() if i < len(row0) else ""
                if c0:
                    last_label = c0
                    propagated_row0.append(c0)
                else:
                    propagated_row0.append(last_label)

            for i in range(max(len(propagated_row0), len(row1))):
                c0 = propagated_row0[i] if i < len(propagated_row0) else ""
                c1 = row1[i].strip() if i < len(row1) else ""
                if c0 and c1:
                    # Avoid "MMLU MMLU" duplication when both rows have
                    # the same text.
                    if c1.lower().startswith(c0.lower()):
                        merged.append(c1)
                    elif c0.lower().startswith(c1.lower()):
                        merged.append(c0)
                    else:
                        merged.append(f"{c0} {c1}")
                else:
                    merged.append(c0 or c1)
            collapsed = [merged] + collapsed[2:]
        return collapsed

    def _parse_models_as_rows(
        self, header: list[str], body: list[list[str]]
    ) -> list[dict]:
        """models are rows, benchmarks are columns."""
        results: list[dict] = []
        modality_suffixes = {
            "pretrained", "finetuned", "instruct", "chat", "rlhf",
            "sft", "base", "fine-tuned", "pre-trained",
        }
        # first column is model name; rest are benchmark columns
        bench_names = [_clean_cell(h) for h in header[1:]]
        expected_cols = len(bench_names)
        total_data_rows = 0
        misaligned_rows = 0
        current_mode_suffix: str | None = None
        for row in body:
            if not row or not row[0]:
                continue
            # Check for thinking mode section headers BEFORE the generic check
            mode_suffix = self._get_thinking_mode_suffix(row)
            if mode_suffix is not None:
                current_mode_suffix = mode_suffix
                self._last_section_header_rows_skipped += 1
                continue
            if self._is_section_header_row(row):
                # Non-thinking-mode section header (e.g. "Math", "Reasoning")
                # within a thinking mode section should NOT reset the mode.
                # Only a thinking/non-thinking header changes the current mode.
                # If we're not in any thinking context, it's just a category
                # header — leave mode as None.
                self._last_section_header_rows_skipped += 1
                continue
            model_name = _clean_cell(row[0])
            model_parts = model_name.split()
            if model_parts and model_parts[-1].lower() in modality_suffixes:
                model_name = " ".join(model_parts[:-1])
            # Handle "Thinking Mode <model>" prefix pattern (e.g. Qwen3 paper)
            _inline_thinking_mode: str | None = None
            if model_name.startswith("Thinking Mode "):
                _inline_thinking_mode = "(Thinking)"
                model_name = model_name[len("Thinking Mode "):]
            elif model_name.startswith("Non-Thinking Mode "):
                _inline_thinking_mode = "(Non-Thinking)"
                model_name = model_name[len("Non-Thinking Mode "):]
            if not model_name or _is_separator(model_name) or not self._looks_like_model_name(model_name):
                continue
            # Append thinking mode suffix to disambiguate models.
            # Priority: inline prefix > section header context.
            effective_mode = _inline_thinking_mode or current_mode_suffix
            if effective_mode:
                model_name = f"{model_name} {effective_mode}"
            # A row should have 1 model-name cell + expected_cols score cells.
            # Fewer columns means pdfplumber collapsed a \multicolumn span,
            # silently shifting all subsequent scores one or more positions left.
            actual_data_cols = len(row) - 1
            total_data_rows += 1
            if actual_data_cols != expected_cols:
                misaligned_rows += 1
            for i, bench in enumerate(bench_names):
                if not bench:
                    continue
                cell_idx = i + 1
                if cell_idx >= len(row):
                    continue
                raw_cell = row[cell_idx].strip()
                # Detect merged multi-column cells: if the header contains
                # multiple space-separated benchmark names and the data cell
                # contains the same number of space-separated numeric values,
                # split them into individual benchmark-score pairs.
                bench_parts = bench.split()
                cell_parts = raw_cell.split()
                if len(bench_parts) > 1 and len(bench_parts) == len(cell_parts):
                    parsed_parts = [_parse_numeric_with_pct(p) for p in cell_parts]
                    if all(v is not None for v, _ in parsed_parts):
                        for bp, (sp, had_pct) in zip(bench_parts, parsed_parts):
                            results.append(
                                {
                                    "model": model_name,
                                    "benchmark": bp,
                                    "score": sp,
                                    "score_type": "absolute",
                                    "relative_to": None,
                                    "_had_pct": had_pct,
                                }
                            )
                        continue
                score, had_pct = _parse_numeric_with_pct(raw_cell)
                if score is None:
                    continue
                results.append(
                    {
                        "model": model_name,
                        "benchmark": bench,
                        "score": score,
                        "score_type": "absolute",
                        "relative_to": None,
                        "_had_pct": had_pct,
                    }
                )
        # Warn when misalignment is widespread — this means \multicolumn spanning
        # has corrupted the column→benchmark mapping for a significant fraction of
        # rows.  The threshold of 15 % is intentionally loose to avoid noise from
        # rare footnote rows that add an extra cell.
        if total_data_rows > 0 and misaligned_rows / total_data_rows > 0.15:
            print(
                f"  warning: {misaligned_rows}/{total_data_rows} data rows have a "
                f"column-count mismatch (expected {expected_cols} benchmark cols). "
                "Likely cause: \\multicolumn spanning — some scores may be "
                "assigned to the wrong benchmark. Review this table manually."
            )
        return results

    def _parse_models_as_columns(
        self, table: list[list[str]]
    ) -> list[dict]:
        """models are columns, benchmarks are rows."""
        results: list[dict] = []
        if len(table[0]) < 2:
            return results

        model_names = [
            _clean_cell(c) if self._looks_like_model_name(_clean_cell(c)) else ""
            for c in table[0][1:]
        ]

        current_mode_suffix: str | None = None
        for row in table[1:]:
            if not row or not row[0]:
                continue
            # Check for thinking mode section headers BEFORE the generic check
            mode_suffix = self._get_thinking_mode_suffix(row)
            if mode_suffix is not None:
                current_mode_suffix = mode_suffix
                self._last_section_header_rows_skipped += 1
                continue
            if self._is_section_header_row(row):
                # Preserve thinking mode context across sub-category headers
                self._last_section_header_rows_skipped += 1
                continue
            bench = _clean_cell(row[0])
            if not bench or _is_separator(bench):
                continue
            for i, model in enumerate(model_names):
                if not model:
                    continue
                effective_model = f"{model} {current_mode_suffix}" if current_mode_suffix else model
                cell_idx = i + 1
                if cell_idx >= len(row):
                    continue
                score, had_pct = _parse_numeric_with_pct(row[cell_idx])
                if score is None:
                    continue
                results.append(
                    {
                        "model": effective_model,
                        "benchmark": bench,
                        "score": score,
                        "score_type": "absolute",
                        "relative_to": None,
                        "_had_pct": had_pct,
                    }
                )
        return results


# ---------------------------------------------------------------------------
# S: paper converter — schema conversion only
# ---------------------------------------------------------------------------


