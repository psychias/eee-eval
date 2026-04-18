"""Docling PDF parser, table extractor, and table repair heuristics."""
from __future__ import annotations

import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from .constants import (
    _BENCHMARK_KEYWORDS_STRONG,
    _DEVELOPER_PATTERNS,
)
from .helpers import _parse_numeric

# ---------------------------------------------------------------------------
# S: table extractor — Docling table detection only
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Docling imports and shared parser
# ---------------------------------------------------------------------------
# Auto-enable HF offline mode BEFORE importing Docling / huggingface_hub,
# because huggingface_hub caches HF_HUB_OFFLINE at import time into a
# module-level constant.  Setting os.environ after import has no effect.
if not os.environ.get("HF_HUB_OFFLINE"):
    _hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    _layout_cache = _hf_home / "hub" / "models--docling-project--docling-layout-heron"
    if _layout_cache.exists():
        os.environ["HF_HUB_OFFLINE"] = "1"
        print("  [init] Docling models cached — auto-enabled HF_HUB_OFFLINE=1")

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
import pandas as pd

# If we auto-set the env var above, huggingface_hub was imported by Docling
# with the correct value.  But if the user set it themselves after Python
# started (e.g. $env:HF_HUB_OFFLINE=1 in the same shell), the constant
# may still be False.  Patch it directly to be safe.
if os.environ.get("HF_HUB_OFFLINE") == "1":
    try:
        import huggingface_hub.constants as _hf_constants
        _hf_constants.HF_HUB_OFFLINE = True
    except ImportError:
        pass


class DoclingParser:
    """Shared Docling PDF parser with per-path result caching.

    Parse each PDF once; all extractors reuse the cached result to avoid
    redundant deep-learning inference across TableExtractor, ProseExtractor,
    LLMFallbackExtractor, and _detect_eval_library.
    """

    _PARSE_TIMEOUT = 300  # seconds — generous for large PDFs

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_table_structure = True
        pipeline_options.do_ocr = False
        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=PyPdfiumDocumentBackend,
                )
            }
        )

    def parse(self, pdf_path: Path) -> Any:
        """Return the cached Docling ConversionResult for *pdf_path*.

        Catches SSL / network errors that occur when Docling tries to
        download models and provides an actionable hint.
        """
        key = str(pdf_path.resolve())
        if key not in self._cache:
            try:
                self._cache[key] = self._converter.convert(key)
            except Exception as exc:
                msg = str(exc)
                if "SSL" in msg or "huggingface" in msg.lower() or "MaxRetryError" in msg:
                    print(
                        "\n  ERROR: Docling failed to download models due to an SSL/network error.\n"
                        "  Fix: set  $env:HF_HUB_OFFLINE=1  (PowerShell) or  export HF_HUB_OFFLINE=1  (bash)\n"
                        "  before running the pipeline. The models must be cached from a prior run.\n",
                        file=sys.stderr,
                    )
                raise
        return self._cache[key]

    def get_full_text(self, pdf_path: Path) -> str:
        """Return the full document as markdown text."""
        return self.parse(pdf_path).document.export_to_markdown()


# Module-level shared parser instance — constructed lazily on first use.
_docling_parser = DoclingParser()


# ---------------------------------------------------------------------------
# S: table extractor — Docling table detection only
# ---------------------------------------------------------------------------


def _split_merged_model_names(cell: str) -> list[str]:
    """Split a header cell containing multiple space-separated model names.

    Uses ``_DEVELOPER_PATTERNS`` to detect where one model name ends and the
    next begins.  Handles Docling's detached-hyphen artefacts (e.g.
    ``"DeepSeek-R1 -Distill-Qwen-32B"`` → single model).

    Returns a list of individual model name strings.  If only one model is
    detected, the original (stripped) text is returned as a single-element
    list.
    """
    cell = cell.strip()
    if not cell:
        return []
    # Re-join detached hyphens first: "GPT-4o -2024-11-20" → "GPT-4o-2024-11-20"
    cell = re.sub(r"\s+-", "-", cell)
    # Insert spaces where Docling concatenated model names without separators.
    # Common pattern: parameter-size suffix immediately followed by start of
    # next model name, e.g. "8BLlama" → "8B Llama", "405BGPT-3.5" → "405B GPT-3.5"
    cell = re.sub(r'(\d+[bBmM])([A-Z])', r'\1 \2', cell)
    # Also handle "InstructGPT" / "TurboNemotron" — insert space before known
    # model family names when preceded by lowercase (concatenation boundary).
    _FAMILY_PREFIXES = (
        "Llama", "GPT", "Claude", "Gemini", "Gemma", "Mistral", "Mixtral",
        "Qwen", "DeepSeek", "Nemotron", "Falcon", "Phi", "Command", "Yi",
        "Vicuna", "Alpaca", "StarCoder", "CodeGen", "Jamba", "DBRX",
        "InternLM", "Baichuan", "OLMo", "Grok", "Solar",
    )
    for prefix in _FAMILY_PREFIXES:
        cell = re.sub(rf'([a-z])({re.escape(prefix)})', r'\1 \2', cell)
    tokens = cell.split()
    if len(tokens) <= 1:
        return [cell]

    models: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        token_lower = token.lower()
        first_seg = token_lower.split("-")[0] if "-" in token_lower else token_lower
        is_developer_start = False
        for pat, _dev in _DEVELOPER_PATTERNS:
            if len(pat) < 3:
                continue
            if first_seg == pat or token_lower.startswith(pat):
                is_developer_start = True
                break
        if is_developer_start and current:
            models.append(current)
            current = [token]
        else:
            current.append(token)
    if current:
        models.append(current)

    if len(models) <= 1:
        return [cell]

    # Detect trailing variant suffixes ("Base", "Instruct", "Chat", etc.)
    # that Docling appends once per model but all land on the last model.
    # E.g. "Gemma-3-4B Qwen2.5-3B Qwen3-4B Base Base Base" → 3 models,
    # 3 trailing "Base" tokens should distribute one per model.
    _VARIANT_SUFFIXES = {"base", "instruct", "chat", "it", "hf", "pretrained", "pre-trained"}
    last_model_tokens = models[-1]
    if len(last_model_tokens) > 1:
        # Find where variant suffix tokens start in the last model
        suffix_start = len(last_model_tokens)
        for i in range(1, len(last_model_tokens)):
            if last_model_tokens[i].lower() in _VARIANT_SUFFIXES:
                suffix_start = i
                break
        trailing_suffixes = last_model_tokens[suffix_start:]
        if trailing_suffixes and all(t.lower() in _VARIANT_SUFFIXES for t in trailing_suffixes):
            num_models = len(models)
            core_last = last_model_tokens[:suffix_start]
            if len(trailing_suffixes) >= num_models:
                # Distribute one suffix per model
                for j, model_tokens in enumerate(models):
                    if j < len(models) - 1:
                        model_tokens.append(trailing_suffixes[j])
                models[-1] = core_last + [trailing_suffixes[num_models - 1]]
            elif len(trailing_suffixes) == 1:
                # Single trailing suffix — append to last model only
                models[-1] = core_last + trailing_suffixes
            else:
                # Mismatch — strip all trailing suffixes from last model
                models[-1] = core_last if core_last else last_model_tokens

    result = [" ".join(m) for m in models]
    return result if len(result) > 1 else [cell]


def _try_expand_merged_model_columns(
    table_rows: list[list[str]],
) -> list[list[str]] | None:
    """Expand models-as-columns tables where Docling merged multiple model
    columns into single cells.

    Detection:
      - At least one header cell splits into ≥ 2 model names (via developer
        prefix boundaries).
      - ≥ 30 % of data rows produce exactly *N* trailing numeric / missing-
        value tokens (where *N* = total model count from the header).

    Reconstruction treats each row as a *flat* sequence of tokens (cell
    boundaries from Docling are unreliable when columns are collapsed) and
    extracts benchmark name + N trailing score-or-dash tokens.

    Returns the rebuilt table, or ``None`` if expansion was not applicable.
    """
    if len(table_rows) < 3:
        return None

    # --- Step 1: split header cells into individual model names ----------
    header = table_rows[0]
    expanded_models: list[str] = []
    any_merged = False
    for cell in header:
        stripped = cell.strip()
        if not stripped:
            continue
        parts = _split_merged_model_names(stripped)
        if len(parts) > 1:
            any_merged = True
            expanded_models.extend(parts)
        # Single-element cells are metadata columns (Category, Benchmark, etc.)
        # — don't include them in the model list.
    if not any_merged or len(expanded_models) < 2:
        return None

    num_models = len(expanded_models)
    print(
        f"    [expand-check] found {num_models} merged model names: "
        f"{expanded_models[:4]}{'...' if num_models > 4 else ''}",
        file=sys.stderr,
    )

    # --- Step 2: verify that data rows match the expected value count ----
    matching_rows = 0
    total_data_rows = 0
    _SKIP_PREFIXES = frozenset({
        "architecture", "#", "total param", "activated param",
        "# total", "# activated",
    })

    for row in table_rows[1:]:
        row_text = " ".join(cell.strip() for cell in row if cell.strip())
        if not row_text:
            continue
        row_lower = row_text.lower().strip()
        if any(row_lower.startswith(sp) for sp in _SKIP_PREFIXES):
            continue
        # skip all-text rows (section headers like "General Tasks")
        tokens = row_text.split()
        if all(_parse_numeric(t) is None and t not in ("-", "–", "—") for t in tokens):
            continue
        total_data_rows += 1

        # pre-process: strip confidence intervals, separate trailing dashes
        cleaned = re.sub(r"\s*(?:\+/?-|±)\s*\d+\.?\d*", "", row_text)
        cleaned = re.sub(r"(\d)-(?=\s|$)", r"\1 -", cleaned)
        ctokens = cleaned.split()
        # extract trailing value tokens (numeric or missing-value dash)
        values: list[str] = []
        remaining = list(ctokens)
        for _ in range(num_models):
            if not remaining:
                break
            last = remaining[-1]
            if _parse_numeric(last) is not None or last in ("-", "–", "—"):
                values.insert(0, remaining.pop())
            else:
                break
        if total_data_rows <= 2:
            _last_tok = repr(ctokens[-1][:20]) if ctokens else "N/A"
            print(
                f"    [expand-row-debug] row_text={row_text[:80]!r} "
                f"values={len(values)} need={num_models} "
                f"ctokens={len(ctokens)} last_tok={_last_tok}",
                file=sys.stderr,
            )
        if len(values) == num_models and remaining:
            # benchmark name should not contain many numbers (garbled row)
            nums_in_bench = sum(
                1 for t in remaining if _parse_numeric(t) is not None
            )
            if nums_in_bench < 2:
                matching_rows += 1

    if total_data_rows < 2 or matching_rows < total_data_rows * 0.3:
        print(
            f"    [expand-check] validation failed: {matching_rows}/{total_data_rows} "
            f"data rows matched (need ≥30%)",
            file=sys.stderr,
        )
        return None

    # --- Step 3: rebuild the table with one column per model -------------
    new_table: list[list[str]] = [["Benchmark"] + expanded_models]

    for row in table_rows[1:]:
        row_text = " ".join(cell.strip() for cell in row if cell.strip())
        if not row_text:
            continue
        row_lower = row_text.lower().strip()
        if any(row_lower.startswith(sp) for sp in _SKIP_PREFIXES):
            continue
        tokens_raw = row_text.split()
        if all(_parse_numeric(t) is None and t not in ("-", "–", "—") for t in tokens_raw):
            # Preserve thinking/non-thinking mode section headers
            if "thinking" in row_lower:
                new_table.append([row_text] + [""] * num_models)
            continue
        cleaned = re.sub(r"\s*(?:\+/?-|±)\s*\d+\.?\d*", "", row_text)
        cleaned = re.sub(r"(\d)-(?=\s|$)", r"\1 -", cleaned)
        ctokens = cleaned.split()
        values = []
        remaining = list(ctokens)
        for _ in range(num_models):
            if not remaining:
                break
            last = remaining[-1]
            if _parse_numeric(last) is not None or last in ("-", "–", "—"):
                values.insert(0, remaining.pop())
            else:
                break
        if len(values) != num_models or not remaining:
            continue
        bench_name = " ".join(remaining)
        if not bench_name.strip():
            continue
        nums_in_bench = sum(
            1 for t in remaining if _parse_numeric(t) is not None
        )
        if nums_in_bench >= 2:
            continue
        new_table.append([bench_name] + values)

    if len(new_table) >= 3:
        print(
            f"  [table-fix] Expanded merged model columns: "
            f"{num_models} models × {len(new_table) - 1} benchmarks"
        )
        return new_table

    return None


def _try_fix_collapsed_table(table_rows: list[list[str]]) -> list[list[str]]:
    """Re-split tables where Docling merged many columns into a few cells.

    When Docling's table structure detection fails, all values for a row may
    end up concatenated in a single cell (e.g. the whole row of benchmarks +
    scores becomes one long string).  This function detects that pattern and
    reconstructs proper columns by:

    1. Finding benchmark keywords in the concatenated header text.
    2. Counting numeric values from the right of each concatenated data row.
    3. Rebuilding with one model column + N benchmark columns.

    Only activates when the table is clearly collapsed (many benchmark
    keywords in header but very few non-empty actual columns).  Tables with
    multi-word benchmark names that cannot be reliably split are left alone
    for the LLM extractor to handle.
    """
    if len(table_rows) < 3:
        return table_rows

    # --- detect collapsed structure ---
    header_text = " ".join(cell.strip() for cell in table_rows[0] if cell.strip())
    header_lower = header_text.lower()

    # Need at least 3 strong benchmark keywords in the header
    bench_kw_count = sum(1 for kw in _BENCHMARK_KEYWORDS_STRONG if kw in header_lower)
    if bench_kw_count < 3:
        return table_rows

    # Check if actual non-empty column count is far smaller than benchmark
    # keyword count — the smoking gun for a collapsed table.
    non_empty_cols_per_row = [
        sum(1 for cell in row if cell.strip()) for row in table_rows
    ]
    avg_non_empty = (
        sum(non_empty_cols_per_row) / len(non_empty_cols_per_row)
        if non_empty_cols_per_row
        else 0
    )
    if avg_non_empty >= bench_kw_count:
        return table_rows  # columns already reasonably separated

    # --- count expected score columns from data rows ---
    score_counts: list[int] = []
    for row in table_rows[1:]:
        row_text = " ".join(cell.strip() for cell in row if cell.strip())
        if not row_text.strip():
            continue
        # Remove confidence intervals before tokenising
        row_text = re.sub(r"\s*(?:\+/?-|±)\s*\d+\.?\d*", "", row_text)
        tokens = row_text.split()
        count = 0
        for t in reversed(tokens):
            if _parse_numeric(t) is not None:
                count += 1
            else:
                break
        if count > 0:
            score_counts.append(count)

    if not score_counts:
        return table_rows

    num_scores = Counter(score_counts).most_common(1)[0][0]
    if num_scores < 2:
        return table_rows

    # --- extract benchmark headers ---
    header_tokens = header_text.split()
    if len(header_tokens) <= num_scores:
        return table_rows  # can't split header

    # Find first benchmark keyword token via startswith or two-word match
    first_bench_idx: int | None = None
    for i, token in enumerate(header_tokens):
        token_lower = token.lower()
        # Single-word keyword match (startswith to handle "Arc-e" matching "arc")
        for kw in _BENCHMARK_KEYWORDS_STRONG:
            if len(kw) >= 3 and (kw == token_lower or token_lower.startswith(kw)):
                first_bench_idx = i
                break
        if first_bench_idx is not None:
            break
        # Two-word keyword match (e.g. "chatbot arena", "mt bench")
        if i < len(header_tokens) - 1:
            two_word = f"{token_lower} {header_tokens[i + 1].lower()}"
            for kw in _BENCHMARK_KEYWORDS_STRONG:
                if " " in kw and kw in two_word:
                    first_bench_idx = i
                    break
        if first_bench_idx is not None:
            break

    if first_bench_idx is None:
        first_bench_idx = len(header_tokens) - num_scores

    bench_headers = header_tokens[first_bench_idx:]

    # Match benchmark header count to score count
    if len(bench_headers) == num_scores:
        pass  # perfect 1:1 match
    elif len(bench_headers) > num_scores:
        if len(bench_headers) / num_scores > 1.5:
            # Too many header tokens relative to scores — likely multi-word
            # benchmark names that we can't reliably split by whitespace.
            # Leave this for the LLM to handle.
            return table_rows
        bench_headers = bench_headers[:num_scores]
    else:
        return table_rows  # fewer headers than scores — can't label

    # --- rebuild table ---
    new_header = ["Model"] + bench_headers
    new_table: list[list[str]] = [new_header]

    for row in table_rows[1:]:
        row_text = " ".join(cell.strip() for cell in row if cell.strip())
        if not row_text.strip():
            continue
        row_text = re.sub(r"\s*(?:\+/?-|±)\s*\d+\.?\d*", "", row_text)
        tokens = row_text.split()
        scores: list[str] = []
        remaining = list(tokens)
        for _ in range(num_scores):
            if not remaining:
                break
            if _parse_numeric(remaining[-1]) is not None:
                scores.insert(0, remaining.pop())
            else:
                break
        model_name = " ".join(remaining)
        if len(scores) == num_scores and model_name.strip():
            new_table.append([model_name] + scores)

    if len(new_table) >= 3:  # header + at least 2 data rows
        # Validate: if most reconstructed "model names" are actually benchmark
        # keywords, this is a benchmarks-as-rows table, not models-as-rows.
        # Skip reconstruction — the table parser or LLM should handle it.
        bench_like = 0
        for rebuilt_row in new_table[1:]:
            name_lower = rebuilt_row[0].lower()
            if any(kw in name_lower for kw in _BENCHMARK_KEYWORDS_STRONG):
                bench_like += 1
        if bench_like > len(new_table[1:]) * 0.4:
            return table_rows

        print(
            f"  [table-fix] Reconstructed collapsed table: "
            f"{len(new_table) - 1} data rows × {num_scores} benchmarks"
        )
        return new_table

    return table_rows


class TableExtractor:
    """extract raw tables from a PDF using Docling."""

    def extract(self, pdf_path: Path) -> list[tuple[int, list[list[str]], str]]:
        """Return list of (page_number, table_rows, context_text) from *pdf_path*.

        *context_text* is the markdown text preceding the table (caption/section header).
        """
        result = _docling_parser.parse(pdf_path)
        tables_out: list[tuple[int, list[list[str]], str]] = []
        doc = result.document
        markdown = doc.export_to_markdown()
        for table in doc.tables:
            try:
                df = table.export_to_dataframe(doc)
            except Exception:
                d = table.export_to_dict()
                rows = d.get("rows") or d.get("data") or []
                df = pd.DataFrame(rows)
            # Prepend column headers as the first row so benchmark keywords
            # (e.g. "MMLU", "GSM8K") are visible to ResultsTableParser.
            header = [str(c) for c in df.columns.tolist()]
            data_rows = df.astype(str).fillna("").values.tolist()
            table_rows = [header] + data_rows
            table_rows = _try_fix_collapsed_table(table_rows)
            page_num = getattr(table, "page_num", 1)
            context_text = getattr(table, "caption", None)
            if not context_text:
                table_md = df.to_string(index=False)
                idx = markdown.find(table_md[:40]) if table_md else -1
                if idx > 0:
                    context_text = markdown[max(0, idx - 300):idx].strip()
                else:
                    context_text = ""
            tables_out.append((page_num, table_rows, context_text or ""))
        return tables_out


# ---------------------------------------------------------------------------
# S: prose extractor — regex-based extraction from running text and captions
# ---------------------------------------------------------------------------


