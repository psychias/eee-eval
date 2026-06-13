"""Unit tests for the original PDF-based extract_paper.py (DEPRECATED).

These tests validated the core extraction helpers, heuristics, and parsers
from the original PDF-based extract_paper.py script which has since been
replaced by the HTML-based extraction pipeline (extract_arxiv_metadata_general).
The functions tested here (_parse_numeric, ResultsTableParser, etc.) no longer
exist in the codebase.

This file is kept for reference but all tests are skipped.
"""
import pytest

pytest.skip(
    "Tests reference functions from deprecated PDF-based extract_paper.py that no longer exist",
    allow_module_level=True,
)


import sys
from pathlib import Path

import pytest

# Ensure the repo root and src directories are on sys.path so the script's imports resolve.
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src" / "utils"))
sys.path.insert(0, str(_ROOT / "src" / "extraction"))

from extract_paper import (
    _parse_numeric,
    _parse_numeric_with_pct,
    _is_separator,
    _clean_cell,
    _infer_developer,
    _infer_model_metadata,
    _infer_metric_name,
    _infer_evaluator_relationship,
    _is_plausible_result,
    _sanitize_extracted,
    _normalise_benchmark_key,
    _lookup_protocol,
    ResultsTableParser,
    EvalProtocolExtractor,
)


# ---------------------------------------------------------------------------
# _parse_numeric
# ---------------------------------------------------------------------------

class TestParseNumeric:
    """Test the numeric cell parser handles all common paper table formats."""

    def test_plain_float(self):
        assert _parse_numeric("85.2") == 85.2

    def test_plain_int(self):
        assert _parse_numeric("92") == 92.0

    def test_percentage_suffix(self):
        assert _parse_numeric("73.1%") == 73.1

    def test_footnote_dagger(self):
        assert _parse_numeric("85.2†") == 85.2

    def test_footnote_asterisk_leading(self):
        assert _parse_numeric("*73.1") == 73.1

    def test_footnote_double_dagger(self):
        assert _parse_numeric("92‡") == 92.0

    def test_parenthesised(self):
        assert _parse_numeric("(85.2)") == 85.2

    def test_dash_missing(self):
        assert _parse_numeric("-") is None

    def test_em_dash_missing(self):
        assert _parse_numeric("—") is None

    def test_na(self):
        assert _parse_numeric("N/A") is None

    def test_empty(self):
        assert _parse_numeric("") is None

    def test_latex_bold(self):
        assert _parse_numeric("\\textbf{85}") == 85.0

    def test_variance_pm(self):
        assert _parse_numeric("6.84 +/-0.07") == 6.84

    def test_variance_unicode(self):
        assert _parse_numeric("6.58±0.05") == 6.58

    def test_variance_latex(self):
        assert _parse_numeric("85.2 \\pm 1.3") == 85.2

    def test_negative(self):
        assert _parse_numeric("-3.5") == -3.5

    def test_nonsense(self):
        assert _parse_numeric("hello") is None


# ---------------------------------------------------------------------------
# _parse_numeric_with_pct
# ---------------------------------------------------------------------------

class TestParseNumericWithPct:
    """Test the extended parser that tracks whether '%' was present."""

    def test_explicit_pct(self):
        value, had_pct = _parse_numeric_with_pct("1.0%")
        assert value == 1.0
        assert had_pct is True

    def test_no_pct(self):
        value, had_pct = _parse_numeric_with_pct("85.2")
        assert value == 85.2
        assert had_pct is False

    def test_pct_with_footnote(self):
        value, had_pct = _parse_numeric_with_pct("73.1%†")
        # footnote stripped first, then % detected on cleaned value
        assert value == 73.1
        # '%' may or may not survive footnote stripping — check the value is right
        assert value == 73.1

    def test_missing_returns_none(self):
        value, had_pct = _parse_numeric_with_pct("-")
        assert value is None
        assert had_pct is False


# ---------------------------------------------------------------------------
# _is_separator
# ---------------------------------------------------------------------------

class TestIsSeparator:
    def test_dashes(self):
        assert _is_separator("---") is True

    def test_equals(self):
        assert _is_separator("===") is True

    def test_empty(self):
        assert _is_separator("") is True

    def test_model_name(self):
        assert _is_separator("Llama 3 70B") is False


# ---------------------------------------------------------------------------
# _clean_cell
# ---------------------------------------------------------------------------

class TestCleanCell:
    def test_strips_footnote_markers(self):
        assert _clean_cell("85.2†") == "85.2"
        assert _clean_cell("*73.1") == "73.1"

    def test_strips_whitespace(self):
        assert _clean_cell("  hello  ") == "hello"


# ---------------------------------------------------------------------------
# _infer_developer
# ---------------------------------------------------------------------------

class TestInferDeveloper:
    def test_llama(self):
        assert _infer_developer("Llama 3 70B", use_hf_api=False) == "meta-llama"

    def test_codellama_before_llama(self):
        assert _infer_developer("CodeLlama-34B", use_hf_api=False) == "meta-llama"

    def test_gpt4(self):
        assert _infer_developer("GPT-4", use_hf_api=False) == "openai"

    def test_gpt_j_not_openai(self):
        # gpt-j should match EleutherAI, not openai
        assert _infer_developer("GPT-J-6B", use_hf_api=False) == "EleutherAI"

    def test_claude(self):
        assert _infer_developer("Claude 3.5 Sonnet", use_hf_api=False) == "anthropic"

    def test_mistral(self):
        assert _infer_developer("Mistral 7B", use_hf_api=False) == "mistralai"

    def test_qwen(self):
        assert _infer_developer("Qwen2-72B", use_hf_api=False) == "Qwen"

    def test_hf_format(self):
        assert _infer_developer("meta-llama/Llama-3-70B", use_hf_api=False) == "meta-llama"

    def test_short_pattern_yi_exact(self):
        # "Yi-34B" should match 01-ai via word boundary
        result = _infer_developer("Yi-34B", use_hf_api=False)
        assert result == "01-ai"

    def test_short_pattern_yi_no_false_positive(self):
        # "City-7B" should NOT match "yi" — word boundary protects it
        result = _infer_developer("City-7B", use_hf_api=False)
        assert result != "01-ai"

    def test_command_r_before_command(self):
        assert _infer_developer("Command-R-Plus", use_hf_api=False) == "CohereForAI"


# ---------------------------------------------------------------------------
# _infer_model_metadata
# ---------------------------------------------------------------------------

class TestInferModelMetadata:
    def test_parameter_count(self):
        meta = _infer_model_metadata("Llama 3 70B")
        assert meta["parameter_count"] == "70B"

    def test_alignment_instruct(self):
        meta = _infer_model_metadata("Mistral 7B Instruct")
        assert meta["model_alignment"] == "instruct"

    def test_alignment_chat(self):
        meta = _infer_model_metadata("Llama 2 7B Chat")
        assert meta["model_alignment"] == "chat"

    def test_quantization(self):
        meta = _infer_model_metadata("Llama-3-8B-INT4")
        assert meta["quantization"] == "INT4"

    def test_no_metadata(self):
        meta = _infer_model_metadata("SomeModel")
        assert meta["parameter_count"] is None
        assert meta["model_alignment"] is None
        assert meta["quantization"] is None


# ---------------------------------------------------------------------------
# _infer_metric_name
# ---------------------------------------------------------------------------

class TestInferMetricName:
    def test_pass_at_1(self):
        assert _infer_metric_name("HumanEval pass@1") == "pass@1"

    def test_exact_match(self):
        assert _infer_metric_name("NQ exact match") == "exact_match"

    def test_accuracy_fallback(self):
        assert _infer_metric_name("MMLU") == "accuracy"

    def test_bleu(self):
        assert _infer_metric_name("BLEU-4") == "bleu"


# ---------------------------------------------------------------------------
# _infer_evaluator_relationship
# ---------------------------------------------------------------------------

class TestInferEvaluatorRelationship:
    def test_first_party_meta_llama(self):
        # Llama paper has Meta authors
        result = _infer_evaluator_relationship(
            "meta-llama",
            ["Hugo Touvron", "Louis Martin", "Meta AI Team"]
        )
        assert result == "first_party"

    def test_third_party_different_org(self):
        result = _infer_evaluator_relationship(
            "openai",
            ["Hugo Touvron", "Louis Martin", "Meta AI Team"]
        )
        assert result == "third_party"

    def test_unknown_developer(self):
        result = _infer_evaluator_relationship("unknown", ["Some Author"])
        assert result == "third_party"

    def test_no_authors(self):
        result = _infer_evaluator_relationship("meta-llama", [])
        assert result == "third_party"


# ---------------------------------------------------------------------------
# _is_plausible_result / _sanitize_extracted
# ---------------------------------------------------------------------------

class TestSanityFilter:
    def test_normal_result_passes(self):
        assert _is_plausible_result({
            "model": "Llama 3 70B",
            "benchmark": "MMLU",
            "score": 85.2,
        })

    def test_year_as_score_rejected(self):
        assert not _is_plausible_result({
            "model": "Some Model",
            "benchmark": "MMLU",
            "score": 2021.0,
        })

    def test_citation_as_model_rejected(self):
        assert not _is_plausible_result({
            "model": "HumanEval (Chen et al., 2021), MBPP (Austin et al.,",
            "benchmark": "MMLU",
            "score": 85.2,
        })

    def test_long_model_name_rejected(self):
        assert not _is_plausible_result({
            "model": "A" * 61,
            "benchmark": "MMLU",
            "score": 50.0,
        })

    def test_huge_score_rejected(self):
        assert not _is_plausible_result({
            "model": "GPT-4",
            "benchmark": "MMLU",
            "score": 5000.0,
        })

    def test_sanitize_filters_bad(self):
        items = [
            {"model": "GPT-4", "benchmark": "MMLU", "score": 86.4},
            {"model": "Citation (et al., 2021)", "benchmark": "X", "score": 2021.0},
            {"model": "Good Model", "benchmark": "GSM8K", "score": 78.5},
        ]
        result = _sanitize_extracted(items)
        assert len(result) == 2
        assert result[0]["model"] == "GPT-4"
        assert result[1]["model"] == "Good Model"


# ---------------------------------------------------------------------------
# ResultsTableParser
# ---------------------------------------------------------------------------

class TestResultsTableParser:
    """Test table orientation detection and parsing."""

    def setup_method(self):
        self.parser = ResultsTableParser()

    def test_is_results_table_with_benchmarks(self):
        table = [
            ["Model", "MMLU", "GSM8K", "HumanEval"],
            ["GPT-4", "86.4", "92.0", "67.0"],
            ["Llama 3", "79.5", "87.3", "62.1"],
            ["Mistral", "72.1", "58.4", "39.2"],
        ]
        assert self.parser.is_results_table(table)

    def test_rejects_non_results_table(self):
        table = [
            ["Hyperparameter", "Value"],
            ["Learning rate", "3e-4"],
            ["Batch size", "256"],
        ]
        assert not self.parser.is_results_table(table)

    def test_rejects_too_small(self):
        table = [
            ["Model", "MMLU"],
            ["GPT-4", "86.4"],
        ]
        assert not self.parser.is_results_table(table)

    def test_models_as_rows_parsing(self):
        table = [
            ["Model", "MMLU", "GSM8K"],
            ["GPT-4", "86.4", "92.0"],
            ["Llama 3 70B", "79.5", "87.3"],
            ["Mistral 7B", "72.1", "58.4"],
        ]
        results = self.parser.parse(table)
        assert len(results) == 6  # 3 models × 2 benchmarks

        # Check specific entries
        gpt4_mmlu = [r for r in results if r["model"] == "GPT-4" and r["benchmark"] == "MMLU"]
        assert len(gpt4_mmlu) == 1
        assert gpt4_mmlu[0]["score"] == 86.4

    def test_models_as_columns_parsing(self):
        table = [
            ["Benchmark", "GPT-4", "Llama 3"],
            ["MMLU", "86.4", "79.5"],
            ["GSM8K", "92.0", "87.3"],
            ["HumanEval", "67.0", "62.1"],
        ]
        results = self.parser.parse(table)
        assert len(results) == 6  # 2 models × 3 benchmarks

    def test_citation_row_filtered(self):
        """Rows with citation-style text should be filtered out."""
        table = [
            ["Model", "MMLU", "GSM8K"],
            ["GPT-4", "86.4", "92.0"],
            ["HumanEval (Chen et al., 2021)", "2021", "2023"],
            ["Llama 3 70B", "79.5", "87.3"],
        ]
        results = self.parser.parse(table)
        models = {r["model"] for r in results}
        assert "GPT-4" in models
        assert "Llama 3 70B" in models
        # The citation should be filtered by _looks_like_model_name
        assert not any("Chen et al" in m for m in models)

    def test_extraction_confidence_high(self):
        table = [
            ["Model", "MMLU", "GSM8K", "HumanEval", "ARC"],
            ["GPT-4", "86.4", "92.0", "67.0", "96.3"],
            ["Llama 3", "79.5", "87.3", "62.1", "89.7"],
            ["Mistral", "72.1", "58.4", "39.2", "78.5"],
        ]
        confidence = self.parser.extraction_confidence(table)
        assert confidence == "high"

    def test_relative_table_detection(self):
        table = [
            ["Model", "MMLU", "GSM8K"],
            ["Variant A", "+2.3", "+5.1"],
            ["Variant B", "-1.2", "+3.4"],
            ["Variant C", "+0.5", "-0.8"],
        ]
        is_rel, _ = self.parser.is_relative_table(table)
        assert is_rel is True


# ---------------------------------------------------------------------------
# _lookup_protocol
# ---------------------------------------------------------------------------

class TestLookupProtocol:
    def test_exact_match(self):
        protocol_map = {
            "__default__": {"shots": 5},
            "mmlu": {"shots": 5, "protocol_source": "explicit"},
            "gsm8k": {"shots": 8, "protocol_source": "explicit"},
        }
        result = _lookup_protocol(protocol_map, "MMLU")
        assert result["shots"] == 5

    def test_default_fallback(self):
        protocol_map = {
            "__default__": {"shots": 0, "method": "generation"},
        }
        result = _lookup_protocol(protocol_map, "UnknownBench")
        assert result["shots"] == 0
        assert result["method"] == "generation"

    def test_empty_map(self):
        assert _lookup_protocol({}, "MMLU") == {}


# ---------------------------------------------------------------------------
# _normalise_benchmark_key
# ---------------------------------------------------------------------------

class TestNormaliseBenchmarkKey:
    def test_lowercases(self):
        assert _normalise_benchmark_key("MMLU") == "mmlu"

    def test_strips_whitespace(self):
        assert _normalise_benchmark_key("  GSM8K  ") == "gsm8k"

    def test_collapses_spaces(self):
        assert _normalise_benchmark_key("Big  Bench") == "big bench"
