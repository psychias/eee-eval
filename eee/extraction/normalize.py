"""Backward-compatible shim — delegates to ``eee.normalization``.

All normalization logic now lives in ``eee/normalization/``.  This file
re-exports the old private names so that existing ``from .normalize import``
statements in the extraction subpackage continue to work.

New code should import directly from ``eee.normalization`` instead.
"""
from eee.normalization.benchmarks import (
    normalize_benchmark as _normalize_benchmark,
    normalize_benchmark_for_matching,
    normalize_benchmark_key as _normalise_benchmark_key,
    normalize_benchmark_with_shots as _normalize_benchmark_with_shots,
    normalize_aime as _normalize_aime,
)
from eee.normalization.models import (
    normalize_model_name as _normalize_model_name,
    normalize_model_id as _normalize_model_id,
    to_hf_model_id as _to_hf_model_id,
    normalize_claude_model_name as _normalize_claude_model_name,
    should_tag_thinking_mode as _should_tag_thinking_mode,
    model_name_matches as _model_name_matches,
    merge_key as _merge_key,
)
from eee.normalization.scores import (
    is_likely_fraction as _is_likely_fraction,
    parse_numeric as _parse_numeric,
    parse_numeric_with_pct as _parse_numeric_with_pct,
)

# Re-export score range data for callers that read it directly
from eee.normalization.scores import _EXPECTED_PERCENTAGE_RANGES

__all__ = [
    "_normalize_benchmark",
    "_normalize_model_name",
    "_normalize_model_id",
    "_should_tag_thinking_mode",
    "_model_name_matches",
    "_merge_key",
    "_normalize_aime",
    "_normalise_benchmark_key",
    "_normalize_benchmark_with_shots",
    "_is_likely_fraction",
    "_parse_numeric",
    "_parse_numeric_with_pct",
    "_EXPECTED_PERCENTAGE_RANGES",
    "normalize_benchmark_for_matching",
]


