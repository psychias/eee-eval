"""Extraction package — split from the monolithic extract_paper.py.

Public API::

    from eee.extraction.pipeline import main
    from eee.extraction.pipeline import PaperExtractionPipeline
"""
import sys
from pathlib import Path

# Ensure repo root + utils/ are importable.
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / 'utils') not in sys.path:
    sys.path.insert(0, str(_ROOT / 'utils'))

# Raise Python 3.11+ integer-string conversion limit (Docling artifacts).
if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)

# Lazy imports — avoids heavy dependency chain when only constants/normalize
# are needed (e.g. from eee.normalization).


def __getattr__(name: str):
    _lazy = {
        "PDFDownloader": ".download",
        "DoclingParser": ".docling_parser",
        "TableExtractor": ".docling_parser",
        "ResultsTableParser": ".table_parser",
        "PaperConverter": ".converter",
        "PaperWriter": ".converter",
        "LLMFallbackExtractor": ".llm_fallback",
        "CoverageStats": ".llm_fallback",
        "ProseExtractor": ".prose",
        "EvalProtocolExtractor": ".protocol",
        "GeminiProtocolExtractor": ".protocol",
        "PaperExtractionPipeline": ".pipeline",
        "main": ".pipeline",
    }
    if name in _lazy:
        import importlib
        mod = importlib.import_module(_lazy[name], __package__)
        val = getattr(mod, name)
        globals()[name] = val  # cache for next access
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "PDFDownloader",
    "DoclingParser",
    "TableExtractor",
    "ResultsTableParser",
    "PaperConverter",
    "PaperWriter",
    "LLMFallbackExtractor",
    "CoverageStats",
    "ProseExtractor",
    "EvalProtocolExtractor",
    "GeminiProtocolExtractor",
    "PaperExtractionPipeline",
    "main",
]
