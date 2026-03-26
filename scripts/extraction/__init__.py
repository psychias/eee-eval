"""Extraction package — split from the monolithic extract_paper.py.

Public API::

    from scripts.extraction.pipeline import main
    from scripts.extraction.pipeline import PaperExtractionPipeline
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
sys.set_int_max_str_digits(0)

from .download import PDFDownloader
from .docling_parser import DoclingParser, TableExtractor
from .table_parser import ResultsTableParser
from .converter import PaperConverter, PaperWriter
from .llm_fallback import LLMFallbackExtractor, CoverageStats
from .prose import ProseExtractor
from .protocol import EvalProtocolExtractor, GeminiProtocolExtractor
from .pipeline import PaperExtractionPipeline, main

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
