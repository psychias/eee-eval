"""Normalization utilities for cross-source matching.

Public API:
- normalize_model_id(raw)                -> lowercased canonical model ID
- normalize_benchmark_for_matching(raw)  -> lowercased canonical benchmark name
"""
from __future__ import annotations

import re


def normalize_model_id(raw: str) -> str:
    """Lowercase, strip trailing -hf, collapse separators to hyphen."""
    s = raw.lower().strip()
    s = re.sub(r"-hf$", "", s)       # hosted HF weights suffix
    s = re.sub(r"[-_\s]+", "-", s)   # unify separators
    return s.strip("-")


def normalize_benchmark_for_matching(raw: str) -> str:
    """Lowercase, strip parenthetical shot info, collapse separators.

    Designed for cross-source collision detection: "BBH (3-Shot)" and
    "BBH" both map to "bbh".
    """
    s = raw.lower().strip()
    s = re.sub(r"\s*\(.*?\)", "", s)                   # (0-shot), (3-Shot) etc.
    s = re.sub(r"[-_]\d+[-_]?shots?$", "", s)          # trailing _5shot
    s = re.sub(r"[-_\s]+", "_", s)                     # unify to underscore
    return s.strip("_")
