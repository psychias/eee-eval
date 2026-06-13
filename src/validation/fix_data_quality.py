"""Post-hoc data quality fixes for all EEE JSON records.

Architecture (SOLID):
  - DataFix (ABC): one fix = one class (OCP, ISP)
  - ParameterLeakFix: removes metadata metrics and out-of-range scores
  - BenchmarkCanonFix: normalizes benchmark names using shared canon map
  - PWCDedupFix: removes duplicate (model, benchmark) within papers_with_code
  - FixRunner: orchestrates all fixes over the data directory
"""
from __future__ import annotations

import json
import re
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.extraction.constants import _BENCHMARK_CANONICAL

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


# ─── Shared utilities ─────────────────────────────────────────────────────────

# Additional canonicalization entries not in constants.py (HF model card variants)
_EXTRA_CANONICAL: dict[str, str] = {
    "truthfulqa": "TruthfulQA",
    "mathqa": "MathQA",
    "pubmedqa": "PubMedQA",
    "sciq": "SciQ",
    "headqa": "HeadQA",
    "logiqa": "LogiQA",
    "webqs": "WebQS",
    "copa": "COPA",
    "multirc": "MultiRC",
}

# Merge: constants.py canonical map + local extras
_CANON_MAP: dict[str, str] = {**_BENCHMARK_CANONICAL, **_EXTRA_CANONICAL}

# Patterns that indicate a metric is model metadata, not a benchmark score
_PARAM_METRIC_PATTERNS = [
    re.compile(r"param", re.IGNORECASE),
    re.compile(r"size", re.IGNORECASE),
    re.compile(r"flop", re.IGNORECASE),
    re.compile(r"training.*(token|step|hour|cost)", re.IGNORECASE),
    re.compile(r"latency", re.IGNORECASE),
    re.compile(r"throughput", re.IGNORECASE),
    re.compile(r"memory", re.IGNORECASE),
]


def _is_metadata_metric(metric_name: str) -> bool:
    """True if metric_name looks like model metadata rather than a benchmark score."""
    return any(p.search(metric_name) for p in _PARAM_METRIC_PATTERNS)


def _is_score_out_of_range(score: float, min_score: float, max_score: float) -> bool:
    """True if score is implausibly outside declared range (parameter leak signal)."""
    if max_score <= 100 and score > 100:
        return True
    if max_score <= 1.0 and score > 1.5:
        return True
    return False


# ─── DataFix ABC ──────────────────────────────────────────────────────────────


@dataclass
class FixStats:
    """Mutable accumulator for fix statistics."""
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def inc(self, key: str, n: int = 1) -> None:
        self.counts[key] += n

    def __getitem__(self, key: str) -> int:
        return self.counts[key]


class DataFix(ABC):
    """Base class for a single data-fix pass."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable fix name."""

    @abstractmethod
    def apply(self, data: dict, source: str, path: Path, stats: FixStats) -> tuple[dict, bool]:
        """Apply fix to a record. Returns (possibly-modified data, was_modified)."""


# ─── Concrete fixes ──────────────────────────────────────────────────────────


class ParameterLeakFix(DataFix):
    """Remove evaluation results that are metadata (parameter counts) or out-of-range."""

    name = "Parameter Leak Removal"

    def apply(self, data: dict, source: str, path: Path, stats: FixStats) -> tuple[dict, bool]:
        eval_results = data.get("evaluation_results", [])
        cleaned = []
        removed = 0

        for er in eval_results:
            metric_name = er.get("metric_config", {}).get("metric_name", "")
            score = er.get("score_details", {}).get("score")
            max_score = er.get("metric_config", {}).get("max_score")
            min_score = er.get("metric_config", {}).get("min_score", 0.0)

            # Remove entries with metadata-like metric names
            if _is_metadata_metric(metric_name):
                removed += 1
                continue

            # Remove entries with scores impossibly above declared range
            if score is not None and max_score is not None:
                if _is_score_out_of_range(score, min_score, max_score):
                    removed += 1
                    continue

            cleaned.append(er)

        if removed > 0:
            data["evaluation_results"] = cleaned
            stats.inc("param_leaks_removed", removed)
            return data, True
        return data, False


class BenchmarkCanonFix(DataFix):
    """Canonicalize benchmark names using the shared canonical map."""

    name = "Benchmark Canonicalization"

    def apply(self, data: dict, source: str, path: Path, stats: FixStats) -> tuple[dict, bool]:
        modified = False
        for er in data.get("evaluation_results", []):
            bench_name = er.get("evaluation_name", "")
            canonical = _CANON_MAP.get(bench_name, bench_name)
            if canonical != bench_name:
                er["evaluation_name"] = canonical
                sd = er.get("source_data", {})
                if sd.get("dataset_name") == bench_name:
                    sd["dataset_name"] = canonical
                stats.inc("benchmarks_canonicalized")
                modified = True
        return data, modified


class PWCDedupFix(DataFix):
    """Deduplicate within-source (model, benchmark) conflicts for papers_with_code.

    This fix operates at the directory level, not per-file.
    It is run as a second pass after per-file fixes.
    """

    name = "PWC Deduplication"

    def apply(self, data: dict, source: str, path: Path, stats: FixStats) -> tuple[dict, bool]:
        # This fix is handled by run_directory_pass() instead
        return data, False

    def run_directory(self, data_dir: Path, stats: FixStats) -> None:
        """Deduplicate across all PWC files."""
        pwc_dir = data_dir / "papers_with_code"
        if not pwc_dir.exists():
            return

        # Build index: (model_id, benchmark) → entries
        index: dict[tuple[str, str], list[dict]] = defaultdict(list)

        for json_path in sorted(pwc_dir.rglob("*.json")):
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            model_id = data.get("model_info", {}).get("id", "?")
            arxiv_date = self._extract_arxiv_date(data)

            for idx, er in enumerate(data.get("evaluation_results", [])):
                bench = er.get("evaluation_name", "?")
                score = er.get("score_details", {}).get("score")
                max_score = er.get("metric_config", {}).get("max_score")
                index[(model_id, bench)].append({
                    "score": score, "max_score": max_score,
                    "arxiv_date": arxiv_date, "file_path": json_path,
                    "er_index": idx,
                })

        # Identify entries to remove (keep most recent with plausible score)
        removals: dict[Path, set[int]] = defaultdict(set)
        for entries in index.values():
            if len(entries) <= 1:
                continue
            sorted_entries = sorted(entries, key=self._sort_key, reverse=True)
            for e in sorted_entries[1:]:
                removals[e["file_path"]].add(e["er_index"])
                stats.inc("pwc_duplicates_removed")

        # Apply removals
        for file_path, indices in removals.items():
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            ers = data.get("evaluation_results", [])
            for idx in sorted(indices, reverse=True):
                if idx < len(ers):
                    ers.pop(idx)
            if not ers:
                file_path.unlink()
                stats.inc("files_deleted_empty")
            else:
                data["evaluation_results"] = ers
                file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
                stats.inc("files_modified")

    @staticmethod
    def _extract_arxiv_date(record: dict) -> str:
        ts = record.get("evaluation_timestamp", "")
        if ts:
            return ts
        arxiv_id = record.get("source_metadata", {}).get("additional_details", {}).get("arxiv_id", "")
        if arxiv_id:
            m = re.match(r"(\d{2})(\d{2})\.", arxiv_id)
            if m:
                return f"{2000 + int(m.group(1))}-{int(m.group(2)):02d}-01"
        return "1970-01-01"

    @staticmethod
    def _sort_key(e: dict) -> tuple:
        plausible = True
        if e["score"] is not None and e["max_score"] is not None:
            plausible = not (e["max_score"] > 0 and e["score"] > e["max_score"] * 1.1)
        return (plausible, e["arxiv_date"])


# ─── FixRunner (orchestration) ────────────────────────────────────────────────


class FixRunner:
    """Orchestrate all data fixes over the data directory."""

    def __init__(self, data_dir: Path = _DATA_DIR,
                 per_file_fixes: list[DataFix] | None = None):
        self._data_dir = data_dir
        self._per_file_fixes = per_file_fixes or [ParameterLeakFix(), BenchmarkCanonFix()]
        self._pwc_dedup = PWCDedupFix()
        self.stats = FixStats()

    def run(self) -> FixStats:
        """Run all fix passes and return accumulated stats."""
        self._run_per_file_pass()
        self._run_directory_pass()
        self._print_summary()
        return self.stats

    def _run_per_file_pass(self) -> None:
        """Pass 1: per-file fixes (param leaks + benchmark canonicalization)."""
        print("Pass 1: Per-file fixes (param leaks + benchmark canonicalization)...")
        source_dirs = [d for d in self._data_dir.iterdir()
                       if d.is_dir() and d.name != "aggregated"]

        for src_dir in sorted(source_dirs):
            source = src_dir.name
            for json_path in sorted(src_dir.rglob("*.json")):
                self.stats.inc("files_scanned")
                self._fix_one_file(json_path, source)

    def _fix_one_file(self, path: Path, source: str) -> None:
        """Apply all per-file fixes to a single JSON file."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return

        any_modified = False
        for fix in self._per_file_fixes:
            data, modified = fix.apply(data, source, path, self.stats)
            any_modified = any_modified or modified

        if any_modified:
            if not data.get("evaluation_results"):
                path.unlink()
                self.stats.inc("files_deleted_empty")
            else:
                path.write_text(json.dumps(data, indent=2), encoding="utf-8")
                self.stats.inc("files_modified")

    def _run_directory_pass(self) -> None:
        """Pass 2: cross-file deduplication for PWC."""
        print("\nPass 2: PWC deduplication...")
        self._pwc_dedup.run_directory(self._data_dir, self.stats)

    def _print_summary(self) -> None:
        print(f"\n{'=' * 60}")
        print("  FIX SUMMARY")
        print(f"{'=' * 60}")
        print(f"  Files scanned:            {self.stats['files_scanned']}")
        print(f"  Parameter leaks removed:  {self.stats['param_leaks_removed']}")
        print(f"  Benchmarks canonicalized: {self.stats['benchmarks_canonicalized']}")
        print(f"  PWC duplicates removed:   {self.stats['pwc_duplicates_removed']}")
        print(f"  Files modified:           {self.stats['files_modified']}")
        print(f"  Files deleted (empty):    {self.stats['files_deleted_empty']}")


# ─── CLI entry point ──────────────────────────────────────────────────────────


def fix_all() -> dict:
    """Run all fixes. Returns stats dict (backward-compatible API)."""
    runner = FixRunner()
    stats = runner.run()
    return dict(stats.counts)


if __name__ == "__main__":
    fix_all()
