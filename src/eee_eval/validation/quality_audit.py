"""Comprehensive data quality audit for EEE JSON records.

Architecture (SOLID):
  - RecordLoader: loads all JSON from data/ (SRP)
  - AuditCheck (ABC): one check = one class (OCP, ISP)
  - AuditRunner: orchestrates checks and collects issues (DI)
  - Reporter: formats and prints results (SRP)

Each check class implements `run(records) -> list[Issue]` and optionally
returns supplementary stats for the reporter.
"""
from __future__ import annotations

import json
import sys
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DATA_DIR = _ROOT / "data"
_SCHEMA_PATH = _ROOT / "eval.schema.json"

# Ground-truth shots per (source, benchmark) — from add_leaderboard_records.py
_KNOWN_SHOTS: dict[tuple[str, str], int] = {
    ("open_llm_leaderboard_v2", "IFEval (strict-prompt)"): 0,
    ("open_llm_leaderboard_v2", "BBH"): 3,
    ("open_llm_leaderboard_v2", "MATH Lvl 5"): 4,
    ("open_llm_leaderboard_v2", "GPQA-Diamond"): 0,
    ("open_llm_leaderboard_v2", "MuSR"): 0,
    ("open_llm_leaderboard_v2", "MMLU-Pro"): 5,
    ("alpacaeval2", "AlpacaEval 2.0"): 0,
    ("alpacaeval2", "AlpacaEval 2.0 LC"): 0,
    ("bfcl", "BFCL-v3"): 0,
    ("bfcl", "BFCL-v3 AST"): 0,
    ("bfcl", "BFCL-v3 Live"): 0,
    ("bigcodebench", "BigCodeBench-Complete"): 0,
    ("bigcodebench", "BigCodeBench-Instruct"): 0,
    ("evalplus", "HumanEval+"): 0,
    ("evalplus", "MBPP+"): 3,
    ("swe_bench", "SWE-Bench-Verified"): 0,
}

# Benchmarks where shots is N/A (preference / arena based)
_SHOTS_NA_BENCHMARKS = {"Arena Elo", "MT-Bench", "WildBench v2", "WildBench v2 (adj)"}


# ─── Data types ───────────────────────────────────────────────────────────────


@dataclass
class Issue:
    severity: str  # CRITICAL, ERROR, WARNING, INFO
    path: str
    message: str


@dataclass
class Record:
    """One loaded JSON file with source provenance."""
    source: str
    path: Path
    data: dict[str, Any]
    parse_error: str | None = None


# ─── RecordLoader (SRP) ──────────────────────────────────────────────────────


class RecordLoader:
    """Load all JSON files from data/ subdirectories."""

    def __init__(self, data_dir: Path = _DATA_DIR, exclude: set[str] | None = None):
        self._data_dir = data_dir
        self._exclude = exclude or {"aggregated", "figures", "html_cache"}

    def load(self) -> list[Record]:
        records: list[Record] = []
        for src_dir in sorted(self._data_dir.iterdir()):
            if not src_dir.is_dir() or src_dir.name in self._exclude:
                continue
            for json_path in src_dir.rglob("*.json"):
                records.append(self._load_one(src_dir.name, json_path))
        return records

    def _load_one(self, source: str, path: Path) -> Record:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return Record(source=source, path=path, data=data)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            return Record(source=source, path=path, data={}, parse_error=str(exc))


# ─── AuditCheck ABC (OCP / ISP) ──────────────────────────────────────────────


class AuditCheck(ABC):
    """Base class for all audit checks."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable check name."""

    @abstractmethod
    def run(self, records: list[Record]) -> list[Issue]:
        """Execute the check and return issues found."""


# ─── Concrete checks ─────────────────────────────────────────────────────────


class SchemaCheck(AuditCheck):
    """Validate all records against eval.schema.json."""

    name = "Schema Validation"

    def __init__(self, schema_path: Path = _SCHEMA_PATH):
        self._schema_path = schema_path

    def run(self, records: list[Record]) -> list[Issue]:
        issues: list[Issue] = []
        try:
            schema = json.loads(self._schema_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return [Issue("SKIP", "N/A", f"Schema not found: {self._schema_path}")]

        for rec in records:
            if rec.parse_error:
                issues.append(Issue("CRITICAL", str(rec.path), f"JSON parse error: {rec.parse_error}"))
                continue
            errs = list(jsonschema.Draft7Validator(schema).iter_errors(rec.data))
            for e in errs:
                issues.append(Issue("ERROR", str(rec.path), f"Schema: {e.message[:200]}"))
        return issues


class ShotsCheck(AuditCheck):
    """Audit the shots field: ground-truth comparison, suspicious zeros, N/A benchmarks."""

    name = "Shots Field Audit"

    def __init__(self):
        self.stats: dict[str, Counter] = defaultdict(Counter)

    def run(self, records: list[Record]) -> list[Issue]:
        issues: list[Issue] = []
        for rec in records:
            if rec.parse_error:
                continue
            for er in rec.data.get("evaluation_results", []):
                issues.extend(self._check_one(rec, er))
        return issues

    def _check_one(self, rec: Record, er: dict) -> list[Issue]:
        issues: list[Issue] = []
        bench = er.get("evaluation_name", "?")
        gen_args = er.get("generation_config", {}).get("generation_args", {})
        shots_val = gen_args.get("shots")
        has_shots = "shots" in gen_args
        key = (rec.source, bench)

        # Track distribution
        self.stats[rec.source][f"shots={shots_val}" if has_shots else "shots=ABSENT"] += 1

        # Ground-truth comparison
        if key in _KNOWN_SHOTS:
            expected = _KNOWN_SHOTS[key]
            if not has_shots:
                issues.append(Issue("WARNING", str(rec.path),
                    f"[{rec.source}/{bench}] shots ABSENT but expected {expected}"))
            elif shots_val != expected:
                issues.append(Issue("ERROR", str(rec.path),
                    f"[{rec.source}/{bench}] shots={shots_val} but expected {expected}"))

        # Suspicious zero on unknown benchmark
        if has_shots and shots_val == 0 and key not in _KNOWN_SHOTS and bench not in _SHOTS_NA_BENCHMARKS:
            issues.append(Issue("INFO", str(rec.path),
                f"[{rec.source}/{bench}] shots=0 — verify genuinely 0-shot"))

        # Shots set on preference-based benchmark
        if has_shots and shots_val is not None and bench in _SHOTS_NA_BENCHMARKS:
            issues.append(Issue("WARNING", str(rec.path),
                f"[{rec.source}/{bench}] shots={shots_val} but benchmark is preference-based"))

        return issues


class HarnessCheck(AuditCheck):
    """Audit eval_library name and version fields."""

    name = "Harness / eval_library Audit"

    def __init__(self):
        self.stats: dict[str, Counter] = defaultdict(Counter)

    def run(self, records: list[Record]) -> list[Issue]:
        issues: list[Issue] = []
        for rec in records:
            if rec.parse_error:
                continue
            lib = rec.data.get("eval_library", {})
            name = lib.get("name", "MISSING")
            version = lib.get("version", "MISSING")
            self.stats[rec.source][(name, version)] += 1

            if name == "MISSING":
                issues.append(Issue("ERROR", str(rec.path),
                    f"[{rec.source}] eval_library.name is MISSING"))
            if version == "unknown":
                issues.append(Issue("INFO", str(rec.path),
                    f"[{rec.source}] eval_library.version='unknown' (harness: {name})"))
            elif version == "MISSING":
                issues.append(Issue("ERROR", str(rec.path),
                    f"[{rec.source}] eval_library.version is MISSING"))
        return issues


class ScoreCheck(AuditCheck):
    """Audit score values: range, outliers, parameter-leak detection."""

    name = "Score Sanity Checks"

    def __init__(self):
        self.ranges: dict[tuple[str, str], list[float]] = defaultdict(list)

    def run(self, records: list[Record]) -> list[Issue]:
        issues: list[Issue] = []
        for rec in records:
            if rec.parse_error:
                continue
            for er in rec.data.get("evaluation_results", []):
                issues.extend(self._check_one(rec, er))
        return issues

    def _check_one(self, rec: Record, er: dict) -> list[Issue]:
        issues: list[Issue] = []
        bench = er.get("evaluation_name", "?")
        score = er.get("score_details", {}).get("score")

        if score is None:
            issues.append(Issue("ERROR", str(rec.path), f"[{rec.source}/{bench}] score is None/missing"))
            return issues

        self.ranges[(rec.source, bench)].append(score)
        metric_cfg = er.get("metric_config", {})
        min_s = metric_cfg.get("min_score")
        max_s = metric_cfg.get("max_score")

        if min_s is not None and score < min_s:
            issues.append(Issue("ERROR", str(rec.path),
                f"[{rec.source}/{bench}] score={score} < min_score={min_s}"))
        if max_s is not None and score > max_s:
            issues.append(Issue("ERROR", str(rec.path),
                f"[{rec.source}/{bench}] score={score} > max_score={max_s}"))
        # Parameter-count leak heuristic
        if max_s is not None and max_s <= 100 and score > 100:
            issues.append(Issue("CRITICAL", str(rec.path),
                f"[{rec.source}/{bench}] score={score} on 0-100 scale — likely parameter leak"))
        if score == 0.0:
            issues.append(Issue("WARNING", str(rec.path),
                f"[{rec.source}/{bench}] score=0.0 — may be missing/default"))
        if min_s is not None and min_s >= 0 and score < 0:
            issues.append(Issue("ERROR", str(rec.path),
                f"[{rec.source}/{bench}] score={score} negative on non-negative benchmark"))
        return issues


class DuplicateCheck(AuditCheck):
    """Detect within-source duplicate (model, benchmark) pairs."""

    name = "Within-Source Duplicates"

    def run(self, records: list[Record]) -> list[Issue]:
        seen: dict[tuple, list[tuple]] = defaultdict(list)
        for rec in records:
            if rec.parse_error:
                continue
            model_id = rec.data.get("model_info", {}).get("id", "?")
            for er in rec.data.get("evaluation_results", []):
                bench = er.get("evaluation_name", "?")
                score = er.get("score_details", {}).get("score")
                seen[(rec.source, model_id, bench)].append((score, str(rec.path)))

        issues: list[Issue] = []
        for key, entries in seen.items():
            if len(entries) <= 1:
                continue
            scores = [e[0] for e in entries]
            if len(set(scores)) > 1:
                issues.append(Issue("WARNING", entries[0][1],
                    f"[{key[0]}/{key[2]}] model={key[1]} {len(entries)} records, different scores: {scores}"))
            else:
                issues.append(Issue("INFO", entries[0][1],
                    f"[{key[0]}/{key[2]}] model={key[1]} {len(entries)} duplicate records (same score)"))
        return issues


class RequiredFieldsCheck(AuditCheck):
    """Check all schema-required fields are present and non-empty."""

    name = "Required Field Completeness"

    _TOP_FIELDS = ["schema_version", "evaluation_id", "retrieved_timestamp",
                   "source_metadata", "model_info", "eval_library", "evaluation_results"]

    def run(self, records: list[Record]) -> list[Issue]:
        issues: list[Issue] = []
        for rec in records:
            if rec.parse_error:
                continue
            issues.extend(self._check_top_level(rec))
            issues.extend(self._check_nested(rec))
        return issues

    def _check_top_level(self, rec: Record) -> list[Issue]:
        issues: list[Issue] = []
        for f in self._TOP_FIELDS:
            if f not in rec.data:
                issues.append(Issue("ERROR", str(rec.path), f"[{rec.source}] Missing: {f}"))
            elif rec.data[f] is None:
                issues.append(Issue("ERROR", str(rec.path), f"[{rec.source}] Null: {f}"))
        return issues

    def _check_nested(self, rec: Record) -> list[Issue]:
        issues: list[Issue] = []
        sm = rec.data.get("source_metadata", {})
        for sf in ["source_type", "source_organization_name", "evaluator_relationship"]:
            if sf not in sm or not sm[sf]:
                issues.append(Issue("ERROR", str(rec.path), f"[{rec.source}] Missing source_metadata.{sf}"))

        el = rec.data.get("eval_library", {})
        for ef in ["name", "version"]:
            if ef not in el:
                issues.append(Issue("ERROR", str(rec.path), f"[{rec.source}] Missing eval_library.{ef}"))

        ers = rec.data.get("evaluation_results", [])
        if not ers:
            issues.append(Issue("ERROR", str(rec.path), f"[{rec.source}] evaluation_results is empty"))
        for er in ers:
            for rf in ["evaluation_name", "source_data", "metric_config", "score_details"]:
                if rf not in er:
                    issues.append(Issue("ERROR", str(rec.path),
                        f"[{rec.source}] Missing evaluation_results[].{rf}"))
        return issues


class BenchmarkNamingCheck(AuditCheck):
    """Detect near-duplicate benchmark names across sources."""

    name = "Benchmark Naming Consistency"

    def __init__(self):
        self.bench_by_source: dict[str, set[str]] = defaultdict(set)

    def run(self, records: list[Record]) -> list[Issue]:
        for rec in records:
            if rec.parse_error:
                continue
            for er in rec.data.get("evaluation_results", []):
                self.bench_by_source[rec.source].add(er.get("evaluation_name", "?"))

        all_benchmarks = sorted({b for bs in self.bench_by_source.values() for b in bs})
        issues: list[Issue] = []
        for i, b1 in enumerate(all_benchmarks):
            norm1 = b1.lower().replace("-", "").replace("_", "").replace(" ", "")
            for b2 in all_benchmarks[i + 1:]:
                norm2 = b2.lower().replace("-", "").replace("_", "").replace(" ", "")
                if norm1 == norm2 and b1 != b2:
                    issues.append(Issue("WARNING", "N/A",
                        f"Near-duplicate benchmarks: '{b1}' vs '{b2}'"))
        return issues


class ModelIdentityCheck(AuditCheck):
    """Check model_info fields for completeness and formatting."""

    name = "Model Identity Checks"

    def run(self, records: list[Record]) -> list[Issue]:
        issues: list[Issue] = []
        for rec in records:
            if rec.parse_error:
                continue
            mi = rec.data.get("model_info", {})
            name = mi.get("name", "")
            mid = mi.get("id", "")
            dev = mi.get("developer", "")

            if not name:
                issues.append(Issue("ERROR", str(rec.path), f"[{rec.source}] model_info.name empty"))
            if not mid:
                issues.append(Issue("ERROR", str(rec.path), f"[{rec.source}] model_info.id empty"))
            if not dev:
                issues.append(Issue("WARNING", str(rec.path),
                    f"[{rec.source}] model_info.developer empty for {mid}"))
            if mid and "/" not in mid:
                issues.append(Issue("WARNING", str(rec.path),
                    f"[{rec.source}] model_info.id='{mid}' missing developer/ prefix"))
        return issues


class MetadataCoverageCheck(AuditCheck):
    """Compute metadata field coverage per source (for reporting)."""

    name = "Metadata Coverage"

    _FIELDS = ["shots", "temperature", "top_p", "prompt_template", "chain_of_thought"]

    def __init__(self):
        self.coverage: dict[str, dict[str, dict[str, int]]] = defaultdict(
            lambda: {f: {"filled": 0, "total": 0} for f in self._FIELDS})
        self.harness_cov: dict[str, dict[str, int]] = defaultdict(
            lambda: {"known": 0, "unknown": 0, "total": 0})

    def run(self, records: list[Record]) -> list[Issue]:
        for rec in records:
            if rec.parse_error:
                continue
            lib_name = rec.data.get("eval_library", {}).get("name", "")
            self.harness_cov[rec.source]["total"] += 1
            if lib_name and lib_name != "unknown":
                self.harness_cov[rec.source]["known"] += 1
            else:
                self.harness_cov[rec.source]["unknown"] += 1

            for er in rec.data.get("evaluation_results", []):
                gen_args = er.get("generation_config", {}).get("generation_args", {})
                for f in self._FIELDS:
                    self.coverage[rec.source][f]["total"] += 1
                    val = gen_args.get(f)
                    if val is not None and val != "" and val != "unknown":
                        self.coverage[rec.source][f]["filled"] += 1
        return []  # No issues — purely informational


# ─── AuditRunner (DI / orchestration) ────────────────────────────────────────


class AuditRunner:
    """Run a sequence of AuditCheck instances and collect all issues."""

    def __init__(self, checks: list[AuditCheck] | None = None):
        self.checks = checks or self._default_checks()
        self.results: dict[str, list[Issue]] = {}

    @staticmethod
    def _default_checks() -> list[AuditCheck]:
        return [
            SchemaCheck(),
            ShotsCheck(),
            HarnessCheck(),
            ScoreCheck(),
            DuplicateCheck(),
            RequiredFieldsCheck(),
            BenchmarkNamingCheck(),
            ModelIdentityCheck(),
            MetadataCoverageCheck(),
        ]

    def run_all(self, records: list[Record]) -> dict[str, list[Issue]]:
        for check in self.checks:
            self.results[check.name] = check.run(records)
        return self.results

    @property
    def all_issues(self) -> list[Issue]:
        return [i for issues in self.results.values() for i in issues]


# ─── Reporter (SRP) ──────────────────────────────────────────────────────────


class Reporter:
    """Format and print audit results to stdout."""

    def __init__(self, records: list[Record], runner: AuditRunner):
        self._records = records
        self._runner = runner

    def print_summary(self) -> None:
        total = len(self._records)
        parse_errors = sum(1 for r in self._records if r.parse_error)
        total_evals = sum(
            len(r.data.get("evaluation_results", []))
            for r in self._records if not r.parse_error)

        source_counts: Counter = Counter()
        source_evals: Counter = Counter()
        for r in self._records:
            source_counts[r.source] += 1
            if not r.parse_error:
                source_evals[r.source] += len(r.data.get("evaluation_results", []))

        print(f"\nTotal files:              {total}")
        print(f"Parse errors:             {parse_errors}")
        print(f"Total evaluation records: {total_evals}")
        print("\nPer-source:")
        for src in sorted(source_counts):
            print(f"  {src:30s} {source_counts[src]:5d} files  {source_evals[src]:5d} eval records")

    def print_check_results(self) -> None:
        for check in self._runner.checks:
            issues = self._runner.results.get(check.name, [])
            self._print_section(check.name)
            self._print_issues_summary(issues)

            # Print supplementary stats from checks that have them
            if hasattr(check, "stats"):
                self._print_stats(check.stats)
            if hasattr(check, "coverage"):
                self._print_coverage(check)
            if hasattr(check, "ranges") and check.ranges:
                self._print_score_ranges(check.ranges)

    def print_final_verdict(self) -> int:
        self._print_section("QUALITY AUDIT SUMMARY")
        all_issues = self._runner.all_issues
        counts = Counter(i.severity for i in all_issues)
        print(f"  CRITICAL: {counts.get('CRITICAL', 0)}")
        print(f"  ERROR:    {counts.get('ERROR', 0)}")
        print(f"  WARNING:  {counts.get('WARNING', 0)}")
        print(f"  INFO:     {counts.get('INFO', 0)}")

        if counts.get("CRITICAL", 0) > 0 or counts.get("ERROR", 0) > 0:
            print("\n  *** DATA HAS ISSUES REQUIRING ATTENTION ***")
            return 1
        print("\n  Data quality: PASS")
        return 0

    # ── helpers ──

    @staticmethod
    def _print_section(title: str) -> None:
        print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")

    @staticmethod
    def _print_issues_summary(issues: list[Issue], limit: int = 15) -> None:
        if not issues:
            print("  No issues.")
            return
        counts = Counter(i.severity for i in issues)
        for sev in ("CRITICAL", "ERROR", "WARNING", "INFO"):
            if counts.get(sev, 0):
                print(f"  {sev}: {counts[sev]}")
        shown = [i for i in issues if i.severity in ("CRITICAL", "ERROR", "WARNING")][:limit]
        for i in shown:
            print(f"    [{i.severity}] {i.message}")
        remaining = len(issues) - len(shown)
        if remaining > 0:
            print(f"    ... and {remaining} more")

    @staticmethod
    def _print_stats(stats: dict) -> None:
        print("\n  Distribution per source:")
        for src in sorted(stats):
            print(f"    {src}:")
            for val, cnt in stats[src].most_common(5):
                label = str(val) if not isinstance(val, tuple) else f"{val[0]} v{val[1]}"
                print(f"      {label:30s} {cnt:5d}")

    @staticmethod
    def _print_coverage(check: MetadataCoverageCheck) -> None:
        header = f"  {'Source':30s} {'harness':>8s} {'shots':>8s} {'temp':>8s} {'top_p':>8s} {'prompt':>8s} {'CoT':>8s}"
        print(f"\n{header}\n  {'-' * (len(header) - 2)}")
        for src in sorted(check.coverage):
            hc = check.harness_cov[src]
            h_pct = f"{hc['known'] / max(hc['total'], 1) * 100:.0f}%"
            row = f"  {src:30s} {h_pct:>8s}"
            for f in check._FIELDS:
                c = check.coverage[src][f]
                pct = c["filled"] / max(c["total"], 1) * 100
                row += f" {pct:7.1f}%"
            print(row)

    @staticmethod
    def _print_score_ranges(ranges: dict[tuple, list[float]], top_n: int = 15) -> None:
        print(f"\n  Score ranges (top {top_n} by count):")
        sorted_r = sorted(ranges.items(), key=lambda x: -len(x[1]))[:top_n]
        for (src, bench), scores in sorted_r:
            print(f"    {src}/{bench}: min={min(scores):.2f} max={max(scores):.2f} "
                  f"mean={sum(scores)/len(scores):.2f} n={len(scores)}")


# ─── CLI entry point ──────────────────────────────────────────────────────────


def main() -> int:
    print("Loading all records...")
    loader = RecordLoader()
    records = loader.load()

    runner = AuditRunner()
    runner.run_all(records)

    reporter = Reporter(records, runner)
    reporter.print_summary()
    reporter.print_check_results()
    return reporter.print_final_verdict()


if __name__ == "__main__":
    sys.exit(main())
