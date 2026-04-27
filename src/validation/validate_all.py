"""
batch validator with auto-fix for all EEE schema JSON files.

iterates over all JSON files in data/, validates each against the official
JSON schema, optionally auto-fixes common issues, re-validates, and writes
a report.

auto-fixes applied (requires --fix flag to write changes back to disk):
  - score stored as string → convert to float
  - missing generation_config → drop (optional field; no insertion needed)
  - schema_version missing → insert "0.2.1"
  - retrieved_timestamp stored as int → convert to string
  - evaluator_relationship "unknown" → replace with "other"
  - source_type "evaluation_platform" → replace with "evaluation_run"

without --fix the script runs in report-only mode: it validates every file,
identifies what could be fixed, but does NOT modify any file on disk.

usage:
    python scripts/validate_all.py [--data-dir data/] [--schema eval.schema.json]
    python scripts/validate_all.py --fix  # apply auto-fixes and write back
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

# add repo root to sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from jsonschema.exceptions import ValidationError
from jsonschema.validators import validator_for

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

_DEFAULT_DATA_DIR = "data"
_DEFAULT_SCHEMA_PATH = "eval.schema.json"
_REPORT_PATH = "VALIDATION_REPORT.json"

_SCHEMA_VERSION = "0.2.1"

# ---------------------------------------------------------------------------
# schema validator loader — single responsibility
# ---------------------------------------------------------------------------


def _load_validator(schema_path: str):
    """load and return a jsonschema validator for *schema_path*."""
    with open(schema_path, "r", encoding="utf-8") as fh:
        schema = json.load(fh)
    cls = validator_for(schema)
    return cls(schema)


# ---------------------------------------------------------------------------
# auto-fixer — single responsibility: repair common schema issues
# ---------------------------------------------------------------------------


class AutoFixer:
    """apply heuristic fixes to a JSON record to make it schema-compliant."""

    def fix(self, data: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        """return (fixed_data, list_of_applied_fix_descriptions)."""
        fixes: list[str] = []

        # fix 1: missing schema_version
        if "schema_version" not in data:
            data["schema_version"] = _SCHEMA_VERSION
            fixes.append("inserted missing schema_version")
        elif data["schema_version"] not in ("0.2.0", "0.2.1"):
            old = data["schema_version"]
            data["schema_version"] = _SCHEMA_VERSION
            fixes.append(f"updated schema_version from {old!r} to {_SCHEMA_VERSION!r}")

        # fix 2: retrieved_timestamp as int → string
        ts = data.get("retrieved_timestamp")
        if isinstance(ts, (int, float)):
            data["retrieved_timestamp"] = str(ts)
            fixes.append("converted retrieved_timestamp from number to string")

        # fix 3: source_metadata.source_type "evaluation_platform" → "evaluation_run"
        sm = data.get("source_metadata", {})
        if sm.get("source_type") == "evaluation_platform":
            sm["source_type"] = "evaluation_run"
            fixes.append("fixed source_type 'evaluation_platform' → 'evaluation_run'")

        # fix 4: source_metadata.evaluator_relationship "unknown" → "other"
        if sm.get("evaluator_relationship") == "unknown":
            sm["evaluator_relationship"] = "other"
            fixes.append("fixed evaluator_relationship 'unknown' → 'other'")

        # fix 5: scores stored as strings in score_details
        for result in data.get("evaluation_results", []):
            sd = result.get("score_details", {})
            score = sd.get("score")
            if isinstance(score, str):
                try:
                    sd["score"] = float(score)
                    fixes.append(
                        f"converted score string → float in {result.get('evaluation_name', '?')!r}"
                    )
                except ValueError:
                    pass

        # fix 6: eval_library missing → insert placeholder
        if "eval_library" not in data:
            data["eval_library"] = {"name": "unknown", "version": "unknown"}
            fixes.append("inserted missing eval_library placeholder")

        # fix 7: model_info.id missing → derive from name
        mi = data.get("model_info", {})
        if mi and "id" not in mi and "name" in mi:
            mi["id"] = mi["name"]
            fixes.append("derived model_info.id from model_info.name")

        return data, fixes


# ---------------------------------------------------------------------------
# validator + fixer orchestrator
# ---------------------------------------------------------------------------


class FileValidator:
    """validate a single JSON file, apply fixes if needed, re-validate.

    Parameters
    ----------
    fix_mode:
        When True (requires ``--fix`` on the CLI), auto-fixes are written
        back to disk and the file is reported as ``"fixed"``.
        When False (default, report-only), fixes are computed in-memory
        and the file is reported as ``"fixable"`` — nothing is written.
    """

    def __init__(
        self,
        schema_validator,
        auto_fixer: AutoFixer,
        fix_mode: bool = False,
    ) -> None:
        self._schema_validator = schema_validator
        self._fixer = auto_fixer
        self._fix_mode = fix_mode

    def validate_file(self, path: Path) -> dict[str, Any]:
        """return a status dict describing the validation outcome for *path*."""
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError as exc:
            return {
                "path": str(path),
                "status": "invalid",
                "error": f"JSONDecodeError: {exc}",
                "fixes_applied": [],
            }

        # first validation pass
        errors = _collect_errors(self._schema_validator, data)
        if not errors:
            return {
                "path": str(path),
                "status": "valid",
                "error": None,
                "fixes_applied": [],
            }

        # apply auto-fixes (in-memory regardless of fix_mode)
        data, fixes = self._fixer.fix(data)
        errors_after = _collect_errors(self._schema_validator, data)

        if not errors_after:
            if self._fix_mode:
                # write fixed data back to disk
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(data, fh, indent=2, ensure_ascii=False)
                return {
                    "path": str(path),
                    "status": "fixed",
                    "error": None,
                    "fixes_applied": fixes,
                }
            # report-only mode: tell the caller what would be fixed
            return {
                "path": str(path),
                "status": "fixable",
                "error": None,
                "fixes_would_apply": fixes,
            }

        # still invalid after fixes
        return {
            "path": str(path),
            "status": "invalid",
            "error": errors_after[0],
            "fixes_applied": fixes,
            "original_errors": errors,
        }


# ---------------------------------------------------------------------------
# batch runner — orchestration only
# ---------------------------------------------------------------------------


class BatchValidator:
    """run FileValidator over all JSON files in a directory tree."""

    def __init__(self, file_validator: FileValidator) -> None:
        self._file_validator = file_validator

    def run(self, data_dir: str) -> dict[str, Any]:
        """validate all .json files under *data_dir*; return a summary dict."""
        paths = sorted(Path(data_dir).rglob("*.json"))

        if not paths:
            print(f"no JSON files found in {data_dir!r}")
            return {"total": 0, "valid": 0, "fixed": 0, "invalid": 0, "files": []}

        print(f"\nvalidating {len(paths)} JSON files in {data_dir!r}...\n")

        results: list[dict] = []
        counts: dict[str, int] = {"valid": 0, "fixed": 0, "invalid": 0}

        for path in paths:
            result = self._file_validator.validate_file(path)
            results.append(result)
            status = result["status"]
            counts[status] = counts.get(status, 0) + 1

            icon = {"valid": "OK", "fixed": "~", "invalid": "!!"}.get(status, "?")
            try:
                print(f"  {icon} {path}")
            except UnicodeEncodeError:
                print(f"  {icon} {str(path).encode('ascii', 'replace').decode()}")
            if result.get("fixes_applied"):
                for fix in result["fixes_applied"]:
                    print(f"      fix: {fix}")
            if status == "invalid":
                print(f"      error: {result['error']}")

        print()
        print(f"results: {counts['valid']} valid, {counts['fixed']} fixed, {counts['invalid']} invalid")

        return {
            "generated_at": str(time.time()),
            "total": len(paths),
            "valid": counts["valid"],
            "fixed": counts["fixed"],
            "invalid": counts["invalid"],
            "pass_rate": round((counts["valid"] + counts["fixed"]) / max(len(paths), 1), 4),
            "files": results,
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _collect_errors(validator, data: dict) -> list[str]:
    """return list of validation error messages for *data*."""
    return [
        f"{type(e).__name__}: {e.message}"
        for e in validator.iter_errors(data)
    ]


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """command-line entry point for the batch validator."""
    import argparse

    parser = argparse.ArgumentParser(
        description="validate all EEE JSON files; use --fix to write corrections"
    )
    parser.add_argument(
        "--data-dir",
        default=_DEFAULT_DATA_DIR,
        help=f"root data directory (default: {_DEFAULT_DATA_DIR!r})",
    )
    parser.add_argument(
        "--schema",
        default=_DEFAULT_SCHEMA_PATH,
        help=f"path to JSON schema file (default: {_DEFAULT_SCHEMA_PATH!r})",
    )
    parser.add_argument(
        "--report",
        default=_REPORT_PATH,
        help=f"path to write VALIDATION_REPORT.json (default: {_REPORT_PATH!r})",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help=(
            "apply auto-fixes and write corrected files back to disk. "
            "Without this flag the script runs in report-only mode and "
            "no files are modified."
        ),
    )
    args = parser.parse_args()

    if args.fix:
        print("running in FIX mode — corrected files will be written to disk.")
    else:
        print(
            "running in REPORT-ONLY mode — no files will be modified. "
            "pass --fix to apply auto-corrections."
        )

    schema_path = Path(args.schema)
    if not schema_path.exists():
        # try relative to repo root
        schema_path = _ROOT / args.schema
    if not schema_path.exists():
        print(f"schema file not found: {args.schema}", file=sys.stderr)
        sys.exit(1)

    validator = _load_validator(str(schema_path))
    fixer = AutoFixer()
    file_validator = FileValidator(validator, fixer, fix_mode=args.fix)
    batch_runner = BatchValidator(file_validator)

    report = batch_runner.run(args.data_dir)

    report_path = Path(args.report)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print(f"\nvalidation report written to {report_path}")

    # exit non-zero if any files remain invalid or have unfixed issues
    if report["invalid"] > 0 or (not args.fix and report.get("fixable", 0) > 0):
        if report["invalid"] > 0:
            sys.exit(1)
        # fixable-but-not-fixed: non-zero but distinguishable exit code
        sys.exit(2)


if __name__ == "__main__":
    main()
