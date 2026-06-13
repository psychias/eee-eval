"""Schema validation and auto-fix for EEE JSON records.

Single Responsibility: Each class does one thing.
  - SchemaValidator: loads schema, validates a dict
  - AutoFixer: applies heuristic repairs to make records schema-compliant
  - FileValidator: orchestrates validate → fix → re-validate for one file
  - BatchValidator: runs FileValidator over a directory tree

Open/Closed: New fix rules → add methods to AutoFixer; new checks → subclass.
Dependency Inversion: BatchValidator depends on FileValidator (injected).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from jsonschema.exceptions import ValidationError
from jsonschema.validators import validator_for

_ROOT = Path(__file__).resolve().parent.parent.parent
_SCHEMA_VERSION = "0.2.1"


# ---------------------------------------------------------------------------
# Schema loading (Single Responsibility)
# ---------------------------------------------------------------------------


class SchemaValidator:
    """Validate dicts against eval.schema.json."""

    def __init__(self, schema_path: Path | str | None = None):
        if schema_path is None:
            schema_path = _ROOT / "eval.schema.json"
        schema_path = Path(schema_path)
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema not found: {schema_path}")
        with schema_path.open(encoding="utf-8") as fh:
            schema = json.load(fh)
        cls = validator_for(schema)
        self._validator = cls(schema)

    def validate(self, data: dict) -> None:
        """Raise ValidationError if data does not conform to schema."""
        self._validator.validate(data)

    def iter_errors(self, data: dict):
        """Yield all validation errors for data."""
        return self._validator.iter_errors(data)

    def is_valid(self, data: dict) -> bool:
        """Return True if data passes schema validation."""
        return self._validator.is_valid(data)

    def collect_errors(self, data: dict) -> list[str]:
        """Return list of error message strings."""
        return [
            f"{type(e).__name__}: {e.message}"
            for e in self._validator.iter_errors(data)
        ]


# ---------------------------------------------------------------------------
# Auto-fixer (Single Responsibility: repair common schema issues)
# ---------------------------------------------------------------------------


class AutoFixer:
    """Apply heuristic fixes to make a JSON record schema-compliant.

    Each fix is an independent method — add new fixes by adding methods
    prefixed with `_fix_`.
    """

    def fix(self, data: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        """Return (fixed_data, list_of_applied_fix_descriptions)."""
        fixes: list[str] = []
        self._fix_schema_version(data, fixes)
        self._fix_retrieved_timestamp(data, fixes)
        self._fix_source_type(data, fixes)
        self._fix_evaluator_relationship(data, fixes)
        self._fix_score_strings(data, fixes)
        self._fix_eval_library(data, fixes)
        self._fix_model_id(data, fixes)
        self._fix_additional_details_types(data, fixes)
        self._fix_legacy_top_level_fields(data, fixes)
        self._fix_generation_args(data, fixes)
        self._fix_metric_config_nulls(data, fixes)
        return data, fixes

    def _fix_schema_version(self, data: dict, fixes: list[str]) -> None:
        if "schema_version" not in data:
            data["schema_version"] = _SCHEMA_VERSION
            fixes.append("inserted missing schema_version")
        elif data["schema_version"] not in ("0.2.0", "0.2.1"):
            old = data["schema_version"]
            data["schema_version"] = _SCHEMA_VERSION
            fixes.append(f"updated schema_version from {old!r} to {_SCHEMA_VERSION!r}")

    def _fix_retrieved_timestamp(self, data: dict, fixes: list[str]) -> None:
        ts = data.get("retrieved_timestamp")
        if isinstance(ts, (int, float)):
            data["retrieved_timestamp"] = str(ts)
            fixes.append("converted retrieved_timestamp from number to string")

    def _fix_source_type(self, data: dict, fixes: list[str]) -> None:
        sm = data.get("source_metadata", {})
        if sm.get("source_type") == "evaluation_platform":
            sm["source_type"] = "evaluation_run"
            fixes.append("fixed source_type 'evaluation_platform' → 'evaluation_run'")
        elif sm.get("source_type") == "paper_html":
            sm["source_type"] = "documentation"
            fixes.append("fixed source_type 'paper_html' → 'documentation'")

    def _fix_evaluator_relationship(self, data: dict, fixes: list[str]) -> None:
        sm = data.get("source_metadata", {})
        if sm.get("evaluator_relationship") == "unknown":
            sm["evaluator_relationship"] = "other"
            fixes.append("fixed evaluator_relationship 'unknown' → 'other'")
        elif sm.get("evaluator_relationship") == "self_reported":
            sm["evaluator_relationship"] = "first_party"
            fixes.append("fixed evaluator_relationship 'self_reported' → 'first_party'")

    def _fix_score_strings(self, data: dict, fixes: list[str]) -> None:
        for result in data.get("evaluation_results", []):
            sd = result.get("score_details", {})
            score = sd.get("score")
            if isinstance(score, str):
                try:
                    sd["score"] = float(score)
                    name = result.get("evaluation_name", "?")
                    fixes.append(f"converted score string → float in {name!r}")
                except ValueError:
                    pass

    def _fix_eval_library(self, data: dict, fixes: list[str]) -> None:
        if "eval_library" not in data:
            data["eval_library"] = {"name": "unknown", "version": "unknown"}
            fixes.append("inserted missing eval_library placeholder")
        else:
            el = data["eval_library"]
            if el.get("version") is None:
                el["version"] = "unknown"
                fixes.append("fixed eval_library.version null → 'unknown'")

    def _fix_additional_details_types(self, data: dict, fixes: list[str]) -> None:
        """Ensure all additional_details values are strings."""
        for obj in [data.get("source_metadata", {}), data.get("eval_library", {})]:
            ad = obj.get("additional_details", {})
            for k, v in list(ad.items()):
                if not isinstance(v, str):
                    ad[k] = str(v)
                    fixes.append(f"converted additional_details[{k!r}] to string")

    def _fix_legacy_top_level_fields(self, data: dict, fixes: list[str]) -> None:
        """Move legacy validated/incorrect_values into source_metadata.additional_details."""
        has_v = "validated" in data
        has_i = "incorrect_values" in data
        if not has_v and not has_i:
            return
        sm = data.setdefault("source_metadata", {})
        ad = sm.setdefault("additional_details", {})
        if has_v:
            ad["validated"] = str(data.pop("validated"))
        if has_i:
            iv = data.pop("incorrect_values")
            ad["incorrect_values"] = "; ".join(str(x) for x in iv) if isinstance(iv, list) else str(iv)
        fixes.append("moved validated/incorrect_values into source_metadata.additional_details")

    def _fix_generation_args(self, data: dict, fixes: list[str]) -> None:
        """Move non-schema keys from generation_args to generation_config.additional_details."""
        _ALLOWED = {"shots", "temperature", "top_p", "top_k", "max_tokens",
                    "execution_command", "reasoning", "reasoning_mode",
                    "chain_of_thought", "prompt_template", "agentic_eval_config",
                    "eval_plan", "eval_limits", "sandbox", "max_attempts",
                    "incorrect_attempt_feedback"}
        for result in data.get("evaluation_results", []):
            gc = result.get("generation_config", {})
            ga = gc.get("generation_args", {})
            bad_keys = [k for k in ga if k not in _ALLOWED]
            if bad_keys:
                ad = gc.setdefault("additional_details", {})
                for k in bad_keys:
                    ad[k] = str(ga.pop(k))
                fixes.append(f"moved {bad_keys} from generation_args to additional_details")
            # Rename n_shot → shots
            if "n_shot" in ga:
                ga["shots"] = ga.pop("n_shot")
                fixes.append("renamed generation_args.n_shot → shots")

    def _fix_metric_config_nulls(self, data: dict, fixes: list[str]) -> None:
        """Replace null max_score/min_score with defaults."""
        for result in data.get("evaluation_results", []):
            mc = result.get("metric_config", {})
            if mc.get("max_score") is None and "max_score" in mc:
                mc["max_score"] = 100.0
                fixes.append("fixed metric_config.max_score null → 100.0")
            if mc.get("min_score") is None and "min_score" in mc:
                mc["min_score"] = 0.0
                fixes.append("fixed metric_config.min_score null → 0.0")

    def _fix_model_id(self, data: dict, fixes: list[str]) -> None:
        mi = data.get("model_info", {})
        if mi and "id" not in mi and "name" in mi:
            mi["id"] = mi["name"]
            fixes.append("derived model_info.id from model_info.name")


# ---------------------------------------------------------------------------
# File-level validation (orchestrates validate → fix → re-validate)
# ---------------------------------------------------------------------------


class FileValidator:
    """Validate a single JSON file, optionally applying auto-fixes.

    Parameters
    ----------
    schema_validator : SchemaValidator
    auto_fixer : AutoFixer
    fix_mode : bool
        When True, write fixes back to disk.
        When False, report what would be fixed without writing.
    """

    def __init__(
        self,
        schema_validator: SchemaValidator,
        auto_fixer: AutoFixer,
        fix_mode: bool = False,
    ) -> None:
        self._schema = schema_validator
        self._fixer = auto_fixer
        self._fix_mode = fix_mode

    def validate_file(self, path: Path) -> dict[str, Any]:
        """Return a status dict for the validation outcome of *path*."""
        data = self._load_json(path)
        if data is None:
            return self._error_result(path, "JSONDecodeError or not a dict")

        # First pass validation
        errors = self._schema.collect_errors(data)
        if not errors:
            return {"path": str(path), "status": "valid", "fixes_applied": []}

        # Attempt auto-fix
        data, fixes = self._fixer.fix(data)
        errors_after = self._schema.collect_errors(data)

        if not errors_after:
            if self._fix_mode:
                self._write_json(path, data)
                return {"path": str(path), "status": "fixed", "fixes_applied": fixes}
            return {"path": str(path), "status": "fixable", "fixes_would_apply": fixes}

        # Still invalid after fixes
        return {
            "path": str(path),
            "status": "invalid",
            "error": errors_after[0],
            "fixes_applied": fixes,
        }

    @staticmethod
    def _load_json(path: Path) -> dict | None:
        try:
            with path.open(encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else None
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    @staticmethod
    def _write_json(path: Path, data: dict) -> None:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

    @staticmethod
    def _error_result(path: Path, error: str) -> dict:
        return {"path": str(path), "status": "invalid", "error": error, "fixes_applied": []}


# ---------------------------------------------------------------------------
# Batch validation (orchestration only)
# ---------------------------------------------------------------------------


class BatchValidator:
    """Run FileValidator over all JSON files in a directory tree."""

    def __init__(self, file_validator: FileValidator) -> None:
        self._file_validator = file_validator

    def run(self, data_dir: str | Path, quiet: bool = False) -> dict[str, Any]:
        """Validate all .json files under *data_dir*; return summary dict."""
        data_dir = Path(data_dir)
        paths = sorted(data_dir.rglob("*.json"))
        # Exclude aggregated directory
        paths = [p for p in paths if "aggregated" not in p.parts]

        if not paths:
            return {"total": 0, "valid": 0, "fixed": 0, "invalid": 0, "files": []}

        if not quiet:
            print(f"\nValidating {len(paths)} JSON files in {data_dir}...\n")

        results: list[dict] = []
        counts = {"valid": 0, "fixed": 0, "fixable": 0, "invalid": 0}

        for path in paths:
            result = self._file_validator.validate_file(path)
            results.append(result)
            status = result["status"]
            counts[status] = counts.get(status, 0) + 1

            if not quiet:
                icon = {"valid": "OK", "fixed": "~", "fixable": "?", "invalid": "!!"}.get(status, "?")
                print(f"  {icon} {path}")
                if result.get("fixes_applied"):
                    for fix in result["fixes_applied"]:
                        print(f"      fix: {fix}")
                if status == "invalid":
                    print(f"      error: {result.get('error', 'unknown')}")

        if not quiet:
            print(f"\nResults: {counts['valid']} valid, {counts['fixed']} fixed, "
                  f"{counts['invalid']} invalid")

        return {
            "generated_at": str(time.time()),
            "total": len(paths),
            **counts,
            "pass_rate": round(
                (counts["valid"] + counts["fixed"]) / max(len(paths), 1), 4
            ),
            "files": results,
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Command-line entry point for batch schema validation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate all EEE JSON files; use --fix to write corrections"
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--schema", default=str(_ROOT / "eval.schema.json"))
    parser.add_argument("--report", default="VALIDATION_REPORT.json")
    parser.add_argument("--fix", action="store_true",
                        help="Apply auto-fixes and write back to disk")
    args = parser.parse_args()

    if args.fix:
        print("Running in FIX mode — corrected files will be written to disk.")
    else:
        print("Running in REPORT-ONLY mode — no files will be modified.")

    schema_validator = SchemaValidator(args.schema)
    fixer = AutoFixer()
    file_val = FileValidator(schema_validator, fixer, fix_mode=args.fix)
    batch = BatchValidator(file_val)

    report = batch.run(args.data_dir)

    report_path = Path(args.report)
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"\nValidation report written to {report_path}")

    if report["invalid"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
