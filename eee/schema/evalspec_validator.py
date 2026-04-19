"""
validator.py — EvalSpec v0.1 Compliance Validator
==================================================
Validates EEE-schema or EvalSpec records against the three-tier
specification defined in §4.5 and Appendix F of the paper.

Usage:
    # Validate a single record
    python validator.py --input record.json --tier required

    # Batch validate a directory
    python validator.py --input records/ --batch --output summary_report.json

    # Validate and save per-record report
    python validator.py --input record.json --output report.json

Output fields:
    compliance_level:           none | required | recommended | full
    completeness_score:         0–5 (matching §5.2 paper metric)
    required_missing:           list of missing Required fields
    recommended_missing:        list of missing Recommended fields
    full_missing:               list of missing Full/Optional fields
    type_errors:                list of type validation errors
    predicted_divergence_risk:  low | medium | high | unknown
    eee_compatibility:          true | false
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# EvalSpec v0.1 field definitions
# ---------------------------------------------------------------------------

# Required fields (EvalSpec-R): failure if missing or null
REQUIRED_FIELDS = [
    "model_id",
    "benchmark_id",
    "scoring_mode",
    "eval_library.name",
    "eval_library.version",
    "n_shot",
    "date",
    "random_seed",
]

# Recommended fields (EvalSpec-Rec): warn if missing
RECOMMENDED_FIELDS = [
    "prompt_template",
    "temperature",
    "top_p",
    "quantization",
    "hardware",
    "software_environment",
]

# Optional / Full fields (EvalSpec-F): info if missing
OPTIONAL_FIELDS = [
    "provenance_link",
    "contamination_check",
    "compute_hours",
]

# Valid scoring_mode values
SCORING_MODES = {"log_likelihood", "generation", "preference", "execution", "other"}

# ISO 8601 date regex (approximate)
ISO_DATE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?)?$"
)

# ---------------------------------------------------------------------------
# EEE → EvalSpec field mapping
# (EEE schema path → EvalSpec field name)
# ---------------------------------------------------------------------------

EEE_TO_EVALSPEC = {
    # Required
    "model_info.id": "model_id",
    "evaluation_results[].evaluation_name": "benchmark_id",
    "evaluation_results[].generation_config.scoring_mode": "scoring_mode",
    "eval_library.name": "eval_library.name",
    "eval_library.version": "eval_library.version",
    "evaluation_results[].generation_config.generation_args.shots": "n_shot",
    "evaluation_timestamp": "date",
    "evaluation_results[].generation_config.generation_args.random_seed": "random_seed",
    # Recommended
    "evaluation_results[].generation_config.generation_args.prompt_template": "prompt_template",
    "evaluation_results[].generation_config.generation_args.temperature": "temperature",
    "evaluation_results[].generation_config.generation_args.top_p": "top_p",
    "model_info.quantization": "quantization",
    "source_metadata.hardware": "hardware",
    "source_metadata.software_environment": "software_environment",
}

# Fields contributing to the 0-5 completeness score used in the paper
# (harness, n_shot, chain-of-thought, temperature, prompt_template)
COMPLETENESS_FIELDS = [
    "eval_library.name",   # harness
    "n_shot",              # n_shot
    "chain_of_thought",    # cot (separate from EvalSpec fields)
    "temperature",         # temperature
    "prompt_template",     # prompt template
]


# ---------------------------------------------------------------------------
# Normalisation utilities
# ---------------------------------------------------------------------------

def get_nested(obj: Any, path: str, default: Any = None) -> Any:
    """Traverse a dot-separated path in a nested dict.
    Handles 'key[]' notation by returning the first element of a list."""
    parts = path.replace("[]", "").split(".")
    cur = obj
    for part in parts:
        if isinstance(cur, dict):
            cur = cur.get(part, default)
        elif isinstance(cur, list):
            cur = cur[0] if cur else default
        else:
            return default
        if cur is None:
            return default
    return cur


def is_nontrivial(value: Any) -> bool:
    """Return True if the value is a non-trivial (non-empty, non-unknown) value."""
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in ("", "unknown", "n/a", "null", "none")
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, bool):
        return True
    return bool(value)


# ---------------------------------------------------------------------------
# EEE record normalisation
# ---------------------------------------------------------------------------

def detect_eee_record(record: dict) -> bool:
    """Return True if the record looks like an EEE schema record."""
    return "evaluation_results" in record


def normalise_eee_to_evalspec(record: dict) -> dict:
    """
    Convert an EEE schema record into a flat EvalSpec-compatible dict.
    Takes the first evaluation_result entry if multiple exist.
    """
    flat: dict[str, Any] = {}

    # model_id
    flat["model_id"] = get_nested(record, "model_info.id") or \
                       get_nested(record, "model_info.name")

    # benchmark_id: use first evaluation result
    results = record.get("evaluation_results", [])
    first_result = results[0] if results else {}
    flat["benchmark_id"] = first_result.get("evaluation_name")

    # scoring_mode: check generation_config
    gen_config = first_result.get("generation_config", {})
    flat["scoring_mode"] = gen_config.get("scoring_mode")

    # eval_library
    el = record.get("eval_library", {})
    flat["eval_library.name"] = el.get("name")
    flat["eval_library.version"] = el.get("version")

    # generation_args
    gen_args = gen_config.get("generation_args", {})
    flat["n_shot"] = gen_args.get("shots") or gen_args.get("n_shot")
    flat["prompt_template"] = gen_args.get("prompt_template")
    flat["temperature"] = gen_args.get("temperature")
    flat["top_p"] = gen_args.get("top_p")
    flat["random_seed"] = gen_args.get("random_seed") or \
                          gen_args.get("seed")

    # date: from evaluation_timestamp or retrieved_timestamp
    flat["date"] = record.get("evaluation_timestamp") or \
                   record.get("retrieved_timestamp")

    # Recommended
    flat["quantization"] = get_nested(record, "model_info.quantization")
    flat["hardware"] = get_nested(record, "source_metadata.hardware")
    flat["software_environment"] = get_nested(
        record, "source_metadata.software_environment")

    # Optional
    flat["provenance_link"] = get_nested(
        record, "source_metadata.additional_details.provenance_link")
    flat["contamination_check"] = get_nested(
        record, "source_metadata.additional_details.contamination_check")
    flat["compute_hours"] = get_nested(
        record, "source_metadata.additional_details.compute_hours")

    # CoT (for completeness score)
    add = gen_config.get("additional_details", {})
    flat["chain_of_thought"] = gen_args.get("chain_of_thought") or \
                               add.get("chain_of_thought") or \
                               add.get("cot")

    return flat


# ---------------------------------------------------------------------------
# Completeness score (0-5, matching paper metric)
# ---------------------------------------------------------------------------

def compute_completeness_score(flat: dict) -> int:
    """
    Compute the 0-5 metadata completeness score used in §5.2 of the paper.
    One point each for: harness (eval_library.name), n_shot, chain_of_thought,
    temperature, prompt_template.
    """
    score = 0
    if is_nontrivial(flat.get("eval_library.name")):
        score += 1
    if is_nontrivial(flat.get("n_shot")):
        score += 1
    if is_nontrivial(flat.get("chain_of_thought")):
        score += 1
    if is_nontrivial(flat.get("temperature")):
        score += 1
    if is_nontrivial(flat.get("prompt_template")):
        score += 1
    return score


# ---------------------------------------------------------------------------
# Divergence risk prediction (based on ρ = −0.850 observational mapping)
# ---------------------------------------------------------------------------

def predict_divergence_risk(completeness: int) -> str:
    """
    Map completeness score to predicted divergence risk.
    Based on the ρ = −0.850 observational association (§5.5):
      completeness 0 → high (|Δ| up to 32 pp)
      completeness 1-2 → medium
      completeness 3-5 → low (|Δ| ≤ 0.23 pp)
    """
    if completeness == 0:
        return "high"
    elif completeness in (1, 2):
        return "medium"
    elif completeness >= 3:
        return "low"
    return "unknown"


# ---------------------------------------------------------------------------
# Type validation
# ---------------------------------------------------------------------------

def type_check(flat: dict) -> list[str]:
    """Return list of type error messages."""
    errors = []

    scoring_mode = flat.get("scoring_mode")
    if scoring_mode is not None and str(scoring_mode).lower() not in SCORING_MODES:
        errors.append(
            f"scoring_mode must be one of {sorted(SCORING_MODES)}; "
            f"got '{scoring_mode}'"
        )

    n_shot = flat.get("n_shot")
    if n_shot is not None:
        try:
            n = int(n_shot)
            if n < 0:
                errors.append(f"n_shot must be a non-negative integer; got {n_shot}")
        except (TypeError, ValueError):
            errors.append(f"n_shot must be a non-negative integer; got '{n_shot}'")

    date_val = flat.get("date")
    if date_val is not None:
        date_str = str(date_val)
        # Accept Unix epoch timestamps as integers
        try:
            int(date_str)
        except ValueError:
            if not ISO_DATE_RE.match(date_str):
                errors.append(
                    f"date must be ISO 8601 or Unix epoch; got '{date_str}'"
                )

    return errors


# ---------------------------------------------------------------------------
# Core validation
# ---------------------------------------------------------------------------

def validate_record(record: dict, tier: str = "required") -> dict:
    """
    Validate a single record and return a compliance report.

    Parameters
    ----------
    record : dict
        A single EEE or EvalSpec record.
    tier : str
        One of 'required', 'recommended', 'full'.

    Returns
    -------
    dict
        Compliance report.
    """
    is_eee = detect_eee_record(record)
    if is_eee:
        flat = normalise_eee_to_evalspec(record)
    else:
        flat = record.copy()

    # --- Required tier ---
    required_missing = [
        f for f in REQUIRED_FIELDS if not is_nontrivial(flat.get(f))
    ]

    # --- Recommended tier ---
    recommended_missing = [
        f for f in RECOMMENDED_FIELDS if not is_nontrivial(flat.get(f))
    ]

    # --- Optional / Full tier ---
    full_missing = [
        f for f in OPTIONAL_FIELDS if not is_nontrivial(flat.get(f))
    ]

    # --- Type errors ---
    type_errors = type_check(flat)

    # --- Compliance level ---
    if required_missing or type_errors:
        compliance_level = "none"
    elif recommended_missing:
        compliance_level = "required"
    elif full_missing:
        compliance_level = "recommended"
    else:
        compliance_level = "full"

    # --- Completeness score (0-5, paper metric) ---
    completeness_score = compute_completeness_score(flat)

    # --- Divergence risk ---
    divergence_risk = predict_divergence_risk(completeness_score)

    # --- EEE compatibility ---
    eee_compatibility = is_eee

    return {
        "compliance_level": compliance_level,
        "completeness_score": completeness_score,
        "required_missing": required_missing,
        "recommended_missing": recommended_missing,
        "full_missing": full_missing,
        "type_errors": type_errors,
        "predicted_divergence_risk": divergence_risk,
        "eee_compatibility": eee_compatibility,
    }


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

def validate_batch(input_dir: Path) -> dict:
    """
    Validate all .json files in a directory.
    Returns a summary report with aggregate completeness statistics.
    """
    json_files = sorted(input_dir.glob("**/*.json"))
    if not json_files:
        return {"error": f"No .json files found in {input_dir}"}

    per_record = []
    for path in json_files:
        try:
            with path.open(encoding="utf-8") as fh:
                record = json.load(fh)
        except (json.JSONDecodeError, OSError) as e:
            per_record.append({
                "file": str(path),
                "error": str(e),
                "compliance_level": "none",
                "completeness_score": 0,
            })
            continue

        report = validate_record(record)
        report["file"] = str(path)
        per_record.append(report)

    n = len(per_record)
    levels = [r.get("compliance_level", "none") for r in per_record]
    scores = [r.get("completeness_score", 0) for r in per_record]

    level_counts = {
        lvl: levels.count(lvl)
        for lvl in ("none", "required", "recommended", "full")
    }
    risk_counts = {
        risk: sum(1 for r in per_record if r.get("predicted_divergence_risk") == risk)
        for risk in ("low", "medium", "high", "unknown")
    }

    mean_completeness = sum(scores) / n if n > 0 else 0.0

    # Compute compliance rates
    def pct(k: str) -> str:
        return f"{100.0 * level_counts[k] / n:.1f}%" if n > 0 else "N/A"

    summary = {
        "n_records": n,
        "compliance_breakdown": {
            "required_tier": {
                "count": level_counts["required"] + level_counts["recommended"] + level_counts["full"],
                "percent": f"{100.0 * (level_counts['required'] + level_counts['recommended'] + level_counts['full']) / n:.1f}%" if n > 0 else "N/A",
            },
            "recommended_tier": {
                "count": level_counts["recommended"] + level_counts["full"],
                "percent": f"{100.0 * (level_counts['recommended'] + level_counts['full']) / n:.1f}%" if n > 0 else "N/A",
            },
            "full_tier": {
                "count": level_counts["full"],
                "percent": pct("full"),
            },
            "none": {
                "count": level_counts["none"],
                "percent": pct("none"),
            },
        },
        "mean_completeness_score": round(mean_completeness, 2),
        "divergence_risk_distribution": risk_counts,
        "per_record": per_record,
    }

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EvalSpec v0.1 compliance validator for EEE-schema records."
    )
    p.add_argument(
        "--input", required=True,
        help="Path to a single JSON record or directory (with --batch)."
    )
    p.add_argument(
        "--tier", choices=["required", "recommended", "full"],
        default="required",
        help="Validation tier (default: required)."
    )
    p.add_argument(
        "--output",
        help="Path to write the report JSON (default: stdout)."
    )
    p.add_argument(
        "--batch", action="store_true",
        help="Batch-validate all .json files in --input directory."
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)

    if args.batch:
        if not input_path.is_dir():
            print(f"Error: --batch requires a directory; got: {input_path}", file=sys.stderr)
            sys.exit(1)
        report = validate_batch(input_path)
    else:
        if not input_path.exists():
            print(f"Error: file not found: {input_path}", file=sys.stderr)
            sys.exit(1)
        with input_path.open(encoding="utf-8") as fh:
            try:
                record = json.load(fh)
            except json.JSONDecodeError as e:
                print(f"Error: invalid JSON in {input_path}: {e}", file=sys.stderr)
                sys.exit(1)
        report = validate_record(record, tier=args.tier)

    output_str = json.dumps(report, indent=2, default=str)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_str, encoding="utf-8")
        print(f"Report written to: {out_path}")
    else:
        print(output_str)


if __name__ == "__main__":
    main()
