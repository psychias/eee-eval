"""
Unified CLI for converting evaluation framework logs into the EEE schema.

Wraps the adapters in eval_converters/ (lm-evaluation-harness, Inspect AI,
HELM) behind a single entry point so they plug into the same pipeline as
paper extraction.

Usage:
    python scripts/convert_eval_logs.py --framework lm_eval   --log_path results.json
    python scripts/convert_eval_logs.py --framework inspect   --log_path data.eval
    python scripts/convert_eval_logs.py --framework helm      --log_path run_dir/
    python scripts/convert_eval_logs.py --framework auto      --log_path path/
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from enum import Enum
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))

from eee_eval.converters.common.adapter import BaseEvaluationAdapter, SupportedLibrary
from eee_eval.converters.lm_eval.adapter import LMEvalAdapter
try:
    from eee_eval.converters.inspect.adapter import InspectAIAdapter
    _HAS_INSPECT = True
except ImportError:
    _HAS_INSPECT = False
    InspectAIAdapter = None  # type: ignore[assignment,misc]
try:
    from eee_eval.converters.helm.adapter import HELMAdapter
    _HAS_HELM = True
except ImportError:
    _HAS_HELM = False
    HELMAdapter = None  # type: ignore[assignment,misc]
from eee_eval.eval_types import EvaluationLog


class _EnumEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


_ADAPTER_MAP: dict[str, type[BaseEvaluationAdapter]] = {
    "lm_eval": LMEvalAdapter,
}
if _HAS_INSPECT:
    _ADAPTER_MAP["inspect"] = InspectAIAdapter
if _HAS_HELM:
    _ADAPTER_MAP["helm"] = HELMAdapter


def _detect_framework(log_path: Path) -> str:
    """Auto-detect which evaluation framework produced *log_path*."""
    if log_path.suffix in (".eval",):
        return "inspect"

    if log_path.is_dir():
        children = {c.name for c in log_path.iterdir()}
        if "run_spec.json" in children and "scenario_state.json" in children:
            return "helm"
        # Check for lm-eval directory layout (results_*.json files)
        if any(c.name.startswith("results") and c.suffix == ".json" for c in log_path.iterdir()):
            return "lm_eval"
        # Check subdirectories for HELM layout
        for sub in log_path.iterdir():
            if sub.is_dir():
                sub_children = {c.name for c in sub.iterdir()}
                if "run_spec.json" in sub_children:
                    return "helm"

    if log_path.is_file() and log_path.suffix == ".json":
        with open(log_path) as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            if "results" in data and "config" in data:
                return "lm_eval"
            if "eval" in data and "samples" in data:
                return "inspect"

    raise ValueError(
        f"Cannot auto-detect framework for {log_path}. "
        "Use --framework lm_eval|inspect|helm explicitly."
    )


def _write_log(log: EvaluationLog, output_dir: Path) -> Path:
    """Write a single EvaluationLog as per-benchmark JSON files (EEE layout)."""
    import re

    def _sanitize(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]", "_", value.replace(" ", "_"))

    record = log.model_dump(mode="json", exclude_none=True)
    model_info = record.get("model_info", {})
    model_id = model_info.get("id", "unknown")
    developer = model_info.get("developer", "unknown")
    model_name = model_id.split("/", 1)[-1] if "/" in model_id else model_id

    paths: list[Path] = []
    for er in record.get("evaluation_results", []):
        bench = er.get("evaluation_name", "unknown")
        split_record = dict(record)
        split_record["evaluation_results"] = [er]

        out_dir = output_dir / _sanitize(bench) / _sanitize(developer) / _sanitize(model_name)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{uuid.uuid4()}.json"

        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(split_record, fh, indent=2, ensure_ascii=False, cls=_EnumEncoder)
        paths.append(out_path)

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert evaluation framework logs to EEE unified schema"
    )
    parser.add_argument(
        "--framework",
        choices=["lm_eval", "inspect", "helm", "auto"],
        default="auto",
        help="Evaluation framework that produced the log (default: auto-detect)",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        required=True,
        help="Path to evaluation log file or directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Output directory (default: data/)",
    )
    parser.add_argument(
        "--source_organization_name",
        type=str,
        default="",
    )
    parser.add_argument(
        "--evaluator_relationship",
        choices=["first_party", "third_party", "collaborative", "other"],
        default="third_party",
    )
    parser.add_argument(
        "--source_organization_url",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--eval_library_version",
        type=str,
        default="unknown",
    )

    args = parser.parse_args()
    log_path = Path(args.log_path)

    if not log_path.exists():
        print(f"Error: {log_path} not found", file=sys.stderr)
        sys.exit(1)

    framework = args.framework
    if framework == "auto":
        framework = _detect_framework(log_path)
        print(f"Auto-detected framework: {framework}")

    adapter_cls = _ADAPTER_MAP[framework]
    adapter = adapter_cls()

    metadata_args = {
        "source_organization_name": args.source_organization_name,
        "evaluator_relationship": args.evaluator_relationship,
        "source_organization_url": args.source_organization_url,
        "eval_library_name": framework,
        "eval_library_version": args.eval_library_version,
    }

    output_dir = Path(args.output_dir)

    if log_path.is_file():
        logs = adapter.transform_from_file(log_path, metadata_args)
    elif log_path.is_dir():
        logs = adapter.transform_from_directory(log_path, metadata_args)
    else:
        print(f"Error: {log_path} is not a file or directory", file=sys.stderr)
        sys.exit(1)

    if not isinstance(logs, list):
        logs = [logs]

    total_files = 0
    for log in logs:
        paths = _write_log(log, output_dir)
        total_files += len(paths)

    print(f"Converted {len(logs)} evaluation(s) → {total_files} JSON files in {output_dir}/")


if __name__ == "__main__":
    main()
