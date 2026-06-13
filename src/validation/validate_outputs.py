"""validate_outputs.py — thin wrapper delegating to schema.py.

DEPRECATED: Use `python -m src.validation.schema` instead.
Kept for backward compatibility with existing task runners.
"""
import sys
from pathlib import Path

from src.validation.schema import SchemaValidator, AutoFixer, FileValidator, BatchValidator

ROOT = Path(__file__).resolve().parent.parent.parent


def main() -> None:
    sv = SchemaValidator()
    fixer = AutoFixer()
    fv = FileValidator(sv, fixer, fix_mode=False)
    bv = BatchValidator(fv)
    report = bv.run(ROOT / "data", quiet=True)

    passed = report["valid"] + report.get("fixed", 0)
    failed = report.get("invalid", 0)

    print(f"schema validation: {passed} passed, {failed} failed")
    if failed:
        for r in report.get("files", []):
            if r.get("status") == "invalid":
                print(f"  FAIL {r['path']}: {r.get('error', 'unknown')}")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
