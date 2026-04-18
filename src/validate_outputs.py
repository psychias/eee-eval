"""validate_outputs.py — schema-check every JSON written under data/."""
import json, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
schema_path = ROOT / "eval.schema.json"

if not schema_path.exists():
    print("  [validation] eval.schema.json not found — skipping")
    sys.exit(0)

from jsonschema.validators import validator_for
schema = json.loads(schema_path.read_text(encoding="utf-8"))
cls = validator_for(schema)
v = cls(schema)

files = [f for f in (ROOT / "data").rglob("*.json")
         if "aggregated" not in str(f)]

passed = failed = 0
for f in files:
    try:
        v.validate(json.loads(f.read_text(encoding="utf-8")))
        passed += 1
    except Exception as e:
        failed += 1
        print(f"  FAIL {f.relative_to(ROOT)}: {e}")

print(f"\n✓ {passed} passed  ✗ {failed} failed  ({len(files)} total)")
sys.exit(1 if failed else 0)
