"""
harness_effect_data.py — compute |delta| split by harness_same/harness_different.
Output: analysis_output/harness_effect.csv
Columns: benchmark, abs_delta, harness_same (bool), model_id, source_a, source_b
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

OUT_DIR = _ROOT / "analysis_output"

_HARNESS_ALIASES = {
    "lm_eval": "lm-evaluation-harness",
    "lm-eval": "lm-evaluation-harness",
    "eleutherai/lm-eval": "lm-evaluation-harness",
    "lm-evaluation-harness": "lm-evaluation-harness",
    "internal": "internal",
    "unknown": "",
}


def normalize_harness(h: str) -> str:
    h = str(h).strip().lower()
    return _HARNESS_ALIASES.get(h, h)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(OUT_DIR / "collision_pairs.csv")
    df["abs_delta"] = df["delta"].abs()

    # Normalize harness names first
    for col in ["harness_a", "harness_b"]:
        df[col] = df[col].apply(normalize_harness)

    df["harness_same"] = df["harness_a"] == df["harness_b"]

    out = df[["benchmark", "abs_delta", "harness_same",
              "model_id", "source_a", "source_b"]].copy()
    out.to_csv(OUT_DIR / "harness_effect.csv", index=False)

    # Print summary
    for bench in sorted(df["benchmark"].unique()):
        sub = df[df["benchmark"] == bench]
        same = sub[sub["harness_same"]]["abs_delta"]
        diff = sub[~sub["harness_same"]]["abs_delta"]
        print(f"{bench}: same={same.median():.4f} (n={len(same)}), "
              f"diff={diff.median():.4f} (n={len(diff)})")

    print(f"\nSaved → {OUT_DIR / 'harness_effect.csv'}")


if __name__ == "__main__":
    main()
