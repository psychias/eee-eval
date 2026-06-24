"""
collision_overlap_matrix.py — build sources × models collision presence matrix.
Output: analysis_output/collision_overlap_matrix.csv
        analysis_output/collision_source_pairs.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))

OUT_DIR = _ROOT / "analysis_output"


def main():
    OUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(OUT_DIR / "collision_pairs.csv")

    # Get all unique sources and models
    sources = sorted(set(df["source_a"].tolist() + df["source_b"].tolist()))
    models = sorted(df["model_id"].unique())

    # Build presence matrix: does source evaluate this model in any collision pair?
    matrix = pd.DataFrame(0, index=sources, columns=models)

    for _, row in df.iterrows():
        matrix.loc[row["source_a"], row["model_id"]] = 1
        matrix.loc[row["source_b"], row["model_id"]] = 1

    # Also save source-pair × model collision existence
    pair_records = []
    for _, row in df.iterrows():
        pair_records.append({
            "source_pair": f"{row['source_a']}|{row['source_b']}",
            "source_a": row["source_a"],
            "source_b": row["source_b"],
            "model_id": row["model_id"],
            "benchmark": row["benchmark"],
            "abs_delta": abs(row["delta"]),
        })

    pairs_df = pd.DataFrame(pair_records)
    matrix.to_csv(OUT_DIR / "collision_overlap_matrix.csv")
    pairs_df.to_csv(OUT_DIR / "collision_source_pairs.csv", index=False)
    print(f"Matrix: {len(sources)} sources × {len(models)} models")
    density = matrix.values.sum() / matrix.size if matrix.size > 0 else 0
    print(f"Density: {density:.1%}")
    print(f"\nSaved → {OUT_DIR / 'collision_overlap_matrix.csv'}")
    print(f"Saved → {OUT_DIR / 'collision_source_pairs.csv'}")


if __name__ == "__main__":
    main()
