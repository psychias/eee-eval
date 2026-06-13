"""
per_benchmark_ols.py — per-benchmark OLS variance decomposition.

For each benchmark with >= 5 collision pairs, fit a separate OLS:
    |delta| ~ harness_differs + n_shot_diff + prompt_template_differs + source_pair

Reports partial R², Cohen's f², p-value, and sample size per benchmark.
Output: analysis_output/per_benchmark_ols.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import numpy as np

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
    df = pd.read_csv(OUT_DIR / "collision_pairs.csv")
    print(f"Loaded {len(df)} collision pairs")

    # Normalize harness names before building predictors
    df["harness_a"] = df["harness_a"].apply(normalize_harness)
    df["harness_b"] = df["harness_b"].apply(normalize_harness)

    results = []
    for bench, grp in df.groupby("benchmark"):
        n = len(grp)
        if n < 5:
            continue

        y = grp["delta"].abs().values
        harness_differs = (grp["harness_a"] != grp["harness_b"]).astype(int).values
        n_shot_a = pd.to_numeric(grp["n_shot_a"], errors="coerce").fillna(0).values
        n_shot_b = pd.to_numeric(grp["n_shot_b"], errors="coerce").fillna(0).values
        n_shot_diff = np.abs(n_shot_a - n_shot_b)
        if "prompt_template_a" in grp.columns:
            pt_differs = (grp["prompt_template_a"] != grp["prompt_template_b"]).astype(int).values
        else:
            pt_differs = np.zeros(n, dtype=int)

        # Simple OLS via numpy
        X = np.column_stack([
            np.ones(n),
            harness_differs,
            n_shot_diff,
            pt_differs,
        ])
        predictor_names = ["harness_differs", "n_shot_diff", "prompt_template_differs"]

        # Full model R²
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        y_hat = X @ beta
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        if ss_tot < 1e-15:
            continue
        r2_full = 1 - ss_res / ss_tot

        # Partial R² for each predictor (drop one at a time)
        for i, pred_name in enumerate(predictor_names):
            col_idx = i + 1  # +1 because col 0 is intercept
            X_reduced = np.delete(X, col_idx, axis=1)
            try:
                beta_red = np.linalg.lstsq(X_reduced, y, rcond=None)[0]
            except np.linalg.LinAlgError:
                continue
            y_hat_red = X_reduced @ beta_red
            ss_res_red = np.sum((y - y_hat_red) ** 2)
            r2_red = 1 - ss_res_red / ss_tot

            partial_r2 = r2_full - r2_red
            f2 = partial_r2 / max(1 - r2_full, 1e-10)

            # Approximate p-value via F-test
            df_num = 1
            df_den = n - X.shape[1]
            if df_den > 0:
                f_stat = (partial_r2 / df_num) / (max(1 - r2_full, 1e-10) / df_den)
                from scipy import stats
                p_val = 1 - stats.f.cdf(f_stat, df_num, df_den)
            else:
                f_stat = float("nan")
                p_val = float("nan")

            results.append({
                "benchmark": bench,
                "predictor": pred_name,
                "n": n,
                "partial_r2": round(partial_r2, 4),
                "f2_cohen": round(f2, 4),
                "f_stat": round(f_stat, 4),
                "p_value": round(p_val, 4),
                "full_model_r2": round(r2_full, 4),
                "beta": round(beta[col_idx], 4),
            })

    out = pd.DataFrame(results)
    out_path = OUT_DIR / "per_benchmark_ols.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
