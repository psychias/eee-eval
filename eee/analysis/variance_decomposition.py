"""
variance_decomposition.py — OLS regression of |score delta| on methodology predictors.

Uses the collision pairs from collision_detection.py and regresses the absolute
score difference on three binary predictors:
  - harness_differs: 1 if harness_a != harness_b
  - n_shot_differs: 1 if n_shot_a != n_shot_b (and both non-empty)
  - prompt_template_differs: 1 if prompt_template_a != prompt_template_b

Reports partial R² for each predictor (type I / sequential SS from statsmodels).

Output: analysis_output/variance_decomp.csv
Columns: predictor, coef, pvalue, partial_r2
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
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


def compute_partial_r2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute partial R² for each predictor via sequential (type I) SS.
    Partial R² ≈ SS_model_with_predictor / SS_total for each term.
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("statsmodels not installed", file=sys.stderr)
        sys.exit(1)

    dep_var = "abs_delta"
    predictors = ["harness_differs", "n_shot_differs", "prompt_template_differs"]

    results_rows = []
    formula = f"{dep_var} ~ " + " + ".join(predictors)
    try:
        model = smf.ols(formula=formula, data=df).fit()
    except Exception as e:
        print(f"OLS failed: {e}", file=sys.stderr)
        return pd.DataFrame()

    ss_total = np.sum((df[dep_var] - df[dep_var].mean()) ** 2)

    for pred in predictors:
        coef = model.params.get(pred, float("nan"))
        pval = model.pvalues.get(pred, float("nan"))
        # Partial R²: correlation of y_hat with predictor = contribution
        # Use the marginal SS approach: fit without this predictor
        other_preds = [p for p in predictors if p != pred]
        if other_preds:
            formula_reduced = f"{dep_var} ~ " + " + ".join(other_preds)
            model_red = smf.ols(formula=formula_reduced, data=df).fit()
            ss_reduced = model_red.ssr
        else:
            ss_reduced = ss_total
        ss_full = model.ssr
        partial_r2 = round((ss_reduced - ss_full) / ss_total, 4)
        results_rows.append({
            "predictor": pred,
            "coef": round(coef, 4),
            "pvalue": round(pval, 4),
            "partial_r2": max(0.0, partial_r2),
        })

    return pd.DataFrame(results_rows)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    collisions_path = OUT_DIR / "collision_pairs.csv"
    if not collisions_path.exists():
        print("Run collision_detection.py first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(collisions_path)
    if df.empty:
        print("No collision pairs found; cannot run OLS.", file=sys.stderr)
        sys.exit(1)

    # Normalize harness names before building predictors
    df["harness_a"] = df["harness_a"].apply(normalize_harness)
    df["harness_b"] = df["harness_b"].apply(normalize_harness)

    df["abs_delta"] = df["delta"].abs()

    # Build binary predictors
    def ne(a, b) -> int:
        a, b = str(a).strip(), str(b).strip()
        if a in ("", "nan") or b in ("", "nan"):
            return 0
        return int(a != b)

    df["harness_differs"] = [
        ne(r.harness_a, r.harness_b) for _, r in df.iterrows()
    ]
    df["n_shot_differs"] = [
        ne(r.n_shot_a, r.n_shot_b) for _, r in df.iterrows()
    ]
    df["prompt_template_differs"] = [
        ne(r.prompt_template_a, r.prompt_template_b) for _, r in df.iterrows()
    ]

    print(f"OLS input: {len(df)} collision pairs")
    print(f"  harness_differs:        {df['harness_differs'].sum()}/{len(df)}")
    print(f"  n_shot_differs:         {df['n_shot_differs'].sum()}/{len(df)}")
    print(f"  prompt_template_differs: {df['prompt_template_differs'].sum()}/{len(df)}")

    results = compute_partial_r2(df)
    if results.empty:
        sys.exit(1)

    out_path = OUT_DIR / "variance_decomp.csv"
    results.to_csv(out_path, index=False)
    print(f"\nVariance decomposition saved → {out_path}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
