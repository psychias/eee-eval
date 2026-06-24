"""
coverage_audit.py — audit metadata field coverage per source directory.

Checks what fraction of evaluation_results records have each methodology
field populated: n_shot, harness (non-"unknown"), prompt_template, temperature,
top_k, reasoning_mode.

Output: analysis_output/coverage_stats.csv
Columns: source, n_records, pct_n_shot, pct_harness, pct_prompt_template,
         pct_temperature, pct_top_k, pct_reasoning_mode
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))

DATA_DIR = _ROOT / "data"
OUT_DIR = _ROOT / "analysis_output"


def audit_source(source_dir: Path) -> dict:
    n_records = 0
    has_n_shot = 0
    has_harness = 0
    has_prompt_template = 0
    has_temperature = 0
    has_top_k = 0
    has_reasoning_mode = 0
    has_chain_of_thought = 0

    for fpath in source_dir.rglob("*.json"):
        # Canonical ArXiv source is data/arxiv_extraction_general/llm/ only;
        # skip audit-sample copies (samples/) and the naive extraction (naive/).
        if "samples" in fpath.parts or "naive" in fpath.parts:
            continue
        try:
            rec = json.loads(fpath.read_text())
        except Exception:
            continue
        if not isinstance(rec, dict):
            continue
        eval_lib = rec.get("eval_library") or {}
        if not isinstance(eval_lib, dict):
            eval_lib = {}
        harness = eval_lib.get("name", "")
        harness_known = harness not in ("", "unknown")

        for result in rec.get("evaluation_results", []):
            n_records += 1
            gen_cfg = result.get("generation_config") or {}
            gen_args = gen_cfg.get("generation_args") or {}
            details = gen_cfg.get("additional_details") or {}

            # Check generation_args first (new pipeline), fall back to
            # additional_details. NB: use an explicit None check, not `or`, so a
            # documented 0-shot (shots == 0) counts as present rather than being
            # swallowed as falsy.
            n_shot_val = gen_args.get("shots")
            if n_shot_val is None:
                n_shot_val = details.get("n_shot", "")
            has_n_shot += 1 if (n_shot_val not in ("", None)) else 0

            has_harness += 1 if harness_known else 0

            # Check both locations for prompt_template
            # "standard" is a generic placeholder, not a real template
            pt = gen_args.get("prompt_template") or details.get("prompt_template", "")
            # coerce lists/dicts to a string for robust matching
            if isinstance(pt, (list, tuple)):
                pt = " ".join(str(x) for x in pt)
            elif not isinstance(pt, str):
                pt = "" if pt is None else str(pt)
            has_prompt_template += 1 if (pt.strip() not in ("", "standard")) else 0
            # chain_of_thought: prompt template names CoT, or an explicit cot flag
            cot_flag = gen_args.get("chain_of_thought") or details.get("chain_of_thought")
            pt_lower = pt.lower()
            is_cot = bool(cot_flag) or ("chain" in pt_lower) or ("cot" in pt_lower)
            has_chain_of_thought += 1 if is_cot else 0

            # Check both locations for temperature. Explicit None check so a
            # documented temperature of 0.0 is not swallowed as falsy.
            temp = gen_args.get("temperature")
            if temp is None:
                temp = details.get("temperature")
            has_temperature += 1 if (temp not in ("", None)) else 0

            # top_k
            top_k = gen_args.get("top_k") or details.get("top_k")
            has_top_k += 1 if (top_k not in ("", None)) else 0

            # reasoning_mode
            rm = gen_args.get("reasoning_mode")
            has_reasoning_mode += 1 if (rm not in ("", None)) else 0

    if n_records == 0:
        return {
            "source": source_dir.name,
            "n_records": 0,
            "pct_n_shot": 0.0,
            "pct_harness": 0.0,
            "pct_prompt_template": 0.0,
            "pct_temperature": 0.0,
            "pct_top_k": 0.0,
            "pct_reasoning_mode": 0.0,
            "pct_chain_of_thought": 0.0,
        }

    return {
        "source": source_dir.name,
        "n_records": n_records,
        "pct_n_shot": round(100 * has_n_shot / n_records, 1),
        "pct_harness": round(100 * has_harness / n_records, 1),
        "pct_prompt_template": round(100 * has_prompt_template / n_records, 1),
        "pct_temperature": round(100 * has_temperature / n_records, 1),
        "pct_top_k": round(100 * has_top_k / n_records, 1),
        "pct_reasoning_mode": round(100 * has_reasoning_mode / n_records, 1),
        "pct_chain_of_thought": round(100 * has_chain_of_thought / n_records, 1),
    }


def main():
    OUT_DIR.mkdir(exist_ok=True)
    rows = []
    for source_dir in sorted(DATA_DIR.iterdir()):
        if source_dir.is_dir():
            rows.append(audit_source(source_dir))

    df = pd.DataFrame(rows)
    df = df.sort_values("n_records", ascending=False)

    out_path = OUT_DIR / "coverage_stats.csv"
    df.to_csv(out_path, index=False)
    print(f"Coverage audit saved -> {out_path}")
    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
