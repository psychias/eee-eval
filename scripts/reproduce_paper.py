#!/usr/bin/env python3
"""
reproduce_paper.py — regenerate every headline number in the paper from the
repository's data and experiment outputs, in one command.

This is the single source of truth for the paper's quantitative claims. It
rebuilds the canonical aggregate from ``data/`` (ArXiv = the per-benchmark
``llm/`` extraction only; audit-sample copies and the naive extraction are
excluded), recomputes the §3–§4 dataset/coverage/audit numbers, and reads the
§5 / Appendix-G experiment results from ``experiments/``. All values are written
to ``analysis_output/paper_numbers.json`` and printed as a report.

Usage
-----
    python reproduce_paper.py            # recompute and write paper_numbers.json
    python reproduce_paper.py --no-rebuild   # skip the (slow) aggregate rebuild

Determinism: everything here is a pure function of the on-disk data plus the
seeded experiment outputs. No network, no randomness. The only non-reproducible
inputs are the upstream dataset-build steps (LLM extraction, leaderboard
fetches), which are documented separately in the README.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ANALYSIS_OUT = ROOT / "analysis_output"
EXP = ROOT / "experiments"

# Sources that make up the "10-source leaderboard subset" (§3.2 / §4.3). ArXiv
# and Papers With Code are not leaderboards and are excluded from the subset;
# both are included in the full-dataset totals.
LEADERBOARD_SOURCES = {
    "open_llm_leaderboard_v2", "alpacaeval2", "bfcl", "bigcodebench",
    "chatbot_arena", "evalplus", "hf_model_card", "wildbench",
    "swe_bench", "mt_bench",
}
ARXIV_SOURCES = {"arxiv_html_llm", "arxiv_html_naive"}


# --------------------------------------------------------------------------- #
# Coverage field predicates — match src/analysis/coverage_audit.py exactly so
# the §4.3 numbers are consistent with the audited methodology.
# --------------------------------------------------------------------------- #
def _present(value) -> bool:
    s = "" if value is None else str(value).strip()
    return s != "" and s.lower() != "unknown" and s.lower() != "nan"


def _prompt_present(value) -> bool:
    s = "" if value is None else str(value).strip()
    return s not in ("", "standard") and s.lower() != "nan"


def _is_cot(cot_flag, prompt_template) -> bool:
    pt = ("" if prompt_template is None else str(prompt_template)).lower()
    flag = str(cot_flag).strip().lower() not in ("", "none", "false", "0", "nan")
    return flag or ("chain" in pt) or ("cot" in pt)


# --------------------------------------------------------------------------- #
# §3 — dataset composition
# --------------------------------------------------------------------------- #
def load_aggregate(rebuild: bool) -> list[dict]:
    """Rebuild the canonical aggregate (optional) and return its rows."""
    if rebuild:
        from eee_eval.analysis import aggregate_results
        aggregate_results.main()
    path = DATA / "aggregated" / "all_results.csv"
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def dataset_numbers(rows: list[dict]) -> dict:
    by_source = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)

    def stats(src_rows):
        return {
            "records": len(src_rows),
            "models": len({r["model_id"] for r in src_rows}),
            "benchmarks": len({r["benchmark"] for r in src_rows}),
        }

    per_source = {s: stats(rs) for s, rs in sorted(by_source.items())}
    lb_rows = [r for r in rows if r["source"] in LEADERBOARD_SOURCES]
    arxiv_rows = [r for r in rows if r["source"] in ARXIV_SOURCES]
    return {
        "total_records": len(rows),
        "total_models": len({r["model_id"] for r in rows}),
        "total_benchmarks": len({r["benchmark"] for r in rows}),
        "leaderboard_subtotal": stats(lb_rows),
        "arxiv": stats(arxiv_rows),
        "per_source": per_source,
    }


def coverage_numbers(rows: list[dict]) -> dict:
    fields = ["shots", "harness", "prompt_template", "temperature", "chain_of_thought"]

    def cover(subset):
        n = len(subset)
        if n == 0:
            return {f: 0.0 for f in fields}
        c = {f: 0 for f in fields}
        for r in subset:
            c["shots"] += _present(r.get("shots"))
            c["harness"] += _present(r.get("harness"))
            c["prompt_template"] += _prompt_present(r.get("prompt_template"))
            c["temperature"] += _present(r.get("temperature"))
            c["chain_of_thought"] += _is_cot(r.get("chain_of_thought"), r.get("prompt_template"))
        return {f: round(100 * c[f] / n, 1) for f in fields}

    subset = [r for r in rows if r["source"] in LEADERBOARD_SOURCES]
    return {"full_dataset": cover(rows), "leaderboard_subset": cover(subset),
            "full_n": len(rows), "subset_n": len(subset)}


def arxiv_score_presence() -> dict:
    """§4.1 structural-presence check over the canonical ArXiv extraction."""
    n = nn = 0
    for f in (DATA / "archiv_paper_extraction" / "llm").rglob("*.json"):
        if f.name in ("summary.json", "metric_configs_from_llm.json", "extraction_results.json"):
            continue
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(rec, dict):
            continue
        for er in rec.get("evaluation_results", []):
            n += 1
            if (er.get("score_details") or {}).get("score") not in ("", None):
                nn += 1
    return {"arxiv_records": n, "non_null_score": nn,
            "pct": round(100 * nn / n, 1) if n else 0.0}


# --------------------------------------------------------------------------- #
# §4 — annotation audits
# --------------------------------------------------------------------------- #
def _read_csv_rows(path: Path) -> list[dict]:
    for enc in ("utf-8-sig", "cp1252"):
        try:
            with open(path, encoding=enc) as fh:
                return list(csv.DictReader(fh))
        except UnicodeDecodeError:
            continue
    return []


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes", "y")


def score_audit() -> dict:
    rows = _read_csv_rows(DATA / "archiv_paper_extraction" / "samples" / "annotation_score_verification.csv")
    # §4.1 score-verification audit. The verdict_type column separates the two
    # axes: 'score' rows are the pipeline's extractions (the precision basis);
    # 'recall' rows are paper-reported scores the pipeline missed (appended as
    # correct=FALSE during the senior-annotator re-audit). agreement_pct is
    # precision over the 'score' rows; recall is captured / reported.
    score_rows = [r for r in rows if (r.get("verdict_type") or "score") == "score"]
    missed = sum(1 for r in rows if r.get("verdict_type") == "recall")
    correct = sum(1 for r in score_rows if _truthy(r.get("correct")))
    captured = len(score_rows)
    reported = captured + missed
    return {"entries": captured, "correct": correct,
            "agreement_pct": round(100 * correct / captured, 1) if captured else 0.0,
            "missed": missed,
            "recall_pct": round(100 * captured / reported, 1) if reported else 0.0}


def harness_audit() -> dict:
    # Precision is computed over the verdict_type rows that 'correct' actually
    # judges: 'harness' rows for the with-harness pool, 'abstention' rows for the
    # without-harness pool. ('score'/'recall' rows carry value/coverage defects
    # surfaced in the re-audit and are excluded from the harness-call precision.)
    def precision(path, vtype):
        rows = [r for r in _read_csv_rows(path) if (r.get("verdict_type") or vtype) == vtype]
        ok = sum(1 for r in rows if _truthy(r.get("correct")))
        return {"rows": len(rows), "correct": ok,
                "pct": round(100 * ok / len(rows), 1) if rows else 0.0}
    base = DATA / "archiv_paper_extraction" / "samples"
    return {"with_harness": precision(base / "annotation_eval_harness_WITH.csv", "harness"),
            "without_harness": precision(base / "annotation_eval_harness_WITHOUT.csv", "abstention")}


# --------------------------------------------------------------------------- #
# §5 / Appendix G — experiments
# --------------------------------------------------------------------------- #
def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _eta_and_f(groups: list[list[float]]):
    from scipy.stats import f_oneway
    vals = [v for g in groups for v in g]
    n = len(vals)
    if n == 0 or sum(len(g) for g in groups) == 0:
        return None
    grand = sum(vals) / n
    ss_tot = sum((v - grand) ** 2 for v in vals)
    ss_btw = sum(len(g) * (sum(g) / len(g) - grand) ** 2 for g in groups if g)
    eta2 = ss_btw / ss_tot if ss_tot else float("nan")
    F = p = float("nan")
    if all(len(g) >= 2 for g in groups) and len(groups) >= 2:
        F, p = (float(x) for x in f_oneway(*groups))
    return {"eta_squared_pct": round(100 * eta2, 1), "F": round(F, 2), "p": p}


def factorial_grid() -> dict:
    rows = [r for r in _read_jsonl(EXP / "controlled_eval" / "results" / "controlled_eval_results.jsonl")
            if r.get("status") == "ok"]
    out = {"n_ok": len(rows)}
    for bench in ("gsm8k", "mmlu"):
        b5 = [r for r in rows if r["benchmark"] == bench and r["n_shot"] == 5]
        fmt = defaultdict(list)
        for r in b5:
            fmt[r["prompt_format"]].append(float(r["score"]))
        out[f"{bench}_format_5shot"] = _eta_and_f([fmt[k] for k in sorted(fmt)])
        # n_shot factor (pooled)
        ns = defaultdict(list)
        for r in rows:
            if r["benchmark"] == bench:
                ns[r["n_shot"]].append(float(r["score"]))
        out[f"{bench}_nshot"] = _eta_and_f([ns[k] for k in sorted(ns)])
    return out


def scoring_mode_e2() -> dict:
    rows = _read_jsonl(EXP / "scoring_mode_eval" / "scoring_mode_results.jsonl")
    by = defaultdict(dict)
    for r in rows:
        mode = r.get("scoring_mode") or r.get("mode") or r.get("method")
        by[r.get("model_id") or r.get("model")][mode] = r.get("score")
    return {"raw_rows": len(rows), "models": {k: v for k, v in by.items()}}


def harness_version_e3() -> dict:
    rows = _read_jsonl(EXP / "harness_reeval" / "version_compare" / "results" / "full_version_runs.jsonl")
    cells = {}
    for r in rows:
        cells[f"{r['model_id'].split('/')[-1]}|{r['lm_eval_version']}|seed{r['seed']}"] = r["score"]
    return {"n_cells": len(rows), "cells": cells}


def judge_sensitivity() -> dict:
    rows = [r for r in _read_jsonl(EXP / "judge_sensitivity" / "results" / "judge_runs.jsonl")
            if r.get("rating") is not None]
    means = defaultdict(list)
    for r in rows:
        means[(r["contestant"], r["judge"])].append(r["rating"])
    mat = {f"{c}|{j}": round(sum(v) / len(v), 2) for (c, j), v in means.items()}
    spreads = {}
    contestants = {c for (c, _j) in means}
    judges = sorted({j for (_c, j) in means})
    for c in contestants:
        vals = [sum(means[(c, j)]) / len(means[(c, j)]) for j in judges if (c, j) in means]
        if len(vals) >= 2:
            spreads[c] = round(max(vals) - min(vals), 2)
    return {"matrix_means": mat,
            "mean_spread": round(sum(spreads.values()) / len(spreads), 2) if spreads else None,
            "max_spread": round(max(spreads.values()), 2) if spreads else None}


def bbh_protocol() -> dict:
    out = {}
    for label, fn in (("full", "bbh_protocol_runs.jsonl"), ("lim100", "bbh_protocol_lim100.jsonl")):
        rows = _read_jsonl(EXP / "harness_reeval" / "version_compare" / "results" / fn)
        out[label] = {r["benchmark"]: r["score"] for r in rows if r.get("status") == "ok"}
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-rebuild", action="store_true",
                    help="skip rebuilding data/aggregated/all_results.csv (use the existing file)")
    args = ap.parse_args()

    print("[reproduce_paper] loading canonical aggregate "
          f"({'using existing' if args.no_rebuild else 'rebuilding from data/'})...")
    rows = load_aggregate(rebuild=not args.no_rebuild)

    numbers = {
        "dataset": dataset_numbers(rows),
        "coverage": coverage_numbers(rows),
        "arxiv_score_presence_4_1": arxiv_score_presence(),
        "score_audit_4_1": score_audit(),
        "harness_audit_4_2": harness_audit(),
        "factorial_grid": factorial_grid(),
        "scoring_mode_e2": scoring_mode_e2(),
        "harness_version_e3": harness_version_e3(),
        "judge_sensitivity": judge_sensitivity(),
        "bbh_protocol": bbh_protocol(),
    }

    ANALYSIS_OUT.mkdir(exist_ok=True)
    out_path = ANALYSIS_OUT / "paper_numbers.json"
    out_path.write_text(json.dumps(numbers, indent=2, default=str), encoding="utf-8")

    d = numbers["dataset"]
    cov = numbers["coverage"]
    print("\n================ PAPER NUMBERS (canonical) ================")
    print(f"Total records:   {d['total_records']:,}   models: {d['total_models']:,}   "
          f"benchmarks: {d['total_benchmarks']:,}")
    print(f"Leaderboard subtotal: {d['leaderboard_subtotal']['records']:,} | "
          f"ArXiv: {d['arxiv']['records']:,}")
    print("Coverage full   (shots/harness/CoT/prompt/temp): "
          + "/".join(str(cov['full_dataset'][k]) for k in
                     ['shots', 'harness', 'chain_of_thought', 'prompt_template', 'temperature']))
    print("Coverage subset (shots/harness/CoT/prompt/temp): "
          + "/".join(str(cov['leaderboard_subset'][k]) for k in
                     ['shots', 'harness', 'chain_of_thought', 'prompt_template', 'temperature']))
    ap41 = numbers["arxiv_score_presence_4_1"]
    print(f"§4.1 ArXiv non-null: {ap41['non_null_score']:,}/{ap41['arxiv_records']:,} = {ap41['pct']}%")
    sa = numbers["score_audit_4_1"]
    print(f"§4.1 score audit: precision {sa['correct']}/{sa['entries']} = {sa['agreement_pct']}% | "
          f"recall {sa['entries']}/{sa['entries'] + sa['missed']} = {sa['recall_pct']}%")
    ha = numbers["harness_audit_4_2"]
    print(f"§4.2 harness with: {ha['with_harness']['correct']}/{ha['with_harness']['rows']} = "
          f"{ha['with_harness']['pct']}% | without abstention: "
          f"{ha['without_harness']['correct']}/{ha['without_harness']['rows']}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
