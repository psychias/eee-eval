"""
Recompute all dataset-level stats with the CANONICAL ArXiv source = llm/ only
(exclude samples/ and naive/). Writes analysis_output/recompute_canonical.txt.

Mirrors coverage_audit.py's field-presence logic so coverage % are consistent
with the paper's methodology.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA = ROOT / "data"
OUT = ROOT / "analysis_output" / "recompute_canonical.txt"

LEADERBOARDS = ["open_llm_leaderboard_v2", "alpacaeval2", "bfcl", "bigcodebench",
                "chatbot_arena", "evalplus", "hf_model_card", "wildbench",
                "swe_bench", "mt_bench"]


def field_flags(rec, er):
    eval_lib = rec.get("eval_library") or {}
    harness = (eval_lib.get("name", "") if isinstance(eval_lib, dict) else "")
    harness_known = harness not in ("", "unknown")
    gc = er.get("generation_config") or {}
    ga = gc.get("generation_args") or {}
    det = gc.get("additional_details") or {}
    n_shot = ga.get("shots") or det.get("n_shot", "")
    pt = ga.get("prompt_template") or det.get("prompt_template", "")
    if isinstance(pt, (list, tuple)):
        pt = " ".join(str(x) for x in pt)
    elif not isinstance(pt, str):
        pt = "" if pt is None else str(pt)
    cot_flag = ga.get("chain_of_thought") or det.get("chain_of_thought")
    ptl = pt.lower()
    is_cot = bool(cot_flag) or ("chain" in ptl) or ("cot" in ptl)
    temp = ga.get("temperature") or det.get("temperature")
    return {
        "n_shot": n_shot not in ("", None),
        "harness": harness_known,
        "prompt_template": pt.strip() not in ("", "standard"),
        "temperature": temp not in ("", None),
        "chain_of_thought": is_cot,
    }


def scan(files):
    n = 0
    cnt = {k: 0 for k in ("n_shot", "harness", "prompt_template", "temperature", "chain_of_thought")}
    models, benches = set(), set()
    nonnull = 0
    for f in files:
        if f.name in ("summary.json", "metric_configs_from_llm.json", "extraction_results.json"):
            continue
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(rec, dict):
            continue
        mid = (rec.get("model_info") or {}).get("id", "")
        for er in rec.get("evaluation_results", []):
            n += 1
            fl = field_flags(rec, er)
            for k in cnt:
                cnt[k] += 1 if fl[k] else 0
            models.add(mid)
            benches.add(er.get("evaluation_name", ""))
            if (er.get("score_details") or {}).get("score") not in ("", None):
                nonnull += 1
    return n, cnt, models, benches, nonnull


def main():
    lines = []
    def out(s=""):
        lines.append(s)

    # Per-source scan
    src_stats = {}
    all_models, all_benches = set(), set()
    for s in LEADERBOARDS + ["papers_with_code"]:
        files = list((DATA / s).rglob("*.json"))
        n, cnt, models, benches, nn = scan(files)
        src_stats[s] = (n, cnt, models, benches, nn)
        all_models |= models; all_benches |= benches

    # ArXiv = llm/ only
    n, cnt, models, benches, nn = scan(list((DATA / "arxiv_extraction_general" / "llm").rglob("*.json")))
    src_stats["arxiv_llm"] = (n, cnt, models, benches, nn)
    all_models |= models; all_benches |= benches

    out("=" * 70)
    out("CANONICAL RECOMPUTE  (ArXiv = llm/ only; samples/ + naive/ excluded)")
    out("=" * 70)
    out(f"{'source':<26}{'records':>9}{'models':>8}{'benches':>8}")
    lb_total = 0
    for s in LEADERBOARDS:
        n, cnt, m, b, nn = src_stats[s]
        out(f"{s:<26}{n:>9}{len(m):>8}{len(b):>8}")
        lb_total += n
    out(f"{'-- leaderboard subtotal':<26}{lb_total:>9}")
    na, _, ma, ba, nna = src_stats["arxiv_llm"]
    out(f"{'arxiv (llm only)':<26}{na:>9}{len(ma):>8}{len(ba):>8}")
    npw, _, mpw, bpw, _ = src_stats["papers_with_code"]
    out(f"{'papers_with_code':<26}{npw:>9}{len(mpw):>8}{len(bpw):>8}")
    total_wpwc = lb_total + na + npw
    total_wopwc = lb_total + na
    out(f"{'TOTAL (incl PWC)':<26}{total_wpwc:>9}")
    out(f"{'TOTAL (excl PWC)':<26}{total_wopwc:>9}")
    out(f"unique models (all incl PWC): {len(all_models)}")
    out(f"unique benchmarks (all incl PWC): {len(all_benches)}")
    out("")
    out(f"ArXiv (llm) non-null score: {nna}/{na} = {100*nna/na:.1f}%")
    out("")

    # Coverage: full (all incl PWC + arxiv-llm) and subset (10 leaderboards)
    fields = ["n_shot", "harness", "prompt_template", "temperature", "chain_of_thought"]
    def agg(sources):
        tot = sum(src_stats[s][0] for s in sources)
        cov = {}
        for fld in fields:
            v = sum(src_stats[s][1][fld] for s in sources)
            cov[fld] = 100 * v / tot if tot else 0
        return tot, cov

    subset_sources = LEADERBOARDS
    full_incl = LEADERBOARDS + ["arxiv_llm", "papers_with_code"]
    full_excl = LEADERBOARDS + ["arxiv_llm"]
    tfi, covfi = agg(full_incl)
    tfe, covfe = agg(full_excl)
    ts, covs = agg(subset_sources)
    label = {"n_shot": "shots", "harness": "harness", "prompt_template": "prompt_template",
             "temperature": "temperature", "chain_of_thought": "chain_of_thought"}
    out("COVERAGE TABLE (paper §4.3)")
    out(f"  full(excl PWC) = {tfe} | full(incl PWC) = {tfi} | subset = {ts}")
    out(f"  {'field':<18}{'full-exclPWC':>14}{'full-inclPWC':>14}{'subset':>10}")
    for fld in fields:
        out(f"  {label[fld]:<18}{covfe[fld]:>14.1f}{covfi[fld]:>14.1f}{covs[fld]:>10.1f}")

    # unique models/benchmarks excluding PWC
    m_excl, b_excl = set(), set()
    for s in full_excl:
        m_excl |= src_stats[s][2]; b_excl |= src_stats[s][3]
    out("")
    out(f"unique models (excl PWC): {len(m_excl)}  | unique benchmarks (excl PWC): {len(b_excl)}")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
