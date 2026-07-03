"""Analyze the GSM8K-on-hf backend control (matched n=4/format, single seed, 5-shot,
full test set). Reports per-model format spreads and the pooled prompt-format ANOVA
on hf, to compare against the vLLM grid's GSM8K format effect and confirm the effect
is not a backend artefact.

The Llama-3.1 cells use the NousResearch mirror, whose chat template breaks the GSM8K
#### parser (scores near the floor), so they are reported but excluded from the
pooled ANOVA (flagged)."""
import json
import os
import statistics
from collections import defaultdict

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "gsm8k_hf_control.jsonl")
FORMATS = ("plain", "instruct", "cot")
MIRROR_ARTIFACT = "meta-llama/Llama-3.1-8B-Instruct"  # NousResearch mirror, near-floor


def anova_oneway(groups):
    vals = [v for g in groups for v in g]
    k = len(groups); N = len(vals)
    if N <= k:
        return None
    gm = statistics.mean(vals)
    ss_between = sum(len(g) * (statistics.mean(g) - gm) ** 2 for g in groups if g)
    ss_within = sum((v - statistics.mean(g)) ** 2 for g in groups for v in g)
    ss_total = ss_between + ss_within
    df_b, df_w = k - 1, N - k
    if df_w == 0 or ss_within == 0 or ss_total == 0:
        return dict(F=float("inf"), eta2=ss_between / ss_total if ss_total else 0.0, dfb=df_b, dfw=df_w)
    F = (ss_between / df_b) / (ss_within / df_w)
    return dict(F=F, eta2=ss_between / ss_total, dfb=df_b, dfw=df_w)


def main():
    rows = [json.loads(l) for l in open(RESULTS, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("status", "ok") == "ok" and r.get("n_shot") == 5]
    by_model = defaultdict(dict)
    for r in rows:
        by_model[r["model_id"]][r["prompt_format"]] = r["score"]

    print(f"GSM8K hf control: {len(by_model)} models, 5-shot, single seed\n")
    print(f"{'model':38} {'plain':>7} {'instr':>7} {'cot':>7} {'spread':>7}")
    for m, d in by_model.items():
        if all(f in d for f in FORMATS):
            sp = max(d.values()) - min(d.values())
            flag = "  <- mirror/near-floor (excluded from ANOVA)" if m == MIRROR_ARTIFACT else ""
            print(f"{m:38} {d['plain']:7.1f} {d['instruct']:7.1f} {d['cot']:7.1f} {sp:7.1f}{flag}")

    # pooled format ANOVA, excluding mirror artifact
    groups = []
    for f in FORMATS:
        groups.append([d[f] for m, d in by_model.items()
                       if m != MIRROR_ARTIFACT and f in d])
    res = anova_oneway(groups)
    print("\nPooled prompt-format ANOVA on hf GSM8K (excl. Llama mirror):")
    if res:
        print(f"  F({res['dfb']},{res['dfw']}) = {res['F']:.2f}   eta^2 = {100*res['eta2']:.1f}%   "
              f"n/format = {len(groups[0])}")
    print("\nCompare: vLLM grid GSM8K format effect = F=13.0, eta^2~27.4% (n=24/format).")


if __name__ == "__main__":
    main()
