"""Powered prompt-format ANOVA for every log-likelihood benchmark, using the
appropriate replication unit for each:

  * Random-few-shot benchmarks (ARC-Challenge, WinoGrande, OpenBookQA) vary by seed,
    so they are powered by SEEDS: 4 models x 3 seeds = n=12 per format.
  * Deterministic fixed-few-shot benchmarks (MMLU, GPQA) do not vary by seed, so they
    are powered by MODELS: 8 models x 1 seed = n=8 per format.

Reports F, p (approx via F-dist survival), and eta^2 per benchmark, for the §G table.
"""
import json
import math
import os
from collections import defaultdict

R = os.path.join(os.path.dirname(__file__), "..", "results")


def load(name):
    p = os.path.join(R, name)
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


def anova(groups):
    vals = [v for g in groups for v in g]
    k, N = len(groups), len(vals)
    if N <= k:
        return None
    gm = sum(vals) / N
    ssb = sum(len(g) * (sum(g) / len(g) - gm) ** 2 for g in groups if g)
    ssw = sum((v - sum(g) / len(g)) ** 2 for g in groups for v in g)
    sst = ssb + ssw
    dfb, dfw = k - 1, N - k
    if ssw == 0 or dfw == 0:
        return dict(F=float("inf"), p=0.0, eta2=100.0, dfb=dfb, dfw=dfw, nf=[len(g) for g in groups])
    F = (ssb / dfb) / (ssw / dfw)
    return dict(F=F, p=_f_sf(F, dfb, dfw), eta2=100 * ssb / sst, dfb=dfb, dfw=dfw, nf=[len(g) for g in groups])


def _betacf(a, b, x):
    MAXIT, EPS = 200, 3e-7
    qab, qap, qam = a + b, a + 1, a - 1
    c = 1.0
    d = 1 - qab * x / qap
    d = 1e-30 if abs(d) < 1e-30 else d
    d = 1 / d
    h = d
    for m in range(1, MAXIT):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d; d = 1e-30 if abs(d) < 1e-30 else d
        c = 1 + aa / c; c = 1e-30 if abs(c) < 1e-30 else c
        d = 1 / d; h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d; d = 1e-30 if abs(d) < 1e-30 else d
        c = 1 + aa / c; c = 1e-30 if abs(c) < 1e-30 else c
        d = 1 / d; de = d * c; h *= de
        if abs(de - 1) < EPS:
            break
    return h


def _betai(a, b, x):
    if x <= 0 or x >= 1:
        return 0.0 if x <= 0 else 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return bt * _betacf(a, b, x) / a
    return 1 - bt * _betacf(b, a, 1 - x) / b


def _f_sf(F, d1, d2):  # P(F > f)
    return _betai(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * F))


FORMATS = ("plain", "instruct", "cot")


def groups_for(rows, bench, key):
    g = {f: [] for f in FORMATS}
    for r in rows:
        if r.get("benchmark") == bench and r.get("status", "ok") == "ok" and r.get("n_shot") == 5 \
                and r.get("prompt_format") in g:
            g[r["prompt_format"]].append(r["score"])
    return [g[f] for f in FORMATS]


def main():
    seed_rows = load("generalization_hf.jsonl") + load("generalization_multiseed.jsonl")
    model_rows = load("mmlu_gpqa_8models.jsonl")
    print(f"{'benchmark':16}{'design':10}{'n/format':10}{'F':>8}{'p':>10}{'eta^2':>8}")
    print("-" * 62)
    for bench, rows, design in [
        ("ARC-Challenge", seed_rows, "seeds"), ("WinoGrande", seed_rows, "seeds"),
        ("OpenBookQA", seed_rows, "seeds"),
        ("MMLU", model_rows, "models"), ("GPQA", model_rows, "models"),
    ]:
        res = anova(groups_for(rows, bench, design))
        if res:
            print(f"{bench:16}{design:10}{'/'.join(map(str,res['nf'])):10}"
                  f"{res['F']:8.2f}{res['p']:10.4f}{res['eta2']:7.1f}%")
        else:
            print(f"{bench:16}{design:10}{'(no data yet)':>30}")
    print("\nReference: GSM8K (generation, vLLM grid) F=13.0, p<0.001, eta^2~27%.")


if __name__ == "__main__":
    main()
