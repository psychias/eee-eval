"""Repeated-measures (within-subject) format ANOVA for the log-likelihood benchmarks.

Prompt format is a WITHIN-subject factor: each subject (a model, or a model x seed
cell) is scored under plain/instruct/cot. A one-way between-groups ANOVA wrongly puts
the large between-subject variance (e.g. a 1.5B vs a 14B model) into the error term
and masks the format effect. The repeated-measures ANOVA blocks on the subject:

  SS_treatment (format), df = k-1
  SS_subject   (block),  df = n-1        <- removed from error
  SS_error = SS_total - SS_treatment - SS_subject, df = (n-1)(k-1)
  F = (SS_treatment/df_t) / (SS_error/df_e)

Also reports partial eta^2 = SS_treatment / (SS_treatment + SS_error) and, as a
non-parametric cross-check, the Friedman test (same blocking). Subjects: models for
the seed-deterministic MMLU/GPQA; (model,seed) cells for ARC/WinoGrande/OpenBookQA.
"""
import json
import math
import os
from collections import defaultdict

R = os.path.join(os.path.dirname(__file__), "..", "results")
FORMATS = ("plain", "instruct", "cot")


def load(name):
    p = os.path.join(R, name)
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


def _betacf(a, b, x):
    c = 1.0; d = 1 - (a + b) * x / (a + 1); d = 1e-30 if abs(d) < 1e-30 else d; d = 1 / d; h = d
    for m in range(1, 200):
        m2 = 2 * m
        aa = m * (b - m) * x / ((a - 1 + m2) * (a + m2)); d = 1 + aa * d; d = 1e-30 if abs(d) < 1e-30 else d
        c = 1 + aa / c; c = 1e-30 if abs(c) < 1e-30 else c; d = 1 / d; h *= d * c
        aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + 1 + m2)); d = 1 + aa * d; d = 1e-30 if abs(d) < 1e-30 else d
        c = 1 + aa / c; c = 1e-30 if abs(c) < 1e-30 else c; d = 1 / d; de = d * c; h *= de
        if abs(de - 1) < 3e-7:
            break
    return h


def f_sf(F, d1, d2):
    x = d2 / (d2 + d1 * F)
    a, b = d2 / 2.0, d1 / 2.0
    if x <= 0 or x >= 1:
        return 0.0 if x <= 0 else 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    return bt * _betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1 - bt * _betacf(b, a, 1 - x) / b


def chi2_sf(x, k):  # survival of chi-square (df=k), regularized upper incomplete gamma
    if x <= 0:
        return 1.0
    a = k / 2.0; xx = x / 2.0
    # series for lower gamma P(a,x); Q = 1-P
    if xx < a + 1:
        term = 1.0 / a; s = term; n = a
        for _ in range(500):
            n += 1; term *= xx / n; s += term
            if abs(term) < abs(s) * 1e-12:
                break
        P = s * math.exp(-xx + a * math.log(xx) - math.lgamma(a))
        return 1 - P
    b = xx + 1 - a; c = 1e30; d = 1 / b; h = d
    for i in range(1, 500):
        an = -i * (i - a); b += 2
        d = an * d + b; d = 1e-30 if abs(d) < 1e-30 else d
        c = b + an / c; c = 1e-30 if abs(c) < 1e-30 else c
        d = 1 / d; de = d * c; h *= de
        if abs(de - 1) < 1e-12:
            break
    return math.exp(-xx + a * math.log(xx) - math.lgamma(a)) * h


def rm_anova(mat):  # mat: list of rows (subjects), each len k (formats)
    n = len(mat); k = len(mat[0])
    grand = sum(v for r in mat for v in r) / (n * k)
    col_means = [sum(r[c] for r in mat) / n for c in range(k)]
    row_means = [sum(r) / k for r in mat]
    ss_t = n * sum((cm - grand) ** 2 for cm in col_means)
    ss_s = k * sum((rm - grand) ** 2 for rm in row_means)
    ss_tot = sum((v - grand) ** 2 for r in mat for v in r)
    ss_e = ss_tot - ss_t - ss_s
    df_t, df_e = k - 1, (n - 1) * (k - 1)
    F = (ss_t / df_t) / (ss_e / df_e) if ss_e > 1e-12 else float("inf")
    p = f_sf(F, df_t, df_e) if ss_e > 1e-12 else 0.0
    peta2 = 100 * ss_t / (ss_t + ss_e) if (ss_t + ss_e) else 0.0
    # Friedman
    Rj = [0.0] * k
    for r in mat:
        order = sorted(range(k), key=lambda c: r[c]); ranks = [0.0] * k; i = 0
        while i < k:
            j = i
            while j + 1 < k and r[order[j + 1]] == r[order[i]]:
                j += 1
            for t in range(i, j + 1):
                ranks[order[t]] = (i + j) / 2 + 1
            i = j + 1
        for c in range(k):
            Rj[c] += ranks[c]
    chi = 12.0 / (n * k * (k + 1)) * sum(x * x for x in Rj) - 3 * n * (k + 1)
    return F, df_t, df_e, p, peta2, chi, chi2_sf(chi, k - 1), n


def subjects(rows, bench, unit):
    d = defaultdict(dict)
    for r in rows:
        if r.get("benchmark") == bench and r.get("status", "ok") == "ok" and r.get("score") is not None \
                and r.get("n_shot") == 5 and r.get("prompt_format") in FORMATS:
            if "Phi-3.5" in r["model_id"]:
                continue
            key = r["model_id"] if unit == "model" else (r["model_id"], r["random_seed"])
            d[key][r["prompt_format"]] = r["score"]
    return [[b[f] for f in FORMATS] for b in d.values() if all(f in b for f in FORMATS)]


def main():
    seed_rows = load("generalization_hf.jsonl") + load("generalization_multiseed.jsonl")
    model_rows = load("mmlu_gpqa_8models.jsonl")
    print(f"{'benchmark':15}{'subj':6}{'n':4}{'F':>8}{'df':>8}{'p (RM)':>10}{'p-eta2':>8}{'Friedman p':>12}")
    print("-" * 71)
    for bench, rows, unit in [
        ("ARC-Challenge", seed_rows, "cell"), ("WinoGrande", seed_rows, "cell"),
        ("OpenBookQA", seed_rows, "cell"),
        ("MMLU", model_rows, "model"), ("GPQA", model_rows, "model"),
    ]:
        m = subjects(rows, bench, unit)
        if len(m) < 2:
            print(f"{bench:15} (insufficient)"); continue
        F, dft, dfe, p, pe, chi, fp, n = rm_anova(m)
        print(f"{bench:15}{unit:6}{n:4}{F:8.2f}{f'({dft},{dfe})':>8}{p:10.4f}{pe:7.0f}%{fp:12.2e}")


if __name__ == "__main__":
    main()
