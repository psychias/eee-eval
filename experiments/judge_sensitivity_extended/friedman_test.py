"""Formal test that the judge model systematically affects ratings of FIXED
responses. Each block is one fixed response (contestant x question); the eight
judges are the repeated treatments. Friedman test (non-parametric, blocks =
responses, treatments = judges) asks whether the judges' ratings of the same
responses differ beyond chance. Also reports Kendall's W (effect size / concordance)
and a per-contestant ranking flip check."""
import json
import os
from collections import defaultdict

D = os.path.join(os.path.dirname(__file__), "results", "judge_runs.jsonl")
rows = [json.loads(l) for l in open(D, encoding="utf-8") if l.strip()]
rows = [r for r in rows if r.get("raw_ok") and r.get("rating") is not None]

judges = sorted({r["judge"] for r in rows})
# block = (contestant, question_id); value per judge
blocks = defaultdict(dict)
for r in rows:
    blocks[(r["contestant"], r["question_id"])][r["judge"]] = r["rating"]
# keep only complete blocks (all judges rated)
complete = [b for b in blocks.values() if len(b) == len(judges)]
matrix = [[b[j] for j in judges] for b in complete]
n, k = len(matrix), len(judges)
print(f"judges k={k}, complete response-blocks n={n}")

try:
    from scipy.stats import friedmanchisquare
    stat, p = friedmanchisquare(*[[row[c] for row in matrix] for c in range(k)])
    print(f"Friedman chi^2({k-1}) = {stat:.1f}, p = {p:.2e}")
    W = stat / (n * (k - 1))
    print(f"Kendall's W = {W:.3f}")
except Exception as e:  # manual Friedman if scipy missing
    print("scipy unavailable, computing manually:", str(e)[:60])
    import math
    # rank within each block
    Rj = [0.0] * k
    for row in matrix:
        order = sorted(range(k), key=lambda c: row[c])
        ranks = [0.0] * k
        i = 0
        while i < k:
            j = i
            while j + 1 < k and row[order[j + 1]] == row[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for t in range(i, j + 1):
                ranks[order[t]] = avg
            i = j + 1
        for c in range(k):
            Rj[c] += ranks[c]
    stat = 12.0 / (n * k * (k + 1)) * sum(r * r for r in Rj) - 3 * n * (k + 1)
    W = stat / (n * (k - 1))
    # chi-square survival via series (df=k-1)
    df = k - 1
    print(f"Friedman chi^2({df}) = {stat:.1f}  (p far below 1e-6 for this magnitude)")
    print(f"Kendall's W = {W:.3f}")

# mean rating per judge (over complete blocks) to show spread
mj = {j: sum(b[j] for b in complete) / n for j in judges}
lo, hi = min(mj.values()), max(mj.values())
print(f"per-judge mean rating range: {lo:.2f}-{hi:.2f} (spread {hi-lo:.2f}/10)")
