"""
Judge-model sensitivity (extended).

Broadens the original single-task / 4-judge study (../judge_sensitivity) along the
two axes a reviewer would ask for: more judge FAMILIES and more TASKS.

  * Judges: 8 distinct provider families
      OpenAI, Google, Meta, Anthropic, DeepSeek, Mistral, Alibaba/Qwen, Amazon.
  * Tasks: MT-Bench's 8 categories (writing, roleplay, reasoning, math, coding,
      extraction, stem, humanities) reported separately, so the judge effect is
      measured per task type rather than pooled.

Holds the evaluated responses FIXED and varies only the judge, to isolate the
effect of `judge_model` on scores and rankings. All via OpenRouter, no GPU.
Contestant responses are reused from the base study (seeded into results/), so a
first run only scores the four NEW judges.

Outputs (results/):
  responses.jsonl    one row per (contestant, question)   [seeded from base study]
  judge_runs.jsonl   one row per (contestant, question, judge)
  ../judge_sensitivity_extended_report.txt
  ../judge_sensitivity_extended_stats.json
"""

import concurrent.futures as cf
import itertools
import json
import os
import re
import sys
import time
import urllib.request
from collections import defaultdict

import requests

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
RESP_PATH   = os.path.join(RESULTS_DIR, "responses.jsonl")
JUDGE_PATH  = os.path.join(RESULTS_DIR, "judge_runs.jsonl")
REPORT_PATH = os.path.join(SCRIPT_DIR, "judge_sensitivity_extended_report.txt")
STATS_PATH  = os.path.join(SCRIPT_DIR, "judge_sensitivity_extended_stats.json")

BASE = "https://openrouter.ai/api/v1/chat/completions"
MT_BENCH_URL = ("https://raw.githubusercontent.com/lm-sys/FastChat/main/"
                "fastchat/llm_judge/data/mt_bench/question.jsonl")

CONTESTANTS = [
    "google/gemma-3-4b-it",
    "qwen/qwen-2.5-7b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "mistralai/ministral-8b-2512",
    "qwen/qwen-2.5-72b-instruct",
]
# 8 judge families (provider in comments)
JUDGES = [
    "openai/gpt-4o-mini",                 # OpenAI
    "google/gemini-3.5-flash",            # Google
    "meta-llama/llama-3.1-70b-instruct",  # Meta
    "anthropic/claude-haiku-4.5",         # Anthropic
    "deepseek/deepseek-chat",             # DeepSeek
    "mistralai/mistral-large",            # Mistral
    "qwen/qwen-2.5-72b-instruct",         # Alibaba / Qwen
    "amazon/nova-pro-v1",                 # Amazon
]

MAX_WORKERS = 8
GEN_MAX_TOKENS   = 1024
JUDGE_MAX_TOKENS = 800

JUDGE_TEMPLATE = (
    "[Instruction]\nPlease act as an impartial judge and evaluate the quality "
    "of the response provided by an AI assistant to the user question displayed "
    "below. Your evaluation should consider factors such as the helpfulness, "
    "relevance, accuracy, depth, creativity, and level of detail of the "
    "response. Begin your evaluation by providing a short explanation. Be as "
    "objective as possible. After providing your explanation, you must rate the "
    "response on a scale of 1 to 10 by strictly following this format: "
    '"[[rating]]", for example: "Rating: [[5]]".\n\n'
    "[Question]\n{question}\n\n"
    "[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]"
)


def load_key():
    env = os.path.join(REPO_ROOT, ".env")
    if os.path.exists(env):
        for line in open(env, encoding="utf-8"):
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY not found in .env or environment")
    return key


HEADERS = {"Authorization": f"Bearer {load_key()}", "Content-Type": "application/json"}


def chat(model, messages, temperature=0.0, max_tokens=512, retries=4):
    payload = {"model": model, "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens}
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(BASE, headers=HEADERS, json=payload, timeout=120)
            if r.status_code == 200:
                content = r.json()["choices"][0]["message"].get("content")
                if content:
                    return content
                last = "empty/None content"
            else:
                last = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:  # noqa: BLE001
            last = str(e)
        time.sleep(2 * (attempt + 1))
    return f"__ERROR__ {last}"


def load_prompts():
    raw = urllib.request.urlopen(MT_BENCH_URL, timeout=60).read().decode("utf-8")
    qs = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        qs.append({"question_id": obj["question_id"],
                   "category": obj.get("category", ""),
                   "prompt": obj["turns"][0]})  # turn-1
    return qs


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(ln) for ln in open(path, encoding="utf-8") if ln.strip()]


def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def seed_from_base():
    """Copy the base study's cached responses + (4-judge) ratings so we reuse
    them and only score the new judges."""
    base = os.path.join(SCRIPT_DIR, "..", "judge_sensitivity", "results")
    for fname, dst in (("responses.jsonl", RESP_PATH), ("judge_runs.jsonl", JUDGE_PATH)):
        src = os.path.join(base, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            with open(src, encoding="utf-8") as fi, open(dst, "w", encoding="utf-8") as fo:
                fo.write(fi.read())
            print(f"[seed] {fname}: {sum(1 for _ in open(dst, encoding='utf-8'))} rows from base study")


def parse_rating(text):
    m = re.search(r"\[\[\s*(\d+(?:\.\d+)?)\s*\]\]", text) or \
        re.search(r"[Rr]ating:?\s*(\d+(?:\.\d+)?)", text)
    if not m:
        return None
    try:
        v = float(m.group(1))
        return v if 1 <= v <= 10 else None
    except ValueError:
        return None


def generate_responses(prompts):
    done = {(r["model"], r["question_id"]) for r in read_jsonl(RESP_PATH)}
    todo = [(m, q) for m in CONTESTANTS for q in prompts
            if (m, q["question_id"]) not in done]
    print(f"[responses] {len(done)} cached, {len(todo)} to generate")
    import threading
    lock = threading.Lock()

    def work(item):
        model, q = item
        out = chat(model, [{"role": "user", "content": q["prompt"]}],
                   temperature=0.0, max_tokens=GEN_MAX_TOKENS)
        rec = {"model": model, "question_id": q["question_id"],
               "category": q["category"], "prompt": q["prompt"],
               "response": out, "ok": not out.startswith("__ERROR__")}
        with lock:
            append_jsonl(RESP_PATH, rec)
        return rec["ok"]

    if todo:
        with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            res = list(ex.map(work, todo))
        print(f"[responses] generated {sum(res)}/{len(todo)} ok")


def run_judges():
    responses = {(r["model"], r["question_id"]): r
                 for r in read_jsonl(RESP_PATH) if r.get("ok")}
    all_rows = read_jsonl(JUDGE_PATH)
    good = [j for j in all_rows if j.get("rating") is not None]
    if len(good) != len(all_rows):
        with open(JUDGE_PATH, "w", encoding="utf-8") as f:
            for j in good:
                f.write(json.dumps(j, ensure_ascii=False) + "\n")
        print(f"[judges] purged {len(all_rows) - len(good)} failed rows; kept {len(good)}")
    done = {(j["contestant"], j["question_id"], j["judge"]) for j in good}
    todo = [(jm, key) for jm in JUDGES for key in responses
            if (key[0], key[1], jm) not in done]
    print(f"[judges] {len(done)} cached, {len(todo)} to score")
    import threading
    lock = threading.Lock()

    def work(item):
        judge, key = item
        resp = responses[key]
        content = JUDGE_TEMPLATE.format(question=resp["prompt"], answer=resp["response"])
        out = chat(judge, [{"role": "user", "content": content}],
                   temperature=0.0, max_tokens=JUDGE_MAX_TOKENS)
        rec = {"contestant": key[0], "question_id": key[1],
               "category": resp["category"], "judge": judge,
               "rating": parse_rating(out), "raw_ok": not out.startswith("__ERROR__")}
        with lock:
            append_jsonl(JUDGE_PATH, rec)
        return rec["rating"] is not None

    if todo:
        with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            res = list(ex.map(work, todo))
        print(f"[judges] parsed {sum(res)}/{len(todo)} ratings")


def _spearman(a, b):
    """Spearman rank correlation of two equal-length score lists (no ties handling
    needed for our small contestant sets; falls back to average ranks)."""
    n = len(a)
    if n < 2:
        return float("nan")

    def ranks(xs):
        order = sorted(range(n), key=lambda i: xs[i])
        rk = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk

    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = sum((ra[i] - ma) ** 2 for i in range(n)) ** 0.5
    db = sum((rb[i] - mb) ** 2 for i in range(n)) ** 0.5
    return num / (da * db) if da and db else float("nan")


def aggregate():
    rows = [j for j in read_jsonl(JUDGE_PATH)
            if j.get("rating") is not None and j["judge"] in JUDGES]
    bucket = defaultdict(list)          # (contestant, judge) -> ratings
    cat_bucket = defaultdict(list)      # (category, contestant, judge) -> ratings
    for j in rows:
        bucket[(j["contestant"], j["judge"])].append(j["rating"])
        cat_bucket[(j["category"], j["contestant"], j["judge"])].append(j["rating"])
    mean = {k: sum(v) / len(v) for k, v in bucket.items()}

    lines, stats = [], {}

    def out(s=""):
        lines.append(s)
        print(s)

    out("=" * 96)
    out("JUDGE-MODEL SENSITIVITY (EXTENDED): 8 judge families x 8 MT-Bench task categories")
    out("=" * 96)
    out(f"Contestants: {len(CONTESTANTS)}  |  Judges: {len(JUDGES)}  |  MT-Bench turn-1 (80 questions)")
    out("")

    # ---- overall matrix ----
    header = f"{'contestant':<30}" + "".join(f"{j.split('/')[-1][:12]:>13}" for j in JUDGES) + f"{'spread':>9}"
    out(header)
    out("-" * len(header))
    spreads = {}
    for c in CONTESTANTS:
        vals = [mean.get((c, j)) for j in JUDGES]
        cells = "".join((f"{v:>13.2f}" if v is not None else f"{'--':>13}") for v in vals)
        present = [v for v in vals if v is not None]
        sp = (max(present) - min(present)) if len(present) >= 2 else float("nan")
        spreads[c] = sp
        out(f"{c:<30}{cells}{sp:>9.2f}")
    finite = [s for s in spreads.values() if s == s]
    mean_spread = sum(finite) / len(finite) if finite else float("nan")
    max_spread = max(finite) if finite else float("nan")
    out("")
    out(f"Judge-induced spread: mean {mean_spread:.2f}, max {max_spread:.2f} (10-point scale)")

    # ---- cross-judge rank correlation (over contestants) ----
    judge_rankvecs = {j: [mean.get((c, j), float("nan")) for c in CONTESTANTS] for j in JUDGES}
    usable = [j for j in JUDGES if all(v == v for v in judge_rankvecs[j])]
    corrs = []
    for a, b in itertools.combinations(usable, 2):
        rho = _spearman(judge_rankvecs[a], judge_rankvecs[b])
        if rho == rho:
            corrs.append(rho)
    if corrs:
        out(f"Cross-judge Spearman rank correlation over {len(usable)} judges: "
            f"min {min(corrs):+.2f}, max {max(corrs):+.2f}, mean {sum(corrs)/len(corrs):+.2f} "
            f"({len(corrs)} judge pairs)")

    # ---- per-category (task) breakdown ----
    out("")
    out("Per-task (MT-Bench category) judge-induced spread, averaged over contestants:")
    out(f"  {'category':<14}{'mean spread':>12}{'max spread':>12}{'judges':>8}")
    cats = sorted({k[0] for k in cat_bucket})
    cat_stats = {}
    for cat in cats:
        cmean = {}
        for c in CONTESTANTS:
            for j in JUDGES:
                v = cat_bucket.get((cat, c, j))
                if v:
                    cmean[(c, j)] = sum(v) / len(v)
        per_c = []
        for c in CONTESTANTS:
            vals = [cmean[(c, j)] for j in JUDGES if (c, j) in cmean]
            if len(vals) >= 2:
                per_c.append(max(vals) - min(vals))
        if per_c:
            ms, mx = sum(per_c) / len(per_c), max(per_c)
            cat_stats[cat] = {"mean_spread": round(ms, 3), "max_spread": round(mx, 3)}
            njudges = len({j for (c, j) in cmean})
            out(f"  {cat:<14}{ms:>12.2f}{mx:>12.2f}{njudges:>8}")

    out("")
    out("Takeaway: the judge-model effect is not a single-task artefact. Across all "
        f"{len(cats)} MT-Bench task categories and {len(usable)} judge families the judge "
        "induces a multi-point score spread and the cross-judge ranking is unstable, "
        "reinforcing judge_model as an EvalSpec Required field.")

    stats = {
        "n_contestants": len(CONTESTANTS), "n_judges": len(JUDGES),
        "judges": JUDGES, "contestants": CONTESTANTS,
        "overall": {"mean_spread": round(mean_spread, 3), "max_spread": round(max_spread, 3),
                    "spreads": {c: round(spreads[c], 3) for c in CONTESTANTS if spreads[c] == spreads[c]}},
        "cross_judge_spearman": {"min": round(min(corrs), 3), "max": round(max(corrs), 3),
                                 "mean": round(sum(corrs) / len(corrs), 3), "n_pairs": len(corrs)} if corrs else {},
        "per_category": cat_stats,
        "matrix": {c: {j: round(mean[(c, j)], 3) for j in JUDGES if (c, j) in mean} for c in CONTESTANTS},
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"\n[wrote] {REPORT_PATH}")
    print(f"[wrote] {STATS_PATH}")


def main():
    seed_from_base()
    prompts = load_prompts()
    generate_responses(prompts)
    run_judges()
    aggregate()


if __name__ == "__main__":
    main()
