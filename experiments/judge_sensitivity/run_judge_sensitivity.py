"""
Judge-model sensitivity (E5).

Holds the evaluated responses FIXED and varies only the judge model, to isolate
the effect of `judge_model` on benchmark scores and rankings. Motivated by the
TruthfulQA Case 2 confound (judge + n_shot entangled) and the four LLM-judge
sources in the dataset (Chatbot Arena, AlpacaEval 2, MT-Bench, WildBench).

Pipeline (all via OpenRouter, no GPU):
  1. Generate one response per (contestant model, MT-Bench question, turn-1).
  2. Score every response with each judge using the canonical MT-Bench
     single-answer 1-10 grading rubric.
  3. Aggregate per (contestant, judge); report the judge-induced score spread
     and the cross-judge ranking (in)stability.

Both stages cache to JSONL so re-runs only fill gaps. Outputs:
  results/responses.jsonl      one row per (contestant, question)
  results/judge_runs.jsonl     one row per (contestant, question, judge)
  judge_sensitivity_report.txt human-readable summary
"""

import concurrent.futures as cf
import json
import os
import re
import sys
import time
import urllib.request

import requests

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
RESP_PATH   = os.path.join(RESULTS_DIR, "responses.jsonl")
JUDGE_PATH  = os.path.join(RESULTS_DIR, "judge_runs.jsonl")
REPORT_PATH = os.path.join(SCRIPT_DIR, "judge_sensitivity_report.txt")

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
JUDGES = [
    "openai/gpt-4o-mini",
    "google/gemini-3.5-flash",
    "meta-llama/llama-3.1-70b-instruct",
    "anthropic/claude-haiku-4.5",
]

MAX_WORKERS = 8
GEN_MAX_TOKENS   = 1024
JUDGE_MAX_TOKENS = 800

# Canonical MT-Bench single-answer grading prompt (FastChat single-v1).
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
                j = r.json()
                content = j["choices"][0]["message"].get("content")
                if content:
                    return content
                last = "empty/None content"  # null content -> retry, then error
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
                   "prompt": obj["turns"][0]})  # turn-1 only
    return qs


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def parse_rating(text):
    m = re.search(r"\[\[\s*(\d+(?:\.\d+)?)\s*\]\]", text)
    if not m:
        m = re.search(r"[Rr]ating:?\s*(\d+(?:\.\d+)?)", text)
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
    lock = __import__("threading").Lock()

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

    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        res = list(ex.map(work, todo))
    print(f"[responses] generated {sum(res)}/{len(todo)} ok")


def run_judges(prompts):
    responses = {(r["model"], r["question_id"]): r
                 for r in read_jsonl(RESP_PATH) if r.get("ok")}
    # Only successful judgments count as done; purge failed rows (e.g. HTTP 402
    # credit errors, unparsed) from the cache so a resume retries exactly them.
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
    lock = __import__("threading").Lock()

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

    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        res = list(ex.map(work, todo))
    print(f"[judges] parsed {sum(res)}/{len(todo)} ratings")


def aggregate():
    rows = [j for j in read_jsonl(JUDGE_PATH) if j.get("rating") is not None]
    # mean[(contestant, judge)] = mean rating
    from collections import defaultdict
    bucket = defaultdict(list)
    for j in rows:
        bucket[(j["contestant"], j["judge"])].append(j["rating"])
    mean = {k: sum(v) / len(v) for k, v in bucket.items()}
    n_by = {k: len(v) for k, v in bucket.items()}

    lines = []
    def out(s=""):
        lines.append(s); print(s)

    out("=" * 84)
    out("JUDGE-MODEL SENSITIVITY (E5): identical responses, judge varied")
    out("=" * 84)
    out(f"Contestants: {len(CONTESTANTS)}  |  Judges: {len(JUDGES)}  |  "
        f"MT-Bench turn-1 questions")
    out("")
    # matrix
    header = f"{'contestant':<34}" + "".join(f"{j.split('/')[-1][:14]:>15}" for j in JUDGES) + f"{'spread':>9}"
    out(header)
    out("-" * len(header))
    contestant_spread = {}
    for c in CONTESTANTS:
        vals = [mean.get((c, j)) for j in JUDGES]
        cells = "".join((f"{v:>15.2f}" if v is not None else f"{'--':>15}") for v in vals)
        present = [v for v in vals if v is not None]
        spread = (max(present) - min(present)) if len(present) >= 2 else float("nan")
        contestant_spread[c] = spread
        out(f"{c:<34}{cells}{spread:>9.2f}")
    out("")
    out("Judge-induced score spread per contestant (max judge mean - min judge mean):")
    for c in CONTESTANTS:
        out(f"  {c:<34} {contestant_spread[c]:.2f} pts (1-10 scale)")
    valid = [s for s in contestant_spread.values() if s == s]
    if valid:
        out(f"  mean spread: {sum(valid)/len(valid):.2f}  |  max spread: {max(valid):.2f}")
    out("")

    # rankings per judge
    out("Ranking per judge (best -> worst by mean rating):")
    rankings = {}
    for j in JUDGES:
        ranked = sorted([c for c in CONTESTANTS if (c, j) in mean],
                        key=lambda c: mean[(c, j)], reverse=True)
        rankings[j] = ranked
        out(f"  {j.split('/')[-1]:<22}: " +
            " > ".join(f"{c.split('/')[-1]}({mean[(c,j)]:.2f})" for c in ranked))
    out("")

    # pairwise rank correlation
    try:
        from scipy.stats import spearmanr
        common = CONTESTANTS
        idx = {c: i for i, c in enumerate(common)}
        out("Pairwise Spearman rank correlation between judges:")
        for a in range(len(JUDGES)):
            for b in range(a + 1, len(JUDGES)):
                ja, jb = JUDGES[a], JUDGES[b]
                ra = [idx[c] for c in rankings[ja]]
                rb_rank = {c: r for r, c in enumerate(rankings[jb])}
                # align on contestants present in both
                cs = [c for c in rankings[ja] if c in rb_rank]
                xa = [rankings[ja].index(c) for c in cs]
                xb = [rb_rank[c] for c in cs]
                if len(cs) >= 3:
                    rho, _ = spearmanr(xa, xb)
                    out(f"  {ja.split('/')[-1]:<20} vs {jb.split('/')[-1]:<20}: rho={rho:+.3f}")
    except ImportError:
        out("(scipy not available; skipped Spearman)")

    out("")
    out(f"Rows scored: {len(rows)}  |  written {JUDGE_PATH}")
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {REPORT_PATH}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    prompts = load_prompts()
    print(f"Loaded {len(prompts)} MT-Bench turn-1 prompts")
    generate_responses(prompts)
    run_judges(prompts)
    aggregate()


if __name__ == "__main__":
    main()
