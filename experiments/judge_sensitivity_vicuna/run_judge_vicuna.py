"""Judge-model sensitivity on a SECOND dataset (Vicuna-Bench), to test whether the
judge effect found on MT-Bench generalizes across benchmarks. Same 5 contestants
and 8 judge families, same fixed-response design (responses generated once at
temperature 0, then every judge scores every response), Vicuna-Bench's 80 single-
turn questions instead of MT-Bench's. Reports the judge-induced spread, cross-judge
Spearman, and a formal Friedman test over the fixed responses. All via OpenRouter,
no GPU. Directly addresses the reviewer's "span more datasets" request."""
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
RESP_PATH = os.path.join(RESULTS_DIR, "responses.jsonl")
JUDGE_PATH = os.path.join(RESULTS_DIR, "judge_runs.jsonl")
STATS_PATH = os.path.join(SCRIPT_DIR, "judge_vicuna_stats.json")

BASE = "https://openrouter.ai/api/v1/chat/completions"
VICUNA_URL = ("https://raw.githubusercontent.com/lm-sys/FastChat/main/"
              "fastchat/llm_judge/data/vicuna_bench/question.jsonl")

CONTESTANTS = [
    "google/gemma-3-4b-it",
    "qwen/qwen-2.5-7b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "mistralai/ministral-8b-2512",
    "qwen/qwen-2.5-72b-instruct",
]
JUDGES = [
    "openai/gpt-4o-mini", "google/gemini-3.5-flash",
    "meta-llama/llama-3.1-70b-instruct", "anthropic/claude-haiku-4.5",
    "deepseek/deepseek-chat", "mistralai/mistral-large",
    "qwen/qwen-2.5-72b-instruct", "amazon/nova-pro-v1",
]
MAX_WORKERS = 8
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
        sys.exit("OPENROUTER_API_KEY not found")
    return key


HEADERS = {"Authorization": f"Bearer {load_key()}", "Content-Type": "application/json"}


def chat(model, messages, max_tokens=800, retries=4):
    payload = {"model": model, "messages": messages, "temperature": 0.0, "max_tokens": max_tokens}
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(BASE, headers=HEADERS, json=payload, timeout=120)
            if r.status_code == 200:
                content = r.json()["choices"][0]["message"].get("content")
                if content:
                    return content
                last = "empty content"
            else:
                last = f"HTTP {r.status_code}: {r.text[:150]}"
        except Exception as e:  # noqa: BLE001
            last = str(e)
        time.sleep(2 * (attempt + 1))
    return f"__ERROR__ {last}"


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


def append_jsonl(p, o):
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")


def parse_rating(text):
    m = re.search(r"\[\[\s*(\d+(?:\.\d+)?)\s*\]\]", text) or re.search(r"[Rr]ating:?\s*(\d+(?:\.\d+)?)", text)
    if not m:
        return None
    try:
        v = float(m.group(1)); return v if 1 <= v <= 10 else None
    except ValueError:
        return None


def load_prompts():
    raw = urllib.request.urlopen(VICUNA_URL, timeout=60).read().decode("utf-8")
    return [{"question_id": json.loads(l)["question_id"],
             "category": json.loads(l).get("category", ""),
             "prompt": json.loads(l)["turns"][0]} for l in raw.splitlines() if l.strip()]


def generate_responses(prompts):
    done = {(r["model"], r["question_id"]) for r in read_jsonl(RESP_PATH)}
    todo = [(m, q) for m in CONTESTANTS for q in prompts if (m, q["question_id"]) not in done]
    print(f"[responses] {len(done)} cached, {len(todo)} to generate", flush=True)
    import threading
    lock = threading.Lock()

    def work(item):
        model, q = item
        out = chat(model, [{"role": "user", "content": q["prompt"]}], max_tokens=1024)
        rec = {"model": model, "question_id": q["question_id"], "category": q["category"],
               "prompt": q["prompt"], "response": out, "ok": not out.startswith("__ERROR__")}
        with lock:
            append_jsonl(RESP_PATH, rec)
        return rec["ok"]
    if todo:
        with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            print(f"[responses] generated {sum(ex.map(work, todo))}/{len(todo)} ok", flush=True)


def run_judges():
    responses = {(r["model"], r["question_id"]): r for r in read_jsonl(RESP_PATH) if r.get("ok")}
    good = [j for j in read_jsonl(JUDGE_PATH) if j.get("rating") is not None]
    with open(JUDGE_PATH, "w", encoding="utf-8") as f:
        for j in good:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")
    done = {(j["contestant"], j["question_id"], j["judge"]) for j in good}
    todo = [(jm, key) for jm in JUDGES for key in responses if (key[0], key[1], jm) not in done]
    print(f"[judges] {len(done)} cached, {len(todo)} to score", flush=True)
    import threading
    lock = threading.Lock()

    def work(item):
        judge, key = item
        resp = responses[key]
        out = chat(judge, [{"role": "user", "content": JUDGE_TEMPLATE.format(question=resp["prompt"], answer=resp["response"])}])
        rec = {"contestant": key[0], "question_id": key[1], "category": resp["category"],
               "judge": judge, "rating": parse_rating(out), "raw_ok": not out.startswith("__ERROR__")}
        with lock:
            append_jsonl(JUDGE_PATH, rec)
        return rec["rating"] is not None
    if todo:
        with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            print(f"[judges] parsed {sum(ex.map(work, todo))}/{len(todo)} ratings", flush=True)


def friedman(matrix):
    n, k = len(matrix), len(matrix[0])
    Rj = [0.0] * k
    for row in matrix:
        order = sorted(range(k), key=lambda c: row[c])
        ranks = [0.0] * k; i = 0
        while i < k:
            j = i
            while j + 1 < k and row[order[j + 1]] == row[order[i]]:
                j += 1
            for t in range(i, j + 1):
                ranks[order[t]] = (i + j) / 2 + 1
            i = j + 1
        for c in range(k):
            Rj[c] += ranks[c]
    stat = 12.0 / (n * k * (k + 1)) * sum(r * r for r in Rj) - 3 * n * (k + 1)
    return stat, k - 1, stat / (n * (k - 1))


def aggregate():
    rows = [j for j in read_jsonl(JUDGE_PATH) if j.get("rating") is not None and j["judge"] in JUDGES]
    bucket = defaultdict(list)
    blocks = defaultdict(dict)
    for j in rows:
        bucket[(j["contestant"], j["judge"])].append(j["rating"])
        blocks[(j["contestant"], j["question_id"])][j["judge"]] = j["rating"]
    mean = {k: sum(v) / len(v) for k, v in bucket.items()}
    spreads = {}
    for c in CONTESTANTS:
        vals = [mean.get((c, j)) for j in JUDGES if mean.get((c, j)) is not None]
        if len(vals) >= 2:
            spreads[c] = max(vals) - min(vals)
    ms = sum(spreads.values()) / len(spreads) if spreads else float("nan")
    rankvecs = {j: [mean.get((c, j)) for c in CONTESTANTS] for j in JUDGES}
    usable = [j for j in JUDGES if all(v is not None for v in rankvecs[j])]

    def spearman(a, b):
        n = len(a); ra = sorted(range(n), key=lambda i: a[i]); rb = sorted(range(n), key=lambda i: b[i])
        ka = [0] * n; kb = [0] * n
        for r, i in enumerate(ra): ka[i] = r
        for r, i in enumerate(rb): kb[i] = r
        ma, mb = sum(ka) / n, sum(kb) / n
        num = sum((ka[i] - ma) * (kb[i] - mb) for i in range(n))
        da = sum((ka[i] - ma) ** 2 for i in range(n)) ** .5; db = sum((kb[i] - mb) ** 2 for i in range(n)) ** .5
        return num / (da * db) if da and db else float("nan")
    corrs = [spearman(rankvecs[a], rankvecs[b]) for a, b in itertools.combinations(usable, 2)]
    complete = [[b[j] for j in JUDGES] for b in blocks.values() if len(b) == len(JUDGES)]
    stat, df, W = friedman(complete) if complete else (float("nan"), 0, float("nan"))
    stats = {"dataset": "vicuna_bench", "n_contestants": len(CONTESTANTS), "n_judges": len(JUDGES),
             "mean_spread": round(ms, 3), "max_spread": round(max(spreads.values()), 3) if spreads else None,
             "cross_judge_spearman": {"min": round(min(corrs), 3), "max": round(max(corrs), 3),
                                      "mean": round(sum(corrs) / len(corrs), 3), "n_pairs": len(corrs)} if corrs else {},
             "friedman": {"chi2": round(stat, 1), "df": df, "n_blocks": len(complete), "kendall_w": round(W, 3)}}
    json.dump(stats, open(STATS_PATH, "w"), indent=2)
    print(json.dumps(stats, indent=2), flush=True)
    print(f"[wrote] {STATS_PATH}", flush=True)


def main():
    generate_responses(load_prompts())
    run_judges()
    aggregate()


if __name__ == "__main__":
    main()
