"""Broaden the "temperature contributes <1% of variance" finding beyond GSM8K to a
panel of generation tasks, via OpenRouter. Tests the reviewer's concern that the
result (parsed short answers, greedy decoding) may not generalize to open-ended
generation.

Panel spans the generation spectrum:
  * GSM8K  (parsed short numeric answer; #### / last-number match)   -- like the original
  * MT-Bench (open-ended, scored 1-10 by a fixed judge)              -- open-ended regime

For each (model, task, temperature in {0.0, 0.7, 1.0}) we draw 2 samples (temp>0 is
stochastic), score them, and report the temperature-induced score spread and the
one-way temperature eta^2 per task. If temperature moves open-ended (MT-Bench)
scores much more than parsed (GSM8K) scores, the <1% finding is task-scoped, exactly
as the paper's caveat states -- now quantified. All via OpenRouter, no GPU."""
import concurrent.futures as cf
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
GEN_PATH = os.path.join(RESULTS_DIR, "gen.jsonl")
STATS_PATH = os.path.join(SCRIPT_DIR, "temperature_panel_stats.json")

BASE = "https://openrouter.ai/api/v1/chat/completions"
MT_BENCH_URL = ("https://raw.githubusercontent.com/lm-sys/FastChat/main/"
                "fastchat/llm_judge/data/mt_bench/question.jsonl")

MODELS = [
    "qwen/qwen-2.5-7b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "mistralai/ministral-8b-2512",
    "qwen/qwen-2.5-72b-instruct",
]
JUDGE = "openai/gpt-4o-mini"          # fixed judge for the open-ended task
TEMPS = [0.0, 0.7, 1.0]
SAMPLES = 2
GSM8K_N = 100                          # subset for tractability (cf. grid limit=200)
MAX_WORKERS = 8

JUDGE_TEMPLATE = (
    "[Instruction]\nAct as an impartial judge and rate the response's quality "
    "(helpfulness, relevance, accuracy, depth). After a short explanation, output "
    'the rating strictly as "[[n]]" with n from 1 to 10.\n\n'
    "[Question]\n{question}\n\n[Answer]\n{answer}")


def load_key():
    env = os.path.join(REPO_ROOT, ".env")
    if os.path.exists(env):
        for line in open(env, encoding="utf-8"):
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    k = os.environ.get("OPENROUTER_API_KEY")
    if not k:
        sys.exit("OPENROUTER_API_KEY not found")
    return k


HEADERS = {"Authorization": f"Bearer {load_key()}", "Content-Type": "application/json"}


def chat(model, prompt, temperature, max_tokens=1024, retries=4):
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "temperature": temperature, "max_tokens": max_tokens}
    last = None
    for a in range(retries):
        try:
            r = requests.post(BASE, headers=HEADERS, json=payload, timeout=150)
            if r.status_code == 200:
                c = r.json()["choices"][0]["message"].get("content")
                if c:
                    return c
                last = "empty"
            else:
                last = f"HTTP {r.status_code}"
        except Exception as e:  # noqa: BLE001
            last = str(e)[:80]
        time.sleep(2 * (a + 1))
    return f"__ERROR__ {last}"


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


def append(p, o):
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")


def load_gsm8k():
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    items = []
    for i, ex in enumerate(ds):
        if i >= GSM8K_N:
            break
        gold = ex["answer"].split("####")[-1].strip().replace(",", "")
        items.append({"qid": f"gsm8k-{i}", "prompt": ex["question"], "gold": gold})
    return items


def load_mtbench():
    raw = urllib.request.urlopen(MT_BENCH_URL, timeout=60).read().decode("utf-8")
    return [{"qid": f"mt-{json.loads(l)['question_id']}", "prompt": json.loads(l)["turns"][0], "gold": None}
            for l in raw.splitlines() if l.strip()]


def num_from(text):
    m = re.search(r"####\s*(-?[\d,]+)", text) or None
    if not m:
        nums = re.findall(r"-?\d[\d,]*", text)
        if not nums:
            return None
        return nums[-1].replace(",", "")
    return m.group(1).replace(",", "")


def parse_rating(text):
    m = re.search(r"\[\[\s*(\d+(?:\.\d+)?)\s*\]\]", text) or re.search(r"[Rr]ating:?\s*(\d+(?:\.\d+)?)", text)
    if not m:
        return None
    try:
        v = float(m.group(1)); return v if 1 <= v <= 10 else None
    except ValueError:
        return None


def generate():
    tasks = {"gsm8k": load_gsm8k(), "mtbench": load_mtbench()}
    done = {(r["model"], r["task"], r["qid"], r["temp"], r["sample"]) for r in read_jsonl(GEN_PATH)}
    todo = []
    for task, items in tasks.items():
        for it in items:
            for m in MODELS:
                for t in TEMPS:
                    for s in range(SAMPLES):
                        if (m, task, it["qid"], t, s) not in done:
                            todo.append((task, it, m, t, s))
    print(f"[gen] {len(done)} cached, {len(todo)} to generate", flush=True)
    import threading
    lock = threading.Lock()

    def work(x):
        task, it, m, t, s = x
        out = chat(m, it["prompt"], t)
        rec = {"model": m, "task": task, "qid": it["qid"], "temp": t, "sample": s,
               "gold": it["gold"], "prompt": it["prompt"], "response": out,
               "ok": not out.startswith("__ERROR__")}
        with lock:
            append(GEN_PATH, rec)
        return rec["ok"]
    if todo:
        with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            print(f"[gen] {sum(ex.map(work, todo))}/{len(todo)} ok", flush=True)


def score():
    rows = [r for r in read_jsonl(GEN_PATH) if r.get("ok")]
    # GSM8K: exact numeric match
    scored = defaultdict(dict)  # (model,task,temp,sample) -> per-item correctness list
    gsm = [r for r in rows if r["task"] == "gsm8k"]
    for r in gsm:
        pred = num_from(r["response"])
        ok = pred is not None and pred == r["gold"]
        scored[(r["model"], "gsm8k", r["temp"], r["sample"])].setdefault("vals", []).append(1.0 if ok else 0.0)
    # MT-Bench: judge each response once (fixed judge, temp 0)
    mt = [r for r in rows if r["task"] == "mtbench"]
    jcache_path = os.path.join(RESULTS_DIR, "judge.jsonl")
    jdone = {(j["model"], j["qid"], j["temp"], j["sample"]) for j in read_jsonl(jcache_path)}
    todo = [r for r in mt if (r["model"], r["qid"], r["temp"], r["sample"]) not in jdone]
    print(f"[judge] {len(jdone)} cached, {len(todo)} to judge", flush=True)
    import threading
    lock = threading.Lock()

    def jwork(r):
        out = chat(JUDGE, JUDGE_TEMPLATE.format(question=r["prompt"], answer=r["response"]), 0.0, 800)
        rec = {"model": r["model"], "qid": r["qid"], "temp": r["temp"], "sample": r["sample"],
               "rating": parse_rating(out)}
        with lock:
            append(jcache_path, rec)
        return rec["rating"] is not None
    if todo:
        with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            print(f"[judge] {sum(ex.map(jwork, todo))}/{len(todo)} rated", flush=True)
    for j in read_jsonl(jcache_path):
        if j["rating"] is not None:
            scored[(j["model"], "mtbench", j["temp"], j["sample"])].setdefault("vals", []).append(j["rating"])
    # cell score = mean over items
    cell = {k: sum(v["vals"]) / len(v["vals"]) for k, v in scored.items() if v.get("vals")}

    def eta2_temp(task):
        # one-way ANOVA: groups = temperatures, obs = (model,sample) cell means
        groups = defaultdict(list)
        for (m, tsk, t, s), val in cell.items():
            if tsk == task:
                groups[t].append(val * (100.0 if task == "gsm8k" else 1.0))
        allv = [v for g in groups.values() for v in g]
        if len(allv) < len(groups) + 1:
            return None
        gm = sum(allv) / len(allv)
        ssb = sum(len(g) * (sum(g) / len(g) - gm) ** 2 for g in groups.values())
        sst = sum((v - gm) ** 2 for v in allv)
        means = {t: round(sum(g) / len(g), 2) for t, g in sorted(groups.items())}
        spread = max(m for m in means.values()) - min(m for m in means.values())
        return {"eta2_pct": round(100 * ssb / sst, 2) if sst else 0.0,
                "temp_means": means, "spread": round(spread, 2), "n_cells": len(allv)}

    stats = {"models": MODELS, "temps": TEMPS, "samples": SAMPLES,
             "gsm8k": eta2_temp("gsm8k"), "mtbench": eta2_temp("mtbench")}
    json.dump(stats, open(STATS_PATH, "w"), indent=2)
    print(json.dumps(stats, indent=2), flush=True)
    print(f"[wrote] {STATS_PATH}", flush=True)


if __name__ == "__main__":
    generate()
    score()
