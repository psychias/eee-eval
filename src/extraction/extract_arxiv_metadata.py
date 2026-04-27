"""Extract evaluation metadata from arxiv HTML papers.

Two approaches:
  1. Naive: regex/keyword search in results/experiments sections
  2. Advanced: Claude Haiku 4.5 via OpenRouter for intelligent extraction

Output:
  data/arxiv_extraction/naive/<model_name>/<uuid>.json
  data/arxiv_extraction/llm/<model_name>/<uuid>.json
  data/arxiv_extraction/summary.json
  data/arxiv_extraction/summary.csv

Usage:
    python scripts/extract_arxiv_metadata.py
"""
import csv
import json
import os
import re
import time
import uuid as _uuid
from datetime import date as _date
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
AUDIT_CSV = ROOT / "data" / "pwc_audit_sample.csv"
OUT_DIR = ROOT / "data" / "arxiv_extraction"

# ── OpenRouter config ──────────────────────────────────────────────────
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
if not OPENROUTER_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY env var is not set. "
        "See .env.example for required configuration."
    )
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HAIKU_MODEL = "anthropic/claude-haiku-4.5"

# ── EvalSpec fields we want to extract ─────────────────────────────────
FIELDS = ["n_shot", "temperature", "prompt_template", "eval_harness",
          "scoring_mode", "num_samples", "decoding_strategy", "seed"]


# ═══════════════════════════════════════════════════════════════════════
# Step 1: Load audit sample & fetch HTML
# ═══════════════════════════════════════════════════════════════════════

def load_audit_sample():
    """Return list of dicts from audit CSV, deduplicated by arxiv_id."""
    rows = []
    seen = set()
    with open(AUDIT_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            aid = r.get("arxiv_id", "").strip()
            if aid and aid not in seen:
                seen.add(aid)
                rows.append(r)
    return rows


def fetch_html(arxiv_id: str) -> str | None:
    """Fetch arxiv HTML version. Returns HTML string or None."""
    base_id = re.sub(r"v\d+$", "", arxiv_id)
    url = f"https://arxiv.org/html/{base_id}"
    try:
        resp = requests.get(url, timeout=30,
                            headers={"User-Agent": "EEE-Eval-Research/1.0"})
        if resp.status_code == 200 and "<html" in resp.text[:500].lower():
            return resp.text
        print(f"  [WARN] {arxiv_id}: HTTP {resp.status_code} or not HTML")
    except Exception as e:
        print(f"  [ERR]  {arxiv_id}: {e}")
    return None


def extract_sections(html: str) -> dict[str, str]:
    """Extract key sections from arxiv HTML paper."""
    soup = BeautifulSoup(html, "html.parser")
    body = soup.get_text(separator="\n", strip=True)

    sections = {}
    headings = soup.find_all(re.compile(r"^h[1-4]$", re.I))
    for h in headings:
        title = h.get_text(strip=True).lower()
        parts = []
        for sib in h.find_next_siblings():
            if sib.name and re.match(r"^h[1-4]$", sib.name, re.I):
                break
            parts.append(sib.get_text(separator="\n", strip=True))
        sections[title] = "\n".join(parts)

    relevant_keys = [
        k for k in sections
        if any(kw in k for kw in ["experiment", "result", "evaluat",
                                   "setup", "setting", "method",
                                   "implementation", "detail",
                                   "appendix", "hyperparameter"])
    ]
    relevant_text = "\n\n".join(
        f"=== {k} ===\n{sections[k]}" for k in relevant_keys
    )
    return {"full": body, "relevant": relevant_text, "sections": sections}


# ═══════════════════════════════════════════════════════════════════════
# Approach 1: Naive keyword/regex extraction
# ═══════════════════════════════════════════════════════════════════════

NSHOT_PAT = re.compile(
    r"(\d+)[- ]?shot|few[- ]?shot|zero[- ]?shot|(\d+)[- ]?example", re.I)
TEMP_PAT = re.compile(
    r"temperature\s*(?:of|=|:)?\s*([0-9]*\.?[0-9]+)", re.I)
HARNESS_PAT = re.compile(
    r"(lm[- _]eval(?:uation)?[- _]harness|eleuther|helm|big-?bench"
    r"|open[- ]?llm[- ]?leaderboard|eval-?plus|human-?eval"
    r"|simple-?evals|alpaca-?eval|mt-?bench)", re.I)
PROMPT_PAT = re.compile(
    r"(chain[- ]?of[- ]?thought|cot|zero[- ]?shot[- ]?cot"
    r"|few[- ]?shot|direct|standard prompt|instruction"
    r"|system prompt|prompt template)", re.I)
SCORING_PAT = re.compile(
    r"(log[- ]?likelihood|multiple[- ]?choice|exact[- ]?match"
    r"|pass@k|greedy|beam|sampling|majority[- ]?voting"
    r"|self[- ]?consistency|maj@)", re.I)
SEED_PAT = re.compile(r"(?:random\s+)?seed\s*(?:of|=|:)?\s*(\d+)", re.I)
SAMPLES_PAT = re.compile(
    r"(?:k\s*=\s*|n\s*=\s*|samples?\s*(?:=|:)\s*)(\d+)", re.I)


def naive_extract(text: str) -> dict:
    """Keyword/regex extraction from paper text."""
    result = {f: None for f in FIELDS}

    shots = NSHOT_PAT.findall(text)
    if shots:
        values = set()
        for m in shots:
            if m[0]:
                values.add(int(m[0]))
            elif m[1]:
                values.add(int(m[1]))
        if "zero" in text.lower():
            values.add(0)
        result["n_shot"] = sorted(values) if values else None

    temps = TEMP_PAT.findall(text)
    if temps:
        result["temperature"] = sorted(set(float(t) for t in temps))

    harnesses = HARNESS_PAT.findall(text)
    if harnesses:
        result["eval_harness"] = list(set(h.lower() for h in harnesses))

    prompts = PROMPT_PAT.findall(text)
    if prompts:
        result["prompt_template"] = list(set(p.lower() for p in prompts))

    scoring = SCORING_PAT.findall(text)
    if scoring:
        result["scoring_mode"] = list(set(s.lower() for s in scoring))

    seeds = SEED_PAT.findall(text)
    if seeds:
        result["seed"] = sorted(set(int(s) for s in seeds))

    samples = SAMPLES_PAT.findall(text)
    if samples:
        result["num_samples"] = sorted(set(int(s) for s in samples))[:5]

    if re.search(r"greedy|temperature\s*(?:=|of)\s*0", text, re.I):
        result["decoding_strategy"] = "greedy"
    elif re.search(r"beam\s*search|num_beams", text, re.I):
        result["decoding_strategy"] = "beam_search"
    elif re.search(r"nucleus|top[- _]?p", text, re.I):
        result["decoding_strategy"] = "nucleus"
    elif re.search(r"top[- _]?k", text, re.I):
        result["decoding_strategy"] = "top_k"

    return result


# ═══════════════════════════════════════════════════════════════════════
# Approach 2: Claude Haiku 4.5 via OpenRouter
# ═══════════════════════════════════════════════════════════════════════

EXTRACTION_PROMPT = """\
You are an expert at reading ML papers. Extract evaluation configuration metadata from the paper text below.

Return a JSON object with these fields (use null if not found):
- n_shot: integer or list of integers (number of few-shot examples)
- temperature: float or list of floats (generation temperature)
- prompt_template: string description (e.g., "chain-of-thought", "direct", "instruction-tuned format")
- eval_harness: string (evaluation framework used, e.g., "lm-evaluation-harness", "HELM", "EvalPlus")
- scoring_mode: string (e.g., "log-likelihood", "exact-match", "pass@k", "majority-voting")
- num_samples: integer (number of samples/generations per problem)
- decoding_strategy: string (e.g., "greedy", "nucleus sampling", "beam search")
- seed: integer or list (random seeds used)

Focus on the EVALUATION SETUP, not training. Look for:
- Explicit mentions of n-shot, few-shot, zero-shot
- Temperature settings for generation
- Which evaluation harness/framework was used
- How answers were scored/evaluated
- Decoding parameters (greedy, sampling, beam search)
- Number of samples or generations per question

IMPORTANT: Only extract what is EXPLICITLY stated. Do not infer or guess.

Return ONLY valid JSON, no other text.

Paper text:
{text}"""


def llm_extract(text: str, arxiv_id: str) -> dict | None:
    """Use Claude Haiku 4.5 to extract eval metadata."""
    try:
        resp = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": HAIKU_MODEL,
                "messages": [
                    {"role": "user",
                     "content": EXTRACTION_PROMPT.format(text=text)}
                ],
                "temperature": 0.0,
            },
            timeout=120,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        content = re.sub(r"```json\s*", "", content)
        content = re.sub(r"```\s*", "", content)
        # Extract first JSON object using brace matching
        start = content.find("{")
        if start == -1:
            return None
        depth = 0
        for i, ch in enumerate(content[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(content[start:i+1])
        return json.loads(content.strip())
    except Exception as e:
        print(f"  [LLM ERR] {arxiv_id}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def make_record(row, extracted, approach):
    """Build an EEE-schema JSON record."""
    today = _date.today().isoformat()
    model_name = row["model_name"]
    safe_name = re.sub(r"[^\w\-.]", "_", model_name)[:80]
    arxiv_id = row["arxiv_id"]
    benchmark = row["benchmark"]
    score = row.get("score", "")

    gen_config = {}
    if extracted:
        for f in FIELDS:
            v = extracted.get(f)
            if v is not None:
                gen_config[f] = v

    src_label = f"arxiv_html_{approach}"
    return {
        "schema_version": "0.2.1",
        "evaluation_id": f"{src_label}/{safe_name}/{arxiv_id}",
        "evaluation_timestamp": today,
        "retrieved_timestamp": today,
        "source_metadata": {
            "source_name": src_label,
            "source_type": "paper_html",
            "source_organization_name": "arXiv",
            "source_organization_url": "https://arxiv.org",
            "evaluator_relationship": "self_reported",
            "extraction_approach": approach,
            "additional_details": {
                "paper_url": row.get("paper_url", ""),
                "arxiv_id": arxiv_id,
            },
        },
        "eval_library": {
            "name": gen_config.get("eval_harness", "unknown"),
            "version": "unknown",
        },
        "model_info": {
            "name": model_name,
            "id": safe_name,
            "developer": "unknown",
        },
        "evaluation_results": [{
            "evaluation_name": benchmark,
            "evaluation_timestamp": today,
            "source_data": {
                "dataset_name": benchmark,
                "source_type": "url",
                "url": [row.get("paper_url", "")],
            },
            "metric_config": {
                "metric_name": "Accuracy",
                "lower_is_better": False,
                "score_type": "continuous",
                "min_score": 0.0,
                "max_score": 100.0,
            },
            "score_details": {
                "score": float(score) if score else None,
            },
            "generation_config": {
                "generation_args": gen_config,
                "additional_details": {"source": src_label},
            },
        }],
    }


def save_record(rec, folder, model_name):
    safe = re.sub(r"[^\w\-.]", "_", model_name)[:80]
    d = OUT_DIR / folder / safe
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{_uuid.uuid4()}.json", "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    papers = load_audit_sample()
    print(f"Loaded {len(papers)} unique papers from audit sample")

    # Create output dirs
    for d in ["naive", "llm"]:
        (OUT_DIR / d).mkdir(parents=True, exist_ok=True)

    results = []

    for i, row in enumerate(papers):
        arxiv_id = row["arxiv_id"]
        print(f"\n[{i+1}/{len(papers)}] {arxiv_id} -- {row['model_name']}")

        html = fetch_html(arxiv_id)
        if not html:
            results.append({
                "arxiv_id": arxiv_id, "model_name": row["model_name"],
                "benchmark": row["benchmark"], "available": False,
                "naive": None, "llm": None,
                "pwc_original": {
                    "n_shot": row.get("n_shot", ""),
                    "prompt_template": row.get("prompt_template", ""),
                    "temperature": row.get("temperature_gen_config", ""),
                    "eval_harness": row.get("eval_harness", ""),
                },
            })
            time.sleep(1)
            continue

        sects = extract_sections(html)
        text = sects["relevant"] or sects["full"]
        print(f"  Text length: {len(text)} chars ({len(sects['relevant'])} relevant)")

        # Approach 1: Naive
        naive = naive_extract(text)
        print(f"  Naive: {json.dumps({k: v for k, v in naive.items() if v}, default=str)}")

        # Approach 2: LLM
        llm_result = llm_extract(text, arxiv_id)
        if llm_result:
            print(f"  LLM:   {json.dumps({k: v for k, v in llm_result.items() if v}, default=str)}")

        # Save per-model JSONs
        save_record(make_record(row, naive, "naive"), "naive", row["model_name"])
        save_record(make_record(row, llm_result, "llm"), "llm", row["model_name"])

        results.append({
            "arxiv_id": arxiv_id, "model_name": row["model_name"],
            "benchmark": row["benchmark"], "available": True,
            "naive": naive, "llm": llm_result,
            "pwc_original": {
                "n_shot": row.get("n_shot", ""),
                "prompt_template": row.get("prompt_template", ""),
                "temperature": row.get("temperature_gen_config", ""),
                "eval_harness": row.get("eval_harness", ""),
            },
        })
        time.sleep(2)

    # ── Summary ────────────────────────────────────────────────────────
    today = _date.today().isoformat()
    html_avail = sum(1 for r in results if r["available"])

    def _count(approach):
        return {
            field: sum(1 for r in results
                       if r[approach] and r[approach].get(field) is not None)
            for field in FIELDS
        }

    summary = {
        "generated": today,
        "total_papers": len(results),
        "html_available": html_avail,
        "naive": _count("naive"),
        "llm": _count("llm"),
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Summary CSV
    with open(OUT_DIR / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arxiv_id", "model_name", "benchmark",
                     "html_available",
                     *[f"naive_{f}" for f in FIELDS],
                     *[f"llm_{f}" for f in FIELDS],
                     "pwc_n_shot", "pwc_prompt_template",
                     "pwc_temperature", "pwc_eval_harness"])
        for r in results:
            row_out = [r["arxiv_id"], r["model_name"], r["benchmark"],
                       r["available"]]
            for field in FIELDS:
                v = r["naive"].get(field) if r["naive"] else None
                row_out.append(json.dumps(v, default=str) if v is not None else "")
            for field in FIELDS:
                v = r["llm"].get(field) if r["llm"] else None
                row_out.append(json.dumps(v, default=str) if v is not None else "")
            for field in ["n_shot", "prompt_template", "temperature_gen_config",
                          "eval_harness"]:
                row_out.append(r.get("pwc_original", {}).get(field, ""))
            w.writerow(row_out)

    # Console
    print("\n" + "=" * 70)
    print("EXTRACTION COVERAGE")
    print("=" * 70)
    print(f"Papers: {len(results)}  |  HTML available: {html_avail}")
    print(f"\n{'Field':20s}  {'Naive':>6s}  {'LLM':>6s}")
    print("-" * 38)
    for field in FIELDS:
        print(f"{field:20s}  {summary['naive'][field]:6d}  {summary['llm'][field]:6d}")
    print(f"\nOutput: {OUT_DIR}")
    print(f"  naive/ -> per-model JSONs (regex extraction)")
    print(f"  llm/   -> per-model JSONs (Haiku extraction)")
    print(f"  summary.json, summary.csv")


if __name__ == "__main__":
    main()
