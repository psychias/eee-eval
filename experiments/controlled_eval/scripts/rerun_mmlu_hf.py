"""Rerun the factorial-grid MMLU cells on the hf backend (all 4 models, replacing the
mixed vLLM set of which 6 cells failed). Faithful to the original vLLM notebook
(Cell 3/6): standard log-likelihood `mmlu` task, format applied via system_instruction
+ apply_chat_template=True, num_fewshot in {0,5}, limit=200 (per subject), temp 0,
seed 42. MMLU is log-likelihood => backend-independent, so hf reproduces the vLLM
numbers and fills the failed cells; the per-benchmark ANOVA only needs MMLU internally
consistent, which an all-hf sweep guarantees.

One model per Colab session (each cell is ~8-35 min; 24 cells > one session). Set
MODELS to the session's model(s). Results print as `RESULT {json}` lines (captured
locally and appended to controlled_eval_results.jsonl). Same prefetch-then-offline
pattern; gated Mistral needs HF_TOKEN injected; Llama uses the ungated NousResearch
mirror but is recorded under its canonical meta-llama id."""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import subprocess
import sys
import json
from datetime import datetime, timezone

# (eval_id, canonical_id_for_record, batch_size) -- set to this session's model(s).
# (eval_id, canonical_id_for_record, batch_size). Run ONE model per Colab session
# (each cell is ~2-16 min; 24 cells exceed one session). The four grid models:
#   ("mistralai/Mistral-7B-Instruct-v0.3", "mistralai/Mistral-7B-Instruct-v0.3", 8)  # gated: set HF_TOKEN
#   ("Qwen/Qwen2.5-7B-Instruct",           "Qwen/Qwen2.5-7B-Instruct",           8)
#   ("Qwen/Qwen2.5-14B-Instruct",          "Qwen/Qwen2.5-14B-Instruct",          4)  # 14B -> batch 4
# Llama-3.1-8B is NOT re-scored here: apply_chat_template=True makes the score
# depend on the chat template, and the only ungated weights (NousResearch mirror)
# ship a different template (diverges up to ~13 pp), so Llama keeps its original
# vLLM grid values (backend-independence holds for the other three: hf vs vLLM
# agree within 0.5 pp, so the hf cells pool with the vLLM Llama cells unbiased).
MODELS = [
    ("Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-14B-Instruct", 4),
]
_TOKEN = os.environ.get("HF_TOKEN", "")  # only needed for gated Mistral

FORMAT_TEMPLATES = {
    "plain":    None,
    "instruct": "You are an expert problem solver. Answer each question carefully and directly.",
    "cot":      "You are an expert problem solver. Before answering, think step by step and show your reasoning. Then state the final answer.",
}
N_SHOTS = [0, 5]
LIMIT = 200
SEED = 42

print("[setup] installing lm-eval==0.4.11 ...", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "lm-eval==0.4.11", "accelerate"], check=True)

SNIPPET = """
import time
from huggingface_hub import snapshot_download
import datasets
REPOS = %r
def retry(fn, *a, **k):
    for i in range(15):
        try:
            return fn(*a, **k)
        except Exception as e:
            print("retry", getattr(fn, "__name__", "fn"), i, str(e)[:80], flush=True)
            time.sleep(20)
    return None
for repo in REPOS:
    print("SNAP_OK" if retry(snapshot_download, repo) else "SNAP_FAIL", repo, flush=True)
cfgs = [c for c in retry(datasets.get_dataset_config_names, "cais/mmlu")
        if c not in ("all", "auxiliary_train")]
for c in cfgs:
    retry(datasets.load_dataset, "cais/mmlu", c)
print("SNAP_OK mmlu", len(cfgs), "configs", flush=True)
""" % ([e for e, _, _ in MODELS],)
online_env = {**os.environ, "HF_HUB_OFFLINE": "0", "HF_DATASETS_OFFLINE": "0",
              "HF_TOKEN": _TOKEN, "HUGGING_FACE_HUB_TOKEN": _TOKEN}
print("[prefetch] caching models + cais/mmlu (online subprocess) ...", flush=True)
subprocess.run([sys.executable, "-c", SNIPPET], env=online_env)
print("[prefetch] done; running lm-eval OFFLINE", flush=True)

import gc

import torch
import lm_eval
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM


def extract(results):
    m = results["results"]["mmlu"]
    v = m.get("acc,none", m.get("acc"))
    return round(float(v) * 100.0, 4)


for eval_id, canon_id, bs in MODELS:
    lm = HFLM(pretrained=eval_id, dtype="bfloat16", batch_size=bs)
    for fmt, sys_inst in FORMAT_TEMPLATES.items():
        for n in N_SHOTS:
            rec = dict(model_id=canon_id, benchmark="mmlu", temperature=0.0,
                       prompt_format=fmt, n_shot=n, random_seed=SEED, score=None,
                       timestamp=datetime.now(timezone.utc).isoformat(),
                       eval_library_version=lm_eval.__version__, limit=LIMIT,
                       backend="hf")
            try:
                results = simple_evaluate(
                    model=lm, tasks=["mmlu"], num_fewshot=n,
                    gen_kwargs=f"seed={SEED}", limit=LIMIT,
                    system_instruction=sys_inst, apply_chat_template=True,
                    fewshot_as_multiturn=False, random_seed=SEED,
                    numpy_random_seed=SEED, torch_random_seed=SEED,
                    write_out=False, log_samples=False, verbosity="ERROR",
                )
                rec["score"] = extract(results)
                rec["status"] = "ok"
                print("RESULT " + json.dumps(rec), flush=True)
            except Exception as exc:  # noqa: BLE001
                rec["status"] = "error"
                rec["reason"] = str(exc)[:200]
                print("RESULT " + json.dumps(rec), flush=True)
                print(f"[cell-error] {canon_id} {fmt} n{n}: {str(exc)[:120]}", flush=True)
    del lm
    gc.collect()
    torch.cuda.empty_cache()

print("[done] session complete", flush=True)
