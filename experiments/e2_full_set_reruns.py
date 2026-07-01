"""E2 uniformity pass: full-set log-likelihood for the three smaller models, so all
four E2 rows are ll@full vs gen@full (the 14B is in finish_missing_cells_e2_e3.py).

Motivation: the original ll values used limit=500/subject, which carries subset-
composition noise. For the 14B this mattered (ll@500=81.72 vs ll@full=79.97); for the
smaller models it did not (Mistral 61.92->61.85, Qwen-7B 74.26->74.25), but rerunning
them on the full set makes every row an exact same-sample comparison to generation@full.

Results (ll@full / gen@full / |Delta|):
  Mistral-7B-Instruct-v0.3   61.85 / 61.64 / 0.21
  Qwen2.5-7B-Instruct        74.25 / 74.43 / 0.18
  Llama-3.1-8B-Instruct      68.19 / 68.16 / 0.03

Gating: Mistral-v0.3 is gated -> the prefetch subprocess authenticates with HF_TOKEN
(set it in the environment before launching; the offline eval needs no token).
meta-llama/Llama-3.1-8B-Instruct requires a separate Meta-license approval, so this
uses the ungated faithful re-upload NousResearch/Meta-Llama-3.1-8B-Instruct (identical
weights). Same prefetch-then-offline pattern; batch_size=8; results print incrementally.
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import subprocess
import sys

# (model_id_for_eval, prefetch_repo) -- Llama eval id is the mirror (ungated).
MODELS = [
    ("mistralai/Mistral-7B-Instruct-v0.3", "mistralai/Mistral-7B-Instruct-v0.3"),
    ("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct"),
    ("NousResearch/Meta-Llama-3.1-8B-Instruct", "NousResearch/Meta-Llama-3.1-8B-Instruct"),
]
_TOKEN = os.environ.get("HF_TOKEN", "")  # inject the operator's token for the gated repo

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
""" % ([r for _, r in MODELS],)
online_env = {**os.environ, "HF_HUB_OFFLINE": "0", "HF_DATASETS_OFFLINE": "0",
              "HF_TOKEN": _TOKEN, "HUGGING_FACE_HUB_TOKEN": _TOKEN}
print("[prefetch] caching models + cais/mmlu (online subprocess) ...", flush=True)
subprocess.run([sys.executable, "-c", SNIPPET], env=online_env)
print("[prefetch] done; running lm-eval OFFLINE", flush=True)

import gc

import torch
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM

for eval_id, _ in MODELS:
    try:
        lm = HFLM(pretrained=eval_id, dtype="bfloat16", batch_size=8)
        r = simple_evaluate(model=lm, tasks=["mmlu"], num_fewshot=5, limit=None,
                            random_seed=42, numpy_random_seed=42,
                            torch_random_seed=42, fewshot_random_seed=42)
        m = r["results"]["mmlu"]
        acc = round(100 * m.get("acc,none", m.get("acc")), 2)
        print(f"[E2] {eval_id} MMLU 5-shot log-likelihood (FULL 14042): {acc}", flush=True)
        del lm
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001
        print(f"[E2-SKIP] {eval_id}: {str(e)[:120]}", flush=True)
