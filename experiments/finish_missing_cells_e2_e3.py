"""
Finish the Appendix-G tables: rerun the two missing cells with lm-eval 0.4.11 (hf).

  E3 (harness-version): Qwen2.5-7B-Instruct, GSM8K full, 5-shot, seed 123, 0.4.11
     -> the previously omitted 0.4.11 / 7B / seed-123 cell. Result: 78.32.
  E2 (scoring-mode):    Qwen2.5-14B-Instruct, MMLU, 5-shot, log-likelihood, full set
     -> the missing 14B log-likelihood cell (table previously had generation only,
        79.79). Result: 79.97  (|Delta| vs generation@full = 0.18 pp). The 14B is
        scored on the FULL set under both modes: its log-likelihood score is
        sample-sensitive (limit=100->80.32, limit=500->81.72, full->79.97), so only
        the full-set ll-vs-gen comparison is free of a sampling confound.

Backend = hf: vLLM's current wheel is CUDA-13 vs Colab's CUDA-12 image; E3 already
uses the hf backend, and MMLU log-likelihood is a deterministic multiple-choice
ranking that is backend-independent.

Two infrastructure issues, both handled by the prefetch-then-offline pattern below:
  1. Memory: the 14B model + MMLU log-likelihood does not fit a 40 GB GPU at the
     batch_size=8 used for the smaller models. The OOM is driven by batch size, NOT
     by the limit (peak memory = batch x seq-len, independent of #examples), so the
     14B cell simply drops to batch_size=4 and keeps limit=500 -- directly comparable
     to the other three E2 cells, just ~2x longer (~20 min for ~49k requests).
  2. HF's cais/mmlu tree API intermittently returns 504s. A successful download is
     not enough on its own, because datasets in ONLINE mode re-validates each subject
     against that same flaky API. Fix: in an ONLINE subprocess, snapshot the models
     and *build* every dataset config (load_dataset, with retries); then run lm-eval
     fully OFFLINE so nothing touches the HF API during the eval.

Run on a Colab A100. Prints each result as it completes.
"""
import os
# Parent runs OFFLINE: transformers + datasets read only from the local cache that
# the online subprocess below populates, so the flaky cais/mmlu API is never hit
# and the (already-cached) models load without a network round-trip.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import subprocess
import sys

print("[setup] installing lm-eval==0.4.11 ...", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "lm-eval==0.4.11", "accelerate"], check=True)

# Prefetch models + datasets in an ONLINE subprocess (retries past the 504s).
# Datasets must be *built* (load_dataset), not just downloaded: snapshot_download
# only caches raw files, whereas datasets in offline mode needs a built arrow cache.
SNIPPET = """
import time
from huggingface_hub import snapshot_download
import datasets
def retry(fn, *a, **k):
    for i in range(15):
        try:
            return fn(*a, **k)
        except Exception as e:
            print("retry", getattr(fn, "__name__", "fn"), i, str(e)[:70], flush=True)
            time.sleep(20)
    raise SystemExit("prefetch failed")
retry(snapshot_download, "Qwen/Qwen2.5-7B-Instruct")
retry(snapshot_download, "Qwen/Qwen2.5-14B-Instruct")
print("SNAP_OK models", flush=True)
retry(datasets.load_dataset, "gsm8k", "main")
cfgs = [c for c in retry(datasets.get_dataset_config_names, "cais/mmlu")
        if c not in ("all", "auxiliary_train")]
for c in cfgs:
    retry(datasets.load_dataset, "cais/mmlu", c)
print("SNAP_OK datasets", len(cfgs), "mmlu configs", flush=True)
"""
online_env = {**os.environ, "HF_HUB_OFFLINE": "0", "HF_DATASETS_OFFLINE": "0"}
print("[prefetch] caching models + gsm8k + cais/mmlu (online subprocess) ...", flush=True)
subprocess.run([sys.executable, "-c", SNIPPET], env=online_env, check=True)
print("[prefetch] done; running lm-eval OFFLINE", flush=True)

import gc

import torch
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM

# -- E3: Qwen2.5-7B-Instruct, GSM8K full, 5-shot, lm-eval 0.4.11, seed 123 --
lm = HFLM(pretrained="Qwen/Qwen2.5-7B-Instruct", dtype="bfloat16", batch_size=48)
r = simple_evaluate(model=lm, tasks=["gsm8k"], num_fewshot=5, limit=None,
                    random_seed=123, numpy_random_seed=123,
                    torch_random_seed=123, fewshot_random_seed=123)
m = r["results"]["gsm8k"]
e3 = round(100 * m.get("exact_match,strict-match", m.get("exact_match,flexible-extract")), 2)
print(f"[E3] Qwen2.5-7B-Instruct GSM8K 5-shot, 0.4.11, seed 123 (full 1319): {e3}", flush=True)
del lm
gc.collect()
torch.cuda.empty_cache()

# -- E2: Qwen2.5-14B-Instruct, MMLU 5-shot, log-likelihood, full test set --
# batch_size=4 so the 14B bf16 model fits a 40 GB GPU; limit=None (full 14042) for an
# exact same-sample comparison to generation@full (the ll score is sample-sensitive).
lm = HFLM(pretrained="Qwen/Qwen2.5-14B-Instruct", dtype="bfloat16", batch_size=4)
r = simple_evaluate(model=lm, tasks=["mmlu"], num_fewshot=5, limit=None,
                    random_seed=42, numpy_random_seed=42,
                    torch_random_seed=42, fewshot_random_seed=42)
m = r["results"]["mmlu"]
e2 = round(100 * m.get("acc,none", m.get("acc")), 2)
print(f"[E2] Qwen2.5-14B-Instruct MMLU 5-shot log-likelihood (full 14042): {e2}", flush=True)
del lm
gc.collect()
torch.cuda.empty_cache()

print(f"\nRESULT  E3_7B_gsm8k_0411_seed123={e3}  E2_14B_mmlu_loglikelihood_full={e2}", flush=True)
