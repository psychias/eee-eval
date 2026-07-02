"""Unify + power the GPQA grid arm: full test set (drop limit=200), single hf
backend, and multiple seeds (more cells for the underpowered benchmark).

Mirrors the MMLU rerun (rerun_mmlu_hf.py): standard log-likelihood GPQA
(gpqa_main_zeroshot for 0-shot, gpqa_main_n_shot for 5-shot), format applied via
system_instruction + apply_chat_template, seeds {42,123,7}, limit=None. One model
per Colab session. GPQA (Idavidrein/gpqa) is a GATED dataset, so the online
prefetch authenticates with HF_TOKEN; the offline eval needs no token.
Llama-3.1 uses the NousResearch mirror (see rerun_mmlu_hf.py caveat)."""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import subprocess
import sys
import json
from datetime import datetime, timezone

MODELS = [("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct", 16)]
_TOKEN = os.environ.get("HF_TOKEN", "")  # gated GPQA dataset (and gated models)
FORMAT_TEMPLATES = {
    "plain": None,
    "instruct": "You are an expert problem solver. Answer each question carefully and directly.",
    "cot": "You are an expert problem solver. Before answering, think step by step and show your reasoning. Then state the final answer.",
}
N_SHOTS = [0, 5]
SEEDS = [42, 123, 7]

print("[setup] installing lm-eval==0.4.11 ...", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lm-eval==0.4.11", "accelerate"], check=True)

SNIPPET = """
import time
from huggingface_hub import snapshot_download
import datasets
REPOS = %r
def retry(fn,*a,**k):
    for i in range(15):
        try: return fn(*a,**k)
        except Exception as e:
            print("retry",getattr(fn,"__name__","fn"),i,str(e)[:80],flush=True); time.sleep(20)
    return None
for repo in REPOS:
    print("SNAP_OK" if retry(snapshot_download,repo) else "SNAP_FAIL",repo,flush=True)
# GPQA main config (gated)
retry(datasets.load_dataset,"Idavidrein/gpqa","gpqa_main")
print("SNAP_OK gpqa",flush=True)
""" % ([e for e, _, _ in MODELS],)
online_env = {**os.environ, "HF_HUB_OFFLINE": "0", "HF_DATASETS_OFFLINE": "0",
              "HF_TOKEN": _TOKEN, "HUGGING_FACE_HUB_TOKEN": _TOKEN}
print("[prefetch] caching model + Idavidrein/gpqa (online) ...", flush=True)
subprocess.run([sys.executable, "-c", SNIPPET], env=online_env)
print("[prefetch] done; running lm-eval OFFLINE", flush=True)

import gc
import torch
import lm_eval
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM


def extract(results):
    for k, m in results["results"].items():
        if k.startswith("gpqa"):
            v = m.get("acc,none", m.get("acc_norm,none", m.get("acc")))
            if v is not None:
                return round(float(v) * 100.0, 4)
    return None


for eval_id, canon_id, bs in MODELS:
    lm = HFLM(pretrained=eval_id, dtype="bfloat16", batch_size=bs)
    for fmt, sys_inst in FORMAT_TEMPLATES.items():
        for n in N_SHOTS:
            task = "gpqa_main_zeroshot" if n == 0 else "gpqa_main_n_shot"
            for seed in SEEDS:
                rec = dict(model_id=canon_id, benchmark="gpqa_main", task_name=task,
                           temperature=0.0, prompt_format=fmt, n_shot=n, random_seed=seed,
                           score=None, timestamp=datetime.now(timezone.utc).isoformat(),
                           eval_library_version=lm_eval.__version__, limit=None, backend="hf")
                try:
                    r = simple_evaluate(model=lm, tasks=[task], num_fewshot=n,
                                        gen_kwargs=f"seed={seed}", limit=None,
                                        system_instruction=sys_inst, apply_chat_template=True,
                                        fewshot_as_multiturn=False, random_seed=seed,
                                        numpy_random_seed=seed, torch_random_seed=seed,
                                        write_out=False, log_samples=False, verbosity="ERROR")
                    rec["score"] = extract(r); rec["status"] = "ok"
                    print("RESULT " + json.dumps(rec), flush=True)
                except Exception as exc:  # noqa: BLE001
                    rec["status"] = "error"; rec["reason"] = str(exc)[:200]
                    print("RESULT " + json.dumps(rec), flush=True)
                    print(f"[cell-error] {canon_id} {fmt} n{n} s{seed}: {str(exc)[:100]}", flush=True)
    del lm; gc.collect(); torch.cuda.empty_cache()
print("[done] session complete", flush=True)
