"""Powered, full-set, single-backend (hf) GPQA rerun -- the GPQA half of the
"beyond GSM8K" fix. GPQA (Idavidrein/gpqa) is small (448 items) so full-set x
multi-seed is cheap; the only obstacle was datasets 3.x dropping loading-script
support (GPQA ships as a loading script). We pin datasets<3, which still runs the
script, and load with an access-granted token. Seeds 42/123/7, three prompt
formats, n=12/format. One model per Colab session; set HF_TOKEN (with GPQA access).
"""
import os
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
_TOKEN = os.environ.get("HF_TOKEN", "")
if _TOKEN:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = _TOKEN

import subprocess
import sys
import json
import time
from datetime import datetime, timezone

MODELS = [("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct", 16)]
FORMAT_TEMPLATES = {
    "plain": None,
    "instruct": "You are an expert problem solver. Answer each question carefully and directly.",
    "cot": "You are an expert problem solver. Before answering, think step by step and show your reasoning. Then state the final answer.",
}
# GPQA is deterministic under fixed few-shot (seed changes nothing), so power the
# format ANOVA across MODELS: single seed, 8 diverse models -> n=8 per format.
SEEDS = [42]
N_SHOT = 5
TASK = "gpqa_main_n_shot"   # lm-eval GPQA main task (multiple-choice log-likelihood), respects num_fewshot

print("[setup] installing lm-eval==0.4.11 + datasets<3 ...", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lm-eval==0.4.11", "accelerate"], check=True)
# datasets 3.x removed loading-script support that GPQA needs; pin back to 2.x.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets==2.21.0"], check=True)

import datasets
print(f"[setup] datasets {datasets.__version__}", flush=True)
# The gated GPQA loading script's internal file download does not pick up the env
# token; an explicit login authenticates the whole huggingface_hub session so the
# script's downloads carry the (access-granted) token.
if _TOKEN:
    from huggingface_hub import login
    login(token=_TOKEN, add_to_git_credential=False)
    print("[setup] hf login done", flush=True)
# sanity: confirm GPQA is now loadable with the granted token
try:
    datasets.load_dataset("Idavidrein/gpqa", "gpqa_main", trust_remote_code=True, token=_TOKEN or None)
    print("[gpqa] dataset loads OK", flush=True)
except Exception as e:  # noqa: BLE001
    print(f"[gpqa] STILL FAILING: {str(e)[:160]}", flush=True)

import gc
import torch
import lm_eval
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM


def extract(results, task):
    m = results["results"].get(task, {})
    for key in ("acc_norm,none", "acc,none", "exact_match,none"):
        if key in m:
            return round(float(m[key]) * 100.0, 4)
    return None


def run_cell(lm, sys_inst, seed):
    last = None
    for attempt in range(3):
        try:
            return simple_evaluate(model=lm, tasks=[TASK], num_fewshot=N_SHOT,
                                   gen_kwargs=f"seed={seed}", limit=None,
                                   system_instruction=sys_inst, apply_chat_template=True,
                                   fewshot_as_multiturn=False, random_seed=seed,
                                   numpy_random_seed=seed, torch_random_seed=seed,
                                   fewshot_random_seed=seed,
                                   write_out=False, log_samples=False, verbosity="ERROR")
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"  [retry {attempt}] seed{seed}: {str(e)[:90]}", flush=True)
            time.sleep(15)
    raise last


for eval_id, canon_id, bs in MODELS:
    lm = HFLM(pretrained=eval_id, dtype="bfloat16", batch_size="auto",
              max_batch_size=bs, trust_remote_code=True)
    for fmt, sys_inst in FORMAT_TEMPLATES.items():
        for seed in SEEDS:
            rec = dict(model_id=canon_id, benchmark="GPQA", task_name=TASK,
                       temperature=0.0, prompt_format=fmt, n_shot=N_SHOT, random_seed=seed,
                       score=None, timestamp=datetime.now(timezone.utc).isoformat(),
                       eval_library_version=lm_eval.__version__, limit=None, backend="hf")
            try:
                r = run_cell(lm, sys_inst, seed)
                rec["score"] = extract(r, TASK); rec["status"] = "ok"
                print("RESULT " + json.dumps(rec), flush=True)
            except Exception as exc:  # noqa: BLE001
                rec["status"] = "error"; rec["reason"] = str(exc)[:200]
                print("RESULT " + json.dumps(rec), flush=True)
                print(f"[cell-error] {canon_id} {fmt} seed{seed}: {str(exc)[:100]}", flush=True)
    del lm; gc.collect(); torch.cuda.empty_cache()
print("[done] session complete", flush=True)
