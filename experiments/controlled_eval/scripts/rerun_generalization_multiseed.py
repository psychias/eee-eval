"""Power up the log-likelihood generalization benchmarks: re-run ARC-Challenge,
WinoGrande, and OpenBookQA at TWO additional seeds (123, 7) beyond the original
seed 42, so each benchmark's prompt-format ANOVA has n=12 per format group
(4 models x 3 seeds) instead of the single-seed n=4. Addresses the reviewer's
"log-likelihood benchmarks are underpowered" concern.

hf backend, full test set, 5-shot, three formats. One model per Colab session."""
import os
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
_TOKEN = os.environ.get("HF_TOKEN", "")
if _TOKEN:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = _TOKEN

import subprocess
import sys
import json
import time
from datetime import datetime, timezone

MODELS = [("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct", 16)]
BENCHES = [("arc_challenge", "ARC-Challenge"),
           ("winogrande", "WinoGrande"),
           ("openbookqa", "OpenBookQA")]
FORMAT_TEMPLATES = {
    "plain": None,
    "instruct": "You are an expert problem solver. Answer each question carefully and directly.",
    "cot": "You are an expert problem solver. Before answering, think step by step and show your reasoning. Then state the final answer.",
}
SEEDS = [123, 7]
N_SHOT = 5

print("[setup] installing lm-eval==0.4.11 ...", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lm-eval==0.4.11", "accelerate"], check=True)

import gc
import torch
import lm_eval
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM


def extract(results, task):
    m = results["results"].get(task, {})
    v = m.get("acc_norm,none", m.get("acc,none", m.get("acc")))
    return round(float(v) * 100.0, 4) if v is not None else None


def run_cell(lm, task, sys_inst, seed):
    last = None
    for attempt in range(4):
        try:
            return simple_evaluate(model=lm, tasks=[task], num_fewshot=N_SHOT,
                                   gen_kwargs=f"seed={seed}", limit=None,
                                   system_instruction=sys_inst, apply_chat_template=True,
                                   fewshot_as_multiturn=False, random_seed=seed,
                                   numpy_random_seed=seed, torch_random_seed=seed,
                                   fewshot_random_seed=seed,
                                   write_out=False, log_samples=False, verbosity="ERROR")
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"  [retry {attempt}] {task} seed{seed}: {str(e)[:80]}", flush=True)
            time.sleep(20)
    raise last


for eval_id, canon_id, bs in MODELS:
    lm = HFLM(pretrained=eval_id, dtype="bfloat16", batch_size=bs)
    for task, label in BENCHES:
        for fmt, sys_inst in FORMAT_TEMPLATES.items():
            for seed in SEEDS:
                rec = dict(model_id=canon_id, benchmark=label, task_name=task,
                           temperature=0.0, prompt_format=fmt, n_shot=N_SHOT, random_seed=seed,
                           score=None, timestamp=datetime.now(timezone.utc).isoformat(),
                           eval_library_version=lm_eval.__version__, limit=None, backend="hf")
                try:
                    r = run_cell(lm, task, sys_inst, seed)
                    rec["score"] = extract(r, task); rec["status"] = "ok"
                    print("RESULT " + json.dumps(rec), flush=True)
                except Exception as exc:  # noqa: BLE001
                    rec["status"] = "error"; rec["reason"] = str(exc)[:200]
                    print("RESULT " + json.dumps(rec), flush=True)
                    print(f"[cell-error] {canon_id} {label} {fmt} seed{seed}: {str(exc)[:100]}", flush=True)
    del lm; gc.collect(); torch.cuda.empty_cache()
print("[done] session complete", flush=True)
