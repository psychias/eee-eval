"""Generalize the factorial grid: add three more benchmarks on a single common
backend (hf), full test set (no limit=200), to test whether the prompt-format
effect and the scoring-mode diagnostic (Sec. 5) hold beyond the original
benchmarks.

New benchmarks (all open, log-likelihood multiple-choice, fit the MMLU-style
format x n-shot design): ARC-Challenge, HellaSwag, WinoGrande. The diagnostic
predicts a small prompt-format effect on these (answer recovered by ranking), in
contrast to the large effect on the generation benchmarks GSM8K/BBH.

Format applied via system_instruction + apply_chat_template; seed 42; limit=None.
Runs ONLINE (these datasets are small and stable, unlike the flaky cais/mmlu, so
no offline-prefetch dance is needed); a retry wraps transient dataset fetches.
One model per Colab session. Mistral is gated: set HF_TOKEN in the environment
(injected for the run); Llama-3.1 uses the NousResearch mirror."""
import os
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
_TOKEN = os.environ.get("HF_TOKEN", "")
if _TOKEN:
    os.environ["HF_TOKEN"] = _TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = _TOKEN

import subprocess
import sys
import json
import time
from datetime import datetime, timezone

MODELS = [("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct", 16)]
BENCHES = [("arc_challenge", "ARC-Challenge"),
           ("hellaswag", "HellaSwag"),
           ("winogrande", "WinoGrande")]
FORMAT_TEMPLATES = {
    "plain": None,
    "instruct": "You are an expert problem solver. Answer each question carefully and directly.",
    "cot": "You are an expert problem solver. Before answering, think step by step and show your reasoning. Then state the final answer.",
}
N_SHOTS = [0, 5]
SEED = 42

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


def run_cell(lm, task, n, sys_inst):
    last = None
    for attempt in range(4):
        try:
            return simple_evaluate(model=lm, tasks=[task], num_fewshot=n,
                                   gen_kwargs=f"seed={SEED}", limit=None,
                                   system_instruction=sys_inst, apply_chat_template=True,
                                   fewshot_as_multiturn=False, random_seed=SEED,
                                   numpy_random_seed=SEED, torch_random_seed=SEED,
                                   write_out=False, log_samples=False, verbosity="ERROR")
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"  [retry {attempt}] {task} n{n}: {str(e)[:80]}", flush=True)
            time.sleep(20)
    raise last


for eval_id, canon_id, bs in MODELS:
    lm = HFLM(pretrained=eval_id, dtype="bfloat16", batch_size=bs)
    for task, label in BENCHES:
        for fmt, sys_inst in FORMAT_TEMPLATES.items():
            for n in N_SHOTS:
                rec = dict(model_id=canon_id, benchmark=label, task_name=task,
                           temperature=0.0, prompt_format=fmt, n_shot=n, random_seed=SEED,
                           score=None, timestamp=datetime.now(timezone.utc).isoformat(),
                           eval_library_version=lm_eval.__version__, limit=None, backend="hf")
                try:
                    r = run_cell(lm, task, n, sys_inst)
                    rec["score"] = extract(r, task); rec["status"] = "ok"
                    print("RESULT " + json.dumps(rec), flush=True)
                except Exception as exc:  # noqa: BLE001
                    rec["status"] = "error"; rec["reason"] = str(exc)[:200]
                    print("RESULT " + json.dumps(rec), flush=True)
                    print(f"[cell-error] {canon_id} {label} {fmt} n{n}: {str(exc)[:100]}", flush=True)
    del lm; gc.collect(); torch.cuda.empty_cache()
print("[done] session complete", flush=True)
