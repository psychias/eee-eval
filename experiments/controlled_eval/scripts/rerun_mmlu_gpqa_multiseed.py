"""Powered, full-set, single-backend (hf) reruns on MMLU and GPQA so the factorial
grid's format claims are as strong there as on GSM8K. Addresses "the strongest
conclusions are limited to GSM8K": full MMLU test set (no limit=200) and full GPQA,
three prompt formats, seeds 123 and 7 (added to the existing seed-42 grid cells to
give n=12 per format), all on the hf backend.

MMLU uses a prefetch-then-offline dance: cais/mmlu intermittently 504s under load,
so we snapshot every config online first, then run lm-eval with HF_HUB_OFFLINE=1.
GPQA (Idavidrein/gpqa) is gated; we attempt it online with the token and report
cleanly if access is denied rather than aborting the MMLU run.

One model per Colab session (swap MODELS). Mistral gated -> set HF_TOKEN."""
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
FORMAT_TEMPLATES = {
    "plain": None,
    "instruct": "You are an expert problem solver. Answer each question carefully and directly.",
    "cot": "You are an expert problem solver. Before answering, think step by step and show your reasoning. Then state the final answer.",
}
SEEDS = [123, 7]
N_SHOT = 5

print("[setup] installing lm-eval==0.4.11 ...", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lm-eval==0.4.11", "accelerate"], check=True)


def prefetch_mmlu():
    """Snapshot the MMLU model-agnostic dataset online (retry) so the eval can run
    fully offline afterwards, dodging the cais/mmlu 504s."""
    import datasets
    for attempt in range(5):
        try:
            datasets.load_dataset("cais/mmlu", "all", trust_remote_code=True)
            print("[prefetch] cais/mmlu 'all' cached", flush=True)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"[prefetch retry {attempt}] {str(e)[:90]}", flush=True)
            time.sleep(30)
    print("[prefetch] FAILED to cache MMLU; will still try online", flush=True)
    return False


def gpqa_available():
    try:
        import datasets
        datasets.load_dataset("Idavidrein/gpqa", "gpqa_main", trust_remote_code=True)
        print("[gpqa] gated dataset accessible", flush=True)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[gpqa] UNAVAILABLE (gated): {str(e)[:120]}", flush=True)
        return False


prefetch_mmlu()
HAVE_GPQA = gpqa_available()

import gc
import torch
import lm_eval
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM

# (lm-eval task, EvalSpec label, metric-key preference)
BENCHES = [("mmlu", "MMLU"), ("gpqa_main_zeroshot", "GPQA")]


def extract(results, task):
    m = results["results"].get(task, {})
    for key in ("acc_norm,none", "acc,none", "exact_match,none"):
        if key in m:
            return round(float(m[key]) * 100.0, 4)
    v = m.get("acc")
    return round(float(v) * 100.0, 4) if v is not None else None


def run_cell(lm, task, sys_inst, seed, offline):
    env_bak = dict(os.environ)
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["HF_DATASETS_OFFLINE"] = "1"
    try:
        last = None
        for attempt in range(3):
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
                time.sleep(15)
        raise last
    finally:
        os.environ.clear(); os.environ.update(env_bak)


for eval_id, canon_id, bs in MODELS:
    lm = HFLM(pretrained=eval_id, dtype="bfloat16", batch_size=bs)
    for task, label in BENCHES:
        if label == "GPQA" and not HAVE_GPQA:
            print(f"[skip] GPQA unavailable for {canon_id}", flush=True)
            continue
        offline = (label == "MMLU")
        for fmt, sys_inst in FORMAT_TEMPLATES.items():
            for seed in SEEDS:
                rec = dict(model_id=canon_id, benchmark=label, task_name=task,
                           temperature=0.0, prompt_format=fmt, n_shot=N_SHOT, random_seed=seed,
                           score=None, timestamp=datetime.now(timezone.utc).isoformat(),
                           eval_library_version=lm_eval.__version__, limit=None, backend="hf")
                try:
                    r = run_cell(lm, task, sys_inst, seed, offline)
                    rec["score"] = extract(r, task); rec["status"] = "ok"
                    print("RESULT " + json.dumps(rec), flush=True)
                except Exception as exc:  # noqa: BLE001
                    rec["status"] = "error"; rec["reason"] = str(exc)[:200]
                    print("RESULT " + json.dumps(rec), flush=True)
                    print(f"[cell-error] {canon_id} {label} {fmt} seed{seed}: {str(exc)[:100]}", flush=True)
    del lm; gc.collect(); torch.cuda.empty_cache()
print("[done] session complete", flush=True)
