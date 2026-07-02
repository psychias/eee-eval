"""Backend control: verify the GSM8K prompt-format effect replicates on the hf
backend, disentangling inference backend from prompt format.

The main grid scores GSM8K under vLLM (generation is slow on hf, which is why the
grid uses vLLM there). To check the format effect is not a vLLM artefact, we re-run
Qwen2.5-7B-Instruct on GSM8K, all three formats (plain/instruct/cot), 5-shot,
greedy, full test set, on the hf backend and compare the plain/instruct/cot gap to
the vLLM grid's. Format applied via system_instruction + apply_chat_template; GSM8K
exact_match strict-match (the #### parser)."""
import os
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import subprocess
import sys
import json
from datetime import datetime, timezone

MODEL = "Qwen/Qwen2.5-7B-Instruct"
FORMAT_TEMPLATES = {
    "plain": None,
    "instruct": "You are an expert problem solver. Answer each question carefully and directly.",
    "cot": "You are an expert problem solver. Before answering, think step by step and show your reasoning. Then state the final answer.",
}
SEED = 42

print("[setup] installing lm-eval==0.4.11 ...", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lm-eval==0.4.11", "accelerate"], check=True)

import gc
import torch
import lm_eval
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM

lm = HFLM(pretrained=MODEL, dtype="bfloat16", batch_size=16)
for fmt, sys_inst in FORMAT_TEMPLATES.items():
    rec = dict(model_id=MODEL, benchmark="gsm8k", backend="hf", prompt_format=fmt,
               n_shot=5, temperature=0.0, random_seed=SEED, limit=None,
               timestamp=datetime.now(timezone.utc).isoformat(),
               eval_library_version=lm_eval.__version__)
    try:
        r = simple_evaluate(model=lm, tasks=["gsm8k"], num_fewshot=5,
                            gen_kwargs="do_sample=False", limit=None,
                            system_instruction=sys_inst, apply_chat_template=True,
                            fewshot_as_multiturn=False, random_seed=SEED,
                            numpy_random_seed=SEED, torch_random_seed=SEED,
                            write_out=False, log_samples=False, verbosity="ERROR")
        m = r["results"]["gsm8k"]
        rec["score"] = round(100 * m.get("exact_match,strict-match",
                                         m.get("exact_match,flexible-extract")), 4)
        rec["status"] = "ok"
        print("RESULT " + json.dumps(rec), flush=True)
    except Exception as exc:  # noqa: BLE001
        rec["status"] = "error"; rec["reason"] = str(exc)[:200]
        print("RESULT " + json.dumps(rec), flush=True)
        print(f"[cell-error] {fmt}: {str(exc)[:120]}", flush=True)
del lm; gc.collect(); torch.cuda.empty_cache()
print("[done]", flush=True)
