"""Backend control at matched n: score GSM8K on the hf backend for the same four
models, three prompt formats, single seed, and full test set as the generalization
arm (ARC-Challenge/WinoGrande/OpenBookQA), so the GSM8K format effect can be read on
the SAME backend and the SAME n=4-per-format design as the log-likelihood benchmarks.

This disentangles inference backend from prompt format: the main grid scores GSM8K
under vLLM (generation is slow on hf), and this control confirms the format effect
is not a vLLM artefact. GSM8K is generation, exact_match strict-match (#### parser);
format via system_instruction + apply_chat_template; greedy (temperature 0); 5-shot.

One model per Colab session (swap MODELS). Mistral is gated: set HF_TOKEN in the
environment. Llama-3.1 uses the NousResearch mirror."""
import os
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
_TOKEN = os.environ.get("HF_TOKEN", "")
if _TOKEN:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = _TOKEN

import subprocess
import sys
import json
from datetime import datetime, timezone

# (eval_id, canonical_id, batch_size) -- one active per session.
MODELS = [("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct", 16)]
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

for eval_id, canon_id, bs in MODELS:
    lm = HFLM(pretrained=eval_id, dtype="bfloat16", batch_size=bs)
    for fmt, sys_inst in FORMAT_TEMPLATES.items():
        rec = dict(model_id=canon_id, benchmark="gsm8k", backend="hf", prompt_format=fmt,
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
            print(f"[cell-error] {canon_id} {fmt}: {str(exc)[:120]}", flush=True)
    del lm; gc.collect(); torch.cuda.empty_cache()
print("[done] session complete", flush=True)
