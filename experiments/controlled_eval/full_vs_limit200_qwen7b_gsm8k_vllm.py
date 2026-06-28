"""
Full-set vs limit=200 robustness check — vLLM backend (like-for-like with the grid).

Same cell as full_vs_limit200_qwen7b_gsm8k.py (Qwen2.5-7B-Instruct, GSM8K, 5-shot,
plain vs cot, temp 0.0) but on the *vLLM* backend the factorial grid used, so the
absolute numbers are directly comparable to the paper's vLLM limit=200 values
(plain ~40.2, cot ~27.7) rather than the hf-backend variant.

vLLM is pinned to 0.6.6.post1 because the current vLLM wheel is built against
CUDA 13 (libcudart.so.13), which Colab's CUDA-12 image cannot load; 0.6.6.post1
ships CUDA-12.1 wheels and pins torch 2.5.1 (pip downgrades torch to match).

Run on a Colab A100 GPU runtime. Reference (paper, vLLM, limit=200, 3 seeds):
plain ~40.2, cot ~27.7.
"""

import subprocess
import sys

# Pin a CUDA-12 vLLM (+ ray, which lm-eval 0.4.11's vLLM wrapper imports). This
# downgrades torch to 2.5.1+cu121 to match the vLLM wheel.
print("[setup] installing lm-eval==0.4.11 + vllm==0.6.6.post1 + ray ...", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "lm-eval==0.4.11", "vllm==0.6.6.post1", "ray"], check=True)

from lm_eval import simple_evaluate
from lm_eval.models.vllm_causallms import VLLM

MODEL = "Qwen/Qwen2.5-7B-Instruct"
TASK  = "gsm8k"          # 5-shot, strict-match (#### parser) — identical to the grid
NSHOT = 5
GEN   = "do_sample=False,temperature=0,seed=42"   # the grid's exact gen settings
SYS = {
    "plain": None,       # model's own default system prompt
    "cot":   ("You are an expert problem solver. Before answering, think step by "
              "step and show your reasoning. Then state the final answer."),
}
REF_LIMIT200 = {"plain": 40.2, "cot": 27.7}   # paper (vLLM), mean over 3 seeds

lm = VLLM(pretrained=MODEL, dtype="bfloat16",
          gpu_memory_utilization=0.90, max_model_len=4096)


def run(fmt, limit):
    res = simple_evaluate(
        model=lm, tasks=[TASK], num_fewshot=NSHOT, gen_kwargs=GEN,
        system_instruction=SYS[fmt], apply_chat_template=True,
        limit=limit, random_seed=42, fewshot_random_seed=42,
    )
    m = res["results"][TASK]
    score = round(100 * m.get("exact_match,strict-match",
                              m.get("exact_match,flexible-extract")), 2)
    tag = "full" if limit is None else f"limit={limit}"
    print(f"  [done] {fmt} {tag}: {score}", flush=True)   # incremental: survives a timeout
    return score


print(f"{'format':<8}{'limit=200':>12}{'full(1319)':>13}{'delta':>9}{'paper@200':>11}", flush=True)
print("-" * 53, flush=True)
for fmt in ("plain", "cot"):
    s200  = run(fmt, 200)
    sfull = run(fmt, None)        # None = full test set
    print(f"{fmt:<8}{s200:>12.2f}{sfull:>13.2f}{sfull - s200:>+9.2f}{REF_LIMIT200[fmt]:>11.1f}", flush=True)
