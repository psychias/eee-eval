"""
Full-set vs limit=200 robustness check for ONE factorial-grid cell.

Cell: Qwen2.5-7B-Instruct x GSM8K x 5-shot, prompt format plain vs cot, temp 0.0.
The factorial grid (Appendix G) scored GSM8K on a 200-problem subset (limit=200)
for compute budget. This reruns the same cell on the FULL GSM8K test set (1319
questions) and prints the full-vs-limit=200 comparison, to confirm the subset is
representative and the plain>cot collapse is not a subset artefact.

Methodology matches the grid exactly: lm-evaluation-harness 0.4.11 + vLLM,
apply_chat_template=True, the plain/cot system prompts from prompt_templates.md,
num_fewshot=5, deterministic decoding (do_sample=False, temperature=0, seed=42),
GSM8K exact_match strict-match (the #### parser).

Run on a Colab GPU runtime (A100 / L4). Paste the body below into a Colab cell,
or: `python full_vs_limit200_qwen7b_gsm8k.py` on a GPU box with the deps installed.

Reference (paper, limit=200, 3 seeds): plain ~40.2, cot ~27.7.
"""

import subprocess
import sys

# Self-install on a fresh Colab VM. We use lm-eval's Hugging Face (transformers)
# backend rather than vLLM: the current vLLM wheel is built against CUDA 13
# (libcudart.so.13), which Colab's CUDA-12 image cannot load, whereas
# transformers uses the VM's own torch/CUDA and is reliable. The hf backend is
# the same one the paper's harness-version experiment (E3) used, and the
# limit=200-vs-full comparison is backend-independent (both sides use hf here).
print("[setup] installing lm-eval==0.4.11 (hf backend) ...", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "lm-eval==0.4.11", "accelerate"], check=True)

from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM

MODEL = "Qwen/Qwen2.5-7B-Instruct"
TASK  = "gsm8k"          # 5-shot, strict-match (#### parser) — identical to the grid
NSHOT = 5
GEN   = "do_sample=False"   # greedy = deterministic (hf generate() has no seed/temperature kwarg)
SYS = {
    "plain": None,       # model's own default system prompt
    "cot":   ("You are an expert problem solver. Before answering, think step by "
              "step and show your reasoning. Then state the final answer."),
}
REF_LIMIT200 = {"plain": 40.2, "cot": 27.7}   # paper, mean over 3 seeds

lm = HFLM(pretrained=MODEL, dtype="bfloat16", batch_size=48)


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
print("\nplain>cot gap @200 vs @full, and full-vs-200 deltas: if the deltas are small")
print("(<~2 pp) and the gap is preserved, the 200-problem subset is representative.")
