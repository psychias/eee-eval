"""Generate the E1 BBH controlled experiment notebook.

Key design insight: lm-eval 0.4.5 uses generate_until for ALL BBH tasks.
There is no loglikelihood BBH variant. OLv2 uses lighteval, whose BBH
implementation uses loglikelihood MC scoring. To reproduce OLv2's ~21pp
number, we need lighteval for Condition A.

Four conditions:
  A  (lighteval):  loglikelihood MC, 18 tasks     → reproduces OLv2's ~21pp
  A' (lm-eval):   direct-answer generation, no CoT, 18 tasks → shows lm-eval's "non-CoT"
  B  (lm-eval):   CoT generation, 23 tasks        → reproduces literature ~55pp
  C  (lm-eval):   CoT generation, 18 tasks        → controls for task coverage
"""
import json
from pathlib import Path

cells = []

def _split_source(source):
    """Split source into lines with \n preserved (nbformat requirement)."""
    lines = source.split("\n")
    # Add \n to all lines except the last
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]

def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": _split_source(source)})

def code(source):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _split_source(source)})


# ═══════════════════════════════════════════════════════════════════════════
md("""# Experiment 1 — BBH: lighteval-style vs Suzgun-style on gemma-7b

**Goal.** Test whether OLv2's BBH score for gemma-7b (~21 pp) and the literature
consensus (~55 pp) can be reproduced on the same hardware with the same model
checkpoint, varying only the evaluation methodology.

If both numbers appear under the predicted conditions, the cross-source gap is
mechanically attributable to `scoring_mode` + `prompt_template`, not model
capability.

**Critical discovery:** lm-eval 0.4.5 uses `generate_until` for **ALL** BBH tasks
(both `bbh_fewshot` and `bbh_cot_fewshot`). There is no loglikelihood BBH variant.
OLv2 uses **lighteval**, whose BBH implementation does loglikelihood MC scoring.
Therefore we need lighteval for Condition A.

| Condition | Harness | Method | Tasks | Expected |
|---|---|---|---|---|
| **A** | lighteval | loglikelihood MC, no CoT | 18 | ~20 pp |
| **A'** | lm-eval | direct-answer generation, no CoT | 18 | TBD |
| **B** | lm-eval | CoT generation + extraction | 23 | ~55 pp |
| **C** | lm-eval | CoT generation + extraction | 18 | ~55 pp |

Condition A' is exploratory — it shows what score you get with lm-eval's
non-CoT BBH tasks. If A' ≈ A, then generation-vs-loglikelihood doesn't
matter much and the gap is all CoT. If A' >> A, then the scoring mode
(loglikelihood vs generation) accounts for part of the gap even without CoT.""")

# ═══════════════════════════════════════════════════════════════════════════
md("## 0 · Preflight: GPU check")

code("""import subprocess, sys
print(subprocess.check_output(
    ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv']
).decode())

import torch
assert torch.cuda.is_available(), 'No GPU detected — this notebook requires an A100.'
gpu_name = torch.cuda.get_device_name(0)
print(f'GPU: {gpu_name}')
if 'A100' not in gpu_name and 'H100' not in gpu_name:
    print(f'WARNING: Expected A100/H100, got {gpu_name}. Proceed with caution.')""")

# ═══════════════════════════════════════════════════════════════════════════
md("""## 1 · Install pinned dependencies

We need BOTH lighteval (for Condition A) and lm-eval (for Conditions A'/B/C).
They share vLLM as the inference backend.""")

code("""%%writefile /content/requirements_E1.txt
lm-eval[vllm]==0.4.5
vllm==0.6.3.post1
accelerate==0.34.2
transformers==4.45.2
datasets==3.0.2
torch==2.4.0
numpy<2.0
sentencepiece
protobuf
matplotlib
scipy""")

code("""!pip install -q --upgrade pip
!pip install -q -r /content/requirements_E1.txt 2>&1 | tail -5
# lighteval installed separately — it may pin different versions of shared deps
!pip install -q 'lighteval[accelerate,vllm]' 2>&1 | tail -5""")

code("""# Record exact installed versions
import importlib.metadata as meta
HARNESS_VERSION    = meta.version('lm_eval')
VLLM_VERSION       = meta.version('vllm')
TRANS_VERSION      = meta.version('transformers')
TORCH_VERSION      = meta.version('torch')
try:
    LIGHTEVAL_VERSION = meta.version('lighteval')
except Exception:
    LIGHTEVAL_VERSION = 'not installed'
print(f'lm-eval:       {HARNESS_VERSION}')
print(f'lighteval:     {LIGHTEVAL_VERSION}')
print(f'vllm:          {VLLM_VERSION}')
print(f'transformers:  {TRANS_VERSION}')
print(f'torch:         {TORCH_VERSION}')""")

# ═══════════════════════════════════════════════════════════════════════════
md("## 2 · Configuration")

code("""import os, json, hashlib, time, datetime, statistics, re
from pathlib import Path

MODEL_ID     = 'google/gemma-7b'
SEED         = 1234
TEMPERATURE  = 0.0
N_SHOT       = 3
BATCH_SIZE   = 'auto'
MAX_GEN_TOKS = 512

OUT_DIR = Path('/content/bbh_E1_results')
OUT_DIR.mkdir(exist_ok=True, parents=True)

RUN_TS = datetime.datetime.utcnow().isoformat() + 'Z'

# ── The 18 lighteval BBH tasks ─────────────────────────────────────────────
LIGHTEVAL_BBH_TASKS = [
    'causal_judgement', 'date_understanding', 'disambiguation_qa',
    'geometric_shapes', 'logical_deduction_five_objects',
    'logical_deduction_seven_objects', 'logical_deduction_three_objects',
    'movie_recommendation', 'navigate', 'reasoning_about_colored_objects',
    'ruin_names', 'salient_translation_error_detection', 'snarks',
    'sports_understanding', 'temporal_sequences',
    'tracking_shuffled_objects_five_objects',
    'tracking_shuffled_objects_seven_objects',
    'tracking_shuffled_objects_three_objects',
]
assert len(LIGHTEVAL_BBH_TASKS) == 18

# ── The 5 additional Suzgun tasks (= 23 total) ────────────────────────────
SUZGUN_EXTRA = [
    'boolean_expressions', 'formal_fallacies', 'hyperbaton',
    'object_counting', 'web_of_lies',
]
SUZGUN_FULL_BBH_TASKS = LIGHTEVAL_BBH_TASKS + SUZGUN_EXTRA
assert len(SUZGUN_FULL_BBH_TASKS) == 23

print(f'Timestamp:       {RUN_TS}')
print(f'Model:           {MODEL_ID}')
print(f'Output dir:      {OUT_DIR}')
print(f'Lighteval tasks: {len(LIGHTEVAL_BBH_TASKS)}')
print(f'Suzgun tasks:    {len(SUZGUN_FULL_BBH_TASKS)}')""")

# ═══════════════════════════════════════════════════════════════════════════
md("""## 3 · Verify lm-eval task names

**CRITICAL:** lm-eval 0.4.5 has `bbh_fewshot_*` and `bbh_cot_fewshot_*`, but
both use `generate_until` — NOT loglikelihood. The difference is only whether
CoT exemplars are included in the prompt.

The prefix auto-detection below handles lm-eval's pipe-delimited table output.""")

code("""# List all BBH-related tasks in the installed lm-eval version
!lm_eval --tasks list 2>&1 | grep -iE 'bbh' | head -60""")

code("""# Parse task list — handle lm-eval's pipe-delimited table format
import subprocess

raw = subprocess.check_output(['lm_eval', '--tasks', 'list'], stderr=subprocess.STDOUT).decode()

# Extract task names from pipe-delimited table rows:
#   |bbh_fewshot_causal_judgement    |lm_eval/tasks/...    |generate_until    |
# We want just the first column (the task name).
bbh_task_names = []
for line in raw.splitlines():
    line = line.strip()
    if 'bbh' not in line.lower():
        continue
    # Split on pipe, take the first non-empty field
    fields = [f.strip() for f in line.split('|') if f.strip()]
    if fields:
        task_name = fields[0]
        bbh_task_names.append(task_name)

# Separate CoT vs non-CoT
cot_tasks = sorted(set(t for t in bbh_task_names if 'cot' in t.lower()))
noncot_tasks = sorted(set(t for t in bbh_task_names if 'cot' not in t.lower()))

print(f'Non-CoT BBH tasks ({len(noncot_tasks)}):')
for t in noncot_tasks:
    print(f'  {t}')

print(f'\\nCoT BBH tasks ({len(cot_tasks)}):')
for t in cot_tasks:
    print(f'  {t}')""")

code("""# Auto-detect prefixes from the CLEAN task names
TASK_PREFIX_A_PRIME = None  # for lm-eval non-CoT (Condition A')
TASK_PREFIX_B = None         # for lm-eval CoT (Conditions B/C)

for t in noncot_tasks:
    # Match: bbh_fewshot_causal_judgement (or similar)
    if t.endswith('_causal_judgement'):
        TASK_PREFIX_A_PRIME = t.replace('_causal_judgement', '')
        break

for t in cot_tasks:
    if t.endswith('_causal_judgement'):
        TASK_PREFIX_B = t.replace('_causal_judgement', '')
        break

print(f'=== AUTO-DETECTED PREFIXES ===')
print(f'  Condition A\\' (lm-eval, non-CoT generation): {TASK_PREFIX_A_PRIME!r}')
print(f'  Condition B/C  (lm-eval, CoT generation):    {TASK_PREFIX_B!r}')

if TASK_PREFIX_A_PRIME is None or TASK_PREFIX_B is None:
    print('\\n*** STOP: Could not auto-detect task prefixes.')
    print('    Inspect the task list above and set prefixes manually.')
    raise RuntimeError('Prefix detection failed')

# Verify all tasks exist for each condition
missing_Ap = [t for t in LIGHTEVAL_BBH_TASKS
              if f'{TASK_PREFIX_A_PRIME}_{t}' not in noncot_tasks]
missing_B  = [t for t in SUZGUN_FULL_BBH_TASKS
              if f'{TASK_PREFIX_B}_{t}' not in cot_tasks]

if missing_Ap:
    print(f'\\n*** WARNING: {len(missing_Ap)} tasks missing from A\\': {missing_Ap}')
else:
    print(f'  All 18 lighteval tasks found for A\\'')

if missing_B:
    print(f'\\n*** WARNING: {len(missing_B)} tasks missing from B: {missing_B}')
else:
    print(f'  All 23 Suzgun tasks found for B')

# Confirm scoring mode
print()
print('NOTE: Both bbh_fewshot_* and bbh_cot_fewshot_* use generate_until.')
print('      Neither uses loglikelihood. For true loglikelihood scoring')
print('      (OLv2\\'s method), we use lighteval in Condition A.')""")

# ═══════════════════════════════════════════════════════════════════════════
md("""## 4 · Condition A — lighteval loglikelihood (18 tasks)

This reproduces OLv2's actual methodology. lighteval evaluates BBH with
loglikelihood scoring (multiple-choice log-probabilities, no text generation).

Expected: ~18-25 pp. This is the OLv2 number.""")

code("""# Check what lighteval BBH tasks are available
# lighteval names them "bigbench_hard:<subtask>" (not "bbh")
!python -m lighteval tasks list 2>&1 | grep -iE 'bigbench_hard|bbh' | head -30""")

code("""# lighteval task identifiers for BBH
# lighteval uses "bigbench_hard:<subtask>" format
# CLI spec: "bigbench_hard:<subtask>|<n_shot>|0"

import subprocess
raw_le = subprocess.check_output(
    ['python', '-m', 'lighteval', 'tasks', 'list'],
    stderr=subprocess.STDOUT
).decode()

le_bbh = []
for line in raw_le.splitlines():
    if 'bigbench_hard' in line.lower() or 'bbh' in line.lower():
        le_bbh.append(line.strip())

print(f'lighteval BBH tasks ({len(le_bbh)}):')
for t in le_bbh[:30]:
    print(f'  {t}')

if not le_bbh:
    print('WARNING: No BBH tasks found. Check lighteval version.')
    print('Searching broader...')
    for line in raw_le.splitlines():
        if 'bench' in line.lower() and 'hard' in line.lower():
            print(f'  {line.strip()}')""")

code("""# Build the lighteval task string for Condition A.
# lighteval BBH tasks are named "bigbench_hard:<subtask>" (NOT "extended|bbh:...")
# CLI task spec format: "bigbench_hard:<subtask>|<n_shot>|0"
# Note: lighteval uses "causal_judgment" (no 'e'), lm-eval uses "causal_judgement" (with 'e')

# Map from our canonical names (lm-eval style) to lighteval subtask names
LIGHTEVAL_BBH_SUBTASK_MAP = {
    'causal_judgement': 'causal_judgment',  # lighteval drops the 'e'
    'date_understanding': 'date_understanding',
    'disambiguation_qa': 'disambiguation_qa',
    'geometric_shapes': 'geometric_shapes',
    'logical_deduction_five_objects': 'logical_deduction_five_objects',
    'logical_deduction_seven_objects': 'logical_deduction_seven_objects',
    'logical_deduction_three_objects': 'logical_deduction_three_objects',
    'movie_recommendation': 'movie_recommendation',
    'navigate': 'navigate',
    'reasoning_about_colored_objects': 'reasoning_about_colored_objects',
    'ruin_names': 'ruin_names',
    'salient_translation_error_detection': 'salient_translation_error_detection',
    'snarks': 'snarks',
    'sports_understanding': 'sports_understanding',
    'temporal_sequences': 'temporal_sequences',
    'tracking_shuffled_objects_five_objects': 'tracking_shuffled_objects_five_objects',
    'tracking_shuffled_objects_seven_objects': 'tracking_shuffled_objects_seven_objects',
    'tracking_shuffled_objects_three_objects': 'tracking_shuffled_objects_three_objects',
}

LE_TASK_SPECS_A = [
    f'bigbench_hard:{LIGHTEVAL_BBH_SUBTASK_MAP[t]}|{N_SHOT}|0'
    for t in LIGHTEVAL_BBH_TASKS
]
print(f'Condition A lighteval task specs ({len(LE_TASK_SPECS_A)}):')
for spec in LE_TASK_SPECS_A[:5]:
    print(f'  {spec}')
print(f'  ... ({len(LE_TASK_SPECS_A)} total)')

le_tasks_str = ','.join(LE_TASK_SPECS_A)""")

code("""# Run lighteval for Condition A (loglikelihood scoring)
OUT_A = OUT_DIR / 'condition_A_lighteval_loglikelihood'
OUT_A.mkdir(exist_ok=True)

t0 = time.time()

# lighteval CLI: positional args (MODEL_ARGS TASKS), options use hyphens
# lighteval uses model_name (not pretrained) and max_model_length (not max_model_len)
!python -m lighteval vllm \\
  "model_name={MODEL_ID},dtype=bfloat16,gpu_memory_utilization=0.90,max_model_length=4096,trust_remote_code=True" \\
  "{le_tasks_str}" \\
  --output-dir {OUT_A} \\
  --save-details

wall_A = time.time() - t0
print(f'\\n=== Condition A (lighteval) wall time: {wall_A/60:.1f} min ===')""")

code("""# If lighteval CLI format is different, try alternative invocations:
# The lighteval API can change between versions. If the cell above fails,
# uncomment and try one of these alternatives:

# Alt 1: lighteval accelerate (non-vLLM) — also uses positional args
# !python -m lighteval accelerate \\
#   "model_name={MODEL_ID},dtype=bfloat16" \\
#   "{le_tasks_str}" \\
#   --output-dir {OUT_A}

# Alt 2: lighteval Python API (most version-resilient)
# from lighteval.logging.evaluation_tracker import EvaluationTracker
# from lighteval.models.vllm.vllm_model import VLLMModelConfig
# from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
# tracker = EvaluationTracker(output_dir=str(OUT_A), save_details=True)
# params = PipelineParameters(launcher_type=ParallelismManager.VLLM)
# config = VLLMModelConfig(
#     model_name=MODEL_ID, dtype="bfloat16",
#     gpu_memory_utilization=0.90, max_model_length=4096,
#     trust_remote_code=True
# )
# pipeline = Pipeline(tasks=le_tasks_str, pipeline_parameters=params,
#                     evaluation_tracker=tracker, model_config=config)
# pipeline.evaluate()
# pipeline.show_results()

# Alt 3: If lighteval's BBH uses different naming, look at:
# !find /content/ -path '*/lighteval/tasks/*bbh*' 2>/dev/null
# and adjust LE_TASK_SPECS_A accordingly.

print('If Condition A failed, try the alternatives above.')""")

# ═══════════════════════════════════════════════════════════════════════════
md("""## 5 · Condition A' — lm-eval direct-answer generation (18 tasks)

lm-eval's `bbh_fewshot_*` tasks: 3-shot, direct answer (no CoT), but using
`generate_until` (NOT loglikelihood). This tells us how much of the gap is
specifically from loglikelihood-vs-generation, independent of CoT.

| Axis | Value |
|---|---|
| scoring_mode | generation (generate_until, answer extraction) |
| prompt_template | fewshot, direct answer, no CoT |
| chain_of_thought | false |
| n_shot | 3 |""")

code("""# Build task list for Condition A' (lm-eval non-CoT)
tasks_Ap = ','.join(f'{TASK_PREFIX_A_PRIME}_{t}' for t in LIGHTEVAL_BBH_TASKS)
print(f'Condition A\\' tasks ({len(LIGHTEVAL_BBH_TASKS)}):')
print(tasks_Ap[:200] + '...')""")

code("""OUT_Ap = OUT_DIR / 'condition_Ap_lmeval_noncot'
OUT_Ap.mkdir(exist_ok=True)

t0 = time.time()

!lm_eval \\
  --model vllm \\
  --model_args pretrained={MODEL_ID},dtype=bfloat16,gpu_memory_utilization=0.90,max_model_len=4096 \\
  --tasks {tasks_Ap} \\
  --num_fewshot {N_SHOT} \\
  --batch_size {BATCH_SIZE} \\
  --seed {SEED} \\
  --output_path {OUT_Ap} \\
  --log_samples \\
  --trust_remote_code

wall_Ap = time.time() - t0
print(f'\\n=== Condition A\\' wall time: {wall_Ap/60:.1f} min ===')""")

# ═══════════════════════════════════════════════════════════════════════════
md("""## 6 · Condition B — Suzgun-style full (CoT generation, 23 tasks)

Canonical BBH from Suzgun et al. (2023). lm-eval's `bbh_cot_fewshot_*` tasks:
3-shot with CoT exemplars, generate_until with answer extraction.

Expected: ~50-58 pp. Wall time ~20-30 min on A100.""")

code("""tasks_B = ','.join(f'{TASK_PREFIX_B}_{t}' for t in SUZGUN_FULL_BBH_TASKS)
print(f'Condition B tasks ({len(SUZGUN_FULL_BBH_TASKS)}):')
print(tasks_B[:200] + '...')""")

code("""OUT_B = OUT_DIR / 'condition_B_suzgun_full'
OUT_B.mkdir(exist_ok=True)

t0 = time.time()

!lm_eval \\
  --model vllm \\
  --model_args pretrained={MODEL_ID},dtype=bfloat16,gpu_memory_utilization=0.90,max_model_len=4096 \\
  --tasks {tasks_B} \\
  --num_fewshot {N_SHOT} \\
  --batch_size {BATCH_SIZE} \\
  --seed {SEED} \\
  --gen_kwargs temperature=0.0,do_sample=False,max_gen_toks=512 \\
  --output_path {OUT_B} \\
  --log_samples \\
  --trust_remote_code

wall_B = time.time() - t0
print(f'\\n=== Condition B wall time: {wall_B/60:.1f} min ===')""")

# ═══════════════════════════════════════════════════════════════════════════
md("""## 7 · Condition C — Suzgun-style, task-matched (CoT, 18 tasks)

Same as B but restricted to A's 18 tasks. Controls for task-coverage confound.""")

code("""tasks_C = ','.join(f'{TASK_PREFIX_B}_{t}' for t in LIGHTEVAL_BBH_TASKS)
print(f'Condition C tasks ({len(LIGHTEVAL_BBH_TASKS)}):')""")

code("""OUT_C = OUT_DIR / 'condition_C_suzgun_18tasks'
OUT_C.mkdir(exist_ok=True)

t0 = time.time()

!lm_eval \\
  --model vllm \\
  --model_args pretrained={MODEL_ID},dtype=bfloat16,gpu_memory_utilization=0.90,max_model_len=4096 \\
  --tasks {tasks_C} \\
  --num_fewshot {N_SHOT} \\
  --batch_size {BATCH_SIZE} \\
  --seed {SEED} \\
  --gen_kwargs temperature=0.0,do_sample=False,max_gen_toks=512 \\
  --output_path {OUT_C} \\
  --log_samples \\
  --trust_remote_code

wall_C = time.time() - t0
print(f'\\n=== Condition C wall time: {wall_C/60:.1f} min ===')""")

# ═══════════════════════════════════════════════════════════════════════════
md("## 8 · Parse results and aggregate")

code("""import glob, json
from pathlib import Path

def load_lmeval_results(out_dir):
    \"\"\"Find and load the lm-eval results JSON.\"\"\"
    files = sorted(Path(out_dir).rglob('results_*.json'))
    if not files:
        raise FileNotFoundError(f'No results_*.json in {out_dir}')
    print(f'Loading: {files[-1]}')
    with open(files[-1]) as f:
        return json.load(f)

def load_lighteval_results(out_dir):
    \"\"\"Find and load lighteval results.
    lighteval saves results in a different format — adjust as needed.\"\"\"
    # lighteval typically saves to results/<model_name>/results_*.json
    # or output_dir/results.json
    for pattern in ['**/results*.json', '**/results.json', '**/*_results.json']:
        files = sorted(Path(out_dir).rglob(pattern.replace('**/', '')))
        if not files:
            files = sorted(Path(out_dir).glob(pattern))
        if files:
            print(f'Loading lighteval: {files[-1]}')
            with open(files[-1]) as f:
                return json.load(f)
    raise FileNotFoundError(f'No lighteval results in {out_dir}')

# Load lm-eval results (A', B, C)
res_Ap = load_lmeval_results(OUT_Ap)
res_B  = load_lmeval_results(OUT_B)
res_C  = load_lmeval_results(OUT_C)

# Load lighteval results (A)
res_A = load_lighteval_results(OUT_A)
print('\\nAll results loaded.')""")

code("""def extract_per_task_acc_lmeval(results_dict, task_prefix):
    \"\"\"Extract per-task accuracy from lm-eval results.\"\"\"
    out = {}
    results = results_dict.get('results', results_dict)
    for tname, metrics in results.items():
        if not tname.startswith(task_prefix):
            continue
        short_name = tname[len(task_prefix) + 1:]
        for key in ['exact_match,flexible-extract', 'exact_match,none',
                     'acc,none', 'acc_norm,none']:
            if key in metrics:
                out[short_name] = float(metrics[key])
                break
        else:
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and 'stderr' not in k:
                    out[short_name] = float(v)
                    break
    return out

def extract_per_task_acc_lighteval(results_dict):
    \"\"\"Extract per-task accuracy from lighteval results.
    lighteval's output format differs from lm-eval. Adjust based on actual output.\"\"\"
    out = {}
    # lighteval may use different structures depending on version:
    # Option 1: results dict keyed by task spec
    # Option 2: nested under 'results' -> task_name -> metric
    if 'results' in results_dict:
        data = results_dict['results']
    else:
        data = results_dict

    for tname, metrics in data.items():
        if not isinstance(metrics, dict):
            continue
        # Extract the BBH subtask name from lighteval's naming
        # e.g. "extended|bbh:causal_judgement|3|0" -> "causal_judgement"
        short = tname
        for task in LIGHTEVAL_BBH_TASKS:
            if task in tname:
                short = task
                break
        if short not in LIGHTEVAL_BBH_TASKS:
            continue
        # Find accuracy metric
        for key in ['acc', 'accuracy', 'exact_match', 'acc_norm',
                     'loglikelihood_acc', 'loglikelihood_acc_norm']:
            if key in metrics:
                val = metrics[key]
                if isinstance(val, (int, float)):
                    out[short] = float(val)
                    break
        else:
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and 'stderr' not in str(k):
                    out[short] = float(v)
                    break
    return out

A_scores  = extract_per_task_acc_lighteval(res_A)
Ap_scores = extract_per_task_acc_lmeval(res_Ap, TASK_PREFIX_A_PRIME)
B_scores  = extract_per_task_acc_lmeval(res_B, TASK_PREFIX_B)
C_scores  = extract_per_task_acc_lmeval(res_C, TASK_PREFIX_B)

print(f'Tasks extracted:')
print(f'  A  (lighteval LL):     {len(A_scores)}/18')
print(f'  A\\' (lm-eval no-CoT):  {len(Ap_scores)}/18')
print(f'  B  (lm-eval CoT 23):  {len(B_scores)}/23')
print(f'  C  (lm-eval CoT 18):  {len(C_scores)}/18')

for label, scores, expected in [('A', A_scores, 18), ('A\\'', Ap_scores, 18),
                                  ('B', B_scores, 23), ('C', C_scores, 18)]:
    if len(scores) != expected:
        print(f'*** WARNING: Expected {expected} tasks for {label}, got {len(scores)}')""")

code("""# Per-task comparison table
hdr = f'{\"task\":42s}  {\"A(LL)\":>7s}  {\"A\\'(gen)\":>7s}  {\"B(CoT)\":>7s}  {\"C(CoT)\":>7s}  {\"B-A\":>6s}'
print(hdr)
print('-' * 90)

for t in LIGHTEVAL_BBH_TASKS:
    a  = A_scores.get(t, float('nan'))
    ap = Ap_scores.get(t, float('nan'))
    b  = B_scores.get(t, float('nan'))
    c  = C_scores.get(t, float('nan'))
    delta = (b - a) if not (a != a or b != b) else float('nan')
    print(f'{t:42s}  {a*100:6.1f}%  {ap*100:6.1f}%  {b*100:6.1f}%  {c*100:6.1f}%  {delta*100:+5.0f}')

print('-' * 90)
print('\\nSuzgun-only tasks (in B only):')
for t in SUZGUN_EXTRA:
    b = B_scores.get(t, float('nan'))
    print(f'  {t:42s}  {b*100:6.1f}%')""")

code("""# Aggregate scores
def mean_pct(d):
    vals = list(d.values())
    return statistics.mean(vals) * 100 if vals else float('nan')

agg_A_18   = mean_pct(A_scores)
agg_Ap_18  = mean_pct(Ap_scores)
agg_B_23   = mean_pct(B_scores)
agg_C_18   = mean_pct(C_scores)

B_on_18 = {t: B_scores[t] for t in LIGHTEVAL_BBH_TASKS if t in B_scores}
agg_B_on_18 = mean_pct(B_on_18)

print('=' * 75)
print('AGGREGATE RESULTS -- gemma-7b on BBH')
print('=' * 75)
print(f'  A.  lighteval loglikelihood (18 tasks):     {agg_A_18:5.2f} pp')
print(f'  A\\'. lm-eval non-CoT gen    (18 tasks):     {agg_Ap_18:5.2f} pp')
print(f'  B.  lm-eval CoT generation  (23 tasks):     {agg_B_23:5.2f} pp')
print(f'  C.  lm-eval CoT generation  (18 tasks):     {agg_C_18:5.2f} pp')
print(f'  B restricted to 18   (sanity check):        {agg_B_on_18:5.2f} pp')
print('-' * 75)
print(f'  Total cross-source gap        (B - A):   {agg_B_23 - agg_A_18:+6.2f} pp')
print(f'  Scoring mode effect (gen-LL)  (A\\'- A):   {agg_Ap_18 - agg_A_18:+6.2f} pp')
print(f'  CoT effect                    (C - A\\'):   {agg_C_18 - agg_Ap_18:+6.2f} pp')
print(f'  Task coverage effect          (B - C):    {agg_B_23 - agg_C_18:+6.2f} pp')
print(f'  Sanity: C vs B-on-18         (should~0):  {agg_C_18 - agg_B_on_18:+6.2f} pp')
print()
print('Decomposition of total gap (B - A):')
print(f'  = scoring_mode (A\\'-A) + CoT (C-A\\') + task_coverage (B-C)')
print(f'  = {agg_Ap_18-agg_A_18:+.1f} + {agg_C_18-agg_Ap_18:+.1f} + {agg_B_23-agg_C_18:+.1f} = {agg_B_23-agg_A_18:+.1f} pp')
print()
print('Reference points:')
print('  OLv2 published:   21.12 pp')
print('  Gemma paper:      55.10 pp (3-shot CoT)')
print('  MAmmoTH2 paper:   57.40 pp')""")

# ═══════════════════════════════════════════════════════════════════════════
md("## 9 · Interpretation")

code("""print('=' * 75)
print('INTERPRETATION')
print('=' * 75)

if 18 <= agg_A_18 <= 25 and 50 <= agg_B_23 <= 58 and 50 <= agg_C_18 <= 58:
    print('OUTCOME: M1 CONFIRMED interventionally.')
    print(f'  A  = {agg_A_18:.1f} pp (predicted 18-25, matches OLv2\\'s ~21 pp)')
    print(f'  A\\' = {agg_Ap_18:.1f} pp (lm-eval non-CoT generation)')
    print(f'  B  = {agg_B_23:.1f} pp (predicted 50-58, matches literature ~55 pp)')
    print(f'  C  = {agg_C_18:.1f} pp (predicted 50-58)')
    print()
    print(f'  Total gap (B-A) = {agg_B_23-agg_A_18:.1f} pp, decomposed as:')
    print(f'    Scoring mode (LL -> gen):  {agg_Ap_18-agg_A_18:+.1f} pp')
    print(f'    CoT effect:                {agg_C_18-agg_Ap_18:+.1f} pp')
    print(f'    Task coverage:             {agg_B_23-agg_C_18:+.1f} pp')
    print()
    print('  The cross-source gap is mechanically attributable to evaluation config.')
elif abs(agg_Ap_18 - agg_A_18) < 3 and agg_B_23 > 40:
    print('OUTCOME: Scoring mode (LL vs gen) has minimal effect.')
    print(f'  A  (LL)  = {agg_A_18:.1f} pp')
    print(f'  A\\' (gen) = {agg_Ap_18:.1f} pp  (delta: {agg_Ap_18-agg_A_18:+.1f})')
    print(f'  B  (CoT) = {agg_B_23:.1f} pp')
    print('  The gap is almost entirely from CoT, not from scoring mode.')
elif agg_A_18 < 15:
    print(f'OUTCOME: A = {agg_A_18:.1f} pp -- lighteval loglikelihood failing.')
    print('  Inspect per-task scores for systematic failure.')
elif agg_B_23 < 40:
    print(f'OUTCOME: B = {agg_B_23:.1f} pp -- CoT extraction failing.')
    print('  Inspect --log_samples outputs.')
else:
    print(f'OUTCOME: Partial -- numbers outside predicted ranges.')
    print(f'  A={agg_A_18:.1f}, A\\'={agg_Ap_18:.1f}, B={agg_B_23:.1f}, C={agg_C_18:.1f}')
    gap = agg_B_23 - agg_A_18
    print(f'  Gap (B-A) = {gap:.1f} pp')
    if gap > 20:
        print('  Substantial gap exists; magnitudes differ from predictions.')""")

# ═══════════════════════════════════════════════════════════════════════════
md("## 10 · Figure: BBH methodology gap (4 conditions)")

code("""import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)

conds  = ['A. lighteval\\n(LL, 18 tasks)',
          'A\\'. lm-eval\\n(gen no-CoT, 18)',
          'C. lm-eval\\n(CoT, 18 tasks)',
          'B. lm-eval\\n(CoT, 23 tasks)']
scores = [agg_A_18, agg_Ap_18, agg_C_18, agg_B_23]
colors = ['#C44E52', '#DD8452', '#8172B2', '#55A868']

bars = ax.bar(conds, scores, color=colors, edgecolor='black', linewidth=0.6, width=0.55)
for b, s in zip(bars, scores):
    ax.text(b.get_x() + b.get_width()/2, s + 1.2, f'{s:.1f}',
            ha='center', fontsize=10, fontweight='bold')

# Reference lines
ax.axhline(21.12, ls='--', lw=0.9, c='#C44E52', alpha=0.7)
ax.text(3.45, 22.2, 'OLv2 reported: 21.12', fontsize=7.5, color='#C44E52', ha='right')
ax.axhline(55.10, ls='--', lw=0.9, c='#55A868', alpha=0.7)
ax.text(3.45, 56.2, 'Gemma paper: 55.10', fontsize=7.5, color='#55A868', ha='right')

# Annotate gap decomposition
mid_LL = (agg_A_18 + agg_Ap_18) / 2
mid_CoT = (agg_Ap_18 + agg_C_18) / 2
ax.annotate('', xy=(1, agg_Ap_18+0.5), xytext=(0, agg_A_18+0.5),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=1.2))
ax.annotate('', xy=(2, agg_C_18+0.5), xytext=(1, agg_Ap_18+0.5),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=1.2))

ax.set_ylim(0, max(scores) * 1.25)
ax.set_ylabel('BBH accuracy (%)')
ax.set_title('gemma-7b on BBH: same model, same hardware, four methodologies', fontsize=11)
ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig(OUT_DIR / 'bbh_methodology_gap.pdf', bbox_inches='tight')
plt.savefig(OUT_DIR / 'bbh_methodology_gap.png', bbox_inches='tight')
plt.show()
print(f'Saved to {OUT_DIR}/bbh_methodology_gap.{{pdf,png}}')""")

# ═══════════════════════════════════════════════════════════════════════════
md("## 11 · Save results_E1.json")

code("""results_E1 = {
    'experiment': 'E1_bbh_lighteval_vs_suzgun',
    'model': MODEL_ID,
    'timestamp': RUN_TS,
    'environment': {
        'harness_version': HARNESS_VERSION,
        'lighteval_version': LIGHTEVAL_VERSION,
        'vllm_version': VLLM_VERSION,
        'transformers_version': TRANS_VERSION,
        'torch_version': TORCH_VERSION,
        'gpu': gpu_name,
        'seed': SEED,
        'temperature': TEMPERATURE,
        'n_shot': N_SHOT,
        'max_gen_toks': MAX_GEN_TOKS,
    },
    'task_prefixes': {
        'condition_A': 'lighteval extended|bbh:<task>|3|0',
        'condition_Ap': TASK_PREFIX_A_PRIME,
        'condition_B_C': TASK_PREFIX_B,
    },
    'scoring_modes': {
        'condition_A': 'loglikelihood (lighteval)',
        'condition_Ap': 'generate_until (lm-eval, no CoT)',
        'condition_B': 'generate_until (lm-eval, CoT)',
        'condition_C': 'generate_until (lm-eval, CoT)',
    },
    'aggregates': {
        'aggregate_A_18_loglikelihood': round(agg_A_18, 4),
        'aggregate_Ap_18_gen_noCot': round(agg_Ap_18, 4),
        'aggregate_B_23_gen_CoT': round(agg_B_23, 4),
        'aggregate_C_18_gen_CoT': round(agg_C_18, 4),
        'aggregate_B_on_18tasks': round(agg_B_on_18, 4),
    },
    'decomposition': {
        'total_gap_B_minus_A': round(agg_B_23 - agg_A_18, 4),
        'scoring_mode_effect_Ap_minus_A': round(agg_Ap_18 - agg_A_18, 4),
        'cot_effect_C_minus_Ap': round(agg_C_18 - agg_Ap_18, 4),
        'task_coverage_effect_B_minus_C': round(agg_B_23 - agg_C_18, 4),
        'sanity_C_minus_B_on_18': round(agg_C_18 - agg_B_on_18, 4),
    },
    'reference_scores': {
        'olv2_published': 21.12,
        'gemma_paper': 55.10,
        'mammoth2_paper': 57.40,
    },
    'per_task': {
        'condition_A': {t: round(v*100, 4) for t, v in A_scores.items()},
        'condition_Ap': {t: round(v*100, 4) for t, v in Ap_scores.items()},
        'condition_B': {t: round(v*100, 4) for t, v in B_scores.items()},
        'condition_C': {t: round(v*100, 4) for t, v in C_scores.items()},
    },
    'wall_time_minutes': {
        'condition_A': round(wall_A / 60, 1),
        'condition_Ap': round(wall_Ap / 60, 1),
        'condition_B': round(wall_B / 60, 1),
        'condition_C': round(wall_C / 60, 1),
    },
}

out_path = OUT_DIR / 'results_E1.json'
with open(out_path, 'w') as f:
    json.dump(results_E1, f, indent=2)

print(f'Saved: {out_path}')
print(json.dumps(results_E1['aggregates'], indent=2))
print(json.dumps(results_E1['decomposition'], indent=2))""")

# ═══════════════════════════════════════════════════════════════════════════
md("## 12 · Summary")

code("""# Determine outcome label
if 18 <= agg_A_18 <= 25 and 50 <= agg_B_23 <= 58:
    outcome = 'M1 CONFIRMED'
elif abs(agg_A_18 - agg_B_23) < 10:
    outcome = 'TASKS NOT DIFFERENTIATED'
elif agg_A_18 < 15:
    outcome = 'CONDITION A FAILURE'
elif agg_B_23 < 40:
    outcome = 'COT EXTRACTION FAILURE'
else:
    outcome = f'PARTIAL (gap = {agg_B_23 - agg_A_18:.1f} pp)'

summary_md = f\"\"\"# Experiment 1 -- BBH Controlled Experiment Summary

**Date:** {RUN_TS}
**Model:** {MODEL_ID}
**Outcome:** {outcome}

## Key Finding
lm-eval 0.4.5 uses generate_until for ALL BBH tasks -- there is no loglikelihood
variant. OLv2 uses lighteval, which does loglikelihood MC scoring. This experiment
uses lighteval for Condition A and lm-eval for the rest, enabling a 3-way
decomposition of the total gap.

## Environment
| Component | Version |
|---|---|
| lm-eval | {HARNESS_VERSION} |
| lighteval | {LIGHTEVAL_VERSION} |
| vLLM | {VLLM_VERSION} |
| transformers | {TRANS_VERSION} |
| GPU | {gpu_name} |

## Headline Numbers
| Condition | Harness | Method | Score (pp) |
|---|---|---|---|
| A  | lighteval | loglikelihood, no CoT, 18 tasks | **{agg_A_18:.2f}** |
| A' | lm-eval   | generation, no CoT, 18 tasks   | **{agg_Ap_18:.2f}** |
| C  | lm-eval   | generation, CoT, 18 tasks      | **{agg_C_18:.2f}** |
| B  | lm-eval   | generation, CoT, 23 tasks      | **{agg_B_23:.2f}** |

## Gap Decomposition (B - A = {agg_B_23 - agg_A_18:+.2f} pp)
- Scoring mode (LL vs gen, no CoT): A' - A = **{agg_Ap_18 - agg_A_18:+.2f} pp**
- CoT effect: C - A' = **{agg_C_18 - agg_Ap_18:+.2f} pp**
- Task coverage: B - C = **{agg_B_23 - agg_C_18:+.2f} pp**

## Reference Comparison
- OLv2 published: 21.12 pp -> Our A: {agg_A_18:.2f} pp (delta: {agg_A_18 - 21.12:+.2f})
- Gemma paper: 55.10 pp -> Our B: {agg_B_23:.2f} pp (delta: {agg_B_23 - 55.10:+.2f})

## Wall Time
- A (lighteval):  {wall_A/60:.1f} min
- A' (lm-eval):   {wall_Ap/60:.1f} min
- B (lm-eval):    {wall_B/60:.1f} min
- C (lm-eval):    {wall_C/60:.1f} min
- Total:          {(wall_A + wall_Ap + wall_B + wall_C)/60:.1f} min

## Files
- results_E1.json -- structured results
- bbh_methodology_gap.pdf -- paper figure
- condition_A_lighteval_loglikelihood/ -- lighteval raw output
- condition_Ap_lmeval_noncot/ -- lm-eval raw output + log_samples
- condition_B_suzgun_full/ -- lm-eval raw output + log_samples
- condition_C_suzgun_18tasks/ -- lm-eval raw output + log_samples
- requirements_E1.txt -- pinned dependencies
\"\"\"

with open(OUT_DIR / 'summary_E1.md', 'w') as f:
    f.write(summary_md)

import shutil
shutil.copy('/content/requirements_E1.txt', OUT_DIR / 'requirements_E1.txt')

print(summary_md)""")

code("""# Download results (Colab)
!cd /content && tar czf bbh_E1_results.tar.gz bbh_E1_results/
try:
    from google.colab import files
    files.download('/content/bbh_E1_results.tar.gz')
except ImportError:
    print('Not in Colab -- results at /content/bbh_E1_results/')""")


# ═══════════════════════════════════════════════════════════════════════════
# Build notebook
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.12"},
        "accelerator": "GPU",
        "gpuClass": "standard"
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = Path(__file__).parent / "E1_bbh_controlled_experiment.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Written {out_path} ({len(cells)} cells)")
