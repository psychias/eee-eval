"""Reproduce OLv2's Gemma-2-9B BBH score (34.0968) under its exact config, to test
whether 34.10 is a deterministic fingerprint of the OLv2 pipeline (SEA-LION provenance
check, Case Study 3). OLv2 evaluated google/gemma-2-9b @ beb0c08e in bfloat16 on the
`leaderboard_bbh` task group (3-shot multiple-choice log-likelihood, acc_norm).

Determinism note: leaderboard_bbh is fixed-few-shot MC log-likelihood, so an exact
reproduction of 34.0968 confirms 34.10 is a reproducible fingerprint of this specific
(model_sha, dtype, harness, task) tuple -- i.e. SEA-LION's matching 34.1 reflects the
same OLv2 pipeline, not an independent measurement. It does NOT by itself separate
"imported the number" from "re-ran the identical pipeline" (both yield 34.0968).

Also runs the CoT-generation protocol arm (bbh_cot_fewshot) to re-confirm the ~34->~68
protocol swing: hitting exactly 34.10 requires OLv2's MC-log-likelihood protocol, so an
independent evaluator choosing a different (equally valid) BBH protocol would not.

Needs: google/gemma-2-9b license accepted on the token's HF account; set HF_TOKEN.
One Colab A100 session."""
import os
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
_TOKEN = os.environ.get("HF_TOKEN", "")
if _TOKEN:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = _TOKEN

import subprocess, sys, json, time
from datetime import datetime, timezone

MODEL = "google/gemma-2-9b"
REVISION = "beb0c08e9eeb0548f3aca2ac870792825c357b7d"  # exact SHA OLv2 evaluated
# (task, EvalSpec label, num_fewshot)  -- OLv2 BBH = leaderboard_bbh, 3-shot MC log-likelihood
ARMS = [("leaderboard_bbh", "BBH-OLv2-protocol (MC log-lik)", 3),
        ("bbh_cot_fewshot", "BBH-CoT-generation", 3)]

print("[setup] installing lm-eval==0.4.11 ...", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lm-eval==0.4.11", "accelerate"], check=True)

if _TOKEN:
    from huggingface_hub import login
    login(token=_TOKEN, add_to_git_credential=False)

import lm_eval
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM


def extract(results, task):
    m = results["results"].get(task, {})
    for key in ("acc_norm,none", "acc,none", "exact_match,none"):
        if key in m:
            return round(float(m[key]) * 100.0, 4)
    # group tasks aggregate under the group name; fall back to any *_norm mean
    return None


lm = HFLM(pretrained=MODEL, revision=REVISION, dtype="bfloat16",
          batch_size="auto", max_batch_size=8)
for task, label, nshot in ARMS:
    rec = dict(model_id=MODEL, revision=REVISION, benchmark="BBH", task_name=task,
               protocol=label, n_shot=nshot, backend="hf",
               eval_library_version=lm_eval.__version__,
               timestamp=datetime.now(timezone.utc).isoformat(), score=None)
    try:
        r = simple_evaluate(model=lm, tasks=[task], num_fewshot=nshot,
                            random_seed=42, numpy_random_seed=42, torch_random_seed=42,
                            fewshot_random_seed=42, write_out=False, log_samples=False,
                            verbosity="ERROR")
        rec["score"] = extract(r, task)
        rec["all_metrics"] = {k: v for k, v in r["results"].get(task, {}).items()
                              if isinstance(v, (int, float))}
        rec["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        rec["status"] = "error"; rec["reason"] = str(exc)[:200]
    print("RESULT " + json.dumps(rec), flush=True)
print("[done] OLv2 raw BBH for google/gemma-2-9b @ beb0c08e was 34.0968; "
      "reproduce under leaderboard_bbh, compare to the CoT-generation arm.", flush=True)
