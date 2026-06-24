#!/usr/bin/env python3
"""Remote eval runner for the lm-evaluation-harness *version* comparison.

Runs on a fresh Colab runtime via `colab run`. It is fully self-contained:
it pip-installs ONE pinned lm-eval version, runs a single (model, task)
evaluation with the `hf` backend, then emits a normalized JSONL record that
matches the schema of ../negative_control_version.jsonl
(model_id, benchmark, lm_eval_version, n_shot, score, status, ...).

One `colab run` == one (version, model, task) cell. The whole point of the
experiment is that CONFIG IS IDENTICAL across the two versions, so every knob
(num_fewshot, limit, seed, dtype) is an explicit forwarded argument rather than
a per-version default.

Example (executed by orchestrate.sh on the remote VM):
    python run_lm_eval_version.py \
        --lm-eval-version 0.4.3 \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --task gsm8k --num-fewshot 5 --limit 200 --seed 42 \
        --output results/qwen1.5b_gsm8k_0.4.3.jsonl
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import traceback


# Identical dependency stack for BOTH lm-eval versions, so the only variable in
# the comparison is the harness version. These are the mid-2024 versions lm-eval
# 0.4.3 shipped against; crucially `huggingface_hub` here still accepts bare
# dataset repo ids like `gsm8k` (newer hf_hub rejects them with HfUriError), and
# transformers 4.44 supports Qwen2.5.
PINNED_DEPS = [
    "numpy==1.26.4",          # transformers 4.44 needs numpy<2
    "transformers==4.44.2",
    "datasets==2.20.0",
    "huggingface_hub==0.23.5",
    "accelerate==0.33.0",
    "peft==0.11.1",           # contemporaneous; new peft imports symbols absent in old hf_hub
]


# lm-eval versions old enough to need the mid-2024 stack (bare-name gsm8k via old
# huggingface_hub, transformers<4.5x). Newer versions require the modern
# transformers `dtype` API and run on their own resolved deps -- the two eras are
# mutually exclusive, so dep pinning is per-version.
OLD_STACK_VERSIONS = {"0.4.0", "0.4.1", "0.4.2", "0.4.3", "0.4.4"}


def pip_install_lm_eval(version: str) -> None:
    """Install one lm-eval version with the dependency stack it actually runs on."""
    # "latest" installs the current release (used for E4 quantisation, where the
    # harness version is not the variable but a modern lm-eval is needed to build
    # a BitsAndBytesConfig compatible with current transformers).
    spec = "lm-eval" if version == "latest" else f"lm-eval=={version}"
    print(f"[runner] installing {spec} ...", flush=True)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", spec]
    )
    if version in OLD_STACK_VERSIONS:
        print(f"[runner] OLD stack pin: {PINNED_DEPS}", flush=True)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", *PINNED_DEPS]
        )
    else:
        # native: keep lm-eval's own resolved deps; just ensure the hf backend dep
        print("[runner] NATIVE stack (lm-eval's resolved deps; +accelerate)",
              flush=True)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "accelerate"]
        )


def run_eval(args) -> dict:
    """Invoke the lm_eval CLI and return the raw results dict."""
    out_dir = os.path.join("_lm_eval_raw", args.lm_eval_version,
                           args.model.replace("/", "__"), args.task)
    os.makedirs(out_dir, exist_ok=True)

    # Only pass an explicit dtype when asked. lm-eval 0.4.11 forwards a literal
    # `dtype=` kwarg to the HF model ctor, which the pinned (old) transformers
    # rejects; omitting it lets each harness version use its own default dtype
    # handling ("auto" -> the model config's dtype), keeping behavior comparable.
    model_args = f"pretrained={args.model}"
    if args.dtype and args.dtype.lower() not in ("auto", "none", ""):
        model_args += f",dtype={args.dtype}"
    if args.quant == "4bit":
        model_args += ",load_in_4bit=True"
    elif args.quant == "8bit":
        model_args += ",load_in_8bit=True"
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", args.task,
        "--batch_size", args.batch_size,
        "--seed", str(args.seed),
        "--output_path", out_dir,
    ]
    # Group tasks like leaderboard_bbh / bbh_cot_fewshot carry a fixed built-in
    # num_fewshot; overriding it errors. --no-fewshot-override lets the task use
    # its native shot count (so both BBH arms stay at their canonical 3-shot).
    if not args.no_fewshot_override:
        cmd += ["--num_fewshot", str(args.num_fewshot)]
    if args.limit is not None and args.limit > 0:
        cmd += ["--limit", str(args.limit)]  # limit<=0 -> full task set
    print(f"[runner] {' '.join(cmd)}", flush=True)
    # capture lm_eval's own output so a failure is diagnosable in the record
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True)
    print(proc.stdout, flush=True)
    if proc.returncode != 0:
        tail = (proc.stdout or "")[-2500:]
        raise RuntimeError(f"lm_eval exited {proc.returncode}. Output tail:\n{tail}")

    # lm_eval writes results_*.json into a timestamped subdir under out_dir
    result_path = _find_results_json(out_dir)
    with open(result_path) as fh:
        return json.load(fh)


def _find_results_json(out_dir: str) -> str:
    for root, _dirs, files in os.walk(out_dir):
        for fn in files:
            if fn.startswith("results") and fn.endswith(".json"):
                return os.path.join(root, fn)
    raise FileNotFoundError(f"no results*.json under {out_dir}")


def extract_score(raw: dict, task: str):
    """Pull the primary accuracy metric for `task` out of lm_eval output.

    Returns (score_percent, metric_name). lm_eval reports fractions (0-1);
    we convert to percent to match the repo's existing records.
    """
    results = raw.get("results", {})
    # For group tasks (bbh_cot_fewshot, leaderboard_bbh) the aggregate row lives
    # under the group key, sometimes only in raw["groups"]. Locate it robustly:
    # exact task key first, then group section, then any key containing the task
    # name, then the first results entry as a last resort.
    node = (results.get(task)
            or raw.get("groups", {}).get(task)
            or next((v for k, v in results.items() if task in k), None)
            or next(iter(results.values()), {}))
    # Prefer exact-match / acc style metrics; take the first non-stderr float.
    preferred = ["exact_match,strict-match", "exact_match,flexible-extract",
                 "exact_match,get-answer", "exact_match,none", "exact_match",
                 "acc_norm,none", "acc,none", "acc_norm", "acc"]
    for key in preferred:
        if key in node and isinstance(node[key], (int, float)):
            return round(float(node[key]) * 100, 4), key
    for key, val in node.items():
        if isinstance(val, (int, float)) and "stderr" not in key:
            return round(float(val) * 100, 4), key
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm-eval-version", required=True)
    ap.add_argument("--model", required=True, help="HF model id")
    ap.add_argument("--task", default="gsm8k")
    ap.add_argument("--num-fewshot", type=int, default=5)
    ap.add_argument("--limit", type=int, default=200,
                    help="cap on eval examples; None for full set")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--quant", default="none", choices=["none", "4bit", "8bit"],
                    help="weight quantisation (E4); installs bitsandbytes when set")
    ap.add_argument("--batch-size", default="auto")
    ap.add_argument("--transformers-pin", default="",
                    help="pin transformers to this version after lm-eval install "
                         "(E4 quant needs a version that still accepts load_in_4bit; "
                         "newer transformers raise TypeError on that kwarg)")
    ap.add_argument("--no-fewshot-override", action="store_true",
                    help="don't pass --num_fewshot (let group tasks like "
                         "leaderboard_bbh/bbh_cot_fewshot use their built-in shots)")
    ap.add_argument("--output", required=True, help="JSONL path to append to")
    args = ap.parse_args()

    record = {
        "model_id": args.model,
        "benchmark": args.task,
        "lm_eval_version": args.lm_eval_version,
        "n_shot": args.num_fewshot,
        "limit": args.limit,
        "seed": args.seed,
        "dtype": args.dtype,
        "quant": args.quant,
        "backend": "hf",
        "score": None,
        "metric": None,
        "status": "error",
        "reason": None,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    try:
        pip_install_lm_eval(args.lm_eval_version)
        if args.transformers_pin:
            print(f"[runner] pinning transformers=={args.transformers_pin}", flush=True)
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet",
                 f"transformers=={args.transformers_pin}"]
            )
        if args.quant in ("4bit", "8bit"):
            print(f"[runner] installing bitsandbytes for {args.quant}", flush=True)
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet", "bitsandbytes"]
            )
        raw = run_eval(args)
        score, metric = extract_score(raw, args.task)
        record["score"] = score
        record["metric"] = metric
        record["status"] = "ok" if score is not None else "no_score"
    except Exception as exc:  # noqa: BLE001 - record the failure, don't crash the batch
        record["reason"] = f"{exc}\n{traceback.format_exc()}"
        print(f"[runner] FAILED: {exc}", file=sys.stderr, flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "a") as fh:
        fh.write(json.dumps(record) + "\n")
    # `colab run` streams stdout back but does not reliably retrieve files, so
    # emit the record on stdout behind a marker the orchestrator greps for.
    print("RESULT_JSON:" + json.dumps(record), flush=True)
    print(f"[runner] done (status={record['status']}, score={record['score']})",
          flush=True)


if __name__ == "__main__":
    main()
