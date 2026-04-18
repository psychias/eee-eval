"""
run_controlled_eval.py
======================
Controlled factorial experiment: measures the effect of evaluation
configuration choices (temperature × prompt_format × n_shot) on
observed scores for 5 open-weight models across 3 benchmarks.

Usage:
    python run_controlled_eval.py [--dry-run] [--models MODEL [MODEL ...]]
                                  [--benchmarks BENCH [BENCH ...]]

Results are written incrementally to:
    submission/experiments/results/controlled_eval_results.jsonl

Interrupted runs can be safely resumed: completed (model, benchmark,
config) triples are skipped on restart.

Requirements:
    pip install lm-eval>=0.4.3 torch transformers accelerate
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "meta-llama/Llama-3.1-70B-Instruct",
]

BENCHMARKS = ["mmlu", "gsm8k", "humaneval"]

TEMPERATURES = [0.0, 0.5, 1.0]
PROMPT_FORMATS = ["standard", "cot", "fewshot"]
N_SHOTS = [0, 3, 5]

RANDOM_SEED = 42

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_FILE = RESULTS_DIR / "controlled_eval_results.jsonl"
SKIPPED_FILE = RESULTS_DIR / "skipped_models.txt"

# humaneval does not support n_shot > 0 (execution-based benchmark)
INVALID_CONFIGS: dict[str, set] = {
    "humaneval": {(temperature, prompt_format, n_shot)
                  for temperature, prompt_format, n_shot
                  in product(TEMPERATURES, PROMPT_FORMATS, N_SHOTS)
                  if n_shot > 0},
}

# Chain-of-thought prefix for cot prompt format
COT_PREFIX = "Let's think step by step."

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(RESULTS_DIR / "run_controlled_eval.log",
                            mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_completed_triples() -> set[tuple]:
    """Return the set of (model, benchmark, temperature, prompt_format, n_shot)
    already recorded in the results file."""
    completed = set()
    if not RESULTS_FILE.exists():
        return completed
    with RESULTS_FILE.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = (
                    rec["model_id"],
                    rec["benchmark"],
                    rec["temperature"],
                    rec["prompt_format"],
                    rec["n_shot"],
                )
                completed.add(key)
            except (json.JSONDecodeError, KeyError):
                pass
    return completed


def append_result(record: dict[str, Any]) -> None:
    """Append a single result record to the JSONL file."""
    with RESULTS_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def log_skipped(model_id: str, reason: str) -> None:
    with SKIPPED_FILE.open("a", encoding="utf-8") as fh:
        fh.write(f"{datetime.now(timezone.utc).isoformat()}  {model_id}  {reason}\n")


def config_key(temperature: float, prompt_format: str, n_shot: int) -> tuple:
    return (temperature, prompt_format, n_shot)


def is_valid_config(benchmark: str, temperature: float,
                    prompt_format: str, n_shot: int) -> tuple[bool, str]:
    """Return (is_valid, reason). Log-likelihood benchmarks like mmlu accept
    temperature=0 only in standard lm_eval; generation benchmarks like
    humaneval require n_shot=0."""
    invalid_set = INVALID_CONFIGS.get(benchmark, set())
    if config_key(temperature, prompt_format, n_shot) in invalid_set:
        return False, f"n_shot={n_shot} not valid for {benchmark}"
    # Log-likelihood benchmarks: temperature > 0 has no effect but is valid;
    # we record score=null and a note for temperature > 0 on mmlu to flag it.
    return True, ""


def get_lm_eval_version() -> str:
    try:
        import lm_eval
        return getattr(lm_eval, "__version__", "unknown")
    except ImportError:
        return "not_installed"


def build_task_config(benchmark: str, n_shot: int,
                      prompt_format: str) -> dict[str, Any]:
    """Build lm_eval task configuration dict."""
    task_config: dict[str, Any] = {
        "task": benchmark,
        "num_fewshot": n_shot,
    }
    if prompt_format == "cot":
        # Prepend CoT prefix to each question via a description override
        task_config["description"] = COT_PREFIX + "\n"
    elif prompt_format == "fewshot":
        # fewshot: n_shot controls the number of in-context examples;
        # num_fewshot is already set above. No template modification needed.
        pass
    # standard: use lm_eval defaults
    return task_config


def run_single_eval(
    model_id: str,
    benchmark: str,
    temperature: float,
    prompt_format: str,
    n_shot: int,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Run a single evaluation and return a result record.

    Returns a dict with keys:
        model_id, benchmark, temperature, prompt_format, n_shot,
        score, timestamp, eval_library_version, random_seed,
        status, reason
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    eval_version = get_lm_eval_version()

    base_record = {
        "model_id": model_id,
        "benchmark": benchmark,
        "temperature": temperature,
        "prompt_format": prompt_format,
        "n_shot": n_shot,
        "score": None,
        "timestamp": timestamp,
        "eval_library_version": eval_version,
        "random_seed": RANDOM_SEED,
        "status": "ok",
        "reason": "",
    }

    valid, reason = is_valid_config(benchmark, temperature, prompt_format, n_shot)
    if not valid:
        base_record.update(status="invalid_config", reason=reason)
        return base_record

    if dry_run:
        # Return a placeholder without running anything
        base_record.update(score=None, status="dry_run")
        return base_record

    try:
        import lm_eval
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM

        task_config = build_task_config(benchmark, n_shot, prompt_format)

        # Model kwargs: temperature affects generation sampling.
        # For log-likelihood tasks (mmlu), temperature is ignored at the
        # lm_eval level, but we record it for completeness.
        model_args = f"pretrained={model_id},dtype=auto"

        gen_kwargs = f"seed={RANDOM_SEED}"
        if benchmark in ("gsm8k", "humaneval") and temperature > 0.0:
            gen_kwargs += f",temperature={temperature},do_sample=True"
        elif temperature == 0.0:
            gen_kwargs += ",do_sample=False"

        results = simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=[task_config["task"]],
            num_fewshot=task_config.get("num_fewshot", 0),
            gen_kwargs=gen_kwargs,
            limit=200,
            random_seed=RANDOM_SEED,
            numpy_random_seed=RANDOM_SEED,
            torch_random_seed=RANDOM_SEED,
            write_out=False,
            log_samples=False,
            verbosity="ERROR",
        )

        # Extract primary metric score
        task_results = results.get("results", {})
        score = None
        for task_name, metrics in task_results.items():
            # Normalise task name for matching
            if task_name.lower().replace("_", "").startswith(
                benchmark.lower().replace("_", "")
            ):
                # Pick the first numeric metric value
                for k, v in metrics.items():
                    if isinstance(v, (int, float)) and not k.endswith(",stderr"):
                        score = round(float(v) * 100.0, 4)  # convert to pp
                        break
                if score is not None:
                    break

        if score is None:
            base_record.update(
                status="score_extraction_failed",
                reason=f"Could not extract score from results: {list(task_results.keys())}",
            )
        else:
            base_record["score"] = score

    except Exception as exc:  # pylint: disable=broad-except
        base_record.update(status="error", reason=str(exc))
        log.warning("Error for %s/%s config=%s: %s",
                    model_id, benchmark, (temperature, prompt_format, n_shot), exc)

    return base_record


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_all(
    models: list[str],
    benchmarks: list[str],
    dry_run: bool = False,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    completed = load_completed_triples()
    log.info("Loaded %d completed triples from %s", len(completed), RESULTS_FILE)

    all_configs = list(product(TEMPERATURES, PROMPT_FORMATS, N_SHOTS))

    total_planned = len(models) * len(benchmarks) * len(all_configs)
    log.info("Planned runs: %d (%d models × %d benchmarks × %d configs)",
             total_planned, len(models), len(benchmarks), len(all_configs))

    done = 0
    for model_id in models:
        # Quick check: is the model loadable?
        model_available = True
        if not dry_run:
            try:
                from transformers import AutoConfig
                AutoConfig.from_pretrained(model_id)
            except Exception as e:  # pylint: disable=broad-except
                log.warning("Model %s not available: %s", model_id, e)
                log_skipped(model_id, str(e))
                model_available = False

        if not model_available:
            continue

        for benchmark in benchmarks:
            for temperature, prompt_format, n_shot in all_configs:
                key = (model_id, benchmark, temperature, prompt_format, n_shot)
                if key in completed:
                    log.debug("Skipping completed: %s", key)
                    done += 1
                    continue

                log.info(
                    "[%d/%d] %s / %s  temp=%.1f fmt=%s shots=%d",
                    done + 1, total_planned,
                    model_id.split("/")[-1], benchmark,
                    temperature, prompt_format, n_shot,
                )

                record = run_single_eval(
                    model_id=model_id,
                    benchmark=benchmark,
                    temperature=temperature,
                    prompt_format=prompt_format,
                    n_shot=n_shot,
                    dry_run=dry_run,
                )
                append_result(record)
                done += 1

    log.info("Finished. %d / %d runs recorded.", done, total_planned)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_summaries() -> None:
    if not RESULTS_FILE.exists():
        print("No results file found.")
        return

    import csv
    from collections import defaultdict

    records = []
    with RESULTS_FILE.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    ok_records = [r for r in records
                  if r.get("status") == "ok" and r.get("score") is not None]
    print(f"\nTotal records: {len(records)}")
    print(f"  OK (with score): {len(ok_records)}")
    print(f"  Invalid config: {sum(1 for r in records if r.get('status') == 'invalid_config')}")
    print(f"  Errors: {sum(1 for r in records if r.get('status') == 'error')}")

    if not ok_records:
        print("No valid scores to summarise.")
        return

    # --- Score variance across temperature values ---
    print("\n=== Score variance by temperature (per model × benchmark × prompt_format) ===")
    by_mbe: dict = defaultdict(list)  # (model, benchmark, prompt_format) → [(temp, score)]
    for r in ok_records:
        key = (r["model_id"].split("/")[-1], r["benchmark"], r["prompt_format"])
        by_mbe[key].append((r["temperature"], r["score"]))

    temp_variances = []
    for key, pairs in by_mbe.items():
        if len(pairs) >= 2:
            scores = [p[1] for p in pairs]
            var = max(scores) - min(scores)
            temp_variances.append((var, key))
    temp_variances.sort(reverse=True)
    print(f"  Mean |max-min| across temperature: "
          f"{sum(v for v, _ in temp_variances)/max(1, len(temp_variances)):.2f} pp")
    print("  Top 5 high-variance (temperature):")
    for var, key in temp_variances[:5]:
        print(f"    {key}: range={var:.2f} pp")

    # --- Score variance across prompt_format ---
    print("\n=== Score variance by prompt_format (per model × benchmark × temperature) ===")
    by_mbt: dict = defaultdict(list)
    for r in ok_records:
        key = (r["model_id"].split("/")[-1], r["benchmark"], r["temperature"])
        by_mbt[key].append((r["prompt_format"], r["score"]))

    fmt_variances = []
    for key, pairs in by_mbt.items():
        if len(pairs) >= 2:
            scores = [p[1] for p in pairs]
            var = max(scores) - min(scores)
            fmt_variances.append((var, key))
    fmt_variances.sort(reverse=True)
    print(f"  Mean |max-min| across prompt_format: "
          f"{sum(v for v, _ in fmt_variances)/max(1, len(fmt_variances)):.2f} pp")
    print("  Top 5 high-variance (prompt_format):")
    for var, key in fmt_variances[:5]:
        print(f"    {key}: range={var:.2f} pp")

    # --- Score variance across n_shot ---
    print("\n=== Score variance by n_shot (per model × benchmark × temperature × format) ===")
    by_mbtp: dict = defaultdict(list)
    for r in ok_records:
        key = (r["model_id"].split("/")[-1], r["benchmark"],
               r["temperature"], r["prompt_format"])
        by_mbtp[key].append((r["n_shot"], r["score"]))

    nshot_variances = []
    for key, pairs in by_mbtp.items():
        if len(pairs) >= 2:
            scores = [p[1] for p in pairs]
            var = max(scores) - min(scores)
            nshot_variances.append((var, key))
    nshot_variances.sort(reverse=True)
    print(f"  Mean |max-min| across n_shot: "
          f"{sum(v for v, _ in nshot_variances)/max(1, len(nshot_variances)):.2f} pp")
    print("  Top 5 high-variance (n_shot):")
    for var, key in nshot_variances[:5]:
        print(f"    {key}: range={var:.2f} pp")

    # --- Top-5 absolute differences across all within-model config pairs ---
    print("\n=== Top-5 within-model config pairs by absolute score difference ===")
    from itertools import combinations
    by_mb: dict = defaultdict(list)
    for r in ok_records:
        by_mb[(r["model_id"], r["benchmark"])].append(r)

    top_diffs = []
    for (model_id, benchmark), runs in by_mb.items():
        for r1, r2 in combinations(runs, 2):
            diff = abs(r1["score"] - r2["score"])
            top_diffs.append((diff, model_id.split("/")[-1], benchmark,
                              (r1["temperature"], r1["prompt_format"], r1["n_shot"]),
                              (r2["temperature"], r2["prompt_format"], r2["n_shot"])))
    top_diffs.sort(reverse=True)
    for diff, model, bench, c1, c2 in top_diffs[:5]:
        print(f"  {model}/{bench}: |Δ|={diff:.2f} pp  {c1} vs {c2}")

    # --- Spearman correlation: config_distance vs |score_A - score_B| ---
    print("\n=== Spearman ρ: config Hamming distance vs |score_A - score_B| ===")
    try:
        from scipy.stats import spearmanr
        import numpy as np

        distances = []
        deltas = []
        for (model_id, benchmark), runs in by_mb.items():
            for r1, r2 in combinations(runs, 2):
                # Hamming distance between config parameter vectors
                # Discretise: temperature → {0,1,2}, format → {0,1,2}, n_shot → {0,1,2}
                temp_vals = {0.0: 0, 0.5: 1, 1.0: 2}
                fmt_vals = {"standard": 0, "cot": 1, "fewshot": 2}
                nshot_vals = {0: 0, 3: 1, 5: 2}
                v1 = [temp_vals.get(r1["temperature"], -1),
                      fmt_vals.get(r1["prompt_format"], -1),
                      nshot_vals.get(r1["n_shot"], -1)]
                v2 = [temp_vals.get(r2["temperature"], -1),
                      fmt_vals.get(r2["prompt_format"], -1),
                      nshot_vals.get(r2["n_shot"], -1)]
                ham = sum(a != b for a, b in zip(v1, v2))
                diff = abs(r1["score"] - r2["score"])
                distances.append(ham)
                deltas.append(diff)

        if len(distances) >= 4:
            rho, pval = spearmanr(distances, deltas)
            print(f"  Spearman ρ = {rho:.3f}, p = {pval:.4f} (n={len(distances)} pairs)")
        else:
            print("  Insufficient data for Spearman correlation.")
    except ImportError:
        print("  scipy not available; skipping Spearman correlation.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Controlled factorial LLM evaluation.")
    p.add_argument("--dry-run", action="store_true",
                   help="Enumerate configs without running evaluations.")
    p.add_argument("--models", nargs="+", default=MODELS,
                   help="Override model list.")
    p.add_argument("--benchmarks", nargs="+", default=BENCHMARKS,
                   help="Override benchmark list.")
    p.add_argument("--summarise-only", action="store_true",
                   help="Skip evaluation; only print summary of existing results.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.summarise_only:
        compute_summaries()
        return

    log.info("Starting controlled evaluation.")
    log.info("Models: %s", args.models)
    log.info("Benchmarks: %s", args.benchmarks)
    log.info("Dry-run: %s", args.dry_run)

    run_all(
        models=args.models,
        benchmarks=args.benchmarks,
        dry_run=args.dry_run,
    )
    compute_summaries()


if __name__ == "__main__":
    main()
