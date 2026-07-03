"""Ready-to-use EvalSpec emission hook for the EleutherAI lm-evaluation-harness.

EvalSpec's premise is that producers already hold the comparability-critical
metadata at evaluation time; the barrier is emitting it, not computing it. This
hook closes that gap for the most common open harness: given the objects
lm-eval-harness already produces at the end of a run (`results`, the task config,
and the model args), it emits an EvalSpec-Required (EvalSpec-R) record per
model x benchmark cell, with the scoring mode read from the task's output type and
the prompt template recorded as a canonical SHA-256 hash.

Usage (drop-in at the end of a harness run):

    from lm_eval import simple_evaluate
    from emission_hooks.lm_eval_to_evalspec import emit_evalspec

    out = simple_evaluate(model=lm, tasks=["gsm8k"], num_fewshot=5, ...)
    records = emit_evalspec(out, model_args={"pretrained": "meta-llama/Llama-3.1-8B"})
    # records -> list of EvalSpec-R dicts, one per task, ready to serialize as JSON-LD

The same three-function shape (`_scoring_mode`, `_prompt_hash`, `emit_evalspec`)
ports to HELM and Inspect by swapping the field-extraction accessors; see the
`src/converters/` adapters for the per-framework field maps.
"""
from __future__ import annotations

import hashlib
from typing import Any


# lm-eval-harness task `output_type` -> EvalSpec controlled-vocabulary scoring_mode.
_OUTPUT_TYPE_TO_SCORING_MODE = {
    "loglikelihood": "log-likelihood",
    "multiple_choice": "log-likelihood",
    "loglikelihood_rolling": "log-likelihood",
    "generate_until": "generation",
}


def _scoring_mode(task_config: dict[str, Any]) -> str:
    """Map the harness output type to the EvalSpec scoring_mode vocabulary.

    This is the field no leaderboard API exposes and that separates the
    log-likelihood-vs-generation collisions in the paper; here it is read
    directly from the harness config rather than inferred."""
    ot = (task_config or {}).get("output_type", "")
    return _OUTPUT_TYPE_TO_SCORING_MODE.get(ot, "unknown")


def _prompt_hash(task_config: dict[str, Any]) -> str | None:
    """SHA-256 of the canonicalized prompt template, or None if unavailable.

    Canonicalization strips leading/trailing whitespace so cosmetic differences
    do not produce spurious template mismatches across runs."""
    tmpl = (task_config or {}).get("doc_to_text")
    if not isinstance(tmpl, str) or not tmpl:
        return None
    canonical = "\n".join(line.rstrip() for line in tmpl.strip().splitlines())
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def emit_evalspec(harness_output: dict[str, Any],
                  model_args: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Emit one EvalSpec-R record per task from an lm-eval-harness result object.

    `harness_output` is the dict returned by `lm_eval.simple_evaluate` (it carries
    `results`, `configs`, `config`, and `versions`). `model_args` is the model
    specification (e.g. `{"pretrained": ...}`); the model id and dtype are read
    from it. Missing values are emitted as the literal "unknown" so absence is
    recorded rather than silently inferred, per the paper's schema convention."""
    model_args = model_args or {}
    run_cfg = harness_output.get("config", {}) or {}
    configs = harness_output.get("configs", {}) or {}
    versions = harness_output.get("versions", {}) or {}
    results = harness_output.get("results", {}) or {}

    model_id = model_args.get("pretrained") or run_cfg.get("model") or "unknown"
    records: list[dict[str, Any]] = []
    for task_name, metrics in results.items():
        tcfg = configs.get(task_name, {})
        metric_key = next((k for k in metrics if "," in k), None)
        record = {
            # EvalSpec-R required fields (EEE paths in comments).
            "model_id": model_id,                                   # model_info
            "benchmark": task_name,                                 # evaluation_name
            "scoring_mode": _scoring_mode(tcfg),                    # generation_config
            "harness_name": "lm-evaluation-harness",                # eval_library
            "harness_version": run_cfg.get("lm_eval_version", "unknown"),
            "n_shot": run_cfg.get("num_fewshot", tcfg.get("num_fewshot", "unknown")),
            "random_seed": run_cfg.get("random_seed", "unknown"),
            "evaluation_timestamp": run_cfg.get("start_time", "unknown"),
            # Recommended-tier extras that lm-eval already knows.
            "prompt_template_hash": _prompt_hash(tcfg) or "unknown",
            "temperature": (run_cfg.get("gen_kwargs") or {}).get("temperature", "unknown"),
            "task_version": versions.get(task_name, "unknown"),
            "metric": metric_key or "unknown",
            "score": metrics.get(metric_key) if metric_key else None,
            # Provenance tag: every value here is source-reported (returned by the
            # harness), not inferred -- the distinction the paper makes first-class.
            "evidence_source": "source-reported",
        }
        records.append(record)
    return records


if __name__ == "__main__":  # tiny self-test on a synthetic harness output
    demo = {
        "config": {"num_fewshot": 5, "lm_eval_version": "0.4.11", "random_seed": 42},
        "configs": {"gsm8k": {"output_type": "generate_until", "doc_to_text": "Q: {{question}}\nA:"}},
        "versions": {"gsm8k": 3.0},
        "results": {"gsm8k": {"exact_match,strict-match": 0.585}},
    }
    import json
    print(json.dumps(emit_evalspec(demo, {"pretrained": "Qwen/Qwen2.5-14B-Instruct"}), indent=2))
