#!/usr/bin/env bash
# Orchestrate the lm-evaluation-harness VERSION comparison on Colab.
#
# Inspired by the TRL-on-Colab-CLI flow (hf.co/blog/sergiopaniego/trl-colab-cli),
# but instead of "QLoRA fine-tune with TRL" each job is "run lm_eval under one
# pinned harness version". The two versions cannot share a Python env, so every
# (version, model) pair gets its own FRESH ephemeral VM via `colab run`; the
# runner pip-installs its pinned version on that VM. Config is identical across
# versions -> any score delta is attributable to the harness version.
#
# Requires: google-colab-cli (Linux/macOS only), authenticated.
#   uv tool install google-colab-cli
#   colab auth --auth oauth2        # one-time interactive login
#
# Usage:  bash orchestrate.sh
set -euo pipefail
cd "$(dirname "$0")"

GPU="${GPU:-A100}"                     # you have credits -> A100, 7B in bf16
TASK="${TASK:-gsm8k}"
NUM_FEWSHOT="${NUM_FEWSHOT:-5}"
LIMIT="${LIMIT:-200}"                  # cap examples for cost; set to 0 for full GSM8K (1319)
SEEDS="${SEEDS:-42 123 1234}"          # multiple seeds for significance
RUN_TIMEOUT="${RUN_TIMEOUT:-7200}"     # colab run code-exec timeout (default 30s is far too low)

VERSIONS=("0.4.3" "0.4.11")
MODELS=(
  "Qwen/Qwen2.5-1.5B-Instruct"         # small
  "Qwen/Qwen2.5-7B-Instruct"           # mid (fp16/bf16 fits A100)
)

RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

# Always forward --limit so the runner gets the real value; runner treats
# limit<=0 as the FULL task set (omitting the flag would hit its 200 default).

for version in "${VERSIONS[@]}"; do
  for model in "${MODELS[@]}"; do
    for seed in $SEEDS; do
      tag="$(echo "$model" | tr '/' '_')_${TASK}_${version}_seed${seed}"
      log="${RESULTS_DIR}/${tag}.log"
      echo "==================================================================="
      echo ">> lm-eval ${version} | ${model} | ${TASK} | seed=${seed} (gpu=${GPU})"
      echo "==================================================================="
      # `colab run` ships run_lm_eval_version.py to a fresh A100 VM, forwards args,
      # streams stdout back. We capture the RESULT_JSON line the runner prints.
      # --dtype auto is REQUIRED: 0.4.11 rejects an explicit dtype= kwarg.
      colab run --gpu "$GPU" --timeout "$RUN_TIMEOUT" run_lm_eval_version.py \
        --lm-eval-version "$version" \
        --model "$model" \
        --task "$TASK" \
        --num-fewshot "$NUM_FEWSHOT" \
        --seed "$seed" \
        --dtype auto \
        --limit "$LIMIT" \
        --output "remote_${tag}.jsonl" 2>&1 | tee "$log" \
        || echo "!! colab run pipeline failed for ${tag} (continuing)"
      # reconstruct the local JSONL from the streamed stdout marker
      grep '^RESULT_JSON:' "$log" | tail -1 | sed 's/^RESULT_JSON://' \
        >> "${RESULTS_DIR}/full_version_runs.jsonl" || echo "!! no RESULT_JSON for ${tag}"
    done
  done
done

echo
echo "All jobs done. Combined -> ${RESULTS_DIR}/full_version_runs.jsonl"
echo "Now run:  python compare_versions.py ${RESULTS_DIR}/full_version_runs.jsonl"
