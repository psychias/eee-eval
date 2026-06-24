#!/usr/bin/env bash
# Remaining E3 jobs only: lm-eval 0.4.11, both models, seeds 42+123 (2-seed cut).
# 0.4.3 / {1.5B,7B} / {42,123} already completed and are in full_version_runs.jsonl.
# Mirrors orchestrate.sh's colab run block; appends to the same results file.
set -uo pipefail
cd /mnt/c/Users/z004knva/Desktop/Projects/eee-eval/experiments/harness_reeval/version_compare || exit 1

GPU=A100
TASK=gsm8k
NUM_FEWSHOT=5
LIMIT=0
RUN_TIMEOUT=10800
VERSION=0.4.11
SEEDS="42 123"
MODELS=("Qwen/Qwen2.5-1.5B-Instruct" "Qwen/Qwen2.5-7B-Instruct")
RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

for model in "${MODELS[@]}"; do
  for seed in $SEEDS; do
    tag="$(echo "$model" | tr '/' '_')_${TASK}_${VERSION}_seed${seed}"
    log="${RESULTS_DIR}/${tag}.log"
    echo "==================================================================="
    echo ">> lm-eval ${VERSION} | ${model} | ${TASK} | seed=${seed} (gpu=${GPU})"
    echo "==================================================================="
    colab run --gpu "$GPU" --timeout "$RUN_TIMEOUT" run_lm_eval_version.py \
      --lm-eval-version "$VERSION" \
      --model "$model" \
      --task "$TASK" \
      --num-fewshot "$NUM_FEWSHOT" \
      --seed "$seed" \
      --dtype auto \
      --limit "$LIMIT" \
      --output "remote_${tag}.jsonl" 2>&1 | tee "$log" \
      || echo "!! colab run pipeline failed for ${tag} (continuing)"
    grep '^RESULT_JSON:' "$log" | tail -1 | sed 's/^RESULT_JSON://' \
      >> "${RESULTS_DIR}/full_version_runs.jsonl" || echo "!! no RESULT_JSON for ${tag}"
  done
done
echo "E3 0.4.11 remainder done -> ${RESULTS_DIR}/full_version_runs.jsonl"
