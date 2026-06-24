#!/usr/bin/env bash
# Re-run the three E3 0.4.11 cells lost to A100 503 (Service Unavailable):
#   1.5B/seed42, 1.5B/seed123, 7B/seed123. Appends to full_version_runs.jsonl.
set -uo pipefail
cd /mnt/c/Users/z004knva/Desktop/Projects/eee-eval/experiments/harness_reeval/version_compare || exit 1

GPU="${GPU:-A100}"
VER=0.4.11
TASK=gsm8k
NUM_FEWSHOT=5
LIMIT=0
RUN_TIMEOUT=10800
RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

# explicit (model|seed) gap list
GAPS=(
  "Qwen/Qwen2.5-1.5B-Instruct|42"
  "Qwen/Qwen2.5-1.5B-Instruct|123"
  "Qwen/Qwen2.5-7B-Instruct|123"
)

for entry in "${GAPS[@]}"; do
  model="${entry%%|*}"; seed="${entry##*|}"
  tag="$(echo "$model" | tr '/' '_')_${TASK}_${VER}_seed${seed}"
  log="${RESULTS_DIR}/${tag}.retry.log"
  echo "==================================================================="
  echo ">> RETRY ${VER} | ${model} | ${TASK} | seed=${seed} (gpu=${GPU})"
  echo "==================================================================="
  colab run --gpu "$GPU" --timeout "$RUN_TIMEOUT" run_lm_eval_version.py \
    --lm-eval-version "$VER" \
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
echo "E3 gap retries done -> ${RESULTS_DIR}/full_version_runs.jsonl"
