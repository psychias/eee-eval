#!/usr/bin/env bash
# E4: quantisation sensitivity. Harness + model + task + n-shot + seeds held
# fixed; only weight quantisation varies. The bf16 baseline is E3's
# 0.4.11 / Qwen2.5-7B full-GSM8K runs, so E4 only needs 4-bit and 8-bit.
# Backs adding a `software_environment`/`quantisation` field to EvalSpec.
set -uo pipefail
cd "$(dirname "$0")"

GPU="${GPU:-L4}"                       # L4 cheaper than A100, fits 7B bf16/quant
TASK="${TASK:-gsm8k}"
NUM_FEWSHOT="${NUM_FEWSHOT:-5}"
LIMIT="${LIMIT:-0}"                    # 0 -> full GSM8K
SEEDS="${SEEDS:-42 123 1234}"
RUN_TIMEOUT="${RUN_TIMEOUT:-7200}"
VER="${VER:-0.4.11}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
QUANTS="${QUANTS:-4bit 8bit}"

RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

for quant in $QUANTS; do
  for seed in $SEEDS; do
    tag="$(echo "$MODEL" | tr '/' '_')_${TASK}_${VER}_${quant}_seed${seed}"
    log="${RESULTS_DIR}/${tag}.log"
    echo "==================================================================="
    echo ">> E4 ${VER} | ${MODEL} | quant=${quant} | seed=${seed} (gpu=${GPU})"
    echo "==================================================================="
    colab run --gpu "$GPU" --timeout "$RUN_TIMEOUT" run_lm_eval_version.py \
      --lm-eval-version "$VER" \
      --model "$MODEL" \
      --task "$TASK" \
      --num-fewshot "$NUM_FEWSHOT" \
      --seed "$seed" \
      --dtype auto \
      --quant "$quant" \
      --limit "$LIMIT" \
      --output "remote_${tag}.jsonl" 2>&1 | tee "$log" \
      || echo "!! colab run pipeline failed for ${tag} (continuing)"
    grep '^RESULT_JSON:' "$log" | tail -1 | sed 's/^RESULT_JSON://' \
      >> "${RESULTS_DIR}/e4_runs.jsonl" || echo "!! no RESULT_JSON for ${tag}"
  done
done
echo "E4 done -> ${RESULTS_DIR}/e4_runs.jsonl"
