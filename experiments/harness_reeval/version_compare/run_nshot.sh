#!/usr/bin/env bash
# N-shot experiment: directly reproduce the §5.2 Belebele case. Model, task,
# harness, format, temperature, scoring mode all held fixed; only n-shot varies.
# Demonstrates that n_shot alone drives a large swing under one benchmark label
# (the case study currently borrows Falcon2's internal 0->5-shot numbers).
# Backs the `n_shot` Required EvalSpec field with a controlled experiment.
set -uo pipefail
cd "$(dirname "$0")"

GPU="${GPU:-L4}"
VER="${VER:-0.4.11}"
MODEL="${MODEL:-mistralai/Mistral-7B-v0.1}"   # BASE model (matches §5.2 era; instruct saturates 0-shot)
TASK="${TASK:-belebele_eng_Latn}"      # English Belebele (multiple-choice, log-likelihood)
LIMIT="${LIMIT:-0}"                     # 0 -> full (900 questions)
SEEDS="${SEEDS:-42 123 1234}"
NSHOTS="${NSHOTS:-0 1 5}"
RUN_TIMEOUT="${RUN_TIMEOUT:-7200}"

RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

for ns in $NSHOTS; do
  for seed in $SEEDS; do
    tag="$(echo "$MODEL" | tr '/' '_')_${TASK}_${VER}_ns${ns}_seed${seed}"
    log="${RESULTS_DIR}/${tag}.log"
    echo "==================================================================="
    echo ">> N-SHOT ${VER} | ${MODEL} | ${TASK} | n_shot=${ns} | seed=${seed} (gpu=${GPU})"
    echo "==================================================================="
    colab run --gpu "$GPU" --timeout "$RUN_TIMEOUT" run_lm_eval_version.py \
      --lm-eval-version "$VER" \
      --model "$MODEL" \
      --task "$TASK" \
      --num-fewshot "$ns" \
      --seed "$seed" \
      --dtype auto \
      --limit "$LIMIT" \
      --output "remote_${tag}.jsonl" 2>&1 | tee "$log" \
      || echo "!! colab run pipeline failed for ${tag} (continuing)"
    grep '^RESULT_JSON:' "$log" | tail -1 | sed 's/^RESULT_JSON://' \
      >> "${RESULTS_DIR}/nshot_runs.jsonl" || echo "!! no RESULT_JSON for ${tag}"
  done
done
echo "N-shot done -> ${RESULTS_DIR}/nshot_runs.jsonl"
