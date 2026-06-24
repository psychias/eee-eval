#!/usr/bin/env bash
# Phase 3: (1) last retry of the E3 7B/0.4.11/seed123 cell (flaky long job);
# (2) matched limit=100 BBH pair for a fair, connection-robust protocol gap
# (full CoT-gen kept dropping the Colab tunnel). leaderboard_bbh full (55.16)
# already landed in bbh_protocol_runs.jsonl as a bonus.
set -uo pipefail
cd /mnt/c/Users/z004knva/Desktop/Projects/eee-eval/experiments/harness_reeval/version_compare || exit 1
GPU=A100; RUN_TIMEOUT=14400; RESULTS_DIR=results
mkdir -p "$RESULTS_DIR"

run_one () {
  local ver="$1" model="$2" task="$3" nfew="$4" extra="$5" seed="$6" limit="$7" out="$8" tag="$9"
  local log="${RESULTS_DIR}/${tag}.log"
  echo "=== ${tag} ==="
  colab run --gpu "$GPU" --timeout "$RUN_TIMEOUT" run_lm_eval_version.py \
    --lm-eval-version "$ver" --model "$model" --task "$task" \
    --num-fewshot "$nfew" $extra --seed "$seed" --dtype auto --limit "$limit" \
    --output "remote_${tag}.jsonl" 2>&1 | tee "$log" \
    || echo "!! colab run failed for ${tag}"
  grep '^RESULT_JSON:' "$log" | tail -1 | sed 's/^RESULT_JSON://' \
    >> "${RESULTS_DIR}/${out}" || echo "!! no RESULT_JSON for ${tag}"
}

# 1) E3 cell, last attempt
run_one 0.4.11 "Qwen/Qwen2.5-7B-Instruct" gsm8k 5 "" 123 0 full_version_runs.jsonl \
  "Qwen_Qwen2.5-7B-Instruct_gsm8k_0.4.11_seed123_retry2"

# 2) matched limit=100 BBH pair (fair gap)
run_one 0.4.11 "Qwen/Qwen2.5-7B-Instruct" bbh_cot_fewshot 3 "--no-fewshot-override" 42 100 \
  bbh_protocol_lim100.jsonl "Qwen_Qwen2.5-7B_bbh_cot_fewshot_lim100"
run_one 0.4.11 "Qwen/Qwen2.5-7B-Instruct" leaderboard_bbh 3 "--no-fewshot-override" 42 100 \
  bbh_protocol_lim100.jsonl "Qwen_Qwen2.5-7B_leaderboard_bbh_lim100"

echo "PHASE3_DONE"
