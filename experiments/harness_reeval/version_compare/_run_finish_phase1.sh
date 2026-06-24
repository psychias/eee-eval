#!/usr/bin/env bash
# Phase 1: finish E3 (missing 0.4.11/7B/seed123) + E6 BBH SMOKE (limit=4) to
# validate group-task score extraction before the full BBH runs.
set -uo pipefail
cd /mnt/c/Users/z004knva/Desktop/Projects/eee-eval/experiments/harness_reeval/version_compare || exit 1
GPU=A100; RUN_TIMEOUT=10800; RESULTS_DIR=results
mkdir -p "$RESULTS_DIR"

run_one () { # ver model task nfew extraflag seed limit outfile tag
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

# 1) E3 missing cell
run_one 0.4.11 "Qwen/Qwen2.5-7B-Instruct" gsm8k 5 "" 123 0 full_version_runs.jsonl \
  "Qwen_Qwen2.5-7B-Instruct_gsm8k_0.4.11_seed123"

# 2) E6 smoke (1.5B, limit 4, both BBH arms)
run_one 0.4.11 "Qwen/Qwen2.5-1.5B-Instruct" bbh_cot_fewshot 3 "--no-fewshot-override" 42 4 \
  bbh_smoke_runs.jsonl "Qwen_Qwen2.5-1.5B_bbh_cot_fewshot_smoke"
run_one 0.4.11 "Qwen/Qwen2.5-1.5B-Instruct" leaderboard_bbh 3 "--no-fewshot-override" 42 4 \
  bbh_smoke_runs.jsonl "Qwen_Qwen2.5-1.5B_leaderboard_bbh_smoke"

echo "PHASE1_DONE"
