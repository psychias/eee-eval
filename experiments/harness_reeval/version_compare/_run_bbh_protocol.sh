#!/usr/bin/env bash
# E6: BBH protocol reproduction. Same model, lm-eval 0.4.11, two scoring
# protocols under the identical "BBH" label:
#   arm A  bbh_cot_fewshot  -> 3-shot chain-of-thought GENERATION (papers' protocol)
#   arm B  leaderboard_bbh  -> 3-shot multiple-choice LOG-LIKELIHOOD (OLv2/lighteval)
# Reproduces in-house the ~35 pp cross-source BBH gap that opens the paper.
# Both are group tasks with built-in 3-shot, so --no-fewshot-override is set.
#
# Defaults: full set on Qwen2.5-7B. Override for a cheap smoke first, e.g.:
#   MODEL=Qwen/Qwen2.5-1.5B-Instruct LIMIT=4 OUTFILE=bbh_smoke_runs.jsonl \
#     RUN_TIMEOUT=3600 bash _run_bbh_protocol.sh
set -uo pipefail
cd /mnt/c/Users/z004knva/Desktop/Projects/eee-eval/experiments/harness_reeval/version_compare || exit 1

GPU="${GPU:-A100}"
VER="${VER:-0.4.11}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
LIMIT="${LIMIT:-0}"                 # 0 -> full BBH suite
SEED="${SEED:-42}"
RUN_TIMEOUT="${RUN_TIMEOUT:-10800}"
OUTFILE="${OUTFILE:-bbh_protocol_runs.jsonl}"
RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

ARMS=("bbh_cot_fewshot" "leaderboard_bbh")

for task in "${ARMS[@]}"; do
  tag="$(echo "$MODEL" | tr '/' '_')_${task}_${VER}_seed${SEED}_lim${LIMIT}"
  log="${RESULTS_DIR}/${tag}.log"
  echo "==================================================================="
  echo ">> ${VER} | ${MODEL} | ${task} | seed=${SEED} limit=${LIMIT} (gpu=${GPU})"
  echo "==================================================================="
  colab run --gpu "$GPU" --timeout "$RUN_TIMEOUT" run_lm_eval_version.py \
    --lm-eval-version "$VER" \
    --model "$MODEL" \
    --task "$task" \
    --num-fewshot 3 \
    --no-fewshot-override \
    --seed "$SEED" \
    --dtype auto \
    --limit "$LIMIT" \
    --output "remote_${tag}.jsonl" 2>&1 | tee "$log" \
    || echo "!! colab run pipeline failed for ${tag} (continuing)"
  grep '^RESULT_JSON:' "$log" | tail -1 | sed 's/^RESULT_JSON://' \
    >> "${RESULTS_DIR}/${OUTFILE}" || echo "!! no RESULT_JSON for ${tag}"
done
echo "BBH protocol arms done -> ${RESULTS_DIR}/${OUTFILE}"
