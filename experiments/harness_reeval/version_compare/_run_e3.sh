#!/usr/bin/env bash
# Launcher for the full E3 sweep (staged file to avoid MSYS var-path mangling).
# Invoked from WSL with a literal path: bash /mnt/c/.../_run_e3.sh
cd /mnt/c/Users/z004knva/Desktop/Projects/eee-eval/experiments/harness_reeval/version_compare || exit 1
export GPU=A100
export LIMIT=0
export RUN_TIMEOUT=10800
export SEEDS="42 123 1234"
export TASK=gsm8k
export NUM_FEWSHOT=5
exec bash orchestrate.sh
