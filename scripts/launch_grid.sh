#!/usr/bin/env bash
set -euo pipefail

CONFIGS_JSONL="${1:-configs.jsonl}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"

python -m finetune.sweep.run_grid \
  --configs_jsonl "${CONFIGS_JSONL}" \
  --num_processes "${NUM_PROCESSES}" \
  --skip_done

