#!/usr/bin/env bash
set -euo pipefail

bash scripts/run_llama31_cross_task_mechanism.sh
bash scripts/run_qwen3_magicoder_functional_controls.sh
bash scripts/run_qwen3_commonsense_mechanism.sh
