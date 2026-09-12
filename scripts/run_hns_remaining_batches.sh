#!/usr/bin/env bash
set -euo pipefail

PYTHON=/dataset1/zailong/envs/peft-sft-lab/bin/python
FIRST=/dataset1/zailong/runs/peft-sft-lab/hns-mechanism-first-batch-20260909
AGGREGATE=/dataset1/zailong/runs/peft-sft-lab/hns-2x4-allmodules-20260909

export PYTHONPATH="$PWD/src"

"$PYTHON" scripts/analyze_hns_paired_inference.py \
  --aggregate_root "$AGGREGATE" --first_batch_root "$FIRST" \
  --output_dir "$FIRST/statistics" --bootstrap_samples 10000 \
  --permutation_samples 100000 --seed 42

bash scripts/run_functional_localization_case.sh qwen_magicoder
bash scripts/run_functional_localization_case.sh qwen_commonsense
bash scripts/run_functional_localization_case.sh llama_tulu

echo "[Done] first-batch statistics and three functional-localization cases"
