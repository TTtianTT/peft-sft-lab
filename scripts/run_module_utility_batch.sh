#!/usr/bin/env bash
set -euo pipefail

PYTHON=/dataset1/zailong/envs/peft-sft-lab/bin/python
ROOT=/dataset1/zailong/runs/peft-sft-lab/hns-module-utility-20260909
export PYTHONPATH="$PWD/src"

"$PYTHON" scripts/analyze_functional_localization_inference.py \
  --run_root /dataset1/zailong/runs/peft-sft-lab/hns-functional-localization-20260909 \
  --output_dir /dataset1/zailong/runs/peft-sft-lab/hns-functional-localization-20260909 \
  --bootstrap_samples 10000 --permutation_samples 100000 --seed 42

bash scripts/run_module_utility_case.sh qwen_magicoder
bash scripts/run_module_utility_case.sh qwen_commonsense
bash scripts/run_module_utility_case.sh llama_tulu

"$PYTHON" scripts/analyze_functional_localization_inference.py \
  --run_root /dataset1/zailong/runs/peft-sft-lab/hns-functional-localization-20260909 \
  --output_dir /dataset1/zailong/runs/peft-sft-lab/hns-functional-localization-20260909/inference \
  --bootstrap_samples 10000 --permutation_samples 100000 --seed 42

"$PYTHON" scripts/analyze_module_utility.py \
  --aggregate_root /dataset1/zailong/runs/peft-sft-lab/hns-2x4-allmodules-20260909 \
  --utility_root "$ROOT/utility" --gradient_root "$ROOT/gradient" \
  --output_dir "$ROOT/analysis" --permutations 10000 --seed 42

"$PYTHON" scripts/analyze_hns_paired_inference.py \
  --aggregate_root /dataset1/zailong/runs/peft-sft-lab/hns-2x4-allmodules-20260909 \
  --first_batch_root /dataset1/zailong/runs/peft-sft-lab/hns-mechanism-first-batch-20260909 \
  --output_dir /dataset1/zailong/runs/peft-sft-lab/hns-mechanism-first-batch-20260909/statistics \
  --bootstrap_samples 10000 --permutation_samples 100000 --seed 42

echo "[Done] module utility batch and adjusted paired inference"
