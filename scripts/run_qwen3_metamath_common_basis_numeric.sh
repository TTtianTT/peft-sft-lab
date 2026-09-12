#!/usr/bin/env bash
set -euo pipefail

PYTHON=/dataset1/zailong/envs/peft-sft-lab/bin/python
BASE=/dataset1/zailong/models/Qwen3-8B
LORA=/dataset1/zailong/models/spectral-surgery/Qwen3-8B-MetaMathQA-50K-LoRA
HNS=/dataset1/zailong/models/spectral-surgery/Qwen3-8B-MetaMathQA-50K-Spectral-Surgery-AllModules-4Plus1
SPLITS=/dataset1/zailong/runs/peft-sft-lab/hns-metamath-direct-dose-20260910/splits
CURRENT=/dataset1/zailong/runs/peft-sft-lab/hns-posthoc-scaling-pilot-20260910/qwen_metamath/eval/validation
PRIOR=/dataset1/zailong/runs/peft-sft-lab/hns-metamath-direct-dose-20260910/eval/validation
ORIGINAL_PER=/dataset1/zailong/runs/peft-sft-lab/hns-posthoc-scaling-pilot-20260910/qwen_metamath/adapters/per-module-spectral-scale
RUN_ROOT=/dataset1/zailong/runs/peft-sft-lab/hns-common-basis-numeric-20260910/qwen_metamath
ADAPTERS="$RUN_ROOT/adapters"
DIAG="$RUN_ROOT/diagnostic"
FORMAL="$RUN_ROOT/formal"

export PYTHONPATH="$PWD/src"
export HF_HOME=/dataset1/zailong/cache/peft-sft-lab/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_BATCH_INVARIANT=1
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p "$RUN_ROOT" "$DIAG" "$FORMAL/calibration" "$FORMAL/validation"

"$PYTHON" scripts/prepare_metamath_numerical_diagnostic.py \
  --current_eval "$CURRENT" \
  --prior_eval "$PRIOR" \
  --split "$SPLITS/validation.jsonl" \
  --output "$DIAG/questions.jsonl" \
  --stable_samples 10 \
  --seed 20260912

"$PYTHON" scripts/build_metamath_common_basis_controls.py \
  --lora_path "$LORA" \
  --hns_path "$HNS" \
  --output_dir "$ADAPTERS" \
  --device cuda

DIAG_VARIANTS=(
  --variant "untouched_lora=$LORA"
  --variant "zero_rebuild=$ADAPTERS/zero-rebuild"
  --variant "common_per_module=$ADAPTERS/common-per-module"
  --variant "common_hns=$ADAPTERS/common-hns"
  --variant "original_factor_permodule=$ORIGINAL_PER"
  --variant "existing_hns=$HNS"
)

"$PYTHON" scripts/eval_gsm8k_deterministic_adapter_set.py \
  --base_model "$BASE" --dataset_path "$DIAG/questions.jsonl" --output_dir "$DIAG/run1" \
  "${DIAG_VARIANTS[@]}" --max_new_tokens 2048 --max_model_len 4096 --max_num_seqs 64 --seed 42
"$PYTHON" scripts/eval_gsm8k_deterministic_adapter_set.py \
  --base_model "$BASE" --dataset_path "$DIAG/questions.jsonl" --output_dir "$DIAG/run2" \
  "${DIAG_VARIANTS[@]}" --max_new_tokens 2048 --max_model_len 4096 --max_num_seqs 64 --seed 42
"$PYTHON" scripts/eval_gsm8k_deterministic_adapter_set.py \
  --base_model "$BASE" --dataset_path "$DIAG/questions.jsonl" --output_dir "$DIAG/reverse" \
  "${DIAG_VARIANTS[@]}" --reverse_variants \
  --max_new_tokens 2048 --max_model_len 4096 --max_num_seqs 64 --seed 42

"$PYTHON" scripts/check_metamath_numerical_gate.py \
  --eval_root "$DIAG" \
  --adapter_manifest "$ADAPTERS/manifest.json" \
  --output "$DIAG/gate.json"

CAL_VARIANTS=(--variant "original_lora=$LORA")
for gamma in 1p00 0p85 0p70 0p60 0p50 0p40; do
  CAL_VARIANTS+=(--variant "global_$gamma=$ADAPTERS/common-global-$gamma")
done
CAL_VARIANTS+=(--variant "common_per_module=$ADAPTERS/common-per-module")
CAL_VARIANTS+=(--variant "common_hns=$ADAPTERS/common-hns")

"$PYTHON" scripts/eval_gsm8k_deterministic_adapter_set.py \
  --base_model "$BASE" --dataset_path "$SPLITS/calibration.jsonl" \
  --output_dir "$FORMAL/calibration/eval" "${CAL_VARIANTS[@]}" \
  --max_new_tokens 2048 --max_model_len 4096 --max_num_seqs 64 --seed 42
"$PYTHON" scripts/analyze_metamath_common_basis_confirm.py \
  --stage calibration --eval_dir "$FORMAL/calibration/eval" \
  --adapter_root "$ADAPTERS" --output_dir "$FORMAL/calibration/analysis"

SELECTION="$FORMAL/calibration/analysis/selection.json"
SELECTED_GLOBAL=$("$PYTHON" -c "import json; print(json.load(open('$SELECTION'))['selected_path'])")

"$PYTHON" scripts/eval_gsm8k_deterministic_adapter_set.py \
  --base_model "$BASE" --dataset_path "$SPLITS/validation.jsonl" \
  --output_dir "$FORMAL/validation/eval" \
  --variant "original_lora=$LORA" \
  --variant "zero_rebuild=$ADAPTERS/zero-rebuild" \
  --variant "selected_global=$SELECTED_GLOBAL" \
  --variant "common_per_module=$ADAPTERS/common-per-module" \
  --variant "common_hns=$ADAPTERS/common-hns" \
  --variant "existing_hns=$HNS" \
  --max_new_tokens 2048 --max_model_len 4096 --max_num_seqs 64 --seed 42
"$PYTHON" scripts/analyze_metamath_common_basis_confirm.py \
  --stage validation --eval_dir "$FORMAL/validation/eval" \
  --adapter_root "$ADAPTERS" --output_dir "$FORMAL/validation/analysis" \
  --selection "$SELECTION"

cp reports/hns_posthoc_lora_normalization_20260910/common_basis_numeric_protocol.md "$RUN_ROOT/protocol.md"
echo "[Done] Common-basis numerical confirmation: $RUN_ROOT"
