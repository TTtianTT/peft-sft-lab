#!/usr/bin/env bash
set -euo pipefail

PYTHON=/dataset1/zailong/envs/peft-sft-lab/bin/python
BASE=/dataset1/zailong/models/Qwen3-8B
LORA=/dataset1/zailong/models/spectral-surgery/Qwen3-8B-MetaMathQA-50K-LoRA
HNS=/dataset1/zailong/models/spectral-surgery/Qwen3-8B-MetaMathQA-50K-Spectral-Surgery-AllModules-4Plus1
GSM8K=/dataset1/zailong/data/peft-sft-lab/cross-task-mechanism/gsm8k-test.jsonl
TRAIN=/dataset1/zailong/data/peft-sft-lab/metamathqa-train.parquet
PRIOR=/dataset1/zailong/runs/peft-sft-lab/hns-signed-gates-20260910/qwen_metamath_reward_paths/rollouts/manifest.json
RUN_ROOT=/dataset1/zailong/runs/peft-sft-lab/hns-metamath-direct-dose-20260910
SPLITS="$RUN_ROOT/splits"
ADAPTERS="$RUN_ROOT/adapters"
CAL_EVAL="$RUN_ROOT/eval/calibration"
VAL_EVAL="$RUN_ROOT/eval/validation"
ANALYSIS="$RUN_ROOT/analysis"

export PYTHONPATH="$PWD/src"
export HF_HOME=/dataset1/zailong/cache/peft-sft-lab/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0
export TORCH_CUDNN_SDPA_DEPRIORITIZED=1

mkdir -p "$RUN_ROOT" "$ANALYSIS/calibration" "$ANALYSIS/validation"

"$PYTHON" scripts/prepare_metamath_direct_dose_splits.py \
  --gsm8k "$GSM8K" \
  --metamath_train "$TRAIN" \
  --exclude_manifest "$PRIOR" \
  --output_dir "$SPLITS" \
  --calibration 256 \
  --validation 512 \
  --seed 20260911

"$PYTHON" scripts/build_metamath_head_dose_adapters.py \
  --lora_path "$LORA" \
  --hns_path "$HNS" \
  --output_dir "$ADAPTERS" \
  --device cuda

"$PYTHON" scripts/eval_gsm8k_adapter_set.py \
  --base_model "$BASE" \
  --dataset_path "$SPLITS/calibration.jsonl" \
  --output_dir "$CAL_EVAL" \
  --variant "lora=$LORA" \
  --variant "zero_rebuild=$ADAPTERS/head-0p00-rebuild" \
  --variant "head_0p25=$ADAPTERS/head-0p25" \
  --variant "scalar_0p25=$ADAPTERS/scalar-0p25" \
  --variant "head_0p50=$ADAPTERS/head-0p50" \
  --variant "scalar_0p50=$ADAPTERS/scalar-0p50" \
  --variant "head_1p00=$ADAPTERS/head-1p00" \
  --variant "scalar_1p00=$ADAPTERS/scalar-1p00" \
  --variant "full_hns=$HNS" \
  --max_new_tokens 2048 \
  --max_model_len 4096 \
  --gpu_memory_utilization 0.85 \
  --seed 42

"$PYTHON" scripts/analyze_metamath_direct_dose.py \
  --stage calibration \
  --eval_dir "$CAL_EVAL" \
  --adapter_manifest "$ADAPTERS/manifest.json" \
  --lora_path "$LORA" \
  --hns_path "$HNS" \
  --output_dir "$ANALYSIS/calibration" \
  --bootstrap 20000 \
  --seed 20260911

SELECTED_HEAD=$("$PYTHON" -c "import json; print(json.load(open('$ANALYSIS/calibration/selection.json'))['selected_head_path'])")
MATCHED_SCALAR=$("$PYTHON" -c "import json; print(json.load(open('$ANALYSIS/calibration/selection.json'))['matched_scalar_path'])")

"$PYTHON" scripts/eval_gsm8k_adapter_set.py \
  --base_model "$BASE" \
  --dataset_path "$SPLITS/validation.jsonl" \
  --output_dir "$VAL_EVAL" \
  --variant "lora=$LORA" \
  --variant "full_hns=$HNS" \
  --variant "selected_head=$SELECTED_HEAD" \
  --variant "matched_scalar=$MATCHED_SCALAR" \
  --max_new_tokens 2048 \
  --max_model_len 4096 \
  --gpu_memory_utilization 0.85 \
  --seed 42

"$PYTHON" scripts/analyze_metamath_direct_dose.py \
  --stage validation \
  --eval_dir "$VAL_EVAL" \
  --adapter_manifest "$ADAPTERS/manifest.json" \
  --lora_path "$LORA" \
  --hns_path "$HNS" \
  --output_dir "$ANALYSIS/validation" \
  --selection "$ANALYSIS/calibration/selection.json" \
  --bootstrap 20000 \
  --seed 20260911

cp reports/hns_metamath_direct_dose_protocol_20260910.md "$RUN_ROOT/protocol.md"
echo "[Done] Qwen MetaMath direct-dose experiment: $RUN_ROOT"
