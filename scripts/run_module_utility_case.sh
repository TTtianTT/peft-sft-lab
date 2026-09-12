#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 {qwen_magicoder|qwen_commonsense|llama_tulu}" >&2
  exit 2
fi

CASE="$1"
PYTHON=/dataset1/zailong/envs/peft-sft-lab/bin/python
MODEL_ROOT=/dataset1/zailong/models/spectral-surgery
DATA_ROOT=/dataset1/zailong/data/peft-sft-lab
RUN_ROOT=/dataset1/zailong/runs/peft-sft-lab/hns-module-utility-20260909
export PYTHONPATH="$PWD/src"
export HF_HOME=/dataset1/zailong/cache/peft-sft-lab/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_CUDNN_SDPA_DEPRIORITIZED=1
export PATH="$(dirname "$PYTHON"):$PATH"

case "$CASE" in
  qwen_magicoder)
    BASE=/dataset1/zailong/models/Qwen3-8B
    LORA="$MODEL_ROOT/Qwen3-8B-Magicoder-50K-LoRA-E1"
    HNS="$MODEL_ROOT/Qwen3-8B-Magicoder-50K-SpectralSurgery-HNS4p1-AllMods"
    DATA="$DATA_ROOT/magicoder-train.parquet"
    DATASET_NAME=ise-uiuc/Magicoder-Evol-Instruct-110K
    CHAT_MODE=non_thinking
    FAST=4
    STABLE=1
    ;;
  qwen_commonsense)
    BASE=/dataset1/zailong/models/Qwen3-8B
    LORA="$MODEL_ROOT/Qwen3-8B-CommonSense170K-LoRA"
    HNS=/dataset1/zailong/runs/peft-sft-lab/hns-allmodules-scope-20260909/Qwen3-8B/commonsense/hns-allmodules-8plus2
    DATA="$DATA_ROOT/commonsense170k-train.parquet"
    DATASET_NAME=commonsense170k
    CHAT_MODE=non_thinking
    FAST=8
    STABLE=2
    ;;
  llama_tulu)
    BASE=/dataset1/zailong/models/Llama-3.1-8B-Instruct
    LORA="$MODEL_ROOT/Llama-3.1-8B-Instruct-InstructionFollowing-LoRA"
    HNS=/dataset1/zailong/runs/peft-sft-lab/hns-allmodules-scope-20260909/Llama-3.1-8B-Instruct/tulu/hns-allmodules-8plus2
    DATA="$DATA_ROOT/cross-task-mechanism/tulu-3-sft-personas-instruction-following-train.parquet"
    DATASET_NAME=tulu_if
    CHAT_MODE=auto
    FAST=8
    STABLE=2
    ;;
  *)
    echo "unknown case: $CASE" >&2
    exit 2
    ;;
esac

GRADIENT="$RUN_ROOT/gradient/$CASE"
UTILITY="$RUN_ROOT/utility/$CASE"
mkdir -p "$RUN_ROOT/gradient" "$RUN_ROOT/utility"
if [[ ! -f "$GRADIENT/spectral_edit_meta.json" ]]; then
  "$PYTHON" -m finetune.spectral_edit.cli sensitivity-hns \
    --base_model "$BASE" --lora_path "$LORA" --out_dir "$GRADIENT" \
    --target_modules all_modules --module_budget 10000 --selection_rule importance \
    --calib_dataset "$DATASET_NAME" --calib_dataset_path "$DATA" \
    --calib_samples 64 --calib_batch_size 2 --calib_shuffle --calib_start 0 \
    --sft_format chat --chat_template_mode "$CHAT_MODE" --max_seq_len 512 \
    --dtype bf16 --fast_steps "$FAST" --stable_steps "$STABLE" --seed 42
else
  echo "[Resume] $GRADIENT/spectral_edit_meta.json"
fi

if [[ ! -f "$UTILITY/module_utility.tsv" ]]; then
  "$PYTHON" scripts/measure_module_intervention_utility.py \
    --base_model "$BASE" --lora_path "$LORA" --hns_path "$HNS" \
    --dataset_path "$DATA" --dataset_name "$DATASET_NAME" --task "$CASE" \
    --output_dir "$UTILITY" --samples 64 --sample_start 64 --batch_size 4 \
    --max_seq_len 512 --seed 42 --dtype bf16 --chat_template_mode "$CHAT_MODE"
else
  echo "[Resume] $UTILITY/module_utility.tsv"
fi

echo "[Done] module utility $CASE"
