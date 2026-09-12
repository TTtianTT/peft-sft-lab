#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/dataset1/zailong/workspace/peft-sft-lab}"
PYTHON="${PYTHON:-/dataset1/zailong/envs/peft-sft-lab/bin/python}"
BASE_MODEL="${BASE_MODEL:-/dataset1/zailong/models/Qwen3-8B}"
SOURCE_LORA="${SOURCE_LORA:-/dataset1/zailong/models/spectral-surgery/Qwen3-8B-Magicoder-50K-LoRA-E1}"
ALL_HNS="${ALL_HNS:-/dataset1/zailong/models/spectral-surgery/Qwen3-8B-Magicoder-50K-SpectralSurgery-HNS4p1-AllMods}"
LOCALIZATION_ROOT="${LOCALIZATION_ROOT:-/dataset1/zailong/runs/peft-sft-lab/Qwen3-8B/magicoder-hns-localization-20260908}"
HUMANEVAL_PATH="${HUMANEVAL_PATH:-/dataset1/zailong/data/peft-sft-lab/humaneval-test.parquet}"
MBPP_PATH="${MBPP_PATH:-/dataset1/zailong/data/peft-sft-lab/mbpp-sanitized-test.parquet}"
OUT_DIR="${OUT_DIR:-/dataset1/zailong/runs/peft-sft-lab/Qwen3-8B/magicoder-hns-interpretability-20260908}"

export HF_HOME="${HF_HOME:-/dataset1/zailong/cache/peft-sft-lab/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO_ROOT"
mkdir -p "$OUT_DIR"

"$PYTHON" scripts/analyze_hns_attention_activations.py \
  --base_model "$BASE_MODEL" \
  --lora_path "$SOURCE_LORA" \
  --hns_path "$ALL_HNS" \
  --middle_hns_path "$LOCALIZATION_ROOT/adapters/localized-hns4p1/hns-layers-middle" \
  --qkv_hns_path "$LOCALIZATION_ROOT/adapters/localized-hns4p1/hns-qkv" \
  --eval_root "$LOCALIZATION_ROOT/eval" \
  --humaneval_path "$HUMANEVAL_PATH" \
  --mbpp_path "$MBPP_PATH" \
  --out_dir "$OUT_DIR" \
  --max_flip 8 \
  --max_stable 4 \
  --max_seq_len 512 \
  --focus_layers 16 17 18
