#!/usr/bin/env bash
set -euo pipefail

BASE_NAME="${1:?usage: $0 BASE_NAME}"
PYTHON=/dataset1/zailong/envs/peft-sft-lab/bin/python
CONFIG="$PWD/configs/hns_forgetting_2x4_20260911.json"
RUN_ROOT=/dataset1/zailong/runs/peft-sft-lab/hns-spectral-standardization-2x4-20260911
ADAPTER_ROOT="$RUN_ROOT/adapters"
EVAL_ROOT="$RUN_ROOT/eval/$BASE_NAME"

case "$BASE_NAME" in
  Qwen3-8B) BASE_PATH=/dataset1/zailong/models/Qwen3-8B ;;
  Llama-3.1-8B-Instruct) BASE_PATH=/dataset1/zailong/models/Llama-3.1-8B-Instruct ;;
  *) echo "unknown base: $BASE_NAME" >&2; exit 2 ;;
esac

export PYTHONPATH="$PWD/src:$PWD/scripts"
export HF_HOME=/dataset1/zailong/cache/peft-sft-lab/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_BATCH_INVARIANT=1
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p "$RUN_ROOT" "$ADAPTER_ROOT" "$EVAL_ROOT"
"$PYTHON" scripts/build_2x4_spectral_standardization.py \
  --config "$CONFIG" --base "$BASE_NAME" --output_dir "$ADAPTER_ROOT" --device cuda

"$PYTHON" scripts/eval_forgetting_matrix_vllm.py \
  --base_model "$BASE_PATH" \
  --variant_manifest "$ADAPTER_ROOT/$BASE_NAME/variant_manifest.json" \
  --config "$CONFIG" --output_dir "$EVAL_ROOT" \
  --tasks magicoder metamath tulu commonsense --diagonal_only \
  --max_model_len 4096 --gpu_memory_utilization 0.90 \
  --max_num_seqs 256 --max_num_batched_tokens 65536 \
  --adapter_block_size 4 --prompt_chunk_short 1024 --prompt_chunk_long 64 --seed 42

"$PYTHON" scripts/score_forgetting_matrix.py --matrix_dir "$EVAL_ROOT" --workers 16
echo "[Done] $BASE_NAME spectral standardization"
