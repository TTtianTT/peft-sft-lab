#!/usr/bin/env bash
set -euo pipefail

BASE_NAME="${1:?usage: $0 BASE_NAME}"
PYTHON=/dataset1/zailong/envs/peft-sft-lab/bin/python
TASK_CONFIG="$PWD/configs/hns_forgetting_2x4_20260911.json"
GRID_CONFIG="$PWD/configs/hns_step_grid_2x4_20260912.json"
RUN_ROOT=/dataset1/zailong/runs/peft-sft-lab/hns-step-grid-2x4-20260912
ADAPTER_ROOT="$RUN_ROOT/adapters"
EVAL_ROOT="$RUN_ROOT/eval/$BASE_NAME"
CONFIG_SNAPSHOT_ROOT="$RUN_ROOT/config/$BASE_NAME"

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

mkdir -p "$RUN_ROOT" "$ADAPTER_ROOT" "$EVAL_ROOT" "$CONFIG_SNAPSHOT_ROOT"
cp "$TASK_CONFIG" "$CONFIG_SNAPSHOT_ROOT/task_config.json"
cp "$GRID_CONFIG" "$CONFIG_SNAPSHOT_ROOT/grid_config.json"

"$PYTHON" scripts/build_hns_step_grid_2x4.py \
  --task_config "$TASK_CONFIG" --grid_config "$GRID_CONFIG" \
  --base "$BASE_NAME" --output_dir "$ADAPTER_ROOT" --device cuda

"$PYTHON" scripts/eval_forgetting_matrix_vllm.py \
  --base_model "$BASE_PATH" \
  --variant_manifest "$ADAPTER_ROOT/$BASE_NAME/variant_manifest.json" \
  --config "$TASK_CONFIG" --output_dir "$EVAL_ROOT" \
  --tasks magicoder metamath tulu commonsense --diagonal_only \
  --max_model_len 4096 --gpu_memory_utilization 0.94 \
  --max_num_seqs 512 --max_num_batched_tokens 65536 \
  --adapter_block_size 11 --prompt_chunk_short 2048 --prompt_chunk_long 128 --seed 42

"$PYTHON" scripts/score_forgetting_matrix.py --matrix_dir "$EVAL_ROOT" --workers 32
echo "[Done] $BASE_NAME HNS step grid"
