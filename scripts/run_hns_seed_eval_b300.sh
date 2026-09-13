#!/usr/bin/env bash
set -euo pipefail
BASE_NAME="${1:?usage: $0 BASE_NAME}"
cd /dataset1/zailong/workspace/peft-sft-lab
PYTHON=/dataset1/zailong/envs/peft-sft-lab/bin/python
RUN_ROOT=/dataset1/zailong/runs/peft-sft-lab/hns-training-seeds-3task-20260912
case "$BASE_NAME" in
  Qwen3-8B|Llama-3.1-8B-Instruct) BASE_PATH="/dataset1/zailong/models/$BASE_NAME" ;;
  *) echo "unknown base: $BASE_NAME" >&2; exit 2 ;;
esac
export PYTHONPATH="$PWD/src:$PWD/scripts"
export HF_HOME=/dataset1/zailong/cache/peft-sft-lab/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE=/dataset1/zailong/cache/peft-sft-lab/datasets
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_BATCH_INVARIANT=1
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# ZMQ adds a UUID; this short path remains below the Unix socket 107-byte limit.
export TMPDIR="/dataset1/zailong/tmp/he-${SLURM_JOB_ID:?}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_CACHE_ROOT="$TMPDIR/vllm-cache"
mkdir -p "$TMPDIR"
for TRAIN_SEED in 43 44; do
  SEED_ROOT="$RUN_ROOT/$BASE_NAME/seed$TRAIN_SEED"
  GRID_ROOT="$SEED_ROOT/hns_step_grid"
  if [[ -f "$GRID_ROOT/eval/generation_manifest.json" && -f "$GRID_ROOT/eval/score_manifest.json" ]]; then
    echo "[Skip] completed evaluation: $BASE_NAME seed=$TRAIN_SEED"
    continue
  fi
  echo "[Eval] $BASE_NAME seed=$TRAIN_SEED $(date -Is)"
  "$PYTHON" scripts/eval_forgetting_matrix_vllm.py \
    --base_model "$BASE_PATH" \
    --variant_manifest "$GRID_ROOT/adapters/$BASE_NAME/variant_manifest.json" \
    --config "$SEED_ROOT/config/task_config.json" --output_dir "$GRID_ROOT/eval" \
    --tasks magicoder metamath tulu --diagonal_only \
    --max_model_len 4096 --gpu_memory_utilization 0.94 \
    --max_num_seqs 1024 --max_num_batched_tokens 65536 \
    --adapter_block_size 11 --max_lora_rank 16 \
    --prompt_chunk_short 4096 --prompt_chunk_long 512 --seed 42
  "$PYTHON" scripts/score_forgetting_matrix.py --matrix_dir "$GRID_ROOT/eval" --workers 32
  echo "[Done] $BASE_NAME seed=$TRAIN_SEED $(date -Is)"
done
