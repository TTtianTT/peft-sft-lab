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
export TMPDIR="/dataset1/zailong/tmp/hf-${SLURM_JOB_ID:?}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_CACHE_ROOT="$TMPDIR/vllm-cache"
mkdir -p "$TMPDIR"
for TRAIN_SEED in 43 44; do
  SEED_ROOT="$RUN_ROOT/$BASE_NAME/seed$TRAIN_SEED"
  GRID_ROOT="$SEED_ROOT/hns_step_grid"
  OUTPUT="$SEED_ROOT/forgetting/eval"
  CONFIG="$SEED_ROOT/forgetting/task_config.json"
  "$PYTHON" scripts/prepare_hns_seed_forgetting.py \
    --seed_config "$SEED_ROOT/config/task_config.json" \
    --reference_config configs/hns_forgetting_2x4_20260911.json \
    --variant_manifest "$GRID_ROOT/adapters/$BASE_NAME/variant_manifest.json" \
    --output "$CONFIG"
  if [[ ! -f "$OUTPUT/generation_manifest.json" ]]; then
    echo "[Eval forgetting] $BASE_NAME seed=$TRAIN_SEED $(date -Is)"
    "$PYTHON" scripts/eval_forgetting_matrix_vllm.py \
      --base_model "$BASE_PATH" \
      --variant_manifest "$GRID_ROOT/adapters/$BASE_NAME/variant_manifest.json" \
      --config "$CONFIG" --output_dir "$OUTPUT" \
      --tasks magicoder metamath tulu commonsense \
      --max_model_len 4096 --gpu_memory_utilization 0.94 \
      --max_num_seqs 1024 --max_num_batched_tokens 65536 \
      --adapter_block_size 11 --max_lora_rank 16 \
      --prompt_chunk_short 4096 --prompt_chunk_long 512 --seed 42
  fi
  "$PYTHON" scripts/score_forgetting_matrix.py --matrix_dir "$OUTPUT" --workers 32
  "$PYTHON" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); rows=json.loads((p/"score_manifest.json").read_text())["records"]; assert len(rows)==136, len(rows); assert len({(r["task"],r["variant"]) for r in rows})==136' "$OUTPUT"
  echo "[Done forgetting] $BASE_NAME seed=$TRAIN_SEED $(date -Is)"
done
