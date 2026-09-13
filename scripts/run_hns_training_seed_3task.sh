#!/usr/bin/env bash
set -euo pipefail

BASE_NAME="${1:?usage: $0 BASE_NAME SEED}"
TRAIN_SEED="${2:?usage: $0 BASE_NAME SEED}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON=/dataset1/zailong/envs/peft-sft-lab/bin/python
RUN_ROOT=/dataset1/zailong/runs/peft-sft-lab/hns-training-seeds-3task-20260912
SEED_ROOT="$RUN_ROOT/$BASE_NAME/seed$TRAIN_SEED"
LORA_ROOT="$SEED_ROOT/lora"
GRID_ROOT="$SEED_ROOT/hns_step_grid"
TASK_CONFIG="$SEED_ROOT/config/task_config.json"
GRID_CONFIG="$REPO_ROOT/configs/hns_step_grid_2x4_20260912.json"

MAGICODER_DATA=/dataset1/zailong/data/peft-sft-lab/magicoder-train.parquet
METAMATH_DATA=/dataset1/zailong/data/peft-sft-lab/metamathqa-train.parquet
TULU_DATA=/dataset1/zailong/data/peft-sft-lab/cross-task-mechanism/tulu-3-sft-personas-instruction-following-train.parquet

case "$BASE_NAME" in
  Qwen3-8B)
    BASE_PATH=/dataset1/zailong/models/Qwen3-8B
    CHAT_MODE=non_thinking
    ;;
  Llama-3.1-8B-Instruct)
    BASE_PATH=/dataset1/zailong/models/Llama-3.1-8B-Instruct
    CHAT_MODE=auto
    ;;
  *)
    echo "unknown base: $BASE_NAME" >&2
    exit 2
    ;;
esac

if [[ "$TRAIN_SEED" != 43 && "$TRAIN_SEED" != 44 ]]; then
  echo "training seed must be 43 or 44, got: $TRAIN_SEED" >&2
  exit 2
fi

export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/scripts"
export HF_HOME=/dataset1/zailong/cache/peft-sft-lab/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE=/dataset1/zailong/cache/peft-sft-lab/datasets
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED="$TRAIN_SEED"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_BATCH_INVARIANT=1

# Compute nodes have a small shared /tmp. Dataset shuffle/index construction can
# exceed it, so keep all large temporary files on /dataset1.
TASK_TMP_ROOT="$RUN_ROOT/tmp/${SLURM_JOB_ID:-manual}-$BASE_NAME-seed$TRAIN_SEED"
export TMPDIR="$TASK_TMP_ROOT"
export TMP="$TASK_TMP_ROOT"
export TEMP="$TASK_TMP_ROOT"

mkdir -p \
  "$LORA_ROOT" "$GRID_ROOT/adapters" "$GRID_ROOT/eval" "$SEED_ROOT/config" \
  "$HF_DATASETS_CACHE" "$TASK_TMP_ROOT"

adapter_complete() {
  local path="$1"
  [[ -f "$path/adapter_config.json" ]] && \
    { [[ -f "$path/adapter_model.safetensors" ]] || [[ -f "$path/adapter_model.bin" ]]; }
}

latest_checkpoint() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    return 0
  fi
  find "$path" -maxdepth 1 -type d -name 'checkpoint-*' -print 2>/dev/null | sort -V | tail -n 1
}

train_lora() {
  local task="$1"
  local dataset="$2"
  local output="$3"
  shift 3

  if adapter_complete "$output"; then
    echo "[Skip] completed LoRA: $BASE_NAME/seed$TRAIN_SEED/$task"
    return
  fi

  local -a resume_args=()
  local checkpoint=""
  checkpoint="$(latest_checkpoint "$output")"
  if [[ -n "$checkpoint" && -f "$checkpoint/trainer_state.json" ]]; then
    resume_args=(--resume_from_checkpoint "$checkpoint")
    echo "[Resume] $task from $checkpoint"
  fi

  echo "[Train] $BASE_NAME seed=$TRAIN_SEED task=$task"
  "$PYTHON" -m finetune.train_sft_peft \
    --base_model "$BASE_PATH" \
    --task "$task" \
    --peft_method lora \
    --dataset_path "$dataset" \
    --output_dir "$output" \
    --train_profile none \
    --seed "$TRAIN_SEED" \
    --dataset_seed 42 \
    --sft_format chat \
    --chat_template_mode "$CHAT_MODE" \
    --target_modules all \
    --r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --pad_to_multiple_of 8 \
    --weight_decay 0 \
    --grad_clip 1 \
    --bf16 \
    --gradient_checkpointing \
    "${resume_args[@]}" \
    "$@"
}

# The 50K subset is fixed with dataset_seed=42. Only optimizer/dropout/sampler
# randomness changes through TRAIN_SEED, so the replicate isolates training seed.
train_lora magicoder "$MAGICODER_DATA" "$LORA_ROOT/magicoder" \
  --max_train_samples 50000 --num_train_epochs 1 \
  --max_seq_len 4096 --per_device_train_batch_size 16 --global_train_batch_size 32 \
  --lr 2e-5 --warmup_ratio 0.05 --lr_scheduler_type cosine \
  --adam_beta1 0.9 --adam_beta2 0.999 \
  --save_strategy epoch --save_total_limit 1 --logging_steps 25

if [[ "$BASE_NAME" == Qwen3-8B ]]; then
  train_lora metamath "$METAMATH_DATA" "$LORA_ROOT/metamath" \
    --max_train_samples 50000 --num_train_epochs 3 \
    --max_seq_len 4096 --per_device_train_batch_size 16 --global_train_batch_size 32 \
    --lr 1e-4 --warmup_ratio 0.05 --lr_scheduler_type cosine \
    --adam_beta1 0.9 --adam_beta2 0.999 \
    --save_strategy epoch --save_total_limit 1 --logging_steps 25

  train_lora if "$TULU_DATA" "$LORA_ROOT/tulu" \
    --num_train_epochs 2 --max_seq_len 4096 \
    --per_device_train_batch_size 16 --global_train_batch_size 128 \
    --lr 4e-4 --warmup_ratio 0.03 --lr_scheduler_type cosine \
    --adam_beta1 0.9 --adam_beta2 0.999 \
    --save_strategy steps --save_steps 100 --save_total_limit 2 --logging_steps 10
else
  # Repository profile documented as matching the original Llama math recipe.
  train_lora metamath "$METAMATH_DATA" "$LORA_ROOT/metamath" \
    --max_train_samples 50000 --num_train_epochs 3 \
    --max_seq_len 1024 --per_device_train_batch_size 64 --global_train_batch_size 768 \
    --lr 1e-4 --warmup_ratio 0.1 --min_lr_ratio 0.01 \
    --lr_scheduler_type cosine_with_min_lr \
    --adam_beta1 0.9 --adam_beta2 0.95 \
    --save_strategy epoch --save_total_limit 1

  # The released seed-42 model card explicitly records seq=1024 and GBS=128.
  train_lora if "$TULU_DATA" "$LORA_ROOT/tulu" \
    --num_train_epochs 2 --max_seq_len 1024 \
    --per_device_train_batch_size 64 --global_train_batch_size 128 \
    --lr 2e-4 --warmup_ratio 0.1 --min_lr_ratio 0.01 \
    --lr_scheduler_type cosine_with_min_lr \
    --adam_beta1 0.9 --adam_beta2 0.95 \
    --save_strategy epoch --save_total_limit 1
fi

"$PYTHON" scripts/write_hns_training_seed_config.py \
  --base "$BASE_NAME" --seed "$TRAIN_SEED" --run_root "$RUN_ROOT" --output "$TASK_CONFIG"
cp "$GRID_CONFIG" "$SEED_ROOT/config/grid_config.json"

"$PYTHON" scripts/build_hns_step_grid_2x4.py \
  --task_config "$TASK_CONFIG" --grid_config "$GRID_CONFIG" \
  --base "$BASE_NAME" --output_dir "$GRID_ROOT/adapters" --device cuda

"$PYTHON" scripts/eval_forgetting_matrix_vllm.py \
  --base_model "$BASE_PATH" \
  --variant_manifest "$GRID_ROOT/adapters/$BASE_NAME/variant_manifest.json" \
  --config "$TASK_CONFIG" --output_dir "$GRID_ROOT/eval" \
  --tasks magicoder metamath tulu --diagonal_only \
  --max_model_len 4096 --gpu_memory_utilization 0.94 \
  --max_num_seqs 512 --max_num_batched_tokens 65536 \
  --adapter_block_size 11 --prompt_chunk_short 2048 --prompt_chunk_long 128 --seed 42

"$PYTHON" scripts/score_forgetting_matrix.py --matrix_dir "$GRID_ROOT/eval" --workers 32
echo "[Done] $BASE_NAME training seed $TRAIN_SEED and full three-task HNS step grid"
