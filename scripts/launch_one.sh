#!/usr/bin/env bash
set -euo pipefail

# Configurable via env vars (or edit defaults below).
NUM_PROCESSES="${NUM_PROCESSES:-${1:-1}}"
BASE_MODEL="${BASE_MODEL:-mistralai/Mistral-7B-v0.1}"
TASK="${TASK:-alpaca}"
PEFT_METHOD="${PEFT_METHOD:-lora}"

MAX_STEPS="${MAX_STEPS:-10}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
LR="${LR:-2e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
SEED="${SEED:-42}"

RUNS_ROOT="${RUNS_ROOT:-runs}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUNS_ROOT/manual-${TASK}-${PEFT_METHOD}-seed${SEED}}"

EXTRA_FLAGS=()
if [[ "${BF16:-1}" == "1" ]]; then EXTRA_FLAGS+=("--bf16"); fi
if [[ "${FP16:-0}" == "1" ]]; then EXTRA_FLAGS+=("--fp16"); fi
if [[ "${GRADIENT_CHECKPOINTING:-1}" == "1" ]]; then EXTRA_FLAGS+=("--gradient_checkpointing"); fi
if [[ "${USE_QLORA:-0}" == "1" ]]; then EXTRA_FLAGS+=("--use_qlora"); fi

accelerate launch --num_processes "${NUM_PROCESSES}" -m finetune.train_sft_peft \
  --base_model "${BASE_MODEL}" \
  --task "${TASK}" \
  --peft_method "${PEFT_METHOD}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --lr "${LR}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --max_seq_len "${MAX_SEQ_LEN}" \
  --seed "${SEED}" \
  "${EXTRA_FLAGS[@]}"

