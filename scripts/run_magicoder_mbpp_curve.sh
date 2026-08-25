#!/usr/bin/env bash
set -euo pipefail

# One-GPU Magicoder LoRA ablation with intermediate MBPP evaluation.
# Each stage resumes the same Trainer state, so the one-epoch cosine schedule
# and data-order state are preserved while the GPU is reused for evaluation.

REPO_DIR="${REPO_DIR:-/root/autodl-tmp/peft-sft-lab}"
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/Llama-3.1-8B-Instruct}"
DATASET_PATH="${DATASET_PATH:-/root/autodl-tmp/magicoder-train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/meta-llama-Llama-3.1-8B-Instruct/code/lora/magicoder-chat-all-linear-lr5e-6-bs192-1ep/seed42}"
SEED="${SEED:-42}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-192}"
EVAL_REQUEST_BATCH_SIZE="${EVAL_REQUEST_BATCH_SIZE:-8}"

cd "$REPO_DIR"
mkdir -p "$OUTPUT_DIR/mbpp_curve"

common_train_args=(
  --base_model "$BASE_MODEL"
  --task code
  --peft_method lora
  --dataset_path "$DATASET_PATH"
  --output_dir "$OUTPUT_DIR"
  --sft_format chat
  --num_train_epochs 1
  --max_seq_len 2048
  --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE"
  --global_train_batch_size "$GLOBAL_BATCH_SIZE"
  --lr 5e-6
  --lr_scheduler_type cosine
  --warmup_ratio 0.10
  --weight_decay 0.0
  --grad_clip 1.0
  --adam_beta1 0.9
  --adam_beta2 0.999
  --r 16
  --lora_alpha 32
  --lora_dropout 0.05
  --target_modules all
  --save_strategy steps
  --save_total_limit 6
  --bf16
  --gradient_checkpointing
  --seed "$SEED"
)

run_train() {
  local stop_step="$1"
  local resume_checkpoint="${2:-}"
  local args=("${common_train_args[@]}" --stop_at_step "$stop_step" --save_steps "$stop_step")
  if [[ -n "$resume_checkpoint" ]]; then
    args+=(--resume_from_checkpoint "$resume_checkpoint")
  fi
  PYTHONPATH=src accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    -m finetune.train_sft_peft "${args[@]}"
}

run_mbpp() {
  local label="$1"
  local adapter_dir="$2"
  local args=(
    --base_model "$BASE_MODEL"
    --output_dir "$OUTPUT_DIR/mbpp_curve/$label"
    --prompt_style chat
    --use_vllm
    --tensor_parallel_size 1
    --vllm_max_model_len 4096
    --vllm_attention_backend FLASH_ATTN
    --vllm_disable_flashinfer_sampler
    --vllm_request_batch_size "$EVAL_REQUEST_BATCH_SIZE"
    --max_new_tokens 512
    --seed "$SEED"
  )
  if [[ -n "$adapter_dir" ]]; then
    args+=(--adapter_dir "$adapter_dir")
  fi
  PYTHONPATH=src python -m finetune.eval.eval_mbpp "${args[@]}"
}

# Keep the base measurement under this run's exact evaluator configuration.
run_mbpp base ""

# A one-step preflight creates run_config.json, which contains the exact number
# of optimizer steps after filtering the local dataset. It avoids hard-coding
# a dataset-size assumption into the checkpoint schedule.
run_train 1
TOTAL_STEPS="$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["total_steps"])' "$OUTPUT_DIR/run_config.json")"
mapfile -t EVAL_STEPS < <(python - "$TOTAL_STEPS" <<'PY'
import math
import sys

total = int(sys.argv[1])
print("\n".join(str(math.ceil(total * fraction)) for fraction in (0.25, 0.50, 0.75, 1.00)))
PY
)

echo "[Plan] total_steps=${TOTAL_STEPS}; MBPP evaluation steps: ${EVAL_STEPS[*]}"
resume_checkpoint="$OUTPUT_DIR/checkpoint-1"
for step in "${EVAL_STEPS[@]}"; do
  run_train "$step" "$resume_checkpoint"
  resume_checkpoint="$OUTPUT_DIR/checkpoint-${step}"
  if [[ ! -d "$resume_checkpoint" ]]; then
    echo "Expected checkpoint was not created: $resume_checkpoint" >&2
    exit 1
  fi
  run_mbpp "step-${step}" "$resume_checkpoint"
done

python - "$OUTPUT_DIR" "${EVAL_STEPS[@]}" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
rows = []
for label in ["base", *[f"step-{step}" for step in sys.argv[2:]]]:
    metrics_path = run_dir / "mbpp_curve" / label / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    rows.append({"checkpoint": label, "pass@1": metrics["pass@1"], "correct": metrics["correct"], "total": metrics["total"]})
(run_dir / "mbpp_curve" / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
print("[Done] MBPP curve:")
for row in rows:
    print(f"  {row['checkpoint']:>12} pass@1={row['pass@1']:.4f} ({row['correct']}/{row['total']})")
PY
