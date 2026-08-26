#!/usr/bin/env bash
set -euo pipefail

# Train one continuous 3-epoch schedule in three resumable stages. Evaluation
# runs only after the training process exits, so HE/MBPP can reuse the GPU.

REPO_DIR="${REPO_DIR:-/root/autodl-tmp/peft-sft-lab}"
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/Llama-3.1-8B-Instruct}"
DATASET_PATH="${DATASET_PATH:-/root/autodl-tmp/magicoder-train.jsonl}"
HUMANEVAL_PATH="${HUMANEVAL_PATH:-/root/autodl-tmp/humaneval-test.parquet}"
MBPP_DATASET_PATH="${MBPP_DATASET_PATH:-}"
RUN_ROOT="${RUN_ROOT:-runs/meta-llama-Llama-3.1-8B-Instruct/code/lora/magicoder-chat-50k-seq4096-gbs32-lr2e-5-r16-a32-3ep/seed42}"
SEED="${SEED:-42}"
EVAL_REQUEST_BATCH_SIZE="${EVAL_REQUEST_BATCH_SIZE:-8}"

cd "$REPO_DIR"
mkdir -p "$RUN_ROOT/eval"

common_train_args=(
  --base_model "$BASE_MODEL"
  --task code
  --peft_method lora
  --dataset_path "$DATASET_PATH"
  --output_dir "$RUN_ROOT"
  --sft_format chat
  --max_train_samples 50000
  --dataset_seed "$SEED"
  --num_train_epochs 3
  --max_seq_len 4096
  --per_device_train_batch_size 1
  --global_train_batch_size 32
  --lr 2e-5
  --lr_scheduler_type cosine
  --warmup_ratio 0.05
  --weight_decay 0.0
  --grad_clip 1.0
  --adam_beta1 0.9
  --adam_beta2 0.999
  --r 16
  --lora_alpha 32
  --lora_dropout 0.05
  --target_modules all
  --logging_steps 25
  --save_strategy epoch
  --save_total_limit 4
  --bf16
  --gradient_checkpointing
  --seed "$SEED"
)

run_train_to_epoch() {
  local epoch="$1"
  local resume_checkpoint="${2:-}"
  local args=("${common_train_args[@]}" --stop_at_epoch "$epoch")
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

latest_checkpoint() {
  find "$RUN_ROOT" -maxdepth 1 -type d -name 'checkpoint-*' -print | sort -V | tail -n 1
}

run_humaneval() {
  local epoch="$1"
  local adapter_dir="$2"
  PYTHONPATH=src python -m finetune.eval.eval_humaneval \
    --base_model "$BASE_MODEL" \
    --adapter_dir "$adapter_dir" \
    --dataset_path "$HUMANEVAL_PATH" \
    --output_dir "$RUN_ROOT/eval/epoch-${epoch}/humaneval" \
    --prompt_style chat \
    --use_vllm \
    --tensor_parallel_size 1 \
    --vllm_max_model_len 4096 \
    --vllm_attention_backend FLASH_ATTN \
    --vllm_disable_flashinfer_sampler \
    --vllm_request_batch_size "$EVAL_REQUEST_BATCH_SIZE" \
    --max_new_tokens 512 \
    --seed "$SEED"
}

run_mbpp() {
  local epoch="$1"
  local adapter_dir="$2"
  local dataset_args=()
  if [[ -n "$MBPP_DATASET_PATH" ]]; then
    dataset_args+=(--dataset_path "$MBPP_DATASET_PATH")
  fi
  PYTHONPATH=src python -m finetune.eval.eval_mbpp \
    --base_model "$BASE_MODEL" \
    --adapter_dir "$adapter_dir" \
    --output_dir "$RUN_ROOT/eval/epoch-${epoch}/mbpp" \
    --prompt_style chat \
    --use_vllm \
    --tensor_parallel_size 1 \
    --vllm_max_model_len 4096 \
    --vllm_attention_backend FLASH_ATTN \
    --vllm_disable_flashinfer_sampler \
    --vllm_request_batch_size "$EVAL_REQUEST_BATCH_SIZE" \
    --max_new_tokens 512 \
    --seed "$SEED" \
    "${dataset_args[@]}"
}

resume_checkpoint=""
for epoch in 1 2 3; do
  echo "[Stage] Training through epoch ${epoch}/3"
  run_train_to_epoch "$epoch" "$resume_checkpoint"
  resume_checkpoint="$(latest_checkpoint)"
  if [[ -z "$resume_checkpoint" || ! -f "$resume_checkpoint/adapter_model.safetensors" ]]; then
    echo "No valid checkpoint found after epoch ${epoch}: ${resume_checkpoint}" >&2
    exit 1
  fi
  echo "[Stage] Evaluating epoch ${epoch}: ${resume_checkpoint}"
  run_humaneval "$epoch" "$resume_checkpoint"
  run_mbpp "$epoch" "$resume_checkpoint"
done

python - "$RUN_ROOT" <<'PY'
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
rows = []
for epoch in (1, 2, 3):
    eval_dir = run_root / "eval" / f"epoch-{epoch}"
    humaneval = json.loads((eval_dir / "humaneval" / "metrics.json").read_text())
    mbpp = json.loads((eval_dir / "mbpp" / "metrics.json").read_text())
    rows.append({
        "epoch": epoch,
        "humaneval_pass@1": humaneval["pass@1"],
        "humaneval_correct": humaneval["correct"],
        "mbpp_pass@1": mbpp["pass@1"],
        "mbpp_correct": mbpp["correct"],
    })

summary_path = run_root / "eval" / "epoch_summary.json"
summary_path.write_text(json.dumps(rows, indent=2) + "\n")
print("[Done] Epoch evaluation summary:")
for row in rows:
    print(
        f"  epoch={row['epoch']} "
        f"HE={row['humaneval_pass@1']:.4f} ({row['humaneval_correct']}/164) "
        f"MBPP={row['mbpp_pass@1']:.4f} ({row['mbpp_correct']}/257)"
    )
print(f"[Done] {summary_path}")
PY
