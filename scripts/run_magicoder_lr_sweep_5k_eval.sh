#!/usr/bin/env bash
set -euo pipefail

# One-factor LR sweep with final code evaluation. "Batch 32" means global
# batch size 32 (per-device batch 1 with accumulation), which is feasible for
# Llama-3.1-8B at sequence length 4096 on a 32 GiB single GPU.

REPO_DIR="${REPO_DIR:-/root/autodl-tmp/peft-sft-lab}"
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/Llama-3.1-8B-Instruct}"
DATASET_PATH="${DATASET_PATH:-/root/autodl-tmp/magicoder-train.jsonl}"
HUMANEVAL_PATH="${HUMANEVAL_PATH:-/root/autodl-tmp/humaneval-test.parquet}"
MBPP_DATASET_PATH="${MBPP_DATASET_PATH:-}"
RUN_ROOT="${RUN_ROOT:-runs/meta-llama-Llama-3.1-8B-Instruct/code/lora/magicoder-lr-sweep-5k-seq4096-gbs32-all-linear/seed42}"
SEED="${SEED:-42}"
EVAL_REQUEST_BATCH_SIZE="${EVAL_REQUEST_BATCH_SIZE:-8}"

cd "$REPO_DIR"
mkdir -p "$RUN_ROOT"

learning_rates=(2e-5 5e-5 1e-4 2e-4)

run_humaneval() {
  local lr="$1"
  local adapter_dir="$2"
  PYTHONPATH=src python -m finetune.eval.eval_humaneval \
    --base_model "$BASE_MODEL" \
    --adapter_dir "$adapter_dir" \
    --dataset_path "$HUMANEVAL_PATH" \
    --output_dir "$RUN_ROOT/lr-${lr}/eval_humaneval" \
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
  local lr="$1"
  local adapter_dir="$2"
  local dataset_args=()
  if [[ -n "$MBPP_DATASET_PATH" ]]; then
    dataset_args+=(--dataset_path "$MBPP_DATASET_PATH")
  fi
  PYTHONPATH=src python -m finetune.eval.eval_mbpp \
    --base_model "$BASE_MODEL" \
    --adapter_dir "$adapter_dir" \
    --output_dir "$RUN_ROOT/lr-${lr}/eval_mbpp" \
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

for lr in "${learning_rates[@]}"; do
  output_dir="$RUN_ROOT/lr-${lr}"
  echo "[Sweep] Training lr=${lr}"
  PYTHONPATH=src accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    -m finetune.train_sft_peft \
    --base_model "$BASE_MODEL" \
    --task code \
    --peft_method lora \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$output_dir" \
    --sft_format chat \
    --max_train_samples 5000 \
    --dataset_seed "$SEED" \
    --num_train_epochs 1 \
    --max_seq_len 4096 \
    --per_device_train_batch_size 1 \
    --global_train_batch_size 32 \
    --lr "$lr" \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.10 \
    --weight_decay 0.0 \
    --grad_clip 1.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --target_modules all \
    --logging_steps 1 \
    --save_strategy epoch \
    --save_total_limit 1 \
    --bf16 \
    --gradient_checkpointing \
    --seed "$SEED"

  run_humaneval "$lr" "$output_dir"
  run_mbpp "$lr" "$output_dir"
done

python - "$RUN_ROOT" "${learning_rates[@]}" <<'PY'
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
rows = []
for lr in sys.argv[2:]:
    run_dir = run_root / f"lr-{lr}"
    checkpoints = sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.removeprefix("checkpoint-")))
    state = json.loads((checkpoints[-1] / "trainer_state.json").read_text())
    losses = [entry["loss"] for entry in state.get("log_history", []) if "loss" in entry]
    humaneval = json.loads((run_dir / "eval_humaneval" / "metrics.json").read_text())
    mbpp = json.loads((run_dir / "eval_mbpp" / "metrics.json").read_text())
    rows.append({
        "learning_rate": lr,
        "first_loss": losses[0],
        "final_loss": losses[-1],
        "min_loss": min(losses),
        "humaneval_pass@1": humaneval["pass@1"],
        "humaneval_correct": humaneval["correct"],
        "mbpp_pass@1": mbpp["pass@1"],
        "mbpp_correct": mbpp["correct"],
    })

(run_root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
print("[Done] LR sweep summary:")
for row in rows:
    print(
        f"  lr={row['learning_rate']:>5} loss={row['final_loss']:.4f} "
        f"HE={row['humaneval_pass@1']:.4f} ({row['humaneval_correct']}/164) "
        f"MBPP={row['mbpp_pass@1']:.4f} ({row['mbpp_correct']}/257)"
    )
PY
