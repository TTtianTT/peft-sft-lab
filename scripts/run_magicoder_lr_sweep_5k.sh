#!/usr/bin/env bash
set -euo pipefail

# Controlled one-factor sweep. Every run receives the same deterministic 5K
# Magicoder subset; learning rate is the only changing training parameter.

REPO_DIR="${REPO_DIR:-/root/autodl-tmp/peft-sft-lab}"
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/Llama-3.1-8B-Instruct}"
DATASET_PATH="${DATASET_PATH:-/root/autodl-tmp/magicoder-train.jsonl}"
RUN_ROOT="${RUN_ROOT:-runs/meta-llama-Llama-3.1-8B-Instruct/code/lora/magicoder-lr-sweep-5k-seq4096-all-linear/seed42}"
SEED="${SEED:-42}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-192}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"

cd "$REPO_DIR"
mkdir -p "$RUN_ROOT"

learning_rates=(5e-6 2e-5 5e-5 1e-4 2e-4)

for lr in "${learning_rates[@]}"; do
  run_name="lr-${lr}"
  output_dir="$RUN_ROOT/$run_name"
  echo "[Sweep] Starting ${run_name} -> ${output_dir}"

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
    --max_seq_len "$MAX_SEQ_LEN" \
    --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
    --global_train_batch_size "$GLOBAL_BATCH_SIZE" \
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
done

python - "$RUN_ROOT" "${learning_rates[@]}" <<'PY'
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
summary = []
for lr in sys.argv[2:]:
    run_dir = run_root / f"lr-{lr}"
    checkpoints = sorted(
        run_dir.glob("checkpoint-*"),
        key=lambda path: int(path.name.removeprefix("checkpoint-")),
    )
    if not checkpoints:
        raise SystemExit(f"No checkpoint found for lr={lr}: {run_dir}")
    state = json.loads((checkpoints[-1] / "trainer_state.json").read_text())
    losses = [
        {"step": row.get("step"), "epoch": row.get("epoch"), "loss": row["loss"], "learning_rate": row.get("learning_rate")}
        for row in state.get("log_history", [])
        if "loss" in row
    ]
    if not losses:
        raise SystemExit(f"No loss logs found for lr={lr}: {checkpoints[-1]}")
    summary.append({
        "learning_rate": lr,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoints[-1]),
        "first_loss": losses[0]["loss"],
        "final_loss": losses[-1]["loss"],
        "min_loss": min(row["loss"] for row in losses),
        "loss_history": losses,
    })

(run_root / "loss_sweep_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("[Done] LR sweep summary:")
for row in summary:
    print(
        f"  lr={row['learning_rate']:>5} "
        f"first={row['first_loss']:.4f} final={row['final_loss']:.4f} min={row['min_loss']:.4f}"
    )
print(f"[Done] detailed trajectories: {run_root / 'loss_sweep_summary.json'}")
PY
