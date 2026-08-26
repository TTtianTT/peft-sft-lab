#!/usr/bin/env bash
set -euo pipefail

# Diagnostic overfit test for the code SFT pipeline. This intentionally uses
# effective batch size 1: with only 32 examples, batch 192 would produce about
# 50 optimizer updates across 50 epochs and could give a false negative.

REPO_DIR="${REPO_DIR:-/root/autodl-tmp/peft-sft-lab}"
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/Llama-3.1-8B-Instruct}"
DATASET_PATH="${DATASET_PATH:-/root/autodl-tmp/magicoder-train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/meta-llama-Llama-3.1-8B-Instruct/code/lora/magicoder-overfit-32-chat-50ep-lr2e-4/seed42}"
SEED="${SEED:-42}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
EPOCHS="${EPOCHS:-50}"

cd "$REPO_DIR"

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
  --output_dir "$OUTPUT_DIR" \
  --sft_format chat \
  --max_train_samples 32 \
  --dataset_seed "$SEED" \
  --num_train_epochs "$EPOCHS" \
  --max_seq_len "$MAX_SEQ_LEN" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --lr 2e-4 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.0 \
  --weight_decay 0.0 \
  --grad_clip 1.0 \
  --adam_beta1 0.9 \
  --adam_beta2 0.999 \
  --r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --target_modules all \
  --save_strategy epoch \
  --save_total_limit 2 \
  --bf16 \
  --gradient_checkpointing \
  --seed "$SEED"

# Trainer state is retained in the final checkpoint. Print the loss trajectory
# directly so this diagnostic does not require TensorBoard or external tools.
python - "$OUTPUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
checkpoints = sorted(
    run_dir.glob("checkpoint-*"),
    key=lambda path: int(path.name.removeprefix("checkpoint-")),
)
if not checkpoints:
    raise SystemExit(f"No Trainer checkpoints found under {run_dir}")

state_path = checkpoints[-1] / "trainer_state.json"
state = json.loads(state_path.read_text())
losses = [entry for entry in state.get("log_history", []) if "loss" in entry]
print("[Overfit] final checkpoint:", checkpoints[-1])
print("[Overfit] loss trajectory (step, epoch, loss):")
for entry in losses:
    print(f"  {entry.get('step', '?'):>5}  {entry.get('epoch', 0):>7.3f}  {entry['loss']:.6f}")
if losses:
    print(f"[Overfit] first_loss={losses[0]['loss']:.6f} final_loss={losses[-1]['loss']:.6f}")
PY
