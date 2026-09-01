#!/usr/bin/env bash
set -euo pipefail

# Qwen2.5-7B-Instruct + Magicoder 5K, one-epoch, three-point LR sweep.
# The training and HumanEval paths both render prompts with the tokenizer's
# built-in chat template. HumanEval uses the OpenCompass-style user message.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/Qwen2.5-7B-Instruct}"
DATASET_PATH="${DATASET_PATH:-/root/autodl-tmp/magicoder-train.jsonl}"
HUMANEVAL_PATH="${HUMANEVAL_PATH:-/root/autodl-tmp/humaneval-test.parquet}"
RUN_ROOT="${RUN_ROOT:-runs/Qwen-Qwen2.5-7B-Instruct/code/lora/magicoder-chat-5k-1ep-seq4096-gbs32-lr-sweep/seed42}"

SEED="${SEED:-42}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
VLLM_REQUEST_BATCH_SIZE="${VLLM_REQUEST_BATCH_SIZE:-32}"
LEARNING_RATES=(5e-6 2e-5 1e-4)

cd "$REPO_DIR"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[ERROR] Missing file: $path" >&2
    exit 1
  fi
}

require_command() {
  local command_name="$1"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "[ERROR] Command not found: $command_name" >&2
    exit 1
  fi
}

require_command python
require_command accelerate
require_file "$BASE_MODEL/config.json"
require_file "$DATASET_PATH"
require_file "$HUMANEVAL_PATH"

EVAL_HELP="$(PYTHONPATH=src python -m finetune.eval.eval_humaneval --help 2>&1)"
if [[ "$EVAL_HELP" != *"--chat_user_prompt_style"* ]] \
  || [[ "$EVAL_HELP" != *"--chat_template_mode"* ]]; then
  echo "[ERROR] eval_humaneval does not support --chat_user_prompt_style." >&2
  echo "        Pull the repository version containing the OpenCompass-style evaluator." >&2
  exit 1
fi

mkdir -p "$RUN_ROOT"

run_humaneval() {
  local label="$1"
  local adapter_dir="${2:-}"
  local eval_dir="$RUN_ROOT/eval/$label"
  local adapter_args=()

  if [[ -f "$eval_dir/metrics.json" ]]; then
    echo "[SKIP] HumanEval already complete: $label"
    return
  fi

  if [[ -n "$adapter_dir" ]]; then
    require_file "$adapter_dir/adapter_model.safetensors"
    require_file "$adapter_dir/adapter_config.json"
    adapter_args=(--adapter_dir "$adapter_dir")
  fi

  echo "[EVAL] $label"
  PYTHONPATH=src python -m finetune.eval.eval_humaneval \
    --base_model "$BASE_MODEL" \
    "${adapter_args[@]}" \
    --dataset_path "$HUMANEVAL_PATH" \
    --split test \
    --output_dir "$eval_dir" \
    --prompt_style chat \
    --chat_user_prompt_style opencompass \
    --chat_template_mode auto \
    --dtype bf16 \
    --use_vllm \
    --tensor_parallel_size 1 \
    --vllm_max_model_len 4096 \
    --vllm_attention_backend FLASH_ATTN \
    --vllm_disable_flashinfer_sampler \
    --vllm_request_batch_size "$VLLM_REQUEST_BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --timeout_s 3.0 \
    --eval_n_workers 4 \
    --seed "$SEED"
}

train_one() {
  local lr="$1"
  local output_dir="$RUN_ROOT/lr-$lr"

  if [[ -f "$output_dir/adapter_model.safetensors" ]] \
    && [[ -f "$output_dir/adapter_config.json" ]]; then
    echo "[SKIP] Adapter already complete: lr=$lr"
    return
  fi

  echo "[TRAIN] lr=$lr -> $output_dir"
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
    --chat_template_mode auto \
    --max_train_samples 5000 \
    --dataset_seed "$SEED" \
    --num_train_epochs 1 \
    --max_seq_len 4096 \
    --per_device_train_batch_size 1 \
    --global_train_batch_size 32 \
    --lr "$lr" \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --weight_decay 0.0 \
    --grad_clip 1.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --target_modules all \
    --logging_steps 5 \
    --save_strategy epoch \
    --save_total_limit 1 \
    --bf16 \
    --gradient_checkpointing \
    --seed "$SEED"
}

print_summary() {
  PYTHONPATH=src python - "$RUN_ROOT" "${LEARNING_RATES[@]}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
learning_rates = sys.argv[2:]
rows = []

base_path = root / "eval" / "base" / "metrics.json"
if base_path.is_file():
    metrics = json.loads(base_path.read_text(encoding="utf-8"))
    rows.append(("base", None, metrics.get("pass@1", 0.0), metrics.get("correct"), metrics.get("total")))

for lr in learning_rates:
    metrics_path = root / "eval" / f"lr-{lr}" / "metrics.json"
    if not metrics_path.is_file():
        continue
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows.append((f"lr-{lr}", lr, metrics.get("pass@1", 0.0), metrics.get("correct"), metrics.get("total")))

print()
print(f"{'experiment':18} {'pass@1':>10} {'correct':>9} {'total':>7}")
print("-" * 48)
for label, _, score, correct, total in rows:
    print(f"{label:18} {score:>10.6f} {str(correct):>9} {str(total):>7}")

trained = [row for row in rows if row[1] is not None]
if trained:
    best = max(trained, key=lambda row: row[2])
    print()
    print(f"Best LR: {best[1]} (HumanEval pass@1={best[2]:.6f}, {best[3]}/{best[4]})")

summary_path = root / "lr_sweep_summary.tsv"
with summary_path.open("w", encoding="utf-8") as handle:
    handle.write("experiment\tlearning_rate\tpass@1\tcorrect\ttotal\n")
    for label, lr, score, correct, total in rows:
        handle.write(f"{label}\t{lr or ''}\t{score}\t{correct}\t{total}\n")
print(f"Summary: {summary_path}")
PY
}

echo "============================================================"
echo "Qwen2.5 Magicoder 5K LR sweep"
echo "Base model:       $BASE_MODEL"
echo "Training dataset: $DATASET_PATH"
echo "HumanEval:        $HUMANEVAL_PATH"
echo "Output root:      $RUN_ROOT"
echo "Learning rates:   ${LEARNING_RATES[*]}"
echo "============================================================"

# Run the unchanged base once so the three fine-tuned models have a directly
# comparable baseline under exactly the same evaluation protocol.
run_humaneval base

for lr in "${LEARNING_RATES[@]}"; do
  train_one "$lr"
  run_humaneval "lr-$lr" "$RUN_ROOT/lr-$lr"
done

print_summary
